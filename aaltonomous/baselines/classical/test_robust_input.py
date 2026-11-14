"""Fast tests for the input-robustness fixes: instruction.max_chars (policy._cap_text), safety.nan_guard
(policy._nan_guard / act), normalize(None). Run:
    PYTHONPATH=. python3 -m unittest aaltonomous.baselines.classical.test_robust_input -v
Flags come from overlays/ab_robust_input.yaml (cand_r17 + both flags on); the flag-off cases use cand_r17.
"""
import time
import unittest

import numpy as np

from .config import hand_postures, load_config
from .normalize import normalize
from .policy import ClassicalPolicy

ON = "ab_robust_input"


def _policy(overlay, logs):
    return ClassicalPolicy(config=load_config(overlay=overlay), log=logs.append)


class TruncationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = []
        cls.pol = _policy(ON, cls.logs)

    def cap(self, text, n):
        self.pol.cfg["instruction"]["max_chars"] = n
        try:
            return self.pol._cap_text(text)
        finally:
            self.pol.cfg["instruction"]["max_chars"] = 2000

    def test_cut_at_last_sentence_boundary(self):
        text = "Put the red objects in the left bin. " * 10
        out = self.cap(text, 100)
        self.assertEqual(out, text[:text[:100].rfind(".") + 1])
        self.assertTrue(out.endswith(".") and len(out) <= 100)

    def test_semicolon_is_a_clause_boundary(self):
        text = "red objects go left; blue objects go right and then some more words follow here"
        out = self.cap(text, 60)
        self.assertEqual(out, text[:text[:60].rfind(";") + 1])

    def test_no_boundary_cuts_at_last_whitespace(self):
        text = "word " * 40
        out = self.cap(text, 22)
        self.assertTrue(out.endswith(" ") and len(out) <= 22 and text.startswith(out))

    def test_no_whitespace_is_a_hard_cut(self):
        self.assertEqual(self.cap("x" * 30, 10), "x" * 10)

    def test_text_that_fits_is_the_same_object(self):
        text = "red left"
        self.assertIs(self.cap(text, 2000), text)

    def test_truncation_is_logged_once_per_call(self):
        self.logs.clear()
        text = "Put the red objects in the left bin. " * 100   # 3.7k chars
        out = self.cap(text, 2000)
        self.assertEqual(self.logs, [f"[instruction] truncated {len(text)}->{len(out)} chars"])

    def test_non_string_is_passed_through(self):
        self.assertEqual(self.cap(7, 2), 7)

    def test_long_input_resolves_fast_with_the_cap(self):
        # the 50k-char finding: 19 s uncapped; the cap must make the mapping cheap
        text = "Put the red objects in the left bin. " * 1352
        t0 = time.perf_counter()
        mapping = self.pol._instruction_mapping({"instruction": text})
        ms = (time.perf_counter() - t0) * 1e3
        self.assertLess(ms, 200.0, f"{ms:.1f} ms")
        self.assertEqual((mapping or {}).get("red"), "left")   # the capped text keeps its first sentence

    def test_flag_values_are_validated(self):
        for bad in (-1, "5", True, 2.5):
            with self.assertRaises(ValueError, msg=repr(bad)):
                load_config(overrides={"instruction.max_chars": bad})
        for bad in ("yes", 1):
            with self.assertRaises(ValueError, msg=repr(bad)):
                load_config(overrides={"safety.nan_guard": bad})


class NanGuardTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.logs = []
        cls.pol = _policy(ON, cls.logs)
        cls.home = cls.pol.home.copy()
        cls.hand = hand_postures(cls.pol.cfg, cls.pol.spec)["open"].copy()

    def setUp(self):
        self.pol.reset({"instruction": ""})
        self.pol._nan_t = -1e9        # the log is rate-limited to 1 line/s: each test starts with a fresh limiter
        self.logs.clear()

    def obs(self, t=0.0, **kw):
        base = {"t": t, "arm_q": self.home.copy(), "hand_q": self.hand.copy(), "head_rgb": None, "head_depth": None}
        base.update(kw)
        return base

    def test_all_finite_obs_is_untouched(self):
        o = self.obs()
        out, hold = self.pol._nan_guard(o)
        self.assertIs(out, o)
        self.assertIsNone(hold)

    def test_nan_arm_q_holds_last_emitted_action(self):
        first = self.pol.act(self.obs(t=0.0))
        last_arm = first["arm_q_target"][-1]
        arm = self.home.copy()
        arm[3] = np.nan
        out = self.pol.act(self.obs(t=0.033, arm_q=arm))
        self.assertTrue(np.isfinite(out["arm_q_target"]).all() and np.isfinite(out["hand_q_target"]).all())
        np.testing.assert_allclose(out["arm_q_target"][-1], last_arm)
        self.assertFalse(out["done"])
        self.assertTrue(any(m.startswith("[nan_guard]") for m in self.logs))

    def test_first_act_nan_joint_latches_without_inventing_home_target(self):
        measured = self.home + 0.05
        measured[0] = np.nan
        out = self.pol.act(self.obs(arm_q=measured))
        self.assertEqual(out["arm_q_target"].shape, (0, 14))
        self.assertEqual(out["hand_q_target"].shape, (0, 14))
        self.assertTrue(self.pol._encoder_latched)
        trusted = self.pol.act(self.obs(t=0.033))
        self.assertFalse(self.pol._encoder_latched)
        self.assertGreater(len(trusted["arm_q_target"]), 0)

    def test_nan_hand_joint_on_first_act_is_not_replaced_by_open_posture(self):
        hand = self.hand + 0.02
        hand[2] = np.inf
        out = self.pol.act(self.obs(hand_q=hand))
        self.assertEqual(out["arm_q_target"].shape, (0, 14))
        self.assertEqual(out["hand_q_target"].shape, (0, 14))

    def test_pretrust_no_action_chunk_queues_no_safety_target(self):
        from aaltonomous.deploy.runner import split_action
        arm = self.home.copy()
        arm[4] = np.nan
        out = self.pol.act(self.obs(arm_q=arm))
        self.assertEqual(split_action(out), [])

    def test_nan_depth_pixels_become_missing_not_an_exception(self):
        depth = np.full((48, 64), 1.0, np.float32)
        depth[0, :5] = np.nan
        depth[1, 1] = np.inf
        out_obs, hold = self.pol._nan_guard(self.obs(head_depth=depth))
        self.assertIsNone(hold)                                 # depth alone does not skip the planner
        d = out_obs["head_depth"]
        self.assertTrue(np.isfinite(d).all())
        self.assertTrue(np.all(d[0, :5] == 0.0) and d[1, 1] == 0.0 and d[2, 2] == 1.0)
        self.assertTrue(np.isnan(depth[0, 0]))                  # the caller's array is not modified
        act = self.pol.act(self.obs(t=0.0, head_depth=depth))
        self.assertTrue(np.isfinite(act["arm_q_target"]).all())

    def test_nan_guard_flag_is_off_by_default(self):
        self.assertFalse(load_config(overlay="cand_r17")["safety"]["nan_guard"])


class NormalizeNoneTests(unittest.TestCase):
    def test_none_is_empty(self):
        self.assertEqual(normalize(None), "")

    def test_empty_stays_empty(self):
        self.assertEqual(normalize(""), "")


if __name__ == "__main__":
    unittest.main()
