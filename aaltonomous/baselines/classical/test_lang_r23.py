"""policy.retry_repeat_ok / policy.late_regrant (timi/lang-r23, default off). Run from outside the repo:
    PYTHONPATH=<repo> python -m pytest <repo>/aaltonomous/baselines/classical/test_lang_r23.py -q"""
import unittest

from .config import FLAGS, load_config
from .test_retry_vary import V1, det, make, miss, obs

STUCK = {(V1, 90), (V1, -90)}


class TestDefaults(unittest.TestCase):
    def test_flags_registered_and_off(self):
        cfg = load_config()["policy"]
        for k in ("policy.retry_repeat_ok", "policy.late_regrant"):
            self.assertIn(k, FLAGS)
            self.assertIs(FLAGS[k][1], False)
            self.assertIs(cfg[k.split(".")[1]], False)


class TestRepeatOk(unittest.TestCase):
    def _stuck(self, **ov):
        d = det("red", 0.30, shape="cube")
        pol = make([d], infeasible=STUCK, overrides={"policy.max_attempts_vary": 3, **ov})
        miss(pol, pol._plan_pick(d, {}))
        return d, pol

    def test_off_refuses(self):
        d, pol = self._stuck()
        p = pol._plan_pick(d, {})
        self.assertFalse(p.ok)
        self.assertIn("repeats a tried", p.reason)

    def test_on_repeats_the_tried_grasp(self):
        d, pol = self._stuck(**{"policy.retry_repeat_ok": True})
        p = pol._plan_pick(d, {})
        self.assertTrue(p.ok, p.reason)

    def test_on_stops_at_the_attempt_limit(self):
        d, pol = self._stuck(**{"policy.retry_repeat_ok": True})
        for _ in range(2):
            p = pol._plan_pick(d, {})
            self.assertTrue(p.ok, p.reason)
            miss(pol, p)
        self.assertFalse(pol._plan_pick(d, {}).ok)      # 3 tries done = max_attempts_vary


class TestLateRegrant(unittest.TestCase):
    def _exhausted(self, **ov):
        d = det("red", 0.30, shape="cube")
        pol = make([d], overrides={"policy.max_attempts_vary": 3, **ov})
        for _ in range(3):
            miss(pol, pol._plan_pick(d, {}))
        self.assertEqual(pol._choose([d], obs(10.0), verify=False)[0], "none")
        return d, pol

    def test_off_gives_up(self):
        d, pol = self._exhausted()
        self.assertEqual(pol._choose_retry([d], obs(10.0), False)[0], "none")

    def test_on_regrants_a_tried_object_when_time_is_left(self):
        d, pol = self._exhausted(**{"policy.late_regrant": True})
        res = pol._choose_retry([d], obs(10.0), False)
        self.assertEqual(res[0], "plan")
        self.assertEqual(pol._late_rg["n"], 1)

    def test_no_regrant_without_time(self):
        d, pol = self._exhausted(**{"policy.late_regrant": True})
        self.assertEqual(pol._choose_retry([d], obs(100.0), False)[0], "none")   # 20 s left < 25
        self.assertEqual(pol._late_rg["n"], 0)

    def test_regrants_are_capped(self):
        d, pol = self._exhausted(**{"policy.late_regrant": True, "policy.late_regrants_max": 1})
        self.assertEqual(pol._choose_retry([d], obs(10.0), False)[0], "plan")
        for _ in range(3):
            miss(pol, pol._plan_pick(d, {}))
        self.assertEqual(pol._choose_retry([d], obs(10.0), False)[0], "none")

    def test_given_up_object_is_readmitted(self):
        d = det("red", 0.30, shape="cube")
        pol = make([d], overrides={"policy.late_regrant": True})
        pol.failed.append(d.top[:2].copy())                  # `skip ...: infeasible` earlier in the episode
        self.assertEqual(pol._choose([d], obs(10.0), verify=False)[0], "none")
        self.assertEqual(pol._choose_retry([d], obs(10.0), False)[0], "plan")
        self.assertEqual(len(pol.failed), 0)

    def test_untouched_object_is_not_a_regrant_target(self):
        d = det("red", 0.30, shape="cube")
        pol = make([d], overrides={"policy.late_regrant": True})
        pol.mapping = {}                                     # no bin for red: abstain, never attempted
        res = pol._choose_retry([d], obs(10.0), False)
        self.assertEqual(res[0], "none")
        self.assertEqual(pol._late_rg["n"], 0)

    def test_other_colour_history_is_kept(self):
        a, b = det("red", 0.30, shape="cube"), det("blue", 0.32, y=-0.1, shape="cube")
        pol = make([a, b], overrides={"policy.max_attempts_vary": 3, "policy.late_regrant": True})
        for _ in range(3):
            miss(pol, pol._plan_pick(a, {}))
        miss(pol, pol._plan_pick(b, {}))
        pol._choose_retry([a], obs(10.0), False)
        self.assertEqual([h["color"] for h in pol.vary_hist], ["blue"])


if __name__ == "__main__":
    unittest.main()
