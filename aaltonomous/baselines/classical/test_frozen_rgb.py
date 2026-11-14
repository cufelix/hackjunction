"""policy.frozen_rgb_guard: python -m unittest aaltonomous.baselines.classical.test_frozen_rgb -v"""
from __future__ import annotations

import unittest

import numpy as np

from .config import load_config
from .policy import ClassicalPolicy


def _policy(guard=True, threaded=False, **extra):
    over = {"policy.frozen_rgb_guard": guard, **extra}
    logs = []
    p = ClassicalPolicy(config=load_config(overrides=over), log=logs.append, threaded=threaded, time_limit=120.0)
    p.arm_cmd, p.hand_cmd = p.home.copy(), p.post["open"].copy()
    p.state = "look"
    p.logs = logs
    return p


def _rgb(seed):
    return np.random.default_rng(seed).integers(0, 255, (48, 64, 3), dtype=np.uint8)


def _obs(rgb, dq=0.0, t=0.0):
    q = np.zeros(14)
    q[0] = dq
    return {"t": t, "arm_q": q, "hand_q": np.zeros(14), "head_rgb": rgb, "head_depth": None}


def _look(p, obs):
    """One look attempt: returns True if the guard discarded it. Real detection is stubbed out."""
    p.state = "look"
    calls = []
    p._look = lambda o, **kw: calls.append(1) or ("none", None, 0)
    p._act(obs)
    return not calls


class TestFrozenRgb(unittest.TestCase):
    def test_a_same_rgb_arms_moved_is_frozen(self):
        p = _policy()
        rgb = _rgb(1)
        self.assertFalse(_look(p, _obs(rgb, 0.0)))
        self.assertTrue(_look(p, _obs(rgb.copy(), 0.5)))
        self.assertEqual(sum("[frozen_rgb]" in l for l in p.logs), 1)
        self.assertEqual(p.state, "settle")   # failed-look path: home, retry

    def test_b_same_rgb_arms_still_is_normal(self):
        p = _policy()
        rgb = _rgb(1)
        self.assertFalse(_look(p, _obs(rgb, 0.0)))
        self.assertFalse(_look(p, _obs(rgb.copy(), 0.01)))
        self.assertFalse(any("[frozen_rgb]" in l for l in p.logs))

    def test_c_changing_rgb_is_normal(self):
        p = _policy()
        self.assertFalse(_look(p, _obs(_rgb(1), 0.0)))
        self.assertFalse(_look(p, _obs(_rgb(2), 0.9)))
        self.assertEqual(p._frozen_looks, 0)

    def test_d_n_consecutive_frozen_looks_stop_the_episode(self):
        p = _policy(**{"policy.frozen_rgb_max_consecutive": 3})
        rgb = _rgb(1)
        _look(p, _obs(rgb, 0.0))
        for i in range(2):
            self.assertTrue(_look(p, _obs(rgb, 0.5)))
            self.assertNotEqual(p.state, "done")
        p.state = "look"
        p._look = lambda *a, **k: self.fail("must not look")
        act = p._act(_obs(rgb, 0.5))
        self.assertEqual(p.state, "done")
        self.assertTrue(act["done"])

    def test_d2_fresh_look_resets_the_streak(self):
        p = _policy()
        rgb = _rgb(1)
        _look(p, _obs(rgb, 0.0))
        _look(p, _obs(rgb, 0.5))
        self.assertEqual(p._frozen_looks, 1)
        _look(p, _obs(_rgb(3), 0.5))
        self.assertEqual(p._frozen_looks, 0)

    def test_e_hash_survives_reset(self):
        p = _policy()
        rgb = _rgb(1)
        _look(p, _obs(rgb, 0.0))
        p.reset()
        p.arm_cmd, p.hand_cmd = p.home.copy(), p.post["open"].copy()
        self.assertEqual(p._frozen_looks, 0)
        self.assertTrue(_look(p, _obs(rgb, 0.5)))

    def test_f_flag_off_is_inert(self):
        p = _policy(guard=False)
        rgb = _rgb(1)
        self.assertFalse(_look(p, _obs(rgb, 0.0)))
        self.assertFalse(_look(p, _obs(rgb, 0.9)))
        self.assertIsNone(p._rgb_ref)
        self.assertFalse(any("frozen_rgb" in l for l in p.logs))

    def test_never_raises_on_bad_obs(self):
        p = _policy()
        o = _obs("not an image", 0.0)
        o["arm_q"] = None
        self.assertFalse(_look(p, o))
        self.assertFalse(_look(p, {"t": 0.0, "arm_q": np.zeros(14), "head_rgb": object(), "hand_q": np.zeros(14)}))

    def test_threaded_path_is_guarded_before_launch(self):
        p = _policy(threaded=True)
        rgb = _rgb(1)
        p.state = "look"
        p._look = lambda o, **kw: ("none", None, 0)
        p._act(_obs(rgb, 0.0))            # fresh: launches the worker
        for _ in range(200):
            if p._job is None or p._job["result"] is not None:
                break
            import time; time.sleep(0.01)
        p._job = None
        p.state = "look"
        p._act(_obs(rgb, 0.5))            # frozen: no job started
        self.assertIsNone(p._job)
        self.assertEqual(p.state, "settle")


if __name__ == "__main__":
    unittest.main()
