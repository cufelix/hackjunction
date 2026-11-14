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


def _obs(rgb, t=0.0):
    return {"t": t, "arm_q": np.zeros(14), "hand_q": np.zeros(14), "head_rgb": rgb, "head_depth": None}


def _pick(p):
    """A pick ran between looks: the same counter _start() bumps. Arms end back at home, as on the robot."""
    p._scene_epoch += 1


def _look(p, obs):
    """One look attempt: returns True if the guard discarded it. Real detection is stubbed out."""
    p.state = "look"
    calls = []
    p._look = lambda o, **kw: calls.append(1) or ("none", None, 0)
    p._act(obs)
    return not calls


class TestFrozenRgb(unittest.TestCase):
    def test_a_same_rgb_after_pick_is_frozen(self):
        p = _policy()
        rgb = _rgb(1)
        self.assertFalse(_look(p, _obs(rgb)))
        _pick(p)
        self.assertTrue(_look(p, _obs(rgb.copy())))
        self.assertEqual(sum("[frozen_rgb]" in l for l in p.logs), 1)
        self.assertEqual(p.state, "settle")   # failed-look path: home, retry

    def test_b_same_rgb_no_pick_is_normal(self):
        p = _policy()
        rgb = _rgb(1)
        self.assertFalse(_look(p, _obs(rgb)))
        self.assertFalse(_look(p, _obs(rgb.copy())))
        self.assertFalse(any("[frozen_rgb]" in l for l in p.logs))

    def test_c_changing_rgb_is_normal(self):
        p = _policy()
        self.assertFalse(_look(p, _obs(_rgb(1))))
        _pick(p)
        self.assertFalse(_look(p, _obs(_rgb(2))))
        self.assertEqual(p._frozen_looks, 0)

    def test_d_n_consecutive_frozen_looks_stop_the_episode(self):
        p = _policy(**{"policy.frozen_rgb_max_consecutive": 3})
        rgb = _rgb(1)
        _look(p, _obs(rgb))
        _pick(p)
        for i in range(2):
            self.assertTrue(_look(p, _obs(rgb)))
            self.assertNotEqual(p.state, "done")
        p.state = "look"
        p._look = lambda *a, **k: self.fail("must not look")
        act = p._act(_obs(rgb))
        self.assertEqual(p.state, "done")
        self.assertTrue(act["done"])

    def test_d2_fresh_look_resets_the_streak(self):
        p = _policy()
        rgb = _rgb(1)
        _look(p, _obs(rgb))
        _pick(p)
        _look(p, _obs(rgb))
        self.assertEqual(p._frozen_looks, 1)
        _look(p, _obs(_rgb(3)))
        self.assertEqual(p._frozen_looks, 0)

    def test_e_hash_survives_reset(self):
        p = _policy()
        rgb = _rgb(1)
        _look(p, _obs(rgb))
        p.reset()                         # new episode = scene re-placed: an identical frame is frozen
        p.arm_cmd, p.hand_cmd = p.home.copy(), p.post["open"].copy()
        self.assertEqual(p._frozen_looks, 0)
        self.assertTrue(_look(p, _obs(rgb)))

    def test_f_flag_off_is_inert(self):
        p = _policy(guard=False)
        rgb = _rgb(1)
        self.assertFalse(_look(p, _obs(rgb)))
        _pick(p)
        self.assertFalse(_look(p, _obs(rgb)))
        self.assertIsNone(p._rgb_ref)
        self.assertFalse(any("frozen_rgb" in l for l in p.logs))

    def test_g_out_and_back_to_home_is_frozen(self):
        """Felix review b1628fa: arm goes out to pick and returns to home, RGB identical -> frozen.
        Drives the real _start() so the epoch bump in the production pick path is covered."""
        p = _policy()
        rgb = _rgb(1)
        self.assertFalse(_look(p, _obs(rgb)))
        before = p.n_started
        p._has_time = lambda *a, **k: True
        plan = type("P", (), {"waypoints": {"grasp": np.array([0.3, 0.0, 0.1])}, "color": "red",
                              "duration": 5.0, "corrective": False})()
        try:
            p._start(plan, 0.0)
        except Exception:
            pass                          # only the counters matter; motion setup may need a full plan
        self.assertEqual(p.n_started, before + 1)
        p.arm_cmd = p.home.copy()         # back at home, displacement ~0
        self.assertTrue(_look(p, _obs(rgb.copy())))
        self.assertEqual(sum("[frozen_rgb]" in l for l in p.logs), 1)

    def test_never_raises_on_bad_obs(self):
        p = _policy()
        o = _obs("not an image")
        self.assertFalse(_look(p, o))
        self.assertFalse(_look(p, {"t": 0.0, "arm_q": np.zeros(14), "head_rgb": object(), "hand_q": np.zeros(14)}))

    def test_threaded_path_is_guarded_before_launch(self):
        p = _policy(threaded=True)
        rgb = _rgb(1)
        p.state = "look"
        p._look = lambda o, **kw: ("none", None, 0)
        p._act(_obs(rgb))                 # fresh: launches the worker
        for _ in range(200):
            if p._job is None or p._job["result"] is not None:
                break
            import time; time.sleep(0.01)
        p._job = None
        _pick(p)
        p.state = "look"
        p._act(_obs(rgb))                 # frozen: no job started
        self.assertIsNone(p._job)
        self.assertEqual(p.state, "settle")


if __name__ == "__main__":
    unittest.main()
