"""calib.plane.apply_height staging can never deadlock: proved open-lift-home -> lift-first-then-open -> rc6 go_to.
Mock world, virtual clock.   python -m unittest aaltonomous.baselines.classical.test_staging_fallback -v
"""
import threading
import time
import unittest

import numpy as np

from .config import load_config
from .mock_world import make_mock_env
from .planner import SafeRoute
from .policy import ClassicalPolicy
from .test_finger_lag import run


def _drive(pol, env, until, max_ticks=900):
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"], "time_limit": 60.0})
    n = 0
    while n < max_ticks and not until(pol):
        pt, ch = getattr(pol, "_pitch_thread", None), pol._jobs.get("counter_height")
        if (pt is not None and pt.is_alive()) or (ch is not None and ch["result"] is None):
            time.sleep(0.002)   # virtual clock: the height fit must commit as it does on a wall clock
            continue
        obs, info = env.step(pol.act(obs))
        n += 1
        if info["episode_over"]:
            break
    return n


def _moving(pol):
    return any("to home" in m for _, m in pol.events)


def _policy(env, overlay="finals_r22"):
    ev = []
    pol = ClassicalPolicy(config=load_config(overlay=overlay), safety=env.filter.cfg, time_limit=60.0,
                          log=lambda m: ev.append(str(m)))
    return pol, ev


class TestStagingNoDeadlock(unittest.TestCase):
    def test_apply_height_on_in_finals(self):
        for ov in ("finals_r21", "finals_r22"):
            self.assertTrue(load_config(overlay=ov)["calib"]["plane"]["apply_height"], ov)

    def test_forced_route_failure_falls_back_to_rc6_within_budget(self):
        env = make_mock_env(seed=0)
        pol, ev = _policy(env)
        pol.planner.safe_home_route = lambda *a, **k: SafeRoute(False, reason="forced")
        _drive(pol, env, _moving)
        self.assertTrue(_moving(pol), "staging deadlocked")
        self.assertTrue(any("[staging] fallback rc6 home route: safe: forced; lift: forced" in m for m in ev), ev)
        t0 = next(t for t, m in pol.events if "pre-motion calibration complete" in m)
        t1 = next(t for t, m in pol.events if "fallback rc6 home route" in m)
        self.assertLessEqual(t1 - t0, 1.5 + 0.3)

    def test_hanging_proof_hits_budget(self):
        env = make_mock_env(seed=0)
        pol, ev = _policy(env)
        gate = threading.Event()
        pol.planner.safe_home_route = lambda *a, **k: gate.wait(30) and SafeRoute(False, reason="late")
        try:
            _drive(pol, env, _moving)
        finally:
            gate.set()
        self.assertTrue(_moving(pol))
        self.assertTrue(any("over 1.5 s budget" in m for m in ev), ev)
        t0 = next(t for t, m in pol.events if "pre-motion calibration complete" in m)
        t1 = next(t for t, m in pol.events if "fallback rc6 home route" in m)
        self.assertLessEqual(t1 - t0, 1.5 + 0.3)

    def test_raising_proof_falls_back(self):
        env = make_mock_env(seed=0)
        pol, ev = _policy(env)

        def boom(*a, **k):
            raise RuntimeError("ik exploded")
        pol.planner.safe_home_route = boom
        _drive(pol, env, _moving)
        self.assertTrue(_moving(pol))
        self.assertTrue(any("RuntimeError: ik exploded" in m and "fallback rc6" in m for m in ev), ev)

    def test_lift_first_proves_where_open_first_does_not(self):
        """Mock start (arm q = 0, hand closed) grazes the centre bin wall while the fingers open at the measured table
        height; lifting first and opening at home is proved by the same keepout check."""
        env = make_mock_env(seed=0)
        pol, _ = _policy(env)
        pl = pol.planner
        pl.scene.counter_z_runtime = 0.0125
        a, h = np.zeros(14), np.zeros(14)
        r1 = pl.safe_home_route(a, h)
        self.assertFalse(r1.ok)
        r2 = pl.safe_home_route(a, h, ("left", "right"), None, False)
        self.assertTrue(r2.ok, r2.reason)
        self.assertEqual([p[0] for p in r2.phases][-1], "open_at_home")

    def test_open_first_default_unchanged_when_proved(self):
        env = make_mock_env(seed=0)
        pol, _ = _policy(env)
        pl = pol.planner
        pl.scene.counter_z_runtime = 0.10
        a = np.array([0, .35, 0, -.5, 0, 0, 0, 0, -.35, 0, -.5, 0, 0, 0], float)
        r = pl.safe_home_route(a, np.zeros(14))
        self.assertTrue(r.ok, r.reason)
        self.assertEqual(r.phases[0][0], "open")


class TestOverlaysPick(unittest.TestCase):
    def test_finals_overlays_pick_and_sort(self):
        for ov in ("finals_r21", "finals_r22"):
            r = run(overlay=ov, time_limit=120.0)
            self.assertGreaterEqual(len(r["starts"]), 1, f"{ov}: no pick started")
            logs = [m for k, m in r["ev"] if k == "log"]
            self.assertFalse(any("holding" in m and "staging" in m for m in logs), ov)
            self.assertTrue(any("to home" in m for m in logs), ov)
            oc = r["env"].backend.outcomes(r["pol"].mapping)
            self.assertGreaterEqual(sum(o != "counter" for o in oc), 1, f"{ov}: nothing sorted {oc}")


if __name__ == "__main__":
    unittest.main()
