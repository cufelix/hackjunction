"""motion.final_approach_scale + grasp.prefer_object_side (default-off grip flags) + grasp.arm_xy_offset as a FLAG.
Run: python -m unittest aaltonomous.baselines.classical.test_grip -v
"""
import unittest

import numpy as np

from aaltonomous.scene.scene import Scene

from .config import FLAGS, OVERLAY_DIR, effective_flags, hand_postures, home_arm, load_config
from .perception import Detection
from .planner import Planner


def _planner(**over):
    cfg = load_config()
    for k, v in over.items():
        sec, key = k.split("__")
        cfg[sec][key] = v
    sc = Scene.load()
    post, home = hand_postures(cfg), home_arm(cfg)
    return Planner(cfg, sc, post, home), sc, post, home


def _det(sc, xy):
    return Detection("red", np.r_[xy, sc.counter_z + sc.object_height()], 1000, 1000, "depth", shape="cube")


def _plan(pl, sc, post, home, xy, bin_name="center"):
    return pl.plan(_det(sc, xy), home, post["open"], bin_name=bin_name)


class TestFlagsOff(unittest.TestCase):
    def test_defaults_are_off(self):
        cfg = load_config()
        self.assertIsNone(cfg["motion"]["final_approach_scale"])
        self.assertFalse(cfg["grasp"]["prefer_object_side"])
        self.assertEqual(cfg["grasp"]["arm_xy_offset"], {"left": [0.0, 0.0], "right": [0.0, 0.0]})
        for k in ("grasp.arm_xy_offset", "grasp.prefer_object_side", "motion.final_approach_scale"):
            self.assertIn(k, FLAGS)
        self.assertFalse([k for k, _, is_on, _, _ in effective_flags(cfg) if is_on and k in (
            "grasp.arm_xy_offset", "grasp.prefer_object_side", "motion.final_approach_scale")])

    def test_flags_off_identical_plans(self):
        a = _planner()
        b = _planner(grasp__prefer_object_side_margin_m=0.0, motion__final_approach_m=0.09)   # params alone change nothing
        for xy, bn in (((0.31, 0.10), "center"), ((0.31, -0.12), "center"), ((0.30, 0.15), "right"), ((0.30, -0.1), "right")):
            p, q = _plan(*a, xy, bn), _plan(*b, xy, bn)
            self.assertEqual(p.ok, q.ok)
            if p.ok:
                self.assertEqual((p.side, p.phases), (q.side, q.phases))
                np.testing.assert_array_equal(p.arm, q.arm)
                np.testing.assert_array_equal(p.hand, q.hand)


class TestFinalApproach(unittest.TestCase):
    def test_only_descend_changes(self):
        a, sc, post, home = _planner()
        s = _planner(motion__final_approach_scale=0.5)
        xy = (0.31, 0.10)
        p, q = _plan(a, sc, post, home, xy), _plan(*s, xy)
        self.assertTrue(p.ok and q.ok)
        self.assertEqual(p.side, q.side)
        self.assertEqual([n for n, *_ in p.phases], [n for n, *_ in q.phases])
        P = {n: (s0, e0) for n, s0, e0 in p.phases}
        Q = {n: (s0, e0) for n, s0, e0 in q.phases}
        # phases before descend: identical samples
        d0 = P["descend"][0]
        np.testing.assert_array_equal(p.arm[:d0], q.arm[:d0])
        self.assertEqual(Q["descend"][0], d0)
        # descend takes longer, ends at the same joint pose
        lp, lq = P["descend"][1] - d0, Q["descend"][1] - d0
        self.assertGreater(lq, lp)
        np.testing.assert_allclose(p.arm[P["descend"][1] - 1], q.arm[Q["descend"][1] - 1], atol=1e-9)
        # every later phase: same samples, only shifted in time
        for name in ("close", "lift", "carry", "open", "retreat", "home"):
            np.testing.assert_allclose(p.arm[P[name][0]:P[name][1]], q.arm[Q[name][0]:Q[name][1]], atol=1e-9)
            np.testing.assert_allclose(p.hand[P[name][0]:P[name][1]], q.hand[Q[name][0]:Q[name][1]], atol=1e-9)
        # the first part of the descend (above the final segment) is not slowed: its speed profile matches stock
        extra = lq - lp
        dt = float(a.dt)
        self.assertLess(extra * dt, 3.0)
        self.assertLess(len(q.arm) * dt, 60.0)

    def test_scale_validation(self):
        for bad in (0.0, 1.5, -1, True, "x"):
            cfg = load_config()
            cfg["motion"]["final_approach_scale"] = bad
            from .config import check_flag_values
            with self.assertRaises(ValueError):
                check_flag_values(cfg)


class TestPreferObjectSide(unittest.TestCase):
    def _order(self, flag, xy, bin_name):
        pl, sc, post, home = _planner(grasp__prefer_object_side=flag)
        calls = []

        def fake(det, side, *a, **k):
            calls.append(side)
            from .planner import Plan
            return Plan(True, color=det.color, side=side)
        pl._plan_side_hints = fake
        pl._lying_reach_on = lambda det: False
        r = pl._plan_core(_det(sc, xy), home, post["open"], bin_name=bin_name)
        return calls, r

    def test_object_side_first_when_both_arms_can_do_it(self):
        # stubbed planner: both arms always succeed -> the order decides. Right bin, object on the left.
        self.assertEqual(self._order(False, (0.31, 0.12), "right")[0], ["right"])
        calls, r = self._order(True, (0.31, 0.12), "right")
        self.assertEqual(calls, ["left"])
        self.assertEqual(r.side, "left")
        calls, r = self._order(True, (0.31, -0.12), "center")
        self.assertEqual(r.side, "right")

    def test_falls_back_when_object_side_arm_fails(self):
        pl, sc, post, home = _planner(grasp__prefer_object_side=True)
        from .planner import Plan
        pl._plan_side_hints = lambda det, side, *a, **k: Plan(side == "right", color=det.color, side=side, reason="x")
        pl._lying_reach_on = lambda det: False
        r = pl._plan_core(_det(sc, (0.31, 0.12)), home, post["open"], bin_name="right")
        self.assertTrue(r.ok)
        self.assertEqual(r.side, "right")

    def test_left_bin_keeps_left_arm(self):
        a, sc, post, home = _planner()
        b = _planner(grasp__prefer_object_side=True)
        xy = (0.31, -0.12)
        p, q = _plan(a, sc, post, home, xy, "left"), _plan(*b, xy, "left")
        self.assertEqual((p.ok, p.side), (q.ok, q.side))
        if p.ok:
            np.testing.assert_array_equal(p.arm, q.arm)

    def test_inside_margin_and_never_less_feasible(self):
        a, sc, post, home = _planner()
        b = _planner(grasp__prefer_object_side=True)
        p, q = _plan(a, sc, post, home, (0.31, 0.02), "right"), _plan(*b, (0.31, 0.02), "right")
        self.assertEqual(p.side, q.side)
        rng = np.random.default_rng(0)
        for _ in range(3):
            xy = (rng.uniform(0.22, 0.42), rng.uniform(-0.25, 0.25))
            for bn in ("left", "center", "right"):
                p, q = _plan(a, sc, post, home, xy, bn), _plan(*b, xy, bn)
                if p.ok:
                    self.assertTrue(q.ok, (xy, bn, q.reason))


class TestOverlays(unittest.TestCase):
    def test_overlays_load(self):
        for stem, fas, pos in (("ab_grip_slow", 0.5, False), ("ab_grip_side", None, True), ("ab_grip_both", 0.5, True)):
            cfg = load_config(overlay=str(OVERLAY_DIR / f"{stem}.yaml"))
            self.assertEqual(cfg["motion"]["final_approach_scale"], fas)
            self.assertEqual(cfg["grasp"]["prefer_object_side"], pos)
            self.assertEqual(cfg["grasp"]["arm_xy_offset"], {"left": [0.0, -0.01], "right": [0.0, 0.0]})
            self.assertEqual(cfg["grasp"]["left_hand_fix"]["shift_z"], 0.01)


if __name__ == "__main__":
    unittest.main()
