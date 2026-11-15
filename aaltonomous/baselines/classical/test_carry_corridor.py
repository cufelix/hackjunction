"""planner.carry_corridor (real-only): a carry whose straight lift -> above_bin line would clip a foreign bin's outer
footprint goes lift -> via (in front of the bin row, in line with the target bin) -> above_bin. Flag off = stock carry.
Run: python -m unittest aaltonomous.baselines.classical.test_carry_corridor -v
"""
import unittest

import numpy as np

from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .perception import Detection
from .planner import Planner

GRASP_LEFT = (0.31, -0.16)      # straight carry to the left bin clips the centre bin's outer footprint
GRASP_CENTRE = (0.31, -0.02)    # straight carry to the centre bin crosses no foreign bin


def _plan(grasp, bin_name, flag, patch_via_fail=False):
    cfg = load_config()
    cfg["planner"]["carry_corridor"] = flag
    sc = Scene.load()
    post, home = hand_postures(cfg), home_arm(cfg)
    pl = Planner(cfg, sc, post, home)
    if patch_via_fail:
        orig = pl._cartesian

        def fake(side, p0, p1, q0, name, plan):
            if name == "carry_via":
                plan.ik[name] = (False, 0.0, 0.0)
                return None
            return orig(side, p0, p1, q0, name, plan)
        pl._cartesian = fake
    top = np.r_[grasp, sc.counter_z + sc.object_height()]
    plan = pl.plan(Detection("red", top, 1000, 1000, "depth"), home, post["open"], bin_name=bin_name, relocate=False)
    return pl, plan


class TestCarryCorridor(unittest.TestCase):
    def test_flag_on_routes_via_the_front_of_the_bin_row(self):
        pl, plan = _plan(GRASP_LEFT, "left", True)
        self.assertTrue(plan.ok, plan.reason)
        self.assertIn("carry_via", plan.waypoints)
        via = plan.waypoints["carry_via"]
        self.assertLessEqual(via[0], 0.37 + 1e-6)
        self.assertAlmostEqual(via[1], pl.drop_xy("left")[1], places=9)
        np.testing.assert_allclose(via[1], plan.waypoints["drop"][1], atol=1e-9)

    def test_flag_off_is_the_stock_carry(self):
        _, plan = _plan(GRASP_LEFT, "left", False)
        self.assertTrue(plan.ok, plan.reason)
        self.assertNotIn("carry_via", plan.waypoints)

    def test_no_via_when_the_straight_carry_is_clear(self):
        _, plan = _plan(GRASP_CENTRE, "center", True)
        self.assertTrue(plan.ok, plan.reason)
        self.assertNotIn("carry_via", plan.waypoints)

    def test_failed_via_leg_falls_back_to_the_stock_carry(self):
        _, plan = _plan(GRASP_LEFT, "left", True, patch_via_fail=True)
        self.assertTrue(plan.ok, plan.reason)
        self.assertNotIn("carry_via", plan.waypoints)
        self.assertNotIn("carry_via", plan.ik)


if __name__ == "__main__":
    unittest.main()
