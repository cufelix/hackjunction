"""planner: novel/generic objects retry a failed drop once at the target bin's near rim (grasp.novel_near_rim_retry,
flag off = stock). Known objects never retry. Clearances are unchanged. Run:
python -m unittest aaltonomous.baselines.classical.test_novel_near_rim -v
"""
import unittest

import numpy as np

from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .perception import Detection
from .planner import Planner

GRASP_CENTRE = (0.31, -0.02)    # plans cleanly to the centre bin with the stock drop


def _setup(flag, novel, fail="stock"):
    """fail: which drop-lower targets are forced to fail: "stock" (the stock release xy), "all" (every lower leg)."""
    cfg = load_config()
    cfg["grasp"]["novel_near_rim_retry"] = flag
    sc = Scene.load()
    post, home = hand_postures(cfg), home_arm(cfg)
    pl = Planner(cfg, sc, post, home)
    stock = pl.drop_xy("center")
    orig = pl._cartesian

    def fake(side, p0, p1, q0, name, plan):
        if name == "lower" and (fail == "all" or np.allclose(p1[:2], stock, atol=1e-9)):
            plan.ik[name] = (False, 0.0, 0.0)
            return None
        return orig(side, p0, p1, q0, name, plan)

    pl._cartesian = fake
    top = np.r_[GRASP_CENTRE, sc.counter_z + sc.object_height()]
    det = Detection("red", top, 1000, 1000, "depth")
    det.novel = novel
    plan = pl.plan(det, home, post["open"], bin_name="center", relocate=False)
    return pl, plan


class TestNovelNearRim(unittest.TestCase):
    def test_novel_flag_on_retries_at_the_near_rim(self):
        pl, plan = _setup(True, True)
        self.assertTrue(plan.ok, plan.reason)
        self.assertIn("drop_near_rim", plan.waypoints)
        stock = pl.drop_xy("center")
        near = pl.drop_xy("center", inset=0.0)
        np.testing.assert_allclose(plan.waypoints["drop_near_rim"][:2], near, atol=1e-9)
        self.assertLessEqual(np.linalg.norm(near), np.linalg.norm(stock) + 1e-9)

    def test_known_object_never_retries(self):
        _, plan = _setup(True, False)
        self.assertFalse(plan.ok)
        self.assertNotIn("drop_near_rim", plan.waypoints)
        self.assertNotIn("near-rim", plan.reason)

    def test_flag_off_no_retry(self):
        _, plan = _setup(False, True)
        self.assertFalse(plan.ok)
        self.assertNotIn("drop_near_rim", plan.waypoints)
        self.assertNotIn("near-rim", plan.reason)

    def test_both_attempts_fail_keeps_original_reason(self):
        _, plan = _setup(True, True, fail="all")
        self.assertFalse(plan.ok)
        self.assertIn("near-rim retry:", plan.reason)
        self.assertNotIn("drop_near_rim", plan.waypoints)


if __name__ == "__main__":
    unittest.main()
