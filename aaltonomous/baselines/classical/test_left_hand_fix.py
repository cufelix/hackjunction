"""grasp.left_hand_fix.shift_z: left-arm triangle grasp z shift (sweep flag, default 0 = off).
Run: python -m unittest aaltonomous.baselines.classical.test_left_hand_fix -v
"""
import os
import unittest
from unittest import mock

import numpy as np

from aaltonomous.scene.scene import Scene

from . import policy_server
from .config import OVERLAY_DIR, hand_postures, home_arm, load_config
from .perception import Detection
from .planner import Planner
from .policy_server import ClassicalServerPolicy
from .test_profile_sim_overrides import OverridePolicy, _info

XY = (0.31, 0.10)   # reachable by both arms with the stock config (checked in the cases below)
SWEEP = {"ab_lh_z_m20": -0.02, "ab_lh_z_m10": -0.01, "ab_lh_z_p10": 0.01, "ab_lh_z_p20": 0.02}


def _grasp_z(side, shape, shift, xy=XY):
    cfg = load_config()
    cfg["grasp"]["left_hand_fix"]["shift_z"] = shift
    sc = Scene.load()
    post, home = hand_postures(cfg), home_arm(cfg)
    pl = Planner(cfg, sc, post, home)
    top = np.r_[xy, sc.counter_z + sc.object_height()]
    det = Detection("red", top, 1000, 1000, "depth", shape=shape)
    plan = pl._plan_side(det, side, "center", home, post["open"])
    assert plan.ok, plan.reason
    return float(plan.waypoints["grasp"][2])


class TestLeftHandFix(unittest.TestCase):
    def test_left_triangle_is_raised_by_the_shift(self):
        base = _grasp_z("left", "triangle", 0.0)
        shifted = _grasp_z("left", "triangle", 0.01)
        self.assertAlmostEqual(shifted - base, 0.01, places=6)

    def test_right_arm_is_unchanged(self):
        self.assertAlmostEqual(_grasp_z("right", "triangle", 0.01), _grasp_z("right", "triangle", 0.0), places=6)

    def test_left_cube_is_unchanged(self):
        self.assertAlmostEqual(_grasp_z("left", "cube", 0.01), _grasp_z("left", "cube", 0.0), places=6)


class TestSweepOverlays(unittest.TestCase):
    def _sim_cfg(self, stem):
        with mock.patch.object(policy_server, "ClassicalPolicy", OverridePolicy), \
                mock.patch.object(policy_server, "detect_profile", return_value=_info("sim")):
            srv = ClassicalServerPolicy(overlay=stem)
            for i in range(5):
                srv.act({"t": i})
        return srv.pol.cfg

    def test_every_sweep_overlay_resolves_its_value_in_default_and_sim(self):
        env = mock.patch.dict(os.environ, {}, clear=False)
        env.start()
        self.addCleanup(env.stop)
        os.environ.pop("AALTO_PROFILE", None)
        for stem, value in SWEEP.items():
            path = OVERLAY_DIR / f"{stem}.yaml"
            self.assertTrue(path.exists(), stem)
            self.assertEqual(load_config(overlay=str(path))["grasp"]["left_hand_fix"]["shift_z"], value, stem)
            self.assertEqual(self._sim_cfg(stem)["grasp"]["left_hand_fix"]["shift_z"], value, stem)


if __name__ == "__main__":
    unittest.main()
