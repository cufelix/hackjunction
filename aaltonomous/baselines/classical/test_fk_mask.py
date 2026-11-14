"""perception.color_gate.fk_mask (FLAG, default off): the FK-posed arms + hands are masked out of detect()."""
import unittest
from unittest import mock

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import home_arm, load_config
from .mock_world import Renderer
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
H_TOP = SC.object_height()


def _perception(fk=True):
    cfg = load_config(overlay="none", overrides={"perception.color_gate.fk_mask": fk})
    return Perception(cfg, SC, CAL, log=lambda *_: None)


def _scene():
    objs = [dict(color="blue", shape="cube", yaw=0.0, size=SC.size, height=H_TOP,
                 pos=SC.mat_to_pelvis([0.45, 0.16, H_TOP / 2]))]
    return Renderer(SC, CAL).render(objs)


class TestFkMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = home_arm(load_config(profile="sim", overlay="none"))

    def test_sim_home_hands_are_in_view_and_masked(self):
        p = _perception()
        m = p._fk_arm_mask((480, 640), self.home)
        self.assertIsNotNone(m)
        self.assertGreater(int(m.sum()), 500)        # SIM home: hands in the head camera image (Timi / Oliver)
        self.assertLess(m.mean(), 0.6)               # a silhouette, not the whole image

    def test_object_under_the_arm_mask_is_dropped(self):
        rgb, depth = _scene()
        p = _perception()
        base = p.detect(rgb, depth)
        self.assertEqual([d.color for d in base], ["blue"])           # no robot_q: unmasked
        with mock.patch.object(p, "_fk_arm_mask", return_value=np.ones((480, 640), bool)):
            self.assertEqual(p.detect(rgb, depth, robot_q=(self.home, None)), [])

    def test_flag_off_ignores_robot_q_and_never_builds_the_mask(self):
        rgb, depth = _scene()
        p = _perception(fk=False)
        with mock.patch.object(p, "_fk_arm_mask", side_effect=AssertionError("mask built with the flag off")):
            self.assertEqual([d.color for d in p.detect(rgb, depth, robot_q=(self.home, None))], ["blue"])

    def test_mask_failure_is_unmasked_not_a_crash(self):
        p = _perception()
        self.assertIsNone(p._fk_arm_mask((480, 640), np.zeros(3)))    # wrong arm_q size -> logged, None


if __name__ == "__main__":
    unittest.main()
