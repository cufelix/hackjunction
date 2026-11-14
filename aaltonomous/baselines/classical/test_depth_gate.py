"""perception.require_depth_for_new: no new pick target from a blob without valid depth. Run:
    python -m unittest aaltonomous.baselines.classical.test_depth_gate -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import Renderer
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
OFF = load_config(overlay="none")
ON = load_config(overlay="none", overrides={"perception.require_depth_for_new": True})


def _objs():
    h = SC.object_height()
    return [dict(color="blue", shape="cube", yaw=0.0, size=SC.size, height=h,
                 pos=SC.mat_to_pelvis([0.40, 0.15, h / 2]))]


class TestDepthGate(unittest.TestCase):
    def test_default_off(self):
        self.assertFalse(OFF["perception"]["require_depth_for_new"])

    def test_blob_without_depth_is_no_new_object(self):
        rgb, _ = Renderer(SC, CAL).render(_objs())
        self.assertEqual(len(Perception(OFF, SC, CAL).detect(rgb, None)), 1)       # stock: invented plane target
        self.assertEqual(Perception(ON, SC, CAL).detect(rgb, None), [])
        _, rej = Perception(ON, SC, CAL).detect(rgb, None, debug=True)
        self.assertEqual([w for _, w in rej], ["no_depth"])

    def test_all_zero_depth_map_too(self):
        rgb, depth = Renderer(SC, CAL).render(_objs())
        self.assertEqual(Perception(ON, SC, CAL).detect(rgb, np.zeros_like(depth)), [])

    def test_blob_with_depth_is_unchanged(self):
        rgb, depth = Renderer(SC, CAL).render(_objs())
        a = Perception(OFF, SC, CAL).detect(rgb, depth)
        b = Perception(ON, SC, CAL).detect(rgb, depth)
        self.assertEqual(len(a), 1)
        self.assertEqual([(d.color, d.method, d.top.tolist()) for d in a], [(d.color, d.method, d.top.tolist()) for d in b])
        self.assertEqual(b[0].method, "depth")

    def test_flag_off_identical_without_depth(self):
        rgb, _ = Renderer(SC, CAL).render(_objs())
        a = Perception(OFF, SC, CAL).detect(rgb, None)
        b = Perception(load_config(overlay="none", overrides={"perception.require_depth_for_new": False}), SC, CAL).detect(rgb, None)
        self.assertEqual([d.top.tolist() for d in a], [d.top.tolist() for d in b])


if __name__ == "__main__":
    unittest.main()
