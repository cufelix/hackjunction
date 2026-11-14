"""perception.color_gate.sat_floor (FLAG, default off): blue (purple) blobs with median S < 110 and yellow blobs with
median S < 45 are shadow / sheen and rejected as "color_gate:sat_floor"; red and orange are unchanged. Run:
    python -m unittest aaltonomous.baselines.classical.test_sat_floor -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import Renderer
from .perception import Detection, Perception

SC, CAL = Scene.load(), Calibration.load()
CAM = load_config(overlay="none")["perception"]["camera"]
REGION_XY = np.array([0.45, 0.155])          # same clear counter spot as test_color_gate
H_TOP = SC.object_height()


def _top(mat_xy, h=H_TOP):
    return SC.mat_to_pelvis([float(mat_xy[0]), float(mat_xy[1]), h])


def _perception(sat_floor=True, enabled=True):
    cfg = load_config(overlay="none", overrides={"perception.color_gate.enabled": enabled,
                                                 "perception.color_gate.sat_floor": sat_floor})
    return Perception(cfg, SC, CAL, log=lambda *_: None)


def _det(color, s_med, top=None):
    top = _top(REGION_XY) if top is None else top
    return Detection(color, np.asarray(top, float), 3000, 500, "depth", sat_med=s_med)


class TestSatFloor(unittest.TestCase):
    def setUp(self):
        self.p = _perception(sat_floor=True)

    def test_flag_is_registered_and_off_by_default(self):
        from .config import FLAGS
        self.assertIn("perception.color_gate.sat_floor", FLAGS)
        cfg = load_config(overlay="none")["perception"]["color_gate"]
        self.assertFalse(cfg["sat_floor"])
        self.assertEqual(cfg["sat_floor_params"], {"blue": 110, "yellow": 45})

    def test_blue_low_s_rejected_high_s_kept(self):
        g = self.p.color_gate_cfg
        self.assertEqual(self.p._color_gate_why(_det("blue", 80), g), "color_gate:sat_floor")
        self.assertIsNone(self.p._color_gate_why(_det("blue", 150), g))

    def test_purple_name_matches_blue_floor(self):
        g = self.p.color_gate_cfg
        self.assertEqual(self.p._color_gate_why(_det("purple", 80), g), "color_gate:sat_floor")
        self.assertIsNone(self.p._color_gate_why(_det("purple", 150), g))

    def test_yellow_low_s_rejected_high_s_kept(self):
        g = self.p.color_gate_cfg
        self.assertEqual(self.p._color_gate_why(_det("yellow", 40), g), "color_gate:sat_floor")
        self.assertIsNone(self.p._color_gate_why(_det("yellow", 50), g))

    def test_red_unchanged(self):
        g = self.p.color_gate_cfg
        off = _perception(sat_floor=False)
        self.assertIsNone(self.p._color_gate_why(_det("red", 60), g))
        self.assertEqual(self.p._color_gate_why(_det("red", 60), g), off._color_gate_why(_det("red", 60), g))

    def test_reject_goes_into_rejected_list_with_reason(self):
        rejected = []
        kept = self.p._color_gate([_det("blue", 80), _det("blue", 150)], rejected)
        self.assertEqual(len(kept), 1)
        self.assertEqual([w for _, w in rejected], ["color_gate:sat_floor"])

    def test_flag_off_keeps_low_s_blob(self):
        off = _perception(sat_floor=False)
        self.assertIsNone(off._color_gate_why(_det("blue", 80), off.color_gate_cfg))
        self.assertIsNone(off._color_gate_why(_det("yellow", 40), off.color_gate_cfg))

    def test_no_measured_s_is_not_rejected(self):
        self.assertIsNone(self.p._color_gate_why(_det("blue", None), self.p.color_gate_cfg))

    def test_sim_render_with_flag_on_keeps_all_objects(self):
        objects = [dict(color=c, shape="cube", yaw=0.0, size=SC.size, height=H_TOP,
                        pos=_top(xy, H_TOP / 2)) for c, xy in
                   zip(("red", "yellow", "blue"), ((0.30, 0.13), (0.45, 0.16), (0.60, 0.18)))]
        rgb, depth = Renderer(SC, CAL).render(objects)
        dets, rej = self.p.detect(rgb, depth, debug=True)
        self.assertEqual(sorted(d.color for d in dets), ["blue", "red", "yellow"])
        self.assertTrue(all(d.sat_med is not None and d.sat_med > 150 for d in dets))
        self.assertNotIn("color_gate:sat_floor", [w for _, w in rej])

    def test_flag_off_detect_identical_to_color_gate_only(self):
        objects = [dict(color=c, shape="cube", yaw=0.0, size=SC.size, height=H_TOP,
                        pos=_top(xy, H_TOP / 2)) for c, xy in
                   zip(("red", "yellow", "blue"), ((0.30, 0.13), (0.45, 0.16), (0.60, 0.18)))]
        rgb, depth = Renderer(SC, CAL).render(objects)
        off = _perception(sat_floor=False)
        base = Perception(load_config(overlay="none", overrides={"perception.color_gate.enabled": True}),
                          SC, CAL, log=lambda *_: None)
        a = off.detect(rgb, depth, debug=True)
        b = base.detect(rgb, depth, debug=True)
        self.assertEqual([repr(d) for d in a[0]], [repr(d) for d in b[0]])
        self.assertEqual([(repr(d), w) for d, w in a[1]], [(repr(d), w) for d, w in b[1]])
        self.assertTrue(all(d.sat_med is None for d in a[0]))


if __name__ == "__main__":
    unittest.main()
