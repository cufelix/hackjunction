"""perception.color_gate.workspace_fail_open and perception.hue_guard.yellow_fix (FLAGS, default off). Run:
    python -m unittest aaltonomous.baselines.classical.test_gate_failopen -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import FLAGS, _get, load_config
from .hue_guard import Cand, judge
from .perception import Detection, Perception

SC, CAL = Scene.load(), Calibration.load()
RANGES = load_config(overlay="none", overrides={"perception.hsv_set": "robust_v2"})["perception"]["hsv_robust_v2"]
WS = "perception.color_gate.workspace_fail_open"
YF = "perception.hue_guard.yellow_fix"


def _perc(fail_open):
    cfg = load_config(overlay="none", overrides={"perception.color_gate.enabled": True, WS: fail_open})
    return Perception(cfg, SC, CAL, log=lambda *_: None)


class TestFlagsRegistered(unittest.TestCase):
    def test_default_off(self):
        cfg = load_config(overlay="none")
        for k in (WS, YF):
            self.assertIn(k, FLAGS)
            self.assertIs(_get(cfg, k), False)


class TestWorkspaceFailOpen(unittest.TestCase):
    def _plane_det(self, p, mat_xy, counter_z):
        """A plane-method detection of an object standing on a counter at counter_z, as seen with the NOMINAL z
        (the ray hits the nominal plane farther out than the object really is)."""
        T = p._T()
        o = T[:3, 3]
        true = SC.mat_to_pelvis([float(mat_xy[0]), float(mat_xy[1]), 0.0])
        true = np.array([true[0], true[1], counter_z + 0.75 * p.height])
        z_top = SC.nominal_counter_z + p.height
        z_aim = z_top - 0.25 * p.height
        s = (z_aim - o[2]) / (true[2] - o[2])
        q = o + s * (true - o)
        return Detection("yellow", np.array([q[0], q[1], z_top]), 3000, 0, "plane")

    def test_off_rejects(self):
        p = _perc(False)
        d = self._plane_det(p, [0.45, 0.155], 0.15)
        self.assertEqual(p._color_gate_workspace(d, p.color_gate_cfg), "color_gate:workspace")

    def test_on_unmeasured_keeps_real_object(self):
        p = _perc(True)
        d = self._plane_det(p, [0.45, 0.155], 0.15)
        p.scene.counter_z_runtime = None
        self.assertIsNone(p._color_gate_workspace(d, p.color_gate_cfg))

    def test_on_unmeasured_still_rejects_far_wall(self):
        p = _perc(True)
        d = self._plane_det(p, [0.45, 0.60], 0.15)                       # far beyond the near bin edge
        self.assertEqual(p._color_gate_workspace(d, p.color_gate_cfg), "color_gate:workspace")

    def test_on_measured_unchanged(self):
        p = _perc(True)
        d = self._plane_det(p, [0.45, 0.155], 0.15)
        p.scene = Scene.load()
        p.scene.counter_z_runtime = 0.15
        self.assertEqual(p._color_gate_workspace(d, p.color_gate_cfg), "color_gate:workspace")


class TestYellowFix(unittest.TestCase):
    def test_off_yellow_hue_labelled_red_is_low_s_red(self):
        v = judge([Cand("red", 40.0, 80.0, 240.0)], RANGES, red_min_s=110.0)
        self.assertEqual(v[0], (None, "low_s_red"))

    def test_on_relabels_yellow(self):
        v = judge([Cand("red", 40.0, 80.0, 240.0)], RANGES, red_min_s=110.0, yellow_fix=True)
        self.assertEqual(v[0], ("yellow", None))

    def test_on_real_low_s_orange_still_abstains(self):
        v = judge([Cand("red", 12.0, 80.0, 200.0)], RANGES, red_min_s=110.0, yellow_fix=True)
        self.assertEqual(v[0], (None, "low_s_red"))

    def test_on_band_hue_still_band(self):
        v = judge([Cand("red", 23.0, 200.0, 200.0)], RANGES, red_min_s=110.0, yellow_fix=True)
        self.assertEqual(v[0], (None, "band"))

    def test_on_good_orange_untouched(self):
        v = judge([Cand("red", 12.0, 190.0, 205.0)], RANGES, red_min_s=110.0, yellow_fix=True)
        self.assertEqual(v[0], ("red", None))


if __name__ == "__main__":
    unittest.main()
