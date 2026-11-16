"""perception.bin_image_guard defence in depth: while the image gate_below_bins is active the nominal in_bin reject
(_reject) stays on; without the flag the image gate replaces it. The colour gate's hedged bin rule (_color_gate_bin)
is deliberately NOT re-enabled: its footprint (nominal + 18 cm towards the robot) deleted real cubes in the venue replay. Run:
    python -m unittest aaltonomous.baselines.classical.test_bin_gate_defence -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .perception import Detection, Perception

SC, CAL = Scene.load(), Calibration.load()
BIN = np.asarray(SC.cfg["bins"]["centers"]["center"], float)


def _perc(guard):
    cfg = load_config(overlay="finals_r25_gen", overrides={"perception.bin_image_guard": guard})
    p = Perception(cfg, SC, CAL, log=lambda *_: None)
    p._image_bin_gate_active = True
    return p


def _det(p, xy):
    top = np.array(SC.mat_to_pelvis([float(xy[0]), float(xy[1]), 0.0]), float)
    top[2] = p.scene.counter_z + 0.9 * p.height   # an object-height top face whatever the overlay's height_range
    return Detection("red", top, 5000, 500, "depth")


class TestBinGateDefence(unittest.TestCase):
    def test_overlay_enables_guard_and_wider_first_run_band(self):
        p = _perc(True)
        self.assertTrue(p.bin_image_guard)
        self.assertEqual(p.gate_below_cfg["first_run_top_frac"], 0.55)
        self.assertEqual(Perception(load_config(overlay="none"), SC, CAL, log=lambda *_: None)
                         .gate_below_cfg.get("first_run_top_frac", 0.45), 0.45)

    def test_guard_on_keeps_in_bin_reject_under_image_gate(self):
        p = _perc(True)
        self.assertEqual(p._reject(_det(p, BIN)), "in_bin:center")

    def test_guard_off_image_gate_skips_in_bin_reject(self):
        p = _perc(False)
        self.assertIsNone(p._reject(_det(p, BIN)))

    def test_colour_gate_bin_rule_stays_off_under_image_gate_even_with_guard(self):
        for guard in (True, False):
            p = _perc(guard)
            self.assertIsNone(p._color_gate_bin(_det(p, BIN), p.color_gate_cfg), guard)

    def test_object_clear_of_bins_still_passes_with_guard(self):
        p = _perc(True)
        far = (0.25, 0.10)   # mat front-left, far from every bin
        d = _det(p, far)
        self.assertIsNone(p._reject(d))
        self.assertIsNone(p._color_gate_bin(d, p.color_gate_cfg))


if __name__ == "__main__":
    unittest.main()
