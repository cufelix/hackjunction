"""scene.counter_z_prior: finals_* fall back to table z 0.15 m / nominal pitch without depth; measured depth wins."""
import unittest
from pathlib import Path

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from . import test_pitch_calibration as tpc
from .config import OVERLAY_DIR, load_config
from .perception import Perception

CAL = Calibration.load()
NOM = Scene.load().counter_z
FINALS = ["finals_r20", "finals_r21"]   # the overlays with apply_height: the prior must never shadow a measurement
RGB = np.zeros((480, 640, 3), np.uint8)


def _perc(overlay):
    sc = Scene.load()      # fresh: the prior lives on the Scene instance
    return Perception(load_config(overlay=overlay), sc, CAL, log=lambda *_: None), sc


class TestDepthPrior(unittest.TestCase):
    def test_finals_overlays_found(self):
        self.assertTrue(all((Path(OVERLAY_DIR) / f"{n}.yaml").exists() for n in FINALS))

    def test_no_depth_uses_prior_and_nominal_pitch(self):
        for name in FINALS:
            with self.subTest(name):
                p, sc = _perc(name)
                for _ in range(3):
                    p.detect(RGB, None)
                self.assertAlmostEqual(sc.counter_z, 0.15, places=9)
                self.assertIsNone(p.pitch_T)
                np.testing.assert_array_equal(p._T(), CAL._T(p.cam))   # pitch correction = 0

    def test_measured_depth_overrides_prior(self):
        sc0 = tpc.SC
        sc0.counter_z_runtime, sc0.counter_z_prior = NOM + 0.10, None
        try:
            rgb, depth = tpc._synthetic_table(-5.0)
        finally:
            sc0.counter_z_runtime = None
        for name in FINALS:
            with self.subTest(name):
                p, sc = _perc(name)
                res = p.calibrate_pitch(rgb, depth)
                self.assertTrue(res.success, res)
                self.assertAlmostEqual(res.pitch_deg, -5.0, delta=1.0)
                np.testing.assert_allclose(p._T(), res.T)     # the measured pose is the one used
                for _ in range(3):
                    p.detect(rgb, depth)
                self.assertAlmostEqual(sc.counter_z, NOM + 0.10, delta=0.01)

    def test_submission_without_depth_is_nominal(self):
        p, sc = _perc("submission")
        for _ in range(3):
            p.detect(RGB, None)
        self.assertEqual(sc.counter_z, NOM)
        self.assertIsNone(p.pitch_T)

    def test_prior_validated(self):
        for bad in (-0.01, 0.31, "x", True):
            with self.assertRaises(ValueError):
                load_config(overrides={"scene.counter_z_prior": bad})
        self.assertEqual(load_config(overrides={"scene.counter_z_prior": 0.3})["scene"]["counter_z_prior"], 0.3)


if __name__ == "__main__":
    unittest.main()
