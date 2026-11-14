"""CPU tests for the flagged per-episode HSV k-means calibration."""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .colour_kmeans import fit_hue_ranges
from .config import load_config
from .mock_world import Renderer
from .perception import Perception


SC, CAL = Scene.load(), Calibration.load()
ROBUST = load_config(overlay="none")["perception"]["hsv_robust"]


def pixels(hues, n=400):
    rng = np.random.default_rng(4)
    out = []
    for h in hues:
        a = np.empty((n, 3), np.uint8)
        a[:, 0] = np.clip(np.rint(rng.normal(h, 1.0, n)), 0, 179)
        a[:, 1] = rng.integers(130, 221, n)
        a[:, 2] = rng.integers(150, 251, n)
        out.append(a)
    return np.concatenate(out)


def hue_inside(ranges, colour, hue):
    return any(lo[0] <= hue <= hi[0] for lo, hi in ranges[colour])


class TestColourKmeans(unittest.TestCase):
    def test_three_clean_clusters_recentre_only_hue_ranges(self):
        fitted, info = fit_hue_ranges(pixels([13, 40, 111]), ROBUST)
        self.assertEqual(info["status"], "adapted")
        np.testing.assert_allclose(info["centres"], [13, 40, 111], atol=1.0)
        for colour, hue in zip(("red", "yellow", "blue"), (13, 40, 111)):
            self.assertTrue(hue_inside(fitted, colour, hue), (colour, fitted[colour]))
            self.assertEqual([(r[0][1:], r[1][1:]) for r in fitted[colour]],
                             [(r[0][1:], r[1][1:]) for r in ROBUST[colour]][:len(fitted[colour])])

    def test_missing_or_close_clusters_fall_back_to_robust(self):
        fitted, info = fit_hue_ranges(pixels([13, 40]), ROBUST)
        self.assertIs(fitted, ROBUST)
        self.assertEqual(info["status"], "fallback")
        fitted, info = fit_hue_ranges(pixels([15, 18, 111]), ROBUST)
        self.assertIs(fitted, ROBUST)
        self.assertIn("too close", info["reason"])

    def test_perception_fits_once_and_resets_each_episode(self):
        cfg = load_config(overlay="none", overrides={"perception.colour_kmeans": True,
                                                       "perception.hsv_set": "robust"})
        h = SC.object_height()
        objects = [dict(color=c, shape="cube", yaw=0.0, size=SC.size, height=h,
                        pos=SC.mat_to_pelvis([x, y, h / 2]))
                   for c, (x, y) in zip(("red", "yellow", "blue"),
                                        ((0.30, 0.10), (0.45, 0.20), (0.34, 0.30)))]
        rgb, depth = Renderer(SC, CAL).render(objects)
        logs = []
        p = Perception(cfg, SC, CAL, log=logs.append)
        self.assertEqual(sorted(d.color for d in p.detect(rgb, depth)), ["blue", "red", "yellow"])
        self.assertTrue(p._colour_kmeans_done)
        self.assertEqual(len([line for line in logs if line.startswith("[colour_kmeans]")]), 1)
        p.detect(rgb, depth)
        self.assertEqual(len([line for line in logs if line.startswith("[colour_kmeans]")]), 1)
        p.reset_calibration()
        self.assertFalse(p._colour_kmeans_done)
        self.assertIs(p.hsv, cfg["perception"]["hsv_robust"])

    def _perc(self, objects):
        cfg = load_config(overlay="none", overrides={"perception.colour_kmeans": True,
                                                       "perception.hsv_set": "robust"})
        rgb, depth = Renderer(SC, CAL).render(objects)
        return Perception(cfg, SC, CAL, log=lambda *_: None), rgb, depth

    def test_retries_when_arm_in_view_and_caps_at_three_attempts(self):
        p, rgb, depth = self._perc([])  # no objects -> fit falls back every time
        p.detect(rgb, depth, arm_clear=False)
        self.assertEqual(p._colour_kmeans_tries, 0)
        for n in (1, 2):
            p.detect(rgb, depth)
            self.assertEqual((p._colour_kmeans_tries, p._colour_kmeans_done), (n, False))
        p.detect(rgb, depth)
        self.assertEqual((p._colour_kmeans_tries, p._colour_kmeans_done), (3, True))
        p.detect(rgb, depth)
        self.assertEqual(p._colour_kmeans_tries, 3)

    def test_no_hsv_robust_key_does_not_raise_when_flags_off(self):
        cfg = load_config(overlay="none")
        del cfg["perception"]["hsv_robust"]
        Perception(cfg, SC, CAL, log=lambda *_: None)

    def test_flag_defaults_off(self):
        cfg = load_config(overlay="none")
        self.assertFalse(cfg["perception"]["colour_kmeans"])
        p = Perception(cfg, SC, CAL)
        self.assertIs(p.hsv, cfg["perception"]["hsv"])


if __name__ == "__main__":
    unittest.main()
