"""perception.online_colour_cal (Timi): hue peaks of the first frame -> shifted HSV ranges. Run:
    python -m unittest aaltonomous/baselines/classical/test_colour_cal.py -v
"""
import copy
import unittest

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .colour_cal import adapt_ranges, find_peaks, hue_histogram
from .config import load_config
from .mock_world import Renderer
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
ROB = {"perception.hsv_set": "robust"}
ON = dict(ROB, **{"perception.online_colour_cal": True})
CFG_ROB, CFG_ON = load_config(overlay="none", overrides=ROB), load_config(overlay="none", overrides=ON)


def hsv2rgb(h, s=200, v=220):
    return cv2.cvtColor(np.array([[[h, s, v]]], np.uint8), cv2.COLOR_HSV2RGB)[0, 0].astype(float)


def scene_with(colors):
    sc = copy.deepcopy(SC)
    sc.cfg["objects"]["colors"] = {k: [int(x) for x in hsv2rgb(*v)] for k, v in colors.items()}
    return sc


def objects(sc, cols):
    h = sc.object_height()
    pos = [(0.30, 0.10), (0.45, 0.20), (0.35, 0.30)]
    return [dict(color=c, shape="cube", yaw=0.1, size=sc.size, height=h, pos=sc.mat_to_pelvis([x, y, h / 2]))
            for c, (x, y) in zip(cols, pos)]


def hist_of(hues, n=300, s=200, v=220):
    hsv = np.zeros((1, n * len(hues), 3), np.uint8)
    for i, h in enumerate(hues):
        hsv[0, i * n:(i + 1) * n] = (h, s, v)
    return hue_histogram(hsv, np.ones(hsv.shape[:2], bool))[0]


class TestColourCal(unittest.TestCase):
    def test_shifted_hues_give_shifted_ranges(self):
        sc = scene_with({"red": (13,), "yellow": (45,), "blue": (111,)})
        rgb, depth = Renderer(sc, CAL).render(objects(sc, ["red", "yellow", "blue"]), np.random.default_rng(0))
        logs = []
        p = Perception(CFG_ON, sc, CAL, log=logs.append)
        self.assertEqual(sorted(d.color for d in p.detect(rgb, depth)), ["blue", "red", "yellow"])
        r = p.colour_cal_info["ranges"]
        self.assertTrue(any("[colour_cal]" in l for l in logs))
        self.assertAlmostEqual(p.colour_cal_info["assigned"]["red"][0], 13, delta=1)
        self.assertAlmostEqual(p.colour_cal_info["assigned"]["yellow"][0], 45, delta=1)
        self.assertTrue(r["red"][0] <= 13 <= r["red"][1] and r["yellow"][0] <= 45 <= r["yellow"][1])
        self.assertEqual(r["red"][1] - r["red"][0], 26)             # width kept ...
        self.assertEqual(r["yellow"][1] - r["yellow"][0], 34)       # ... unless an overlap forced a cut
        self.assertLess(r["red"][1], r["yellow"][0])
        self.assertLessEqual(abs(r["blue"][0] - 100), 8)
        p.reset_calibration()                                        # per episode
        self.assertIs(p.hsv, CFG_ON["perception"]["hsv_robust"])

    def test_no_objects_no_change(self):
        rgb, depth = Renderer(SC, CAL).render([])
        p = Perception(CFG_ON, SC, CAL, log=lambda *_: None)
        self.assertEqual(p.detect(rgb, depth), [])
        self.assertIs(p.hsv, CFG_ON["perception"]["hsv_robust"])
        self.assertEqual(p.colour_cal_info["peaks"], [])

    def test_no_depth_waits_for_a_valid_frame(self):
        rgb, depth = Renderer(SC, CAL).render([])
        p = Perception(CFG_ON, SC, CAL, log=lambda *_: None)
        p.detect(rgb, None)
        self.assertFalse(p._colour_cal_done)

    def test_missing_class_keeps_its_robust_range(self):
        new, info = adapt_ranges(CFG_ROB["perception"]["hsv_robust"], hist_of([12]))
        base = CFG_ROB["perception"]["hsv_robust"]
        self.assertEqual(new["yellow"], base["yellow"])
        self.assertEqual(new["blue"], base["blue"])
        self.assertNotEqual(new["red"], base["red"])

    def test_shift_is_clamped_to_8(self):
        new, info = adapt_ranges(CFG_ROB["perception"]["hsv_robust"], hist_of([60, 125]))   # 60: not yellow-ish enough
        self.assertEqual(info["ranges"]["yellow"], [26, 60])        # centre 35 -> 43 max
        new, info = adapt_ranges(CFG_ROB["perception"]["hsv_robust"], hist_of([175]))
        self.assertGreaterEqual(info["ranges"]["red"][0], -18)

    def test_ranges_never_overlap(self):
        base = CFG_ROB["perception"]["hsv_robust"]
        for hues in ([14, 44], [15, 30], [16, 52], [11, 36, 112], [8, 26]):
            new, info = adapt_ranges(base, hist_of(hues))
            iv = sorted(info["ranges"].values())
            for a, b in zip(iv, iv[1:]):
                self.assertLess(a[1], b[0], (hues, info["ranges"]))

    def test_flags_off_identical(self):
        self.assertFalse(load_config(overlay="none")["perception"]["online_colour_cal"])
        rgb, depth = Renderer(SC, CAL).render(objects(SC, ["red", "yellow", "blue"]), np.random.default_rng(1))
        a = Perception(CFG_ROB, SC, CAL)
        b = Perception(load_config(overlay="none", overrides=ROB), SC, CAL)
        da, db = a.detect(rgb, depth), b.detect(rgb, depth)
        self.assertIs(a.hsv, CFG_ROB["perception"]["hsv_robust"])
        self.assertEqual([(d.color, d.u, d.v) if hasattr(d, "u") else d.color for d in da],
                         [(d.color, d.u, d.v) if hasattr(d, "u") else d.color for d in db])
        self.assertTrue(all(np.array_equal(x.top, y.top) for x, y in zip(da, db)))

    def test_sim_colours_barely_move(self):
        rgb, depth = Renderer(SC, CAL).render(objects(SC, ["red", "yellow", "blue"]), np.random.default_rng(2))
        p = Perception(CFG_ON, SC, CAL, log=lambda *_: None)
        self.assertEqual(sorted(d.color for d in p.detect(rgb, depth)), ["blue", "red", "yellow"])


if __name__ == "__main__":
    unittest.main()
