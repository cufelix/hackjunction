"""perception.bin_hull_observed (purple recall): observed bin mass bounds the object mask instead of the nominal hull."""
from __future__ import annotations

import unittest

import cv2
import numpy as np

from .config import FLAGS, _get, load_config
from .test_gatefix import SC, fake_perception, rgb_frame

KEY = "perception.bin_hull_observed"
MARGIN = "perception.bin_hull_observed_params.margin_px"
FOLLOW = "perception.bin_hull_observed_params.gate_follow"


def draw(hsv):
    cv2.rectangle(hsv, (40, 0), (600, 120), (5, 110, 110), -1)        # observed bin / tape mass, bottom edge row 120
    cv2.rectangle(hsv, (280, 230), (329, 269), (115, 180, 200), -1)    # purple cube on the counter, under the nominal hull


class TestBinHullObserved(unittest.TestCase):
    def test_registered_default_off_and_in_the_overlay(self):
        for k in (KEY, MARGIN, FOLLOW):
            self.assertIn(k, FLAGS)
        self.assertFalse(_get(load_config(overlay="none"), KEY))
        self.assertFalse(_get(load_config(overlay="finals_r22"), KEY))
        ab = load_config(overlay="ab_purple_recall")
        self.assertTrue(_get(ab, KEY))
        self.assertEqual(_get(ab, MARGIN), 20)

    def test_flag_off_the_nominal_hull_hides_the_cube(self):
        p = fake_perception()
        self.assertNotIn("blue", [d.color for d in p.detect(rgb_frame(draw), None)])
        self.assertIsNone(p._obs_hull_row)

    def test_flag_on_the_cube_is_found(self):
        p = fake_perception(**{KEY: True})
        self.assertEqual([d.color for d in p.detect(rgb_frame(draw), None)], ["blue"])
        self.assertIsNotNone(p._obs_hull_row)
        self.assertFalse(p._image_bin_gate_active)       # gate_below_bins itself stays off

    def test_margin_keeps_pixels_near_the_bin_masked(self):
        near = lambda hsv: (cv2.rectangle(hsv, (40, 0), (600, 120), (5, 110, 110), -1),
                            cv2.rectangle(hsv, (280, 125), (329, 150), (115, 180, 200), -1))
        blue = lambda m: [d for d in fake_perception(**{KEY: True, MARGIN: m}).detect(rgb_frame(near), None)
                          if d.color == "blue"]
        self.assertEqual(len(blue(3)), 1)
        self.assertEqual(blue(40), [])

    def test_no_bin_mass_in_view_falls_back_to_the_nominal_hull(self):
        p = fake_perception(**{KEY: True})
        blank = rgb_frame(lambda hsv: cv2.rectangle(hsv, (280, 230), (329, 269), (115, 180, 200), -1))
        self.assertNotIn("blue", [d.color for d in p.detect(blank, None)])
        self.assertIsNone(p._obs_hull_row)

    def test_gate_follow_controls_the_colour_gate_bin_rules(self):
        on = fake_perception(**{KEY: True})
        on.detect(rgb_frame(draw), None)
        self.assertTrue(on._bins_from_image())
        off = fake_perception(**{KEY: True, FOLLOW: False})
        off.detect(rgb_frame(draw), None)
        self.assertFalse(off._bins_from_image())

    def test_margin_validated(self):
        for bad in (-1, 61, True, "3"):
            with self.assertRaises(ValueError):
                load_config(overlay="none", overrides={MARGIN: bad})

    def test_invalid_flag_type_rejected(self):
        with self.assertRaises(ValueError):
            load_config(overlay="none", overrides={KEY: "yes"})


if __name__ == "__main__":
    unittest.main()
