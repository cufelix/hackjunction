"""Synthetic projection tests for calib.fk_xcheck."""
import types
import unittest

import numpy as np

from .fk_xcheck import FkHandXCheck, project_with_correction


class _Calibration:
    def __init__(self):
        K = np.array([[605.0, 0.0, 320.0], [0.0, 605.0, 240.0], [0.0, 0.0, 1.0]])
        # OpenCV optical axes expressed in pelvis coordinates: z forward=x, x right=-y, y down=-z.
        R = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = R, [0.0, 0.0, 0.35]
        self.T = T
        self.cams = {"head": types.SimpleNamespace(K=K, width=640, height=480, depth_scale=0.001,
                                                    T_pelvis_optical=T)}

    def _T(self, cam):
        return self.T


CFG = {
    "pitch_range_deg": 12.0, "yaw_range_deg": 4.0, "fine_step_deg": 0.25,
    "max_v": 105, "max_s": 100, "open_px": 1, "min_area_px": 5, "max_area_px": 500,
    "association_px": 170, "score_truncate_px": 80, "match_px": 5,
    "min_points": 3, "good_points": 6, "min_coverage": 0.5,
    "depth_range_m": [0.15, 2.0], "min_valid_depth_px": 5, "depth_tolerance_m": 0.15,
}


class TestFkXCheck(unittest.TestCase):
    def setUp(self):
        self.cal = _Calibration()
        # Palm/finger-link-like points in front of the head, spread over both image axes.
        self.points = np.array([[x, y, z] for x, y, z in (
            (.70, -.13, .40), (.73, -.08, .33), (.76, -.03, .43), (.80, .02, .36),
            (.84, .07, .45), (.88, .12, .38), (.72, .12, .28), (.77, .07, .23),
            (.82, .02, .27), (.87, -.03, .22), (.79, -.09, .25), (.86, -.14, .30))])
        self.check = FkHandXCheck(dict(CFG), self.cal)

    def _render(self, pitch):
        import cv2
        rgb = np.full((480, 640, 3), 235, np.uint8)
        for u, v, z in project_with_correction(self.points, self.cal, "head", pitch, 0.0):
            if z > 0 and 0 <= u < 640 and 0 <= v < 480:
                cv2.circle(rgb, (round(u), round(v)), 2, (35, 35, 35), -1)
        return rgb

    def test_recovers_known_pitch_errors_within_one_degree(self):
        for pitch in (-9.0, -4.0, 0.0, 3.0):
            with self.subTest(pitch=pitch):
                got = self.check.estimate(self._render(pitch), self.points)
                self.assertTrue(got.visible, got)
                self.assertLessEqual(abs(got.pitch_deg - pitch), 1.0, got)
                self.assertLessEqual(abs(got.yaw_deg), 1.0, got)

    def test_no_hand_is_noop_with_low_confidence(self):
        got = self.check.estimate(np.full((480, 640, 3), 235, np.uint8), self.points)
        self.assertFalse(got.visible)
        self.assertEqual((got.pitch_deg, got.yaw_deg), (0.0, 0.0))
        self.assertLess(got.confidence, 0.1)

    def test_online_disagreement_over_two_degrees_lowers_confidence(self):
        rgb = self._render(3.0)
        standalone = self.check.estimate(rgb, self.points)
        disagreed = self.check.estimate(rgb, self.points, online_pitch_deg=-3.0)
        self.assertGreater(disagreed.disagreement_deg, 2.0)
        self.assertLess(disagreed.confidence, standalone.confidence)


if __name__ == "__main__":
    unittest.main()
