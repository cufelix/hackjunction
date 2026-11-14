"""perception.bins_from_image: image-space bin rims / masks (rim_detect), tested on frames rendered from our camera
model with the bins shifted away from the nominal geometry and orange tape on the rims."""
from __future__ import annotations

import copy
import time
import unittest

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import FLAGS, _get, load_config
from .perception import Perception
from .rim_detect import detect_bin_rims

SC, CAL = Scene.load(), Calibration.load()
H_OBJ = SC.object_height()
WALL = float(SC.cfg["bins"]["wall_height"])
HALF = np.asarray(SC.cfg["bins"]["outer_size"], float) / 2
FACE, TAPE, INNER, COUNTER = (5, 190, 100), (7, 175, 205), (20, 40, 120), (12, 35, 175)
OBJ_HSV = {"blue": (115, 180, 200), "yellow": (28, 200, 220), "red": (3, 200, 200)}


def make_perception(**over):
    base = {"perception.hsv_set": "robust_v2", "perception.classify_shape": False,
            "perception.color_gate.enabled": False, "perception.hue_guard.enabled": False}
    base.update(over)
    return Perception(load_config(profile="none", overlay="none", overrides=base), SC, CAL, log=lambda _m: None)


def _proj(p, pts):
    return np.round(p._project(SC.mat_to_pelvis(np.asarray(pts, float)))[:, :2]).astype(np.int32)


def _box_hull(p, cx, cy, hx, hy, z0, z1):
    pts = [[cx + sx * hx, cy + sy * hy, z] for sx in (-1, 1) for sy in (-1, 1) for z in (z0, z1)]
    return cv2.convexHull(_proj(p, pts))


def near_rim_row(p, cx, cy, u=None):
    """Image row of the near wall's counter edge for a bin at mat centre (cx, cy)."""
    return float(p._project(SC.mat_to_pelvis(np.array([[cx, cy - HALF[1], 0.0]])))[0, 1])


def render(p, shift_y, objects=(), tape=True, bins=("left", "center", "right")):
    """HSV frame: counter, three orange bins moved `shift_y` m away, bright tape on the near-wall bottom edge."""
    hsv = np.zeros((480, 640, 3), np.uint8)
    hsv[:] = COUNTER
    for n in bins:
        cx, cy = SC.cfg["bins"]["centers"][n]
        cy += shift_y
        cv2.fillConvexPoly(hsv, _box_hull(p, cx, cy, HALF[0], HALF[1], 0.0, WALL), INNER)
        face = _proj(p, [[cx - HALF[0], cy - HALF[1], 0.0], [cx + HALF[0], cy - HALF[1], 0.0],
                         [cx + HALF[0], cy - HALF[1], WALL], [cx - HALF[0], cy - HALF[1], WALL]])
        cv2.fillConvexPoly(hsv, face, FACE)
        if tape:
            cv2.line(hsv, tuple(int(v) for v in face[0]), tuple(int(v) for v in face[1]), TAPE, 7)
    for o in objects:
        cx, cy, z0, z1, hh, col = o
        cv2.fillConvexPoly(hsv, _box_hull(p, cx, cy, hh, hh, z0, z1), OBJ_HSV[col])
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def front_cube(shift):
    """Blue cube 2 cm in front of the REAL (shifted) centre bin."""
    return (0.45, 0.365 + shift - HALF[1] - 0.02 - 0.0225, 0.0, H_OBJ, 0.0225, "blue")


def inside_cube(shift):
    return (0.45, 0.365 + shift, 0.003, 0.003 + H_OBJ, 0.0225, "red")


class TestRimDetect(unittest.TestCase):
    def test_rim_lines_follow_bins_shifted_5_10_14_cm(self):
        p = make_perception()
        for s in (0.0, 0.05, 0.10, 0.14):
            res = detect_bin_rims(cv2.cvtColor(render(p, s), cv2.COLOR_RGB2HSV), expect_w_px=p._nominal_bin_px()[0][2])
            self.assertEqual(len(res.bins), 3, s)
            for rb, n in zip(sorted(res.bins, key=lambda b: b.u0), ("right", "center", "left")):
                cx, cy = SC.cfg["bins"]["centers"][n]
                want = near_rim_row(p, cx, cy + s)
                got = float(rb.v_at(0.5 * (rb.u0 + rb.u1)))
                self.assertTrue(-3.0 <= got - want <= 6.5, (s, n, got, want))   # line = tape bottom edge

    def test_tape_blobs_are_found_only_as_tape(self):
        p = make_perception()
        hsv = cv2.cvtColor(render(p, 0.10, [front_cube(0.10)]), cv2.COLOR_RGB2HSV)
        res = detect_bin_rims(hsv, expect_w_px=p._nominal_bin_px()[0][2])
        cube_row = near_rim_row(p, 0.45, 0.365 + 0.10) + 25
        self.assertFalse(res.tape[int(cube_row):].any(), "the object must never be tape")


class TestPerceptionIntegration(unittest.TestCase):
    def detect(self, shift, objects, flag=True, **over):
        p = make_perception(**{"perception.bins_from_image": flag, **over})
        dets, rej = p.detect(render(p, shift, objects), None, debug=True)
        return p, dets, rej

    def test_object_in_front_of_shifted_bins_is_kept_object_inside_excluded(self):
        for s in (0.05, 0.10, 0.14):
            p, dets, rej = self.detect(s, [front_cube(s), inside_cube(s)])
            self.assertEqual([d.color for d in dets], ["blue"], (s, [(d.color, d.uv) for d in dets]))
            self.assertTrue(p._image_bin_gate_active)
            row = near_rim_row(p, 0.45, 0.365 + s)
            self.assertGreater(dets[0].uv[1], row)
            self.assertFalse(any(d.color == "red" for d in dets))

    def test_flag_off_loses_the_front_object_the_flag_recovers(self):
        s = 0.14
        _p, off, _ = self.detect(s, [front_cube(s)], flag=False)
        _p, on, _ = self.detect(s, [front_cube(s)], flag=True)
        self.assertEqual([d.color for d in off], [])
        self.assertEqual([d.color for d in on], ["blue"])

    def test_tape_is_never_a_target_and_is_logged(self):
        p, dets, rej = self.detect(0.10, [], **{"perception.reject_log": True})
        self.assertEqual(dets, [])
        p2 = make_perception(**{"perception.bins_from_image": True, "perception.tape_value_cut": False})
        # a lone thin tape strip under the rim (as on the real bins) is dropped even when it is bright and compact
        frame = render(p2, 0.10, tape=True)
        hsv = cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
        row = int(near_rim_row(p2, 0.45, 0.465)) + 14
        cv2.rectangle(hsv, (300, row), (420, row + 8), (7, 175, 205), -1)
        dets, rejected = p2.detect(cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB), None, debug=True)
        self.assertEqual(dets, [])
        self.assertIn("rim_tape", [why for _d, why in rejected])

    def test_trusted_runtime_bins_win_over_the_image(self):
        p = make_perception(**{"perception.bins_from_image": True, "perception.color_gate.live_bins": True})
        p.calibration = type("C", (), {"success": True})()
        p.detect(render(p, 0.10, [front_cube(0.10)]), None)
        self.assertFalse(p._image_bin_gate_active)

    def test_rims_not_found_falls_back_to_the_nominal_path(self):
        p = make_perception(**{"perception.bins_from_image": True})
        blank = cv2.cvtColor(np.full((480, 640, 3), COUNTER, np.uint8), cv2.COLOR_HSV2RGB)
        p.detect(blank, None)
        self.assertFalse(p._image_bin_gate_active)
        self.assertIsNone(p.bin_rims)

    def test_depth_path_uses_the_image_masks_too(self):
        s = 0.14
        p = make_perception(**{"perception.bins_from_image": True})
        rgb = render(p, s, [front_cube(s)])
        self.assertEqual([d.color for d in p.detect(rgb, np.zeros((480, 640), np.uint16))], ["blue"])

    def test_rim_distance_hint_is_off_by_default_and_tracks_the_shift_when_on(self):
        p = make_perception(**{"perception.bins_from_image": True})
        p.detect(render(p, 0.10), None)
        self.assertEqual(p.bin_rim_hint, {})
        p = make_perception(**{"perception.bins_from_image": True,
                               "perception.bins_from_image_params": {"hint": True}})
        p.detect(render(p, 0.10), None)
        self.assertTrue(p.bin_rim_hint)
        for n, (dx, dy) in p.bin_rim_hint.items():
            self.assertAlmostEqual(dx, 0.10, delta=0.02, msg=n)
        self.assertEqual(set(p._bin_centres_mat()), set(SC.bin_names))


class TestFlagsAndTiming(unittest.TestCase):
    def test_registered_default_off_and_overlay_adds_exactly_the_flag(self):
        self.assertIn("perception.bins_from_image", FLAGS)
        a, b = load_config(overlay="finals_r22_gate"), load_config(overlay="finals_r22_bins")
        self.assertEqual({k for k in FLAGS if _get(a, k) != _get(b, k)}, {"perception.bins_from_image"})
        self.assertFalse(_get(load_config(overlay="none"), "perception.bins_from_image"))

    def test_flags_off_detection_is_byte_identical_without_the_new_config(self):
        cfg = load_config(overlay="finals_r22_gate")
        legacy = copy.deepcopy(cfg)
        legacy["perception"].pop("bins_from_image"), legacy["perception"].pop("bins_from_image_params")
        p = make_perception()
        rgb = render(p, 0.0, [front_cube(0.0)])
        depth = np.zeros((480, 640), np.uint16)
        for d in (None, depth):
            a = Perception(cfg, SC, CAL, log=lambda _m: None).detect(rgb, d, debug=True)
            b = Perception(legacy, SC, CAL, log=lambda _m: None).detect(rgb, d, debug=True)
            self.assertEqual([repr(x) for x in a[0]], [repr(x) for x in b[0]])
            self.assertEqual([(repr(x), w) for x, w in a[1]], [(repr(x), w) for x, w in b[1]])

    def test_rim_detection_cost_under_4ms_per_frame(self):
        p = make_perception()
        hsv = cv2.cvtColor(render(p, 0.10, [front_cube(0.10)]), cv2.COLOR_RGB2HSV)
        w = p._nominal_bin_px()[0][2]
        detect_bin_rims(hsv, expect_w_px=w)
        t = []
        for _ in range(30):
            t0 = time.perf_counter()
            detect_bin_rims(hsv, expect_w_px=w)
            t.append(time.perf_counter() - t0)
        self.assertLess(float(np.median(t)) * 1e3, 4.0, f"median {np.median(t) * 1e3:.2f} ms")


if __name__ == "__main__":
    unittest.main()
