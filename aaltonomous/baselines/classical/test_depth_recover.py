"""Synthetic RGB-D tests for perception.depth_blob_recover.

Run with:
    python -m unittest aaltonomous.baselines.classical.test_depth_recover -v
"""
import dataclasses
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .depth_recover import _shape_hint, recover_depth_blobs
from .mock_world import Renderer
from .perception import Detection, Perception
from .policy import ClassicalPolicy


class TestDepthBlobRecover(unittest.TestCase):
    def setUp(self):
        self.scene = Scene.load()
        self.scene.counter_z_runtime = None
        self.cal = Calibration.load()
        self.renderer = Renderer(self.scene, self.cal)
        self.empty_rgb, self.empty_depth = self.renderer.render([])

    def _perception(self, enabled=True, **overrides):
        values = {"perception.depth_blob_recover": enabled, "perception.hsv_set": "robust"}
        values.update(overrides)
        cfg = load_config(overlay="none", overrides=values)
        return Perception(cfg, self.scene, self.cal, log=lambda *_: None)

    @staticmethod
    def _rgb(h, s, v=210):
        return cv2.cvtColor(np.uint8([[[h, s, v]]]), cv2.COLOR_HSV2RGB)[0, 0]

    def _frame(self, color, shape, hsv, xy=(0.30, 0.16), height=None, z0=0.0, yaw=0.0):
        height = self.scene.object_height() if height is None else float(height)
        obj = {"color": color, "shape": shape, "yaw": yaw, "size": self.scene.size, "height": height,
               "pos": self.scene.mat_to_pelvis([xy[0], xy[1], z0 + height / 2])}
        rgb, depth = self.renderer.render([obj])
        changed = (depth != self.empty_depth) | np.any(rgb != self.empty_rgb, axis=2)
        self.assertTrue(changed.any(), "synthetic object must be visible")
        rgb[changed] = self._rgb(*hsv)
        return rgb, depth

    def test_low_saturation_purple_cube_recovered(self):
        rgb, depth = self._frame("blue", "cube", (125, 40, 210))
        self.assertEqual(self._perception(False).detect(rgb, depth), [])
        dets = self._perception().detect(rgb, depth)
        self.assertEqual([d.color for d in dets], ["blue"])       # internal blue class = real purple piece
        self.assertEqual(dets[0].method, "depth")
        self.assertGreaterEqual(dets[0].color_conf, 0.45)

    def test_low_saturation_orange_prism_recovered(self):
        rgb, depth = self._frame("red", "triangle", (12, 40, 210), yaw=0.35)
        self.assertEqual(self._perception(False).detect(rgb, depth), [])
        dets = self._perception().detect(rgb, depth)
        self.assertEqual([d.color for d in dets], ["red"])        # internal red class = real orange piece
        self.assertEqual(dets[0].method, "depth")

    def test_flat_tape_below_1p2cm_not_recovered(self):
        rgb, depth = self._frame("red", "cube", (12, 40, 160), height=0.005)
        self.assertEqual(self._perception().detect(rgb, depth), [])

    def test_blob_inside_bin_not_recovered(self):
        centre = self.scene.cfg["bins"]["centers"]["center"]
        floor = float(self.scene.cfg["bins"]["floor_thickness"])
        rgb, depth = self._frame("blue", "cube", (125, 40, 210), xy=centre, z0=floor)
        dets, rejected = self._perception().detect(rgb, depth, debug=True, include_bins=True)
        self.assertEqual(dets, [])
        self.assertFalse(any(d is not None and d.color == "blue" and why.startswith("depth_recover:")
                             for d, why in rejected))

    def test_unknown_colour_abstains(self):
        rgb, depth = self._frame("blue", "cube", (70, 180, 180))
        self.assertEqual(self._perception().detect(rgb, depth), [])

    def test_fk_arm_mask_excludes_depth_blob(self):
        rgb, depth = self._frame("blue", "cube", (125, 40, 210))
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        arm_mask = (depth != self.empty_depth) | np.any(rgb != self.empty_rgb, axis=2)
        accepted, _, _, _ = recover_depth_blobs(self._perception(), rgb, depth, hsv, arm_mask=arm_mask)
        self.assertEqual(accepted, [])

    def test_policy_supplies_joint_state_for_recovery_fk_mask(self):
        policy = object.__new__(ClassicalPolicy)
        policy.cfg = load_config(overlay="none", overrides={"perception.depth_blob_recover": True})
        arm, hand = np.zeros(14), np.ones(14)
        out = policy._fk_kw({"arm_q": arm, "hand_q": hand})
        np.testing.assert_array_equal(out["robot_q"][0], arm)
        np.testing.assert_array_equal(out["robot_q"][1], hand)

    def test_flat_elongated_footprint_can_supply_optional_prism_hint(self):
        component = np.zeros((20, 20), bool)
        for row in range(4, 16):
            component[row, 5:5 + (row - 3) // 2] = True
        heights = np.full(int(component.sum()), 0.045)
        det = Detection("red", np.zeros(3), 0, len(heights), "depth")
        _shape_hint(det, component, heights,
                    {"shape_hint": True, "shape_flatness_m": 0.008, "shape_prism_aspect": 1.35})
        self.assertEqual((det.shape, det.shape_path), ("triangle", "depth_blob_hint"))

    def test_relaxed_recovery_survives_legacy_hue_and_sat_gates(self):
        rgb, depth = self._frame("red", "triangle", (12, 40, 210), yaw=0.35)
        perception = self._perception(**{"perception.hue_guard.enabled": True,
                                         "perception.color_gate.enabled": True,
                                         "perception.color_gate.sat_floor": True})
        self.assertEqual([d.color for d in perception.detect(rgb, depth)], ["red"])

    def test_existing_colour_detection_is_not_duplicated(self):
        rgb, depth = self._frame("yellow", "cube", (28, 180, 210))
        dets = self._perception().detect(rgb, depth)
        self.assertEqual([d.color for d in dets], ["yellow"])

    def test_flag_off_path_does_not_call_recovery_and_matches_explicit_false(self):
        rgb, depth = self._frame("yellow", "cube", (28, 180, 210))
        default = Perception(load_config(overlay="none"), self.scene, self.cal, log=lambda *_: None)
        explicit = Perception(load_config(overlay="none", overrides={"perception.depth_blob_recover": False}),
                              self.scene, self.cal, log=lambda *_: None)
        with mock.patch("aaltonomous.baselines.classical.depth_recover.recover_depth_blobs",
                        side_effect=AssertionError("flag-off path called recovery")) as recover:
            a, b = default.detect(rgb, depth), explicit.detect(rgb, depth)
        recover.assert_not_called()
        self.assertEqual([dataclasses.asdict(d) | {"top": d.top.tolist()} for d in a],
                         [dataclasses.asdict(d) | {"top": d.top.tolist()} for d in b])

    def test_recovery_median_under_5ms(self):
        rgb, depth = self._frame("blue", "cube", (125, 40, 210))
        perception = self._perception()
        hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
        for _ in range(3):
            recover_depth_blobs(perception, rgb, depth, hsv)
        elapsed = []
        for _ in range(10):
            started = time.perf_counter()
            result = recover_depth_blobs(perception, rgb, depth, hsv)
            elapsed.append((time.perf_counter() - started) * 1000.0)
            self.assertEqual(len(result[0]), 1)
        self.assertLess(float(np.median(elapsed)), 5.0, elapsed)


if __name__ == "__main__":
    unittest.main()
