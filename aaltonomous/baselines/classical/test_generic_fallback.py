"""Integration tests for the opt-in unknown-object grasp path."""
import unittest
from unittest.mock import patch
from types import SimpleNamespace

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .mock_world import Renderer
from .perception import Detection, Perception
from .planner import Plan, Planner
from .policy import ClassicalPolicy


SC, CAL = Scene.load(), Calibration.load()


def _frame():
    layout = SC.cfg["layout"]
    mat_xy = [float(np.mean(layout["region_x"])), float(np.mean(layout["region_y"]))]
    height = SC.object_height()
    obj = {"color": "red", "shape": "cube", "yaw": 0.37, "size": SC.size, "height": height,
           "pos": SC.mat_to_pelvis([mat_xy[0], mat_xy[1], height / 2.0])}
    return Renderer(SC, CAL).render([obj])


class TestGenericPerception(unittest.TestCase):
    def _detect(self, classifier_result, depth=True):
        cfg = load_config(overrides={"grasp.generic_fallback": True})
        rgb, dep = _frame()
        if not depth:
            dep = np.zeros_like(dep)
        with patch("aaltonomous.baselines.classical.perception.classify_top_face_diag",
                   return_value=classifier_result):
            detections = Perception(cfg, SC, CAL).detect(rgb, dep)
        self.assertEqual(len(detections), 1)
        return detections[0]

    def test_unknown_shape_gets_metric_proposal(self):
        det = self._detect((None, 0.0, {"path": "abstain_unknown"}))
        self.assertTrue(det.generic_required)
        self.assertIsNone(det.shape)
        self.assertEqual(det.shape_path, "generic")
        self.assertGreater(det.generic_confidence, 0.5)
        self.assertAlmostEqual(det.generic_width, SC.size, delta=0.008)
        self.assertTrue(np.isfinite(det.yaw))

    def test_low_confidence_known_shape_uses_generic_path(self):
        det = self._detect(("cube", 0.2, {"path": "metric"}))
        self.assertTrue(det.generic_required)
        self.assertIsNone(det.shape)  # do not apply a low-confidence shape-specific mapping
        self.assertIsNotNone(det.generic_width)

    def test_no_depth_cannot_propose_and_must_remain_an_abstention(self):
        det = self._detect((None, 0.0, {"path": "abstain_plane"}), depth=False)
        self.assertTrue(det.generic_required)
        self.assertIsNone(det.generic_width)

    def test_flag_off_does_not_call_proposer(self):
        rgb, depth = _frame()
        with patch("aaltonomous.baselines.classical.generic_grasp.propose_grasp",
                   side_effect=AssertionError("flag-off path called generic proposer")):
            detections = Perception(load_config(overlay="none"), SC, CAL).detect(rgb, depth)
        self.assertEqual(len(detections), 1)
        self.assertFalse(detections[0].generic_required)


class TestGenericPolicyAndPosture(unittest.TestCase):
    def _policy(self, mapping):
        policy = ClassicalPolicy.__new__(ClassicalPolicy)
        policy.cfg = load_config(overrides={"grasp.generic_fallback": True,
                                            "policy.bin_lookup": "resolve_bin"})
        policy.mapping = mapping
        policy._bin_misses = []
        policy.log = lambda _message: None
        return policy

    @staticmethod
    def _detection(width):
        det = Detection("red", np.array([0.5, 0.0, 0.8]), 500, 400, "depth", yaw=0.2)
        det.generic_required = True
        det.generic_width = width
        det.generic_confidence = 0.9
        return det

    def test_generic_destination_uses_colour_rule_and_abstains_without_one(self):
        det = self._detection(0.04)
        self.assertEqual(self._policy({"red": "left", ("red", "cube"): "right"})._bin_of(det), "left")
        self.assertIsNone(self._policy({("red", "cube"): "right"})._bin_of(det))
        self.assertEqual(self._policy({"default": "center"})._bin_of(det), "center")
        self.assertIsNone(self._policy({"red": "abstain", "default": "center"})._bin_of(det))
        det.generic_width = None
        self.assertIsNone(self._policy({"red": "left"})._bin_of(det))

    def test_posture_is_selected_by_measured_width(self):
        planner = Planner.__new__(Planner)
        planner.cfg = load_config(overrides={"grasp.generic_fallback": True})
        narrow = planner._posture_choice(self._detection(0.030))
        wide = planner._posture_choice(self._detection(0.050))
        self.assertEqual(narrow[0], "closed_demo")
        self.assertEqual(wide[0], planner.cfg["grasp"]["close_posture"])
        self.assertEqual(narrow[4], "generic_width")

    def test_generic_yaw_is_tried_before_legacy_hints(self):
        planner = Planner.__new__(Planner)
        planner.cfg = load_config(overrides={"grasp.generic_fallback": True})
        original_hint = np.array([1.0, 0.0, 0.0])
        planner.kin = SimpleNamespace(hint=original_hint.copy(), w_hint=0.3)
        seen = []

        def plan_side(_det, _side, _bin, _arm, _hand, **_kw):
            seen.append(planner.kin.hint.copy())
            return Plan(True)

        planner._plan_side = plan_side
        planner._finger_err = lambda *_args: 0.0
        planner._plan_side_legacy = lambda *_args, **_kw: self.fail("legacy hints ran before valid generic yaw")
        det = self._detection(0.04)
        result = planner._plan_side_hints(det, "left", "left", np.zeros(14), np.zeros(14))
        self.assertTrue(result.ok)
        finger_axis = np.arctan2(seen[0][1], seen[0][0])
        axis_error = abs(float((finger_axis - det.yaw + np.pi / 2.0) % np.pi - np.pi / 2.0))
        self.assertLess(axis_error, 1e-12)
        np.testing.assert_array_equal(planner.kin.hint, original_hint)
        self.assertEqual(planner.kin.w_hint, 0.3)

    def test_far_yaw_abstains_without_legacy_orientation(self):
        planner = Planner.__new__(Planner)
        planner.cfg = load_config(overrides={"grasp.generic_fallback": True})
        planner.kin = SimpleNamespace(hint=np.array([1.0, 0.0, 0.0]), w_hint=0.3)
        planner._plan_side = lambda *_args, **_kw: self.fail("far generic yaw started IK")
        planner._plan_side_legacy = lambda *_args, **_kw: self.fail("generic proposal used legacy orientation")
        det = self._detection(0.04)
        det.yaw = np.pi / 2.0

        result = planner._plan_side_hints(det, "left", "left", np.zeros(14), np.zeros(14))
        self.assertFalse(result.ok)
        self.assertIn("outside natural range", result.reason)

    def test_missing_or_misaligned_yaw_abstains(self):
        planner = Planner.__new__(Planner)
        planner.cfg = load_config(overrides={"grasp.generic_fallback": True})
        planner.kin = SimpleNamespace(hint=np.array([1.0, 0.0, 0.0]), w_hint=0.3)
        det = self._detection(0.04)
        det.yaw = None
        self.assertFalse(planner._plan_side_generic(det, "left", "left", np.zeros(14), np.zeros(14)).ok)

        det.yaw = 0.2
        planner._plan_side = lambda *_args, **_kw: Plan(True)
        planner._finger_err = lambda *_args: 30.0
        result = planner._plan_side_generic(det, "left", "left", np.zeros(14), np.zeros(14))
        self.assertFalse(result.ok)
        self.assertIn("yaw error", result.reason)

    def test_generic_plan_reuses_safe_trajectory_and_respects_joint_limits(self):
        cfg = load_config(overrides={"grasp.generic_fallback": True})
        post, home = hand_postures(cfg), home_arm(cfg)
        planner = Planner(cfg, SC, post, home)
        layout = SC.cfg["layout"]
        top = SC.mat_to_pelvis([np.mean(layout["region_x"]) + 0.08,
                                np.mean(layout["region_y"]), SC.object_height()])
        det = Detection("red", top, 1000, 1000, "depth", yaw=0.2)
        det.generic_required, det.generic_width, det.generic_confidence = True, 0.030, 0.9

        plan = planner.plan(det, home, post["open"], relocate=False)

        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual(plan.posture[0], "closed_demo")
        self.assertEqual(plan.safety_clips, 0)
        self.assertTrue(np.all(plan.arm >= planner.spec.arm_lower - 1e-9))
        self.assertTrue(np.all(plan.arm <= planner.spec.arm_upper + 1e-9))
        self.assertTrue(np.all(plan.hand >= planner.spec.hand_lower - 1e-9))
        self.assertTrue(np.all(plan.hand <= planner.spec.hand_upper + 1e-9))

    def test_generic_plan_uses_measured_object_height(self):
        cfg = load_config(overrides={"grasp.generic_fallback": True})
        post, home = hand_postures(cfg), home_arm(cfg)
        planner = Planner(cfg, SC, post, home)
        layout = SC.cfg["layout"]
        xy = SC.mat_to_pelvis([np.mean(layout["region_x"]) + 0.08,
                               np.mean(layout["region_y"]), 0.0])[:2]
        zs = []
        for height in (0.045, 0.070):
            det = Detection("red", np.r_[xy, SC.counter_z + height], 1000, 1000, "depth", yaw=0.2)
            det.generic_required, det.generic_width, det.generic_confidence = True, min(height, 0.050), 0.9
            self.assertAlmostEqual(planner._object_height(det), height)
            plan = planner.plan(det, home, post["open"], relocate=False)
            self.assertTrue(plan.ok, (height, plan.reason))
            zs.append(plan.waypoints["grasp"][2])
            self.assertLessEqual(plan.waypoints["grasp"][2], det.top[2] - 0.008 + 1e-9)
        self.assertGreater(zs[1] - zs[0], 0.005)

        short = Detection("red", np.r_[xy, SC.counter_z + 0.025], 1000, 1000, "depth", yaw=0.2)
        short.generic_required, short.generic_width, short.generic_confidence = True, 0.025, 0.9
        plan = planner.plan(short, home, post["open"], relocate=False)
        self.assertFalse(plan.ok)
        self.assertIn("above the object top", plan.reason)

    def test_generic_width_config_is_safety_bounded(self):
        with self.assertRaises(ValueError):
            load_config(overrides={"grasp.generic_max_width": 0.2})


if __name__ == "__main__":
    unittest.main()
