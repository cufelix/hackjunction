"""CPU-only tests for EV placement confidence and the per-object motion budget."""
import unittest
from unittest.mock import patch

import numpy as np

from .config import load_config
from .perception import BinCalibration, colour_confidence
from .test_time_triage import det, far_near_dets, make, obs


def cpu_make(*args, **kwargs):
    """Build the scripted policy without requiring the optional third_party URDF."""
    with patch("aaltonomous.baselines.classical.policy.ArmKinematics", return_value=object()):
        return make(*args, **kwargs)


class TestColourConfidence(unittest.TestCase):
    def setUp(self):
        self.ranges = load_config(overlay="none")["perception"]["hsv_robust"]
        self.blob = np.ones((20, 20), np.uint8)

    def score(self, hsv, color):
        image = np.empty((20, 20, 3), np.uint8)
        image[:] = hsv
        return colour_confidence(image, self.blob, self.ranges, color, min_area_px=120)

    def test_clear_orange_object_is_confident(self):
        self.assertGreater(self.score((13, 160, 250), "red"), 0.9)

    def test_dim_orange_tape_like_blob_is_rejected_by_brightness(self):
        self.assertLess(self.score((5, 160, 119), "red"), 0.1)

    def test_purple_center_beats_blue_hue_edge(self):
        purple = self.score((111, 150, 240), "blue")
        blue_edge = self.score((100, 100, 200), "blue")
        self.assertGreater(purple, 0.9)
        self.assertLess(blue_edge, 0.6)

    def test_low_brightness_and_mixed_blob_reduce_confidence(self):
        self.assertLess(self.score((111, 150, 100), "blue"), 0.6)
        image = np.empty((20, 20, 3), np.uint8)
        image[:] = (111, 150, 240)
        image[:10] = (60, 10, 100)
        self.assertLess(colour_confidence(image, self.blob, self.ranges, "blue"), 0.6)

    def test_small_blob_has_low_support(self):
        image = np.empty((20, 20, 3), np.uint8)
        image[:] = (111, 150, 240)
        blob = np.zeros((20, 20), np.uint8)
        blob[:6, :6] = 1
        self.assertLess(colour_confidence(image, blob, self.ranges, "blue", min_area_px=120), 0.6)


class TestPlacementGate(unittest.TestCase):
    def test_new_policy_flags_default_off(self):
        cfg = load_config(overlay="none")
        self.assertFalse(cfg["policy"]["place_gate"])
        self.assertIsNone(cfg["policy"]["object_time_budget_s"])

    def test_low_colour_candidate_is_skipped_for_the_next_object(self):
        logs = []
        red, blue = far_near_dets()
        red.color_conf, blue.color_conf = 0.4, 0.95
        pol = cpu_make({"red": 5.0, "blue": 5.0}, [red, blue], order="largest", place_gate=True)
        pol.log = logs.append

        action = pol.act(obs(0.0))

        self.assertFalse(action["done"])
        self.assertEqual([m.split()[1] for _, m in pol.events if m.startswith("execute")], ["blue"])
        self.assertEqual(len(pol.gate_skipped), 1)
        self.assertEqual(len([line for line in logs if line.startswith("[gate] skip red")]), 1)

    def test_self_calibration_confidence_gates_but_nominal_bins_pass(self):
        d = det("red", 0.30)
        d.color_conf = 0.9
        pol = cpu_make({"red": 5.0}, [d], place_gate=True)
        self.assertTrue(pol._place_gate_allows(d, "left", remember=False))  # nominal bins = 1.0
        pol.calibration = BinCalibration(True, True, 0.4, np.zeros(2), 1000, 0.005, 3.0, "synthetic")
        self.assertFalse(pol._place_gate_allows(d, "left", remember=False))

    def test_mapping_confidence_is_part_of_bin_confidence(self):
        d = det("red", 0.30)
        d.color_conf = 0.9
        pol = cpu_make({"red": 5.0}, [d], place_gate=True)
        pol.mapping_conf = {"red": 0.49}
        self.assertFalse(pol._place_gate_allows(d, "left", remember=False))


class TestObjectTimeBudget(unittest.TestCase):
    def test_second_attempt_that_would_exceed_budget_is_abandoned(self):
        red, blue = far_near_dets()
        pol = cpu_make({"red": 20.0, "blue": 5.0}, [red, blue], order="largest", object_time_budget_s=35.0)
        pol.act(obs(0.0))                       # red uses 20 s of its cumulative budget
        pol.state, pol.queue_a, pol.queue_h = "look", None, None

        action = pol.act(obs(20.0))             # another 20 s would exceed 35; blue is tried instead

        self.assertFalse(action["done"])
        executed = [m.split()[1] for _, m in pol.events if m.startswith("execute")]
        self.assertEqual(executed, ["red", "blue"])
        self.assertEqual(pol.planner.calls, ["red", "red", "blue"])
        self.assertEqual(len(pol.budget_abandoned), 1)
        self.assertAlmostEqual(pol.object_time_tracks[0]["spent_s"], 20.0)

    def test_first_plan_larger_than_budget_is_never_started(self):
        red, blue = far_near_dets()
        pol = cpu_make({"red": 20.0, "blue": 5.0}, [red, blue], order="largest", object_time_budget_s=10.0)
        pol.act(obs(0.0))
        executed = [m.split()[1] for _, m in pol.events if m.startswith("execute")]
        self.assertEqual(executed, ["blue"])
        self.assertEqual(pol.planner.calls, ["red", "blue"])
        self.assertEqual(pol.object_time_tracks[0]["color"], "blue")


if __name__ == "__main__":
    unittest.main()
