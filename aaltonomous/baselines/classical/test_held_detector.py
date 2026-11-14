"""Real-Dex3 held detection from synthetic traces based on REAL_DEX3_REPORT.md."""
import unittest
from types import SimpleNamespace

import numpy as np

from .held import HeldDetector
from .policy import ClassicalPolicy


def trace(side, index_sum, middle_sum, thumb0_native=0.0):
    """(cmd, q) with requested closing-positive two-joint finger errors."""
    cmd = np.zeros(14)
    off = 0 if side == "left" else 7
    sign = -1.0 if side == "left" else 1.0
    cmd[off:off + 7] = -0.4 if side == "left" else 0.4
    pairs = ({"middle": (3, 4), "index": (5, 6)} if side == "left"
             else {"index": (3, 4), "middle": (5, 6)})
    native = np.zeros(14)
    for finger, total in (("index", index_sum), ("middle", middle_sum)):
        native[off + np.asarray(pairs[finger])] = total / (2.0 * sign)
    native[off] = thumb0_native
    return cmd, cmd - native


class TestFingerSumHeldDetector(unittest.TestCase):
    def detector_with_reference(self, side, threshold=0.045):
        det = HeldDetector("finger_sum", threshold)
        det.start(0.0)
        # Real free tracking bias: state is ~0.02 rad more closed per joint, so the
        # closing-positive reference sum is -0.04 rad for each finger.
        for tick in range(1, 31):
            cmd, q = trace(side, -0.04, -0.04, thumb0_native=0.46)
            det.observe(tick / 30.0, cmd, q)
        return det

    def test_left_order_middle_before_index_and_cube_is_held(self):
        det = self.detector_with_reference("left")
        cmd, q = trace("left", 0.19, 0.02, thumb0_native=0.70)
        held, scores = det.held("left", cmd, q)
        self.assertTrue(held)
        self.assertAlmostEqual(scores["index"], 0.23)
        self.assertAlmostEqual(scores["middle"], 0.06)
        self.assertAlmostEqual(scores["finger_sum"], 0.23)

    def test_right_order_index_before_middle_and_cube_is_held(self):
        det = self.detector_with_reference("right")
        cmd, q = trace("right", 0.19, 0.02, thumb0_native=0.70)
        held, scores = det.held("right", cmd, q)
        self.assertTrue(held)
        self.assertAlmostEqual(scores["index"], 0.23)
        self.assertAlmostEqual(scores["middle"], 0.06)
        self.assertAlmostEqual(scores["finger_sum"], 0.23)

    def test_open_and_empty_closed_are_not_held_but_cube_is(self):
        for side in ("left", "right"):
            det = self.detector_with_reference(side)
            # Open/free repeats the reference; empty close is at most +0.03 rad
            # after bias subtraction; the cube is ~0.23 rad after subtraction.
            for sums, want in (((-0.04, -0.04), False), ((-0.01, -0.01), False), ((0.19, 0.02), True)):
                cmd, q = trace(side, *sums, thumb0_native=0.70)
                self.assertEqual(det.held(side, cmd, q)[0], want, (side, sums))

    def test_thumb0_gap_is_excluded(self):
        for side in ("left", "right"):
            det = self.detector_with_reference(side)
            cmd, q = trace(side, -0.01, -0.01, thumb0_native=0.70)
            self.assertFalse(det.held(side, cmd, q)[0])

    def test_threshold_is_configurable(self):
        cmd, q = trace("right", 0.01, -0.04)
        low = self.detector_with_reference("right", threshold=0.045)
        high = self.detector_with_reference("right", threshold=0.055)
        self.assertTrue(low.held("right", cmd, q)[0])   # 0.01 - (-0.04) = 0.05
        self.assertFalse(high.held("right", cmd, q)[0])

    def test_reference_averages_only_the_first_second(self):
        det = HeldDetector("finger_sum", 0.045)
        det.start(10.0)
        for tick in range(1, 31):
            cmd, q = trace("left", -0.02 if tick <= 15 else -0.06, -0.04)
            det.observe(10.0 + tick / 30.0, cmd, q)
        reference = det.reference
        # Mean index sum is -0.04; middle is -0.04. A late outlier is ignored.
        cmd, q = trace("left", 1.0, 1.0)
        det.observe(11.1, cmd, q)
        np.testing.assert_allclose(det.reference, reference)
        cmd, q = trace("left", -0.04, -0.04)
        held, scores = det.held("left", cmd, q)
        self.assertFalse(held)
        self.assertAlmostEqual(scores["index"], 0.0)
        self.assertAlmostEqual(scores["middle"], 0.0)

    def test_finger_sum_waits_at_least_point_eight_seconds(self):
        det = HeldDetector("finger_sum", 0.045)
        self.assertEqual(det.evaluation_tick(close_start=10, close_end=20, hz=30.0), 34)
        stall = HeldDetector("stall", 0.045)
        self.assertEqual(stall.evaluation_tick(close_start=10, close_end=20, hz=30.0), 20)


class TestLegacyStallHeldDetector(unittest.TestCase):
    def test_current_max_absolute_error_rule_is_unchanged(self):
        det = HeldDetector("stall", 0.045)
        cmd, q = np.zeros(14), np.zeros(14)
        q[0] = 0.009999
        self.assertFalse(det.held("left", cmd, q)[0])
        q[0] = 0.010
        self.assertTrue(det.held("left", cmd, q)[0])


class TestPolicyHeldWiring(unittest.TestCase):
    @staticmethod
    def policy_with_detector(side):
        pol = ClassicalPolicy.__new__(ClassicalPolicy)
        pol.held_detector = TestPolicyHeldWiring.detector_with_reference(side)
        # Synthetic trace targets are already comfortably inside these bounds.
        pol._held_hand_lo = np.full(14, -1.0)
        pol._held_hand_hi = np.full(14, 1.0)
        return pol

    @staticmethod
    def detector_with_reference(side):
        det = HeldDetector("finger_sum", 0.045)
        det.start(0.0)
        for tick in range(1, 31):
            cmd, q = trace(side, -0.04, -0.04, thumb0_native=0.46)
            det.observe(tick / 30.0, cmd, q)
        return det

    def test_finger_sum_does_not_override_still_on_table_retry(self):
        pol = self.policy_with_detector("right")
        pol.queue_a = pol.queue_h = None
        # The bias-corrected sum says held (0.01 - reference -0.04 = 0.05),
        # while every legacy tracking error is below 0.010 rad. Retry-vary must
        # retain its own legacy result so the next look can retry an object that
        # is still on the table; finger_sum is support/logging only.
        cmd, q = trace("right", 0.01, 0.01)
        self.assertTrue(pol.held_detector.held("right", cmd, q)[0])
        pol.hand_cmd = cmd
        hist = {"color": "red", "xy": np.array([0.3, 0.1]), "held": None}
        pol.cfg = {"policy": {"retry_assoc_radius": 0.06}}
        pol.vary_hist = [hist]
        plan = SimpleNamespace(side="right", abort=None)
        pol._abort = {"n": 0, "close_end": 0, "plan": plan, "hist": hist}
        pol._abort_watch({"hand_q": q})
        self.assertFalse(hist["held"])
        still_on_table = SimpleNamespace(color="red", top=np.array([0.31, 0.1, 0.0]))
        self.assertEqual(pol._hist_for(still_on_table), [hist])

    def test_finger_sum_uses_last_policy_command_with_safety_position_clamp(self):
        pol = ClassicalPolicy.__new__(ClassicalPolicy)
        pol.hand_cmd = np.array([-2.0, -0.4, 0.5, 2.0])
        spec = SimpleNamespace(hand_lower=np.array([-1.0, -0.5, 0.0, 0.0]),
                               hand_upper=np.array([1.0, 0.5, 1.0, 0.1]))
        # Wide joints use the configured 0.04 rad margin. The 0.1 rad range is
        # capped at a quarter-range margin, exactly as SafetyFilter._shrink.
        safety = SimpleNamespace(hand_pos_margin=0.04)
        pol._held_hand_lo, pol._held_hand_hi = pol._safety_hand_limits(spec, safety)
        pol.held_detector = HeldDetector("finger_sum", 0.045)
        np.testing.assert_allclose(pol._held_command(), [-0.96, -0.4, 0.5, 0.075])
        pol.held_detector = HeldDetector("stall", 0.045)
        np.testing.assert_array_equal(pol._held_command(), pol.hand_cmd)

    def test_grasp_diag_logs_finger_sum_held_value(self):
        for sums, word in (((-0.01, -0.01), "no"), ((0.19, 0.02), "yes")):
            pol = self.policy_with_detector("right")
            pol.queue_a = pol.queue_h = None
            pol.cfg = {"grasp": {"close_posture": "closed"}}
            pol.post = {"closed": np.zeros(14)}
            logs = []
            pol.log = logs.append
            cmd, q = trace("right", *sums, thumb0_native=0.70)
            pol.hand_cmd = cmd
            plan = SimpleNamespace(side="right", posture=None, color="red", target_bin="right")
            samples = {key: (tick, q, cmd) for key, tick in (("close", 40), ("lift", 50))}
            pol._gdiag = {"n": 0, "close_end": 40, "lift_end": 50, "t": 1.0,
                          "plan": plan, "samples": samples}
            pol._grasp_diag({"hand_q": q})
            self.assertEqual(len(logs), 1)
            self.assertIn(f"held={word}", logs[0])
            self.assertIn("finger_sum=", logs[0])

    def test_grasp_diag_headline_matches_close_decision_not_lift(self):
        pol = self.policy_with_detector("right")
        pol.queue_a = pol.queue_h = None
        pol.cfg = {"grasp": {"close_posture": "closed"}}
        pol.post = {"closed": np.zeros(14)}
        logs = []
        pol.log = logs.append
        close_cmd, close_q = trace("right", -0.01, -0.01)
        lift_cmd, lift_q = trace("right", 0.19, 0.02)
        pol.hand_cmd = lift_cmd
        plan = SimpleNamespace(side="right", posture=None, color="red", target_bin="right")
        pol._gdiag = {"n": 0, "close_end": 40, "lift_end": 50, "t": 1.0, "plan": plan,
                      "samples": {"close": (40, close_q, close_cmd), "lift": (50, lift_q, lift_cmd)}}
        pol._grasp_diag({"hand_q": lift_q})
        self.assertIn("held=no", logs[0])
        self.assertIn("sample_held=no", logs[0])
        self.assertIn("sample_held=yes", logs[0])

    def test_grasp_diag_does_not_finish_before_point_eight_second_close_sample(self):
        pol = self.policy_with_detector("right")
        pol.cfg = {"grasp": {"close_posture": "closed"}}
        pol.post = {"closed": np.zeros(14)}
        pol.log = lambda _: None
        pol.queue_a = np.zeros((20, 14))
        plan = SimpleNamespace(side="right", posture=None, color="red", target_bin="right")
        pol._gdiag = {"n": 50, "close_end": 40, "lift_end": 30, "t": 1.0,
                      "plan": plan, "samples": {}}
        cmd, q = trace("right", 0.19, 0.02)
        pol.hand_cmd = cmd
        pol._grasp_diag({"hand_q": q})
        self.assertIsNotNone(pol._gdiag)
        self.assertIn("lift", pol._gdiag["samples"])
        self.assertNotIn("close", pol._gdiag["samples"])


if __name__ == "__main__":
    unittest.main()
