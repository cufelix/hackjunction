"""Real-Dex3 held detection from synthetic traces based on REAL_DEX3_REPORT.md."""
import unittest
from types import SimpleNamespace

import numpy as np

from .held import HeldDetector
from .policy import ClassicalPolicy
from .test_retry_vary import det, make, obs


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
    def test_m3_empty_close_reference_requires_steady_samples_after_point_four_seconds(self):
        det = HeldDetector("finger_sum", 0.045)
        det.require_empty_close_calibration(0.0, steady_samples=3, max_velocity_rad_s=0.08,
                                            hold_after_settle_s=0.4, timeout_s=3.0)
        cmd, q = trace("left", -0.04, -0.04)
        self.assertEqual(det.sample_empty_close(0.1, cmd, q), "pending")
        self.assertEqual(det.sample_empty_close(0.2, cmd, q), "pending")
        self.assertEqual(det.sample_empty_close(0.59, cmd, q), "pending")
        self.assertEqual(det.sample_empty_close(0.61, cmd, q), "pending")
        self.assertEqual(det.sample_empty_close(0.71, cmd, q), "pending")
        self.assertEqual(det.sample_empty_close(0.81, cmd, q), "ready")
        self.assertTrue(det.ready)

    def test_m3_motion_resets_reference_and_invalid_calibration_disables_abort(self):
        det = HeldDetector("finger_sum", 0.045)
        det.require_empty_close_calibration(0.0, steady_samples=2, max_velocity_rad_s=0.08,
                                            hold_after_settle_s=0.4, timeout_s=1.0)
        cmd, q = trace("left", -0.04, -0.04)
        det.sample_empty_close(0.1, cmd, q)
        moving = q.copy()
        moving[3] += 0.2
        det.sample_empty_close(0.2, cmd, moving)
        self.assertFalse(det.ready)
        self.assertEqual(det.sample_empty_close(1.1, cmd, q), "invalid")
        self.assertFalse(det.ready)
        self.assertIn("timed out", det.calibration_reason)

    def test_m3_four_samples_fit_real_chunk_cadence_after_lag(self):
        det = HeldDetector("finger_sum", 0.045)
        det.require_empty_close_calibration(0.0, steady_samples=4, max_velocity_rad_s=0.08,
                                            hold_after_settle_s=0.4, timeout_s=5.0)
        cmd, q = trace("left", -0.04, -0.04)
        # Hand is still moving through 1.65 s (close trajectory + 300--380 ms lag).
        for i, t in enumerate((0.10, 0.43, 0.76, 1.09, 1.42, 1.65)):
            moving = q.copy()
            moving[3:7] += 0.1 * i
            self.assertEqual(det.sample_empty_close(t, cmd, moving), "pending")
        settled = moving
        statuses = [det.sample_empty_close(t, cmd, settled) for t in (1.98, 2.31, 2.64, 2.97, 3.30, 3.63)]
        self.assertEqual(statuses[-1], "ready")
        self.assertTrue(det.ready)

    def test_m3_rejects_nonidentical_index_middle_commands(self):
        det = HeldDetector("finger_sum", 0.045)
        det.require_empty_close_calibration(0.0, steady_samples=2, max_velocity_rad_s=0.08,
                                            hold_after_settle_s=0.4, timeout_s=2.0)
        cmd, q = trace("left", -0.04, -0.04)
        cmd[3] += 0.1
        self.assertEqual(det.sample_empty_close(0.1, cmd, q), "invalid")
        self.assertFalse(det.ready)
        self.assertIn("differ", det.calibration_reason)

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

    def test_max_of_samples_uses_the_existing_threshold(self):
        det = self.detector_with_reference("right")
        empty = trace("right", -0.01, -0.01)
        held = trace("right", 0.19, 0.02)
        verdict, scores = det.held_max("right", [empty, held, empty])
        self.assertTrue(verdict)
        self.assertAlmostEqual(scores["finger_sum"], 0.23)


class TestLegacyStallHeldDetector(unittest.TestCase):
    def test_current_max_absolute_error_rule_is_unchanged(self):
        det = HeldDetector("stall", 0.045)
        cmd, q = np.zeros(14), np.zeros(14)
        q[0] = 0.009999
        self.assertFalse(det.held("left", cmd, q)[0])
        q[0] = 0.010
        self.assertTrue(det.held("left", cmd, q)[0])


class TestEmptyGraspAbort(unittest.TestCase):
    """grasp.held_detector=finger_sum: an empty close is aborted at the close check; unsure never aborts."""

    def started(self, sums=(-0.01, -0.01), detector=None):
        d = det("red", 0.30, shape="cube")
        pol = make([d], overrides={"grasp.held_detector": "finger_sum"})
        pol.held_detector = detector or TestPolicyHeldWiring.detector_with_reference("left")
        pol._held_hand_lo, pol._held_hand_hi = np.full(14, -1.0), np.full(14, 1.0)
        pol.planner.go_to = lambda *a, **k: (np.zeros((5, 14)), np.zeros((5, 14)))
        cmd, q = trace("left", *sums, thumb0_native=0.70)
        pol._start(pol._plan_pick(d, {}), 0.0)                  # 60-tick plan, close check at tick 34
        pol.hand_cmd = cmd
        pol.queue_a, pol.queue_h = np.zeros((60 - 34, 14)), np.zeros((60 - 34, 14))
        return d, pol, q

    def test_empty_close_is_aborted_and_the_retry_varies_the_grasp(self):
        d, pol, q = self.started(sums=(-0.01, -0.01))
        pol._grasp_diag({"hand_q": q})
        self.assertIsNone(pol._gdiag)
        self.assertIs(pol.vary_hist[-1]["held"], False)          # the miss the retry logic reads
        self.assertEqual(pol._attempt_count(d.top[:2], d.color), 1)
        pol.queue_a = pol.queue_h = None                         # the home path has finished
        res = pol._choose([d], obs(), verify=False)
        self.assertEqual(res[0], "plan")
        self.assertTrue(pol.planner.calls[-1])                   # varied posture/yaw, not the stock grasp

    def test_aborted_empty_close_still_writes_a_grasp_diag_line_with_scores(self):
        _, pol, q = self.started(sums=(-0.01, -0.01))
        logs = []
        pol.log = logs.append
        pol._grasp_diag({"hand_q": q})
        self.assertIsNone(pol._gdiag)
        diag = [m for m in logs if m.startswith("[grasp_diag]")]
        self.assertEqual(len(diag), 1, logs)
        self.assertIn("held=no aborted=1", diag[0])
        self.assertIn("close@", diag[0])
        self.assertIn("lift=n/a", diag[0])
        for field in ("finger_sum", "index_sum", "middle_sum", "threshold", "sample_held", "stall_vs_closed",
                      "cmd_vs_closed"):
            self.assertIn(f"{field}=", diag[0])
        self.assertRegex(diag[0], r"finger_sum=-?\d+\.\d+ index_sum=-?\d+\.\d+ middle_sum=-?\d+\.\d+")
        self.assertTrue(any(m.startswith("[held_abort]") for m in logs))

    def test_diag_formatting_error_never_raises_into_the_abort(self):
        _, pol, q = self.started(sums=(-0.01, -0.01))
        logs = []
        pol.log = logs.append
        pol._diag_body = lambda g: 1 / 0
        pol._grasp_diag({"hand_q": q})
        self.assertIsNone(pol._gdiag)
        self.assertIs(pol.vary_hist[-1]["held"], False)
        self.assertTrue(any("format_error=ZeroDivisionError" in m for m in logs))

    def test_held_close_is_not_aborted(self):
        _, pol, q = self.started(sums=(0.19, 0.02))
        n = len(pol.queue_a)
        pol._grasp_diag({"hand_q": q})
        self.assertEqual(len(pol.queue_a), n)
        self.assertIsNone(pol.vary_hist[-1]["held"])

    def test_unsure_is_never_empty(self):
        _, pol, q = self.started(detector=HeldDetector("finger_sum", 0.045))   # no bias reference yet
        n = len(pol.queue_a)
        pol._grasp_diag({"hand_q": q})
        self.assertEqual(len(pol.queue_a), n)
        self.assertIsNone(pol.vary_hist[-1]["held"])
        for bad in (q[:7], np.where(np.arange(14) == 3, np.nan, q)):   # missing or non-finite joint data
            _, pol, _ = self.started()
            pol._gdiag["lift_end"] = 10 ** 6                       # lift sample never taken: no finalize on bad data
            n = len(pol.queue_a)
            pol._grasp_diag({"hand_q": bad})
            self.assertEqual(len(pol.queue_a), n)
            self.assertIsNotNone(pol._gdiag)
            self.assertIsNone(pol.vary_hist[-1]["held"])

    def test_invalid_empty_close_calibration_disables_early_abort(self):
        detector = HeldDetector("finger_sum", 0.045)
        detector.require_empty_close_calibration(0.0, steady_samples=2, max_velocity_rad_s=0.08,
                                                 hold_after_settle_s=0.4, timeout_s=1.0)
        cmd, q = trace("left", -0.04, -0.04)
        detector.sample_empty_close(0.1, cmd, q)
        self.assertEqual(detector.sample_empty_close(1.1, cmd, q), "invalid")
        _, pol, empty_q = self.started(sums=(-0.01, -0.01), detector=detector)
        self.assertFalse(pol._held_abort_due(pol._gdiag, empty_q, pol.hand_cmd))

    def test_stall_mode_never_aborts(self):
        _, pol, q = self.started()
        pol.held_detector = HeldDetector("stall", 0.045)
        self.assertFalse(pol._held_abort_due({"plan": SimpleNamespace(side="left")}, q, pol.hand_cmd))

    def test_read_through_carry_keeps_a_held_object_and_uses_carry_samples(self):
        _, pol, held_q = self.started(sums=(0.19, 0.02))
        pol.cfg["held"]["read_through_carry"] = True
        pol._gdiag["carry"] = (30, 60)
        pol._gdiag["samples"] = {"close": (20, held_q, pol.hand_cmd),
                                  "lift": (30, held_q, pol.hand_cmd)}
        pol.queue_a = pol.queue_h = np.zeros((60 - 40, 14))
        pol._grasp_diag({"hand_q": held_q})
        self.assertIsNotNone(pol._gdiag)
        self.assertIn("carry", pol._gdiag["samples"])
        self.assertEqual(len(pol._gdiag["samples"]["carry"]), 1)

    def test_slip_during_lift_aborts_before_any_carry_command(self):
        _, pol, empty_q = self.started(sums=(-0.01, -0.01))
        logs = []
        pol.log = logs.append
        pol.cfg["held"]["read_through_carry"] = True
        pol._gdiag["carry"] = (30, 60)
        held_cmd, held_q = trace("left", 0.19, 0.02, thumb0_native=0.70)
        pol._gdiag["samples"] = {"close": (20, held_q, held_cmd)}
        pol.hand_cmd = held_cmd
        pol.queue_a = pol.queue_h = np.zeros((60 - 30, 14))
        pol._grasp_diag({"hand_q": empty_q})
        self.assertIsNone(pol._gdiag)
        # rc7 (fixcal C4): the carry is cancelled and a proved open -> vertical lift -> home recovery is requested
        self.assertEqual(len(pol.queue_a), 0)
        self.assertEqual(pol._empty_recovery["side"], "left")
        self.assertIs(pol.vary_hist[-1]["held"], False)
        self.assertTrue(any("post-lift sample empty" in m for m in logs))

    def test_action_chunk_stops_exactly_at_lift_end_before_the_read(self):
        _, pol, _ = self.started(sums=(0.19, 0.02))
        pol.cfg["held"]["read_through_carry"] = True
        pol._gdiag["lift_end"] = 30
        pol._gdiag["carry"] = (30, 60)
        pol.queue_a = pol.queue_h = np.zeros((60 - 22, 14))
        action = pol._pop()
        self.assertEqual(len(action["arm_q_target"]), 8)
        self.assertEqual(len(pol.queue_a), 30)  # next act reads hand_q before queueing tick 30 of carry


class TestPolicyHeldWiring(unittest.TestCase):
    def test_m3_calibration_does_not_replace_the_home_settle_queue(self):
        pol = ClassicalPolicy(log=None, overrides={"grasp.held_detector": "finger_sum",
                                                   "grasp.held_calibration.enabled": True})
        pol.state = "look"  # settle switches to look while its hold queue is still draining
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        pol.queue_a = np.repeat(pol.arm_cmd[None], 2, axis=0)
        pol.queue_h = np.repeat(pol.hand_cmd[None], 2, axis=0)
        self.assertIsNone(pol._held_calibration_tick({"hand_q": pol.hand_cmd.copy()}, 1.0))
        self.assertIsNone(pol._held_cal_phase)
        self.assertEqual(len(pol.queue_a), 2)

        pol.queue_a = pol.queue_h = np.empty((0, 14))
        action = pol._held_calibration_tick({"hand_q": pol.hand_cmd.copy()}, 1.1)
        self.assertEqual(pol.state, "held_calibration")
        self.assertEqual(pol._held_cal_phase, "sample")
        self.assertGreater(len(action["hand_q_target"]), 0)

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
