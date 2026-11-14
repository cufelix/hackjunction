"""Focused regressions for the opt-in finals done/look/planner hardening."""
from __future__ import annotations

import threading
import time
import unittest

import numpy as np

from .config import load_config
from .perception import Detection
from .planner import Plan
from .policy import ClassicalPolicy


def _obs(t=0.0, policy=None):
    arm = np.zeros(14) if policy is None else policy.arm_cmd
    hand = np.zeros(14) if policy is None else policy.hand_cmd
    return {"t": t, "arm_q": arm, "hand_q": hand, "head_rgb": None, "head_depth": None}


def _ready(**overrides):
    policy = ClassicalPolicy(config=load_config(overrides=overrides), log=None, time_limit=120.0)
    policy.arm_cmd, policy.hand_cmd = policy.home.copy(), policy.post["open"].copy()
    policy.state = "look"
    return policy


def _plan(color="red"):
    return Plan(True, side="left", color=color, target_bin="left", arm=np.zeros((30, 14)),
                hand=np.zeros((30, 14)), waypoints={"grasp": np.array([0.3, 0.0, 0.8])},
                ik={"grasp": (True, 0.0, 0.0)}, safety_clips=0)


class TestDoneConfirmation(unittest.TestCase):
    ON = {"policy.done_confirm": True}

    def test_empty_then_objects_reappear_is_not_done(self):
        policy = _ready(**self.ON)
        first = policy._after_look(("none", None, 0), 1.0)
        self.assertFalse(first["done"])
        self.assertEqual(policy._clear_looks, 1)

        policy.require_confirm = True
        again = policy._after_look(("plan", _plan(), 1), 2.0)
        self.assertFalse(again["done"])
        self.assertEqual(policy.state, "await")
        self.assertEqual(policy._clear_looks, 0)

    def test_requires_count_and_elapsed_span(self):
        policy = _ready(**self.ON)
        self.assertFalse(policy._after_look(("none", None, 0), 1.0)["done"])
        self.assertFalse(policy._after_look(("none", None, 0), 10.9)["done"])
        self.assertTrue(policy._after_look(("none", None, 0), 11.0)["done"])

    def test_look_exception_retries_and_is_not_clear(self):
        policy = _ready(**self.ON)
        policy._after_look(("none", None, 0), 1.0)
        action = policy._after_look(("error", "camera read"), 2.0)
        self.assertFalse(action["done"])
        self.assertEqual(policy.state, "settle")
        self.assertEqual((policy._clear_looks, policy._clear_since), (0, None))

    def test_time_limit_still_ends_retries(self):
        policy = _ready(**self.ON)
        policy.time_limit = 12.0
        action = policy._after_look(("error", "camera read"), 12.0)
        self.assertTrue(action["done"])
        self.assertEqual(policy.state, "done")


class _BlockedDetector:
    def __init__(self):
        self.release = threading.Event()
        self.calls = []
        self.reset_called = False

    def detect(self, *args, **kwargs):
        self.calls.append(time.monotonic())
        self.release.wait(2.0)
        return []

    def reset_calibration(self):
        self.reset_called = True


class TestDetectDeadline(unittest.TestCase):
    def test_deadline_expiry_is_failed_look_and_act_does_not_wait(self):
        policy = _ready(**{"perception.look_timeout_s": 0.02})
        detector = _BlockedDetector()
        policy.perception = detector
        replacements = []

        def rotate_detector(_state=None):
            fresh = _BlockedDetector()
            fresh.calls, fresh.release = detector.calls, detector.release
            replacements.append(fresh)
            policy.perception = fresh

        policy._replace_stale_perception = rotate_detector
        try:
            start = time.perf_counter()
            first = policy.act(_obs(1.0, policy))
            first_ms = (time.perf_counter() - start) * 1000.0
            self.assertFalse(first["done"])
            self.assertLess(first_ms, 1000.0 / 27.0)

            time.sleep(0.03)
            action = policy.act(_obs(1.1, policy))
            self.assertFalse(action["done"])
            self.assertEqual(policy.state, "settle")
            self.assertTrue(any("deadline exceeded" in message for _, message in policy.events), policy.events)
            self.assertEqual(len(replacements), 1)

            # A native call cannot be killed safely. Launch one isolated retry, but never exceed the configured
            # number of in-flight detector calls if both calls wedge.
            policy.queue_a = policy.queue_h = None
            policy.state = "look"
            policy.act(_obs(1.2, policy))
            limit = time.monotonic() + 0.2
            while len(detector.calls) < 2 and time.monotonic() < limit:
                time.sleep(0.001)
            self.assertEqual(len(detector.calls), 2)
            time.sleep(0.03)
            policy.act(_obs(1.3, policy))
            self.assertEqual(len(replacements), 2)
            policy.queue_a = policy.queue_h = None
            policy.state = "look"
            policy.act(_obs(1.4, policy))
            self.assertEqual(len(detector.calls), 2)
        finally:
            detector.release.set()

    def test_reset_preserves_timed_out_call_and_isolates_next_episode(self):
        policy = _ready(**{"perception.look_timeout_s": 0.02})
        detector = _BlockedDetector()
        policy.perception = detector
        try:
            policy.act(_obs(1.0, policy))
            time.sleep(0.03)
            policy.act(_obs(1.1, policy))
            self.assertEqual(len(policy._stale_detect_jobs), 1)
            self.assertIsNot(policy.perception, detector)
            self.assertIs(policy.perception.scene, policy.planner.scene)

            policy.reset()
            self.assertEqual(len(policy._stale_detect_jobs), 1)
            self.assertIsNot(policy.perception, detector)
            self.assertFalse(detector.reset_called)
            self.assertEqual(len(detector.calls), 1)
        finally:
            detector.release.set()

    def test_retry_height_calibration_and_planner_share_live_scene(self):
        policy = _ready(**{"perception.look_timeout_s": 0.02, "calib.plane.apply_height": True})
        old_perception, old_scene = policy.perception, policy.scene
        corrected_T = policy.perception.current_T()
        corrected_T[0, 3] += 0.012
        policy.perception.set_camera_pose(corrected_T)
        stable = policy._perception_snapshot(policy.perception)
        policy._replace_stale_perception(stable)
        self.assertIsNot(policy.perception, old_perception)
        self.assertIsNot(policy.scene, old_scene)
        self.assertIs(policy.perception.scene, policy.scene)
        self.assertIs(policy.planner.scene, policy.scene)
        np.testing.assert_array_equal(policy.perception.current_T(), corrected_T)

        measured = policy.scene.nominal_counter_z + 0.04
        policy.perception._plane_height = lambda *_a, **_k: (measured, 0.0, 1000, None)
        for _ in range(3):   # integ-all: fixcal commits the height only after N consistent looks
            policy.perception.update_counter_height(np.zeros((2, 2), dtype=np.float32))
        self.assertAlmostEqual(policy.planner.scene.counter_z_runtime, measured)


class TestPerTargetGuard(unittest.TestCase):
    ON = {"planner.per_target_guard": True, "policy.order": "largest"}

    @staticmethod
    def _det(color, px):
        return Detection(color, np.array([0.30, 0.05 if color == "red" else -0.05, 0.80]), px, px, "depth")

    def test_one_target_throws_and_another_still_executes(self):
        policy = _ready(**self.ON)
        red, blue = self._det("red", 900), self._det("blue", 500)

        def planning(det, *_args, **_kwargs):
            if det.color == "red":
                raise RuntimeError("IK backend failed")
            return _plan(det.color)

        policy._plan_pick = planning
        result = policy._choose([red, blue], _obs(0.0, policy), verify=False)
        self.assertEqual((result[0], result[1].color), ("plan", "blue"))
        self.assertEqual(policy.failed, [])
        action = policy._after_look(result, 0.0)
        self.assertFalse(action["done"])
        self.assertEqual(policy.state, "executing")

    def test_infeasible_target_is_cooled_not_episode_fatal(self):
        policy = _ready(**self.ON)
        red = self._det("red", 900)
        calls = []

        def planning(det, *_args, **_kwargs):
            calls.append(det.color)
            return Plan(False, color=det.color, reason="no safe IK") if len(calls) == 1 else _plan(det.color)

        policy._plan_pick = planning
        result = policy._choose([red], _obs(0.0, policy), verify=False)
        self.assertEqual(result[0], "relook")
        self.assertTrue(policy._target_guarded(red, 0.5))
        self.assertFalse(policy._after_look(result, 0.0)["done"])

        # A fresh look while the target is cooling must not turn one target failure into episode done.
        cooling = policy._choose([red], _obs(0.5, policy), verify=False)
        self.assertEqual(cooling[0], "relook")
        self.assertEqual(calls, ["red"])
        self.assertFalse(policy._after_look(cooling, 0.5)["done"])

        retried = policy._choose([red], _obs(2.1, policy), verify=False)
        self.assertEqual((retried[0], retried[1].color), ("plan", "red"))
        self.assertEqual(calls, ["red", "red"])

    def test_guard_does_not_relook_when_no_target_failed(self):
        policy = _ready(**self.ON)
        unmapped = self._det("red", 900)
        policy.mapping = {}
        result = policy._choose([unmapped], _obs(0.0, policy), verify=False)
        self.assertEqual(result[0], "none")
        self.assertEqual(policy._target_cooldowns, [])

    def test_throwing_correction_does_not_block_counter_target(self):
        policy = _ready(**{**self.ON, "policy.verify_placement": True})
        wrong, blue = self._det("red", 900), self._det("blue", 500)

        class Frame:
            def detect(self, *_args, **_kwargs):
                return [blue], [(wrong, "in_bin:right")]

        policy.perception = Frame()
        policy._review_bins = lambda *_args: [(wrong, "right", "left")]

        def planning(det, *_args, **_kwargs):
            if det.color == "red":
                raise RuntimeError("corrective IK failed")
            return _plan(det.color)

        policy._plan_pick = planning
        result = policy._look(_obs(0.0, policy))
        self.assertEqual((result[0], result[1].color), ("plan", "blue"))
        self.assertTrue(policy._target_guarded(wrong, 0.5))


class TestActLatency(unittest.TestCase):
    def test_outstanding_detector_polling_keeps_27_hz_margin(self):
        rows = {}
        detectors = []
        try:
            for name, override in (("off", {}), ("on", {"perception.look_timeout_s": 5.0})):
                policy = _ready(**override)
                policy.threaded = True
                detector = _BlockedDetector()
                detectors.append(detector)
                policy.perception = detector
                policy.act(_obs(1.0, policy))
                samples = []
                for _ in range(150):
                    start = time.perf_counter()
                    policy.act(_obs(1.0, policy))
                    samples.append((time.perf_counter() - start) * 1000.0)
                rows[name] = float(np.percentile(samples, 95))
            self.assertLess(rows["on"], 1000.0 / 27.0, rows)
            self.assertLess(rows["on"], rows["off"] + 2.0, rows)
        finally:
            for detector in detectors:
                detector.release.set()


if __name__ == "__main__":
    unittest.main()
