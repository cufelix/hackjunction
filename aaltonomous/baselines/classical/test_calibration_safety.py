"""Regression tests for rc6 review C5/H3/C4: pre-motion calibration, frame snapshots, safe recovery."""
import unittest
import threading
import time
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aaltonomous.scene.scene import Scene

from . import bin_vote
from .config import load_config
from .perception import PitchCalibration
from .planner import Planner, SafeRoute
from .policy import ClassicalPolicy


class TestFrameTransform(unittest.TestCase):
    def test_one_frame_keeps_one_immutable_transform_and_version(self):
        cfg = load_config(overrides={"perception.online_pitch_cal": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        p = pol.perception
        frame = p.frame_transform()
        point = np.array([0.35, 0.0, p.scene.counter_z + 0.05])
        with p.use_frame_transform(frame):
            before = p._project(point)
            T = p.current_T(); T[0, 3] += 0.05
            p.commit_pitch(PitchCalibration(True, True, 0.0, 1, 0.0, 0.0, "test", T=T))
            during = p._project(point)
            self.assertEqual(p._frame_transform.get().version, frame.version)
        after = p._project(point)
        np.testing.assert_array_equal(during, before)
        self.assertFalse(np.array_equal(after, before))
        self.assertFalse(frame.T.flags.writeable)

    def test_vote_rim_and_tape_consumers_share_the_supplied_frame(self):
        cfg = load_config(overrides={"perception.online_pitch_cal": True, "bin.vote.enabled": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        p, seen = pol.perception, []
        frame = p.frame_transform()

        def rim(perception, depth, centers, vote_cfg):
            seen.append(("rim", perception._frame_transform.get().version))
            T = perception.current_T(); T[1, 3] += 0.04
            perception.commit_pitch(PitchCalibration(True, True, 0.0, 1, 0.0, 0.0, "mid-frame", T=T))
            return {name: None for name in centers}

        def tape(perception, rgb, depth, centers, vote_cfg):
            seen.append(("tape", perception._frame_transform.get().version))
            return {name: None for name in centers}

        with mock.patch.object(bin_vote, "rim_votes", side_effect=rim), \
                mock.patch.object(bin_vote, "tape_votes", side_effect=tape):
            bin_vote.vote_bins(p, pol.planner,
                               [{"head_rgb": None, "head_depth": None, "transform": frame}], cfg,
                               log=lambda _: None)
        self.assertEqual(seen, [("rim", frame.version), ("tape", frame.version)])


class TestCalibrationPhase(unittest.TestCase):
    def test_reset_enters_hold_only_phase_before_startup_route(self):
        cfg = load_config(overrides={"perception.online_pitch_cal": True,
                                     "calib.plane.apply_height": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        pol.reset({"instruction": ""})
        self.assertEqual(pol.state, "calibrate")
        pol.planner.go_to = mock.Mock(side_effect=AssertionError("motion before calibration"))
        pol.perception.detect = mock.Mock(side_effect=AssertionError("detect before calibration"))
        obs = {"t": 7.0, "arm_q": pol.home.copy(), "hand_q": pol.post["open"].copy(),
               "head_rgb": None, "head_depth": None}
        out = pol.act(obs)
        self.assertEqual(pol.state, "calibrate")
        pol.planner.go_to.assert_not_called()
        pol.perception.detect.assert_not_called()
        np.testing.assert_array_equal(out["arm_q_target"][0], obs["arm_q"])

    def test_safety_shell_drops_previous_episode_target_before_calibration_hold(self):
        cfg = load_config(overrides={"perception.online_pitch_cal": True,
                                     "policy.safety_shell": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        pol._shell_last = (np.ones(14), np.ones(14))
        pol.reset({"instruction": ""})
        self.assertIsNone(pol._shell_last)
        measured_arm = pol.home.copy()
        measured_hand = pol.post["open"].copy()
        out = pol.act({"t": 3.0, "arm_q": measured_arm, "hand_q": measured_hand,
                       "head_rgb": None, "head_depth": None})
        np.testing.assert_array_equal(out["arm_q_target"][0], measured_arm)
        np.testing.assert_array_equal(out["hand_q_target"][0], measured_hand)

    def test_pitch_wait_timeout_is_from_phase_entry_and_never_joins(self):
        cfg = load_config(overrides={"perception.online_pitch_cal": True})
        pol = ClassicalPolicy(config=cfg, log=None, threaded=False)
        pol._calib_phase_t0 = 10.0
        pol.pitch_calibration = None
        pol._pitch_thread = mock.Mock()
        self.assertTrue(pol._await_pitch_cal(10.5))
        self.assertFalse(pol._await_pitch_cal(11.0))
        pol._pitch_thread.join.assert_not_called()

    def test_preflight_poll_stays_above_27_hz_while_worker_is_blocked(self):
        cfg = load_config(overrides={"perception.online_pitch_cal": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        pol.reset({"instruction": ""})
        gate = threading.Event()
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.full((480, 640), 800, np.uint16)
        obs = {"t": 0.0, "arm_q": pol.home.copy(), "hand_q": pol.post["open"].copy(),
               "head_rgb": rgb, "head_depth": depth}
        def slow(*_):
            gate.wait(5.0)
            return pol.perception.make_pitch_failure("test")
        with mock.patch.object(pol.perception, "compute_pitch", side_effect=slow):
            elapsed = []
            for k in range(5):
                obs["t"] = 0.1 * k
                started = time.perf_counter(); pol.act(obs); elapsed.append(time.perf_counter() - started)
            gate.set()
            pol._pitch_thread.join(5.0)
        self.assertLess(max(elapsed), 1.0 / 27.0, elapsed)

    def test_height_fit_poll_stays_above_27_hz_while_worker_is_blocked(self):
        cfg = load_config(overrides={"calib.plane.apply_height": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        pol.reset({"instruction": ""})
        gate = threading.Event()
        obs = {"t": 0.0, "arm_q": pol.home.copy(), "hand_q": pol.post["open"].copy(),
               "head_rgb": None, "head_depth": np.full((480, 640), 800, np.uint16)}

        def slow(*_):
            gate.wait(5.0)
            return pol.perception.measure_counter_height(None)

        with mock.patch.object(pol, "_measure_counter_height", side_effect=slow):
            elapsed = []
            for k in range(5):
                obs["t"] = 0.1 * k
                started = time.perf_counter(); pol.act(obs); elapsed.append(time.perf_counter() - started)
            gate.set()
            for _ in range(100):
                job = pol._jobs.get("counter_height")
                if job is None or job["result"] is not None:
                    break
                time.sleep(0.001)
        self.assertLess(max(elapsed), 1.0 / 27.0, elapsed)

    def test_height_commit_is_shared_with_planner_keepout_before_route(self):
        cfg = load_config(overrides={"calib.plane.apply_height": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        z = pol.scene.nominal_counter_z + 0.09
        pol.planner.commit_counter_height(z)
        self.assertEqual(pol.scene.counter_z, z)
        self.assertEqual(pol.planner.counter_keepout_z, z)

    def test_startup_uses_checked_staging_route_not_go_to_when_height_is_enabled(self):
        cfg = load_config(overrides={"calib.plane.apply_height": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        pol.reset({"instruction": ""})
        pol.state = "init"
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        a = np.repeat(pol.home[None], 2, 0)
        h = np.repeat(pol.post["open"][None], 2, 0)
        route = SafeRoute(True, a, h, [("vertical_lift", 0, 1), ("home", 1, 2)])
        pol._bg = mock.Mock(return_value=("ok", route))
        pol.planner.go_to = mock.Mock(side_effect=AssertionError("unchecked startup route"))
        out = pol._act({"t": 2.0, "arm_q": pol.home, "hand_q": pol.post["open"],
                        "head_rgb": None, "head_depth": None})
        pol.planner.go_to.assert_not_called()
        self.assertEqual(pol.state, "settle")
        np.testing.assert_array_equal(out["arm_q_target"][-1], a[-1])

    def test_real_urdf_staging_route_lifts_for_a_higher_runtime_table(self):
        cfg = load_config(overrides={"calib.plane.apply_height": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        pol.planner.commit_counter_height(pol.scene.nominal_counter_z + 0.10)
        route = pol.planner.safe_home_route(pol.home, pol.post["open"], ("left", "right"))
        self.assertTrue(route.ok, route.reason)
        self.assertIn("vertical_lift", [name for name, _, _ in route.phases])
        self.assertEqual(pol.planner.dry_run_safety(pol.home, pol.post["open"], route.arm, route.hand), 0)


class TestEmptyGraspRecovery(unittest.TestCase):
    def setUp(self):
        cfg = load_config(overrides={"grasp.held_detector": "finger_sum"})
        self.pol = ClassicalPolicy(config=cfg, log=None)
        self.pol.reset({"instruction": ""})
        self.pol.arm_cmd = self.pol.home.copy()
        self.pol.hand_cmd = self.pol.post["closed"].copy()
        self.pol.queue_a = np.ones((5, 14)); self.pol.queue_h = np.ones((5, 14))
        plan = SimpleNamespace(color="red", side="left", target_bin="left", source_bin=None)
        self.g = {"plan": plan, "t": 3.0}

    def test_abort_never_calls_direct_joint_space_home(self):
        self.pol.planner.go_to = mock.Mock(side_effect=AssertionError("direct home forbidden"))
        self.pol._abort_empty_grasp(self.g)
        self.pol.planner.go_to.assert_not_called()
        self.assertEqual(len(self.pol.queue_a), 0)
        self.assertEqual(self.pol._empty_recovery["side"], "left")

    def test_proved_route_is_published_and_unproved_route_holds(self):
        self.pol._abort_empty_grasp(self.g)
        a = np.repeat(self.pol.arm_cmd[None], 3, 0)
        h = np.repeat(self.pol.post["open"][None], 3, 0)
        route = SafeRoute(True, a, h, [("open", 0, 1), ("vertical_lift", 1, 2), ("home", 2, 3)])
        self.pol._bg = mock.Mock(return_value=("ok", route))
        out = self.pol._empty_recovery_step(3.1)
        self.assertIsNone(self.pol._empty_recovery)
        self.assertEqual(self.pol.state, "executing")
        np.testing.assert_array_equal(out["hand_q_target"][-1], h[-1])

        self.pol._empty_recovery = {"arm": self.pol.arm_cmd.copy(), "hand": self.pol.hand_cmd.copy(),
                                    "side": "left", "source_bin": None, "t": 4.0}
        self.pol._jobs = {}
        self.pol._bg = mock.Mock(return_value=("ok", SafeRoute(False, reason="keepout")))
        held = self.pol._empty_recovery_step(4.1)
        self.assertEqual(self.pol.state, "empty_hold")
        np.testing.assert_array_equal(held["arm_q_target"][0], self.pol.arm_cmd)

    def test_planner_route_orders_open_vertical_lift_home_and_dry_runs(self):
        cfg, scene = load_config(), Scene.load()

        class Kin:
            def grasp_pose(self, side, arm):
                i = 0 if side == "left" else 7
                return np.array([0.30, -0.20 if side == "right" else 0.20,
                                 scene.counter_z + 0.20 + arm[i]]), np.array([0.0, 0.0, -1.0])

            def solve(self, side, target, seed, **_):
                out = np.array(seed, float); out[0 if side == "left" else 7] = target[2] - scene.counter_z - 0.20
                return out, True, 0.0, 0.0

            def hand_points(self, side, arm, hand):
                p = self.grasp_pose(side, arm)[0].copy(); p[2] -= 0.02
                return np.repeat(p[None], 8, 0)

        zero = np.zeros(14)
        planner = Planner(cfg, scene, {"open": zero.copy()}, zero.copy(), kin=Kin())
        planner.dry_run_safety = mock.Mock(return_value=0)
        start = zero.copy(); start[0] = -0.15
        route = planner.safe_home_route(start, np.ones(14), sides=("left",))
        self.assertTrue(route.ok, route.reason)
        self.assertEqual([name for name, _, _ in route.phases], ["open", "vertical_lift", "home"])
        planner.dry_run_safety.assert_called_once()


if __name__ == "__main__":
    unittest.main()
