"""Focused unit tests for the default-off conservative profile flags."""
import types
import unittest

import numpy as np

from .planner import Planner
from .policy import ClassicalPolicy


class _PlanRecorder:
    def __init__(self):
        self.object_speed_scale = None
        self.seen = []

    def plan(self, *_args, **_kwargs):
        self.seen.append(self.object_speed_scale)
        return types.SimpleNamespace(ok=True)


class _FailingPlanRecorder(_PlanRecorder):
    def plan(self, *_args, **_kwargs):
        self.seen.append(self.object_speed_scale)
        raise RuntimeError("test planner failure")


class TestFirstObjectSpeedScale(unittest.TestCase):
    @staticmethod
    def policy(scale):
        p = ClassicalPolicy.__new__(ClassicalPolicy)
        p.cfg = {"policy": {"first_object_speed_scale": scale, "retry_vary_grasp": False}}
        p.planner = _PlanRecorder()
        p.arm_cmd = np.zeros(14)
        p.hand_cmd = np.zeros(14)
        p.n_started = 0
        return p

    def test_only_first_started_pick_is_scaled_and_planner_state_is_restored(self):
        p = self.policy(0.7)
        p._plan_pick(object(), {})
        self.assertEqual(p.planner.seen, [0.7])
        self.assertIsNone(p.planner.object_speed_scale)
        p.n_started = 1
        p._plan_pick(object(), {})
        self.assertEqual(p.planner.seen, [0.7, None])

    def test_null_is_the_exact_unscaled_planner_path(self):
        p = self.policy(None)
        p._plan_pick(object(), {})
        self.assertEqual(p.planner.seen, [None])

    def test_planner_state_is_restored_when_planning_raises(self):
        p = self.policy(0.7)
        p.planner = _FailingPlanRecorder()
        with self.assertRaisesRegex(RuntimeError, "test planner failure"):
            p._plan_pick(object(), {})
        self.assertEqual(p.planner.seen, [0.7])
        self.assertIsNone(p.planner.object_speed_scale)

    def test_planner_scales_every_arm_motion_phase(self):
        from aaltonomous.scene.scene import Scene
        from .config import hand_postures, home_arm, load_config
        from .perception import Detection

        cfg, scene = load_config(overlay="none"), Scene.load()
        home, post = home_arm(cfg), hand_postures(cfg)
        layout = scene.cfg["layout"]
        top = scene.mat_to_pelvis([np.mean(layout["region_x"]), np.mean(layout["region_y"]),
                                   scene.object_height()])
        det = Detection("red", top, 1000, 1000, "depth")
        plans = []
        for scale in (None, 0.7):
            planner = Planner(cfg, scene, post, home)
            planner.object_speed_scale = scale
            plan = planner.plan(det, home, post["open"], relocate=False)
            self.assertTrue(plan.ok, plan.reason)
            plans.append(plan)
        stock_ticks = {name: end - start for name, start, end in plans[0].phases}
        slow_ticks = {name: end - start for name, start, end in plans[1].phases}
        for name in ("to_pregrasp", "descend", "lift", "carry", "retreat", "home"):
            self.assertGreater(slow_ticks[name], stock_ticks[name], name)
        self.assertEqual(slow_ticks["close"], stock_ticks["close"])
        self.assertEqual(slow_ticks["open"], stock_ticks["open"])


class _IK:
    def __init__(self, ok):
        self.ok = ok
        self.calls = []

    def solve(self, side, target, seed, **kwargs):
        self.calls.append((side, np.asarray(target).copy(), np.asarray(seed).copy(), kwargs))
        return np.zeros(7), self.ok, 0.0, 0.0


class TestAdjustedDrop(unittest.TestCase):
    @staticmethod
    def planner(ok=True, dz=0.04, margin=0.02):
        p = Planner.__new__(Planner)
        p.cfg = {"grasp": {"drop_height_extra_m": dz, "drop_lateral_margin_m": margin}}
        p.home = np.zeros(14)
        p.kin = _IK(ok)
        p.bin_xy = lambda _name: np.array([0.505, 0.26])
        return p

    def test_adjusted_target_moves_toward_bin_centre_and_requires_ik(self):
        p = self.planner(ok=True)
        xy, z, adjusted = p._adjusted_drop_target("left", "left", [0.453, 0.20], 0.1225, np.zeros(7))
        np.testing.assert_allclose(xy, [0.453, 0.22])
        self.assertAlmostEqual(z, 0.1625)
        self.assertTrue(adjusted)
        np.testing.assert_allclose(p.kin.calls[0][1], [0.453, 0.22, 0.1625])

    def test_ik_failure_falls_back_to_original_target(self):
        p = self.planner(ok=False)
        xy, z, adjusted = p._adjusted_drop_target("left", "left", [0.453, 0.20], 0.1225, np.zeros(7))
        np.testing.assert_array_equal(xy, [0.453, 0.20])
        self.assertEqual(z, 0.1225)
        self.assertFalse(adjusted)

    def test_both_flags_off_skip_even_the_ik_check(self):
        p = self.planner(ok=True, dz=0.0, margin=0.0)
        xy, z, adjusted = p._adjusted_drop_target("left", "left", [0.453, 0.20], 0.1225, np.zeros(7))
        np.testing.assert_array_equal(xy, [0.453, 0.20])
        self.assertEqual((z, adjusted, p.kin.calls), (0.1225, False, []))

    def test_adjusted_path_failure_retries_complete_stock_drop_leg(self):
        from aaltonomous.scene.scene import Scene
        from .config import hand_postures, home_arm, load_config
        from .perception import Detection

        cfg = load_config(overlay="none", overrides={"grasp.drop_height_extra_m": 0.04,
                                                     "grasp.drop_lateral_margin_m": 0.02})
        scene = Scene.load()
        home, post = home_arm(cfg), hand_postures(cfg)
        layout = scene.cfg["layout"]
        top = scene.mat_to_pelvis([np.mean(layout["region_x"]), np.mean(layout["region_y"]),
                                   scene.object_height()])
        planner = Planner(cfg, scene, post, home)
        stock_xy = planner.drop_xy("left")
        cartesian = planner._cartesian
        forced = []

        def fail_first_lower(side, p0, p1, seed, name, plan):
            if name == "lower" and not forced:
                forced.append(np.asarray(p1).copy())
                plan.ik[name] = (False, 0.0, 0.0)
                plan.reason = "forced adjusted lower failure"
                return None
            return cartesian(side, p0, p1, seed, name, plan)

        planner._cartesian = fail_first_lower
        plan = planner.plan(Detection("red", top, 1000, 1000, "depth"), home, post["open"], relocate=False)
        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual(len(forced), 1)
        self.assertIn("drop_fallback", plan.waypoints)
        self.assertNotIn("drop_adjusted", plan.waypoints)
        np.testing.assert_allclose(plan.waypoints["drop"][:2], stock_xy)


class TestJointOffsetLog(unittest.TestCase):
    def test_uses_own_last_steady_command_only_and_stops_after_first_second(self):
        logs = []
        p = ClassicalPolicy.__new__(ClassicalPolicy)
        p.cfg = {"debug": {"log_joint_offset": True}}
        p.log = logs.append
        p._joint_offset_t0 = None
        p._last_policy_arm_cmd = p._last_policy_hand_cmd = None
        p._last_policy_cmd_steady = False
        arm = np.full((6, 14), 0.5)
        hand = np.full((6, 14), -0.25)
        p._remember_policy_action(arm, hand, False, steady=True)
        obs = {"arm_q": np.full(14, 0.4), "hand_q": np.full(14, -0.2),
               "arm_q_target": np.full(14, 99.0), "hand_q_target": np.full(14, 99.0)}
        p._joint_offset_diag(obs, 0.2)
        self.assertEqual(len(logs), 1)
        self.assertIn("arm=[0.1, 0.1", logs[0])
        self.assertIn("hand=[-0.05, -0.05", logs[0])

        moving = np.vstack([np.zeros((1, 14)), np.ones((1, 14))])
        p._remember_policy_action(moving, moving, False)
        p._joint_offset_diag(obs, 0.4)
        self.assertEqual(len(logs), 1)

        one_tick_trajectory = np.ones((1, 14))
        p._remember_policy_action(one_tick_trajectory, one_tick_trajectory, False)
        p._joint_offset_diag(obs, 0.6)
        self.assertEqual(len(logs), 1)

        p._remember_policy_action(arm, hand, False, steady=True)
        p._joint_offset_diag(obs, 1.3)
        self.assertEqual(len(logs), 1)

    def test_disabled_diagnostic_does_not_retain_commands_or_change_action_bytes(self):
        p = ClassicalPolicy.__new__(ClassicalPolicy)
        p.cfg = {"debug": {"log_joint_offset": False}}
        p._last_policy_arm_cmd = p._last_policy_hand_cmd = None
        p._last_policy_cmd_steady = False
        arm = np.arange(42, dtype=float).reshape(3, 14)
        hand = -arm
        out = p._remember_policy_action(arm, hand, False, steady=True)
        self.assertEqual(out["arm_q_target"].tobytes(), arm.tobytes())
        self.assertEqual(out["hand_q_target"].tobytes(), hand.tobytes())
        self.assertIsNone(p._last_policy_arm_cmd)
        self.assertIsNone(p._last_policy_hand_cmd)
        self.assertFalse(p._last_policy_cmd_steady)


if __name__ == "__main__":
    unittest.main()
