"""M4 acceleration-aware hand ramps and final stateful arm/hand output caps."""
import unittest

import numpy as np

from .config import load_config
from .planner import MotionOutputLimiter, Planner, ramp


DT = 1.0 / 30.0
CAPS = {"arm_vel": 0.85, "arm_acc": 4.0, "hand_vel": 1.5, "hand_acc": 8.0}


def rates(path, start, dt=DT):
    return np.diff(np.vstack([np.asarray(start, float)[None], path]), axis=0) / dt


class TestMotionOutputCaps(unittest.TestCase):
    def test_flag_off_keeps_the_legacy_linear_hand_ramp(self):
        a, b = np.zeros(2), np.array([0.2, -0.1])
        got = ramp(a, b, 1.2, DT)
        n = int(np.ceil(np.abs(b - a).max() / (1.2 * DT) + 1e-9))
        expected = a + (b - a) * (np.arange(1, n + 1) / n)[:, None]
        np.testing.assert_array_equal(got, expected)
        self.assertIsNone(load_config(profile="none", overlay="none")["motion"]["output_caps"])

    def test_finals_caps_clamp_both_profiles_without_speeding_up_real(self):
        sim = load_config(profile="sim", overlay="finals_r21")["motion"]
        real = load_config(profile="none", overlay="finals_r21")["motion"]
        self.assertEqual(sim["output_caps"], CAPS)
        self.assertEqual((sim["arm_max_vel"], sim["arm_max_acc"], sim["hand_max_vel"], sim["hand_max_acc"]),
                         (0.85, 4.0, 1.5, 8.0))
        self.assertEqual((real["arm_max_vel"], real["arm_max_acc"], real["hand_max_vel"], real["hand_max_acc"]),
                         (0.5, 1.5, 1.2, 8.0))

    def test_acceleration_shaped_hand_ramp_obeys_velocity_and_acceleration(self):
        start, target = np.zeros(14), np.ones(14)
        path = ramp(start, target, CAPS["hand_vel"], DT, amax=CAPS["hand_acc"])
        velocity = rates(path, start)
        acceleration = np.diff(np.vstack([np.zeros((1, 14)), velocity]), axis=0) / DT
        self.assertLessEqual(float(np.abs(velocity).max()), CAPS["hand_vel"] + 1e-9)
        self.assertLessEqual(float(np.abs(acceleration).max()), CAPS["hand_acc"] + 1e-8)
        np.testing.assert_allclose(path[-1], target)

    def test_home_and_abort_go_to_uses_acceleration_shaped_hand_ramp(self):
        planner = Planner.__new__(Planner)
        planner.cfg = {"motion": {"arm_max_vel": CAPS["arm_vel"], "arm_max_acc": CAPS["arm_acc"],
                                  "hand_max_vel": CAPS["hand_vel"], "hand_max_acc": CAPS["hand_acc"],
                                  "output_caps": CAPS}}
        planner.dt = DT
        zero, arm_target, hand_target = np.zeros(14), np.ones(14), -np.ones(14)
        arm, hand = planner.go_to(zero, zero, arm_target, hand_target)
        for path, start, vmax, amax in ((arm, zero, CAPS["arm_vel"], CAPS["arm_acc"]),
                                        (hand, zero, CAPS["hand_vel"], CAPS["hand_acc"])):
            velocity = rates(path, start)
            acceleration = np.diff(np.vstack([np.zeros((1, 14)), velocity]), axis=0) / DT
            self.assertLessEqual(float(np.abs(velocity).max()), vmax + 1e-9)
            self.assertLessEqual(float(np.abs(acceleration).max()), amax + 1e-8)
        np.testing.assert_allclose(arm[-1], arm_target)
        np.testing.assert_allclose(hand[-1], hand_target)

    def test_stateful_output_caps_survive_queue_replacement(self):
        limiter = MotionOutputLimiter(0.85, 4.0, 1.5, 8.0, DT)
        zero = np.zeros(14)
        normal_a, normal_h = np.ones((10, 14)), np.ones((10, 14)) * 2.0
        a1, h1 = limiter.limit(normal_a, normal_h, zero, zero, now=0.0)
        # Abruptly replace the queue with a home/abort route in the opposite direction.
        abort_a, abort_h = -np.ones((10, 14)), -np.ones((10, 14)) * 2.0
        a2, h2 = limiter.limit(abort_a, abort_h, zero, zero, now=10 * DT)
        arm = np.vstack([a1, a2])
        hand = np.vstack([h1, h2])
        arm_v, hand_v = rates(arm, zero), rates(hand, zero)
        arm_acc = np.diff(np.vstack([np.zeros((1, 14)), arm_v]), axis=0) / DT
        hand_acc = np.diff(np.vstack([np.zeros((1, 14)), hand_v]), axis=0) / DT
        self.assertLessEqual(float(np.abs(arm_v).max()), 0.85 + 1e-9)
        self.assertLessEqual(float(np.abs(hand_v).max()), 1.5 + 1e-9)
        self.assertLessEqual(float(np.abs(arm_acc).max()), 4.0 + 1e-8)
        self.assertLessEqual(float(np.abs(hand_acc).max()), 8.0 + 1e-8)

    def test_invalid_or_faster_output_caps_fail_at_load(self):
        for caps in ({"arm_vel": 0.86, "arm_acc": 4.0, "hand_vel": 1.5, "hand_acc": 8.0},
                     {"arm_vel": 0.85, "arm_acc": 4.1, "hand_vel": 1.5, "hand_acc": 8.0},
                     {"arm_vel": 0.85, "arm_acc": 4.0, "hand_vel": 1.6, "hand_acc": 8.0},
                     {"arm_vel": 0.85, "arm_acc": 4.0, "hand_vel": 1.5}):
            with self.subTest(caps=caps), self.assertRaises(ValueError):
                load_config(overrides={"motion.output_caps": caps})


if __name__ == "__main__":
    unittest.main()
