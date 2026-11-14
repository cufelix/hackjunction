"""motion.arm_vel_cap: rad/s cap on arm_max_vel, arm_max_acc scaled by the same ratio, null = profile, never faster."""
import unittest

from .config import load_config


class TestArmVelCap(unittest.TestCase):
    def test_off_is_the_profile(self):
        a, b = load_config(profile="sim", overlay="none"), load_config(profile="sim", overlay="cand_r17")
        self.assertIsNone(b["motion"]["arm_vel_cap"])
        self.assertEqual((a["motion"]["arm_max_vel"], a["motion"]["arm_max_acc"]), (0.97, 8.0))
        self.assertEqual(b["motion"]["arm_max_vel"], 0.97)

    def test_cap_scales_vel_and_acc(self):
        for v in (80, 85, 90):
            mo = load_config(profile="sim", overlay=f"ab_vel{v}")["motion"]
            self.assertAlmostEqual(mo["arm_max_vel"], v / 100)
            self.assertAlmostEqual(mo["arm_max_acc"], 8.0 * (v / 100) / 0.97)

    def test_cap_never_speeds_up_a_slower_profile(self):
        mo = load_config(profile="none", overrides={"motion.arm_vel_cap": 0.8})["motion"]
        self.assertEqual((mo["arm_max_vel"], mo["arm_max_acc"]), (0.5, 1.5))

    def test_out_of_range_fails_at_load(self):
        with self.assertRaises(ValueError):
            load_config(overrides={"motion.arm_vel_cap": 1.2})

    def test_ab_vel_overlays_differ_from_cand_r17_only_by_the_cap(self):
        from .config import FLAGS, _get
        base = load_config(profile="sim", overlay="cand_r17")
        for v in (80, 85, 90):
            cfg = load_config(profile="sim", overlay=f"ab_vel{v}")
            diff = [k for k in FLAGS if _get(cfg, k) != _get(base, k)]
            self.assertEqual(diff, ["motion.arm_vel_cap"])


if __name__ == "__main__":
    unittest.main()
