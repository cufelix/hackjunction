"""Feature flags + the submission overlay (integration branch). Run:
    python -m unittest aaltonomous.baselines.classical.test_flags -v
"""
import tempfile
import unittest
from pathlib import Path

from .config import BAKED_OVERLAY, FLAGS, OVERLAY_DIR, STOCK, _get, effective_flags, load_config, load_overlay


class TestFlags(unittest.TestCase):
    def test_every_flag_defaults_off(self):
        cfg = load_config(overlay="none")
        for k, (_, off, _) in FLAGS.items():
            if off != STOCK:
                self.assertEqual(_get(cfg, k), off, k)
        self.assertFalse(any(on for _, _, on, _, _ in effective_flags(cfg)))

    def test_no_baked_overlay_in_the_repo(self):
        self.assertFalse(BAKED_OVERLAY.exists(), "overlay.yaml is only created inside the submission image")

    def test_shipped_submission_overlay_is_stock_plus_the_ab_proven_flags(self):
        stock = load_config(overlay="none")
        sub = load_config(overlay="submission")
        self.assertEqual(sub.pop("overlay"), "submission")
        on = {k for k, _, is_on, _, _ in effective_flags(load_config(overlay="submission")) if is_on}
        self.assertEqual(on, {"instruction.resolve", "policy.bin_lookup", "grasp.close_posture", "hand.closed_tight",
                              "grasp.point_in_palm", "motion.close_hold_s"})        # A/B: sort_language 54/74 vs 1/74 (box A, twice)
        sub["instruction"]["resolve"] = False
        sub["policy"]["bin_lookup"] = "colour"
        for k in ("close_posture", "point_in_palm"):                       # pinch_v1 grasp (Modal 60/82 vs 52/82)
            sub["grasp"][k] = stock["grasp"][k]
        sub["hand"]["closed_tight"] = stock["hand"]["closed_tight"]
        sub["motion"]["close_hold_s"] = stock["motion"]["close_hold_s"]
        self.assertEqual(sub, stock)
        self.assertEqual(set(load_overlay(OVERLAY_DIR / "submission.yaml")["flags"]), set(FLAGS))

    def test_overlay_flips_only_its_flags(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.yaml"
            p.write_text("name: x\nflags:\n  policy.verify_placement: true\n  policy.max_total_attempts: 12\n")
            cfg = load_config(overlay=str(p))
        on = {k for k, _, is_on, _, _ in effective_flags(cfg) if is_on}
        self.assertEqual(on, {"policy.verify_placement", "policy.max_total_attempts"})
        self.assertEqual(cfg["profile"], "sim")      # applied on top of the sim profile
        self.assertEqual(cfg["motion"]["arm_max_vel"], load_config(overlay="none")["motion"]["arm_max_vel"])

    def test_ab_overlays_flip_exactly_their_flags(self):
        expect = {"ab_retry": {"policy.retry_pass"}, "ab_maxatt4": {"policy.max_attempts"},
                  "ab_resolver": {"instruction.resolve"}, "ab_verify": {"policy.verify_placement"},
                  "ab_triage": {"policy.time_triage", "policy.max_total_attempts"},
                  "ab_hold06": {"motion.close_hold_s"}, "ab_hold12": {"motion.close_hold_s"},
                  "ab_slots": {"grasp.drop_slots"}, "ab_hold12_slots": {"motion.close_hold_s", "grasp.drop_slots"},
                  "ab_carry05": {"motion.carry_scale"}, "ab_calib": {"perception.self_calibration.enabled"},
                  "ab_yaff": {"perception.y_affine"},
                  "ab_pinch_hybrid": {"grasp.close_posture", "hand.closed_tight", "grasp.point_in_palm",
                                      "motion.close_hold_s", "perception.classify_shape",
                                      "grasp.close_posture_by_shape"},
                  "ab_pinch_hybrid_apex": {"grasp.close_posture", "hand.closed_tight", "grasp.point_in_palm",
                                           "motion.close_hold_s", "perception.classify_shape",
                                           "grasp.close_posture_by_shape", "grasp.posture_apex_gate"},
                  "ab_pinch_hybrid_yaff": {"grasp.close_posture", "hand.closed_tight", "grasp.point_in_palm",
                                           "motion.close_hold_s", "perception.classify_shape",
                                           "grasp.close_posture_by_shape",
                                           "perception.y_affine"},
                  "ab_pinch_demo_stall": {"grasp.close_posture", "hand.closed_tight", "grasp.point_in_palm",
                                          "motion.lift_on_stall.enabled"},
                  "ab_lang_full": {"instruction.resolve", "perception.classify_shape", "perception.no_blob_shape",
                                   "policy.bin_lookup"},
                  "ab_verify_lang": {"policy.verify_placement", "policy.verify_yellow_unknown",
                                     "policy.max_corrections_per_episode"},
                  "ab_prism_narrow": {"instruction.resolve", "policy.bin_lookup", "grasp.point_in_palm", "grasp.close_posture",
                              "hand.closed_tight", "motion.close_hold_s", "perception.hsv_set", "perception.split_blobs",
                              "perception.classify_shape", "grasp.close_posture_by_shape", "grasp.posture_apex_gate"} | {"grasp.prism_posture"},
                  "ab_prism_contact": {"instruction.resolve", "policy.bin_lookup", "grasp.point_in_palm", "grasp.close_posture",
                              "hand.closed_tight", "motion.close_hold_s", "perception.hsv_set", "perception.split_blobs",
                              "perception.classify_shape", "grasp.close_posture_by_shape", "grasp.posture_apex_gate",
                              "grasp.prism_posture", "grasp.close_to_contact"},
                  "ab_prism_cc25": {"instruction.resolve", "policy.bin_lookup", "grasp.point_in_palm", "grasp.close_posture",
                              "hand.closed_tight", "motion.close_hold_s", "perception.hsv_set", "perception.split_blobs",
                              "perception.classify_shape", "grasp.close_posture_by_shape", "grasp.posture_apex_gate"} | {"grasp.close_to_contact", "grasp.contact_squeeze_rad"},
                  "ab_prism_cc40": {"instruction.resolve", "policy.bin_lookup", "grasp.point_in_palm", "grasp.close_posture",
                              "hand.closed_tight", "motion.close_hold_s", "perception.hsv_set", "perception.split_blobs",
                              "perception.classify_shape", "grasp.close_posture_by_shape", "grasp.posture_apex_gate"} | {"grasp.close_to_contact", "grasp.contact_squeeze_rad"},
                  "ab_prism_offset": {"instruction.resolve", "policy.bin_lookup", "grasp.point_in_palm", "grasp.close_posture",
                              "hand.closed_tight", "motion.close_hold_s", "perception.hsv_set", "perception.split_blobs",
                              "perception.classify_shape", "grasp.close_posture_by_shape", "grasp.posture_apex_gate"} | {"grasp.prism_offset", "perception.triangle_pose"},
                  "ab_realcolour": {"instruction.resolve", "policy.bin_lookup", "grasp.point_in_palm",
                                    "grasp.close_posture", "hand.closed_tight", "motion.close_hold_s",
                                    "perception.hsv_set", "perception.split_blobs"}}
        for name, keys in expect.items():
            keys = keys | {"policy.grasp_diag_log"}     # passive diagnostics ride along in every A/B arm
            cfg = load_config(overlay=name)
            self.assertEqual(cfg["overlay"], name)
            on = {k for k, _, is_on, _, _ in effective_flags(cfg) if is_on}
            self.assertEqual(on, keys, name)
        self.assertEqual(load_config(overlay="ab_maxatt4")["policy"]["max_attempts"], 4)
        self.assertEqual(load_config(overlay="ab_triage")["policy"]["max_total_attempts"], 10)

    def test_every_overlay_loads_and_its_postures_are_in_urdf_limits(self):
        from .config import hand_postures, home_arm
        from .kinematics import ArmKinematics
        for f in sorted(OVERLAY_DIR.glob("*.yaml")):
            cfg = load_config(overlay=str(f))
            hand_postures(cfg)                            # raises outside joint_spec (URDF) limits
            home_arm(cfg)
        for name, pip in (("ab_pinch_v1", [0.0862, -0.0864, 0.0143]), ("ab_pinch_demo", [0.1094, -0.0694, 0.0142]),
                          ("ab_pinch_v1_yaff", [0.0862, -0.0864, 0.0143]),
                          ("ab_pinch_demo_yaff", [0.1094, -0.0694, 0.0142]),
                          ("ab_pinch_demo_yaw", [0.1094, -0.0694, 0.0142])):
            cfg = load_config(overlay=name)
            self.assertEqual(cfg["grasp"]["close_posture"], "closed_tight")
            kin = ArmKinematics(cfg)
            self.assertEqual(list(kin.offset["left"]), pip)
            self.assertEqual(list(kin.offset["right"]), [pip[0], -pip[1], pip[2]])   # right = [x, +y, z]

    def test_thumb0_nonzero_fails_until_the_mirror_is_fixed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t0.yaml"
            p.write_text("flags:\n  grasp.close_posture: closed_tight\n"
                         "  hand.closed_tight: [-0.67, 0.92, 0.15, -0.28, -0.02, -0.19, -0.05]\n")
            with self.assertRaisesRegex(ValueError, "thumb0≠0 requires fixing mirror_left_to_right"):
                load_config(overlay=str(p))
        with self.assertRaisesRegex(ValueError, "keep thumb0 sign"):
            load_config(overrides={"hand.preshape": [0.3, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4]})

    def test_hard_limits(self):
        """Overlays can never make anything faster than the stock sim profile."""
        self.assertFalse([k for k in FLAGS if k.startswith("motion.") and k not in
                          ("motion.carry_scale", "motion.close_hold_s", "motion.lift_on_stall.enabled",
                           "motion.lift_on_stall.timeout_s")])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fast.yaml"
            for bad in ("motion.carry_scale: 1.5", "motion.carry_scale: 0.0", "grasp.drop_slot_overflow: pile"):
                p.write_text("flags:\n  " + bad + "\n")
                with self.assertRaises(ValueError, msg=bad):
                    load_config(overlay=str(p))
        self.assertEqual(load_config(overlay="ab_hold12")["motion"]["close_hold_s"], 1.2)
        self.assertEqual(load_config(overlay="none")["motion"]["close_hold_s"], 0.25)   # sim profile

    def test_placeholder_overlay_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tbd.yaml"
            p.write_text("placeholder: true\nflags:\n  grasp.close_posture: closed_tight\n")
            with self.assertRaises(ValueError):
                load_config(overlay=str(p))                # values TBD: must not run by accident
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pinch.yaml"
            p.write_text("flags:\n  grasp.close_posture: closed_tight\n  grasp.point_in_palm: [0.099, -0.084, 0.0]\n"
                         "  hand.closed_tight: [0.0, -0.1, 1.2, -1.1, -0.7, -1.1, -0.7]\n")
            cfg = load_config(overlay=str(p))
            self.assertEqual(cfg["grasp"]["close_posture"], "closed_tight")
            p.write_text("flags:\n  grasp.point_in_palm: [0.1, 0.0]\n")
            with self.assertRaises(ValueError):
                load_config(overlay=str(p))

    def test_unknown_overlay_key_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "typo.yaml"
            p.write_text("flags:\n  policy.verify_placment: true\n")
            with self.assertRaises(ValueError):
                load_config(overlay=str(p))
            p.write_text("flags:\n  motion.arm_max_vel: 2.0\n")   # not a flag: overlays cannot retune
            with self.assertRaises(ValueError):
                load_config(overlay=str(p))


if __name__ == "__main__":
    unittest.main()
