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

    def test_shipped_submission_overlay_is_the_stock_config(self):
        stock = load_config(overlay="none")
        sub = load_config(overlay="submission")
        self.assertEqual(sub.pop("overlay"), "submission")
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
                  "ab_carry05": {"motion.carry_scale"},
                  "ab_verify_lang": {"policy.verify_placement", "policy.verify_yellow_unknown",
                                     "policy.max_corrections_per_episode"}}
        for name, keys in expect.items():
            keys = keys | {"policy.grasp_diag_log"}     # passive diagnostics ride along in every A/B arm
            cfg = load_config(overlay=name)
            self.assertEqual(cfg["overlay"], name)
            on = {k for k, _, is_on, _, _ in effective_flags(cfg) if is_on}
            self.assertEqual(on, keys, name)
        self.assertEqual(load_config(overlay="ab_maxatt4")["policy"]["max_attempts"], 4)
        self.assertEqual(load_config(overlay="ab_triage")["policy"]["max_total_attempts"], 10)

    def test_hard_limits(self):
        """Overlays can never make anything faster than the stock sim profile."""
        self.assertFalse([k for k in FLAGS if k.startswith("motion.") and k not in
                          ("motion.carry_scale", "motion.close_hold_s")])
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fast.yaml"
            for bad in ("motion.carry_scale: 1.5", "motion.carry_scale: 0.0", "grasp.drop_slot_overflow: pile"):
                p.write_text("flags:\n  " + bad + "\n")
                with self.assertRaises(ValueError, msg=bad):
                    load_config(overlay=str(p))
        self.assertEqual(load_config(overlay="ab_hold12")["motion"]["close_hold_s"], 1.2)
        self.assertEqual(load_config(overlay="none")["motion"]["close_hold_s"], 0.25)   # sim profile

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
