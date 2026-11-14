"""tools/release_gate.py: submission overlay must equal the chosen rcN flags."""
import sys
import unittest
from unittest import mock
from pathlib import Path

from .config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import release_gate  # noqa: E402


class TestReleaseGate(unittest.TestCase):
    def test_rc1_fails_because_it_differs_from_promoted_finals(self):
        lines = []
        self.assertFalse(release_gate.gate("rc1", out=lines.append))
        self.assertEqual(lines[-1], "RELEASE GATE FAIL")
        self.assertTrue(any(l.startswith("instruction.strict: submission=") for l in lines))

    def test_identical_overlay_passes(self):
        lines = []
        self.assertTrue(release_gate.gate("submission", out=lines.append))
        self.assertEqual(lines, ["RELEASE GATE PASS: submission == submission"])

    def test_diff_flags_ignores_overlay_name_and_reports_flag_diffs(self):
        a, b = load_config(overlay="submission"), load_config(overlay="submission")
        b["overlay"] = "other"
        self.assertEqual(release_gate.diff_flags(a, b), [])
        b["policy"]["max_attempts"] = 9
        self.assertEqual(release_gate.diff_flags(a, b), [("policy.max_attempts", a["policy"]["max_attempts"], 9)])

    def test_pitch_cal_conflict_fails_even_when_flags_match(self):
        cfg = load_config(overrides={"perception.online_pitch_cal": True, "perception.y_affine": [0.02, 0.0]})
        lines = []
        with mock.patch.object(release_gate, "load_config", lambda overlay: cfg):
            self.assertFalse(release_gate.gate("rc1", out=lines.append))
        self.assertTrue(lines[0].startswith("CONFLICT: perception.online_pitch_cal is ON"))
        self.assertEqual(lines[-1], "RELEASE GATE FAIL")

    def test_hand_cal_conflict_rules(self):
        from .config import hand_cal_conflicts
        on = {"perception.hand_calibration.enabled": True}
        self.assertEqual(hand_cal_conflicts(load_config(overrides=on)), [])
        self.assertEqual(hand_cal_conflicts(load_config(overrides={**on, "perception.online_pitch_cal": True})), [])
        self.assertEqual(len(hand_cal_conflicts(load_config(overrides={**on, "perception.y_affine": [0.02, 0.0]}))), 1)
        self.assertEqual(len(hand_cal_conflicts(load_config(overrides={**on, "perception.xy_offset": [0.0, 0.01]}))), 1)
        self.assertEqual(hand_cal_conflicts(load_config(overrides={"perception.y_affine": [0.02, 0.0]})), [])


if __name__ == "__main__":
    unittest.main()
