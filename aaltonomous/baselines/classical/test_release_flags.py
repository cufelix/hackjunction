"""C1 build-time finals allow-list."""
import sys
import unittest
from pathlib import Path
from unittest import mock

from .config import load_config

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
import check_release_flags  # noqa: E402


class _Server:
    def __init__(self, profile, overlay):
        self.config_snapshot = load_config(profile=profile, overlay=overlay)


class TestReleaseFlags(unittest.TestCase):
    def test_finals_r22_is_the_reviewed_allow_list(self):
        lines = []
        with mock.patch.object(check_release_flags, "ClassicalServerPolicy", _Server):
            self.assertTrue(check_release_flags.check("finals_r22", out=lines.append))
        self.assertEqual(lines, ["RELEASE FLAGS PASS"])

    def test_missing_flag_fails(self):
        lines = []
        with mock.patch.object(check_release_flags, "ClassicalServerPolicy", _Server):
            self.assertFalse(check_release_flags.check("finals_r20", out=lines.append))
        self.assertTrue(any("motion.output_caps" in line for line in lines))
        self.assertEqual(lines[-1], "RELEASE FLAGS FAIL")

    def test_submission_dockerfile_runs_real_profile_assertion(self):
        dockerfile = (Path(__file__).resolve().parents[3] / "tools" / "submission.Dockerfile").read_text()
        self.assertIn("AALTO_PROFILE=none", dockerfile)
        self.assertIn("python3 tools/check_release_flags.py", dockerfile)


if __name__ == "__main__":
    unittest.main()
