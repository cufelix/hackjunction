"""promote_overlay.py and promote.sh: submission.yaml takes the rc overlay's effective flags, comments kept.

Works only in temp copies of the repo. Config is loaded in child processes, so the copied code is what runs.
The bash checks skip when bash is missing.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
PY = sys.executable
PARITY = ("import json, sys; from aaltonomous.baselines.classical.config import FLAGS, _get, load_config; "
          "c = load_config(overlay=sys.argv[1]); print(json.dumps({k: _get(c, k) for k in FLAGS}, sort_keys=True))")
ENV = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class PromoteTest(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="promote-test-"))
        self.root = self.tmp / "repo"
        shutil.copytree(REPO / "aaltonomous", self.root / "aaltonomous",
                        ignore=shutil.ignore_patterns("__pycache__"))
        shutil.copytree(REPO / "tools", self.root / "tools")
        self.sub = self.root / "aaltonomous/baselines/classical/overlays/submission.yaml"
        self.rc1 = self.root / "aaltonomous/baselines/classical/overlays/rc1.yaml"

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def child_flags(self, overlay_path):
        r = subprocess.run([PY, "-c", PARITY, str(overlay_path)], cwd=self.root, env=ENV,
                           capture_output=True, text=True, check=True)
        return json.loads(r.stdout)

    def promote(self, name):
        return subprocess.run([PY, str(self.root / "tools/promote_overlay.py"), name, str(self.sub)],
                              cwd=self.root, env=ENV, capture_output=True, text=True)

    def test_rewrite_gives_rc1_flag_parity_and_keeps_comments(self):
        before = self.sub.read_text().splitlines()
        r = self.promote("rc1")
        self.assertEqual(r.returncode, 0, r.stderr)
        self.assertIn("CHANGED", r.stdout)
        after = self.sub.read_text().splitlines()
        self.assertEqual(len(before), len(after))
        self.assertEqual(sum("#" in x for x in before), sum("#" in x for x in after))
        self.assertEqual(self.child_flags(self.sub), self.child_flags(self.rc1))

    def test_second_run_changes_nothing(self):
        self.assertEqual(self.promote("rc1").returncode, 0)
        digest = sha(self.sub)
        r = self.promote("rc1")
        self.assertEqual((r.returncode, r.stdout.strip()), (0, "NO CHANGES"))
        self.assertEqual(sha(self.sub), digest)

    def test_gate_passes_after_promotion(self):
        self.assertEqual(self.promote("rc1").returncode, 0)
        r = subprocess.run([PY, "tools/release_gate.py", "rc1"], cwd=self.root, env=ENV,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 0, r.stdout + r.stderr)

    @unittest.skipUnless(shutil.which("bash"), "bash not available")
    def test_unknown_overlay_exits_1_and_changes_nothing(self):
        digest = sha(self.sub)
        env = dict(ENV, PYTHON=PY)
        r = subprocess.run(["bash", "tools/promote.sh", "no_such_rc"], cwd=self.root, env=env,
                           capture_output=True, text=True)
        self.assertEqual(r.returncode, 1, r.stderr)
        self.assertEqual(sha(self.sub), digest)
        self.assertFalse((self.root / "runs").exists())
        self.assertFalse((self.root / "RELEASE.md").exists())


if __name__ == "__main__":
    unittest.main()
