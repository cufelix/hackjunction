"""Tests for hand-in helpers that do not require a Docker daemon."""
from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.handin_lib import (
    image_metrics,
    parse_registry_digest,
    policy_hz_rt,
    registry_repository,
    require_urdf,
    validate_push_reference,
    write_handin_record,
)


ROOT = Path(__file__).resolve().parents[1]


class TestHandin(unittest.TestCase):
    def test_slow_sim_rate_arithmetic_and_margin(self):
        self.assertAlmostEqual(policy_hz_rt(10, 0.30), 90.0)
        self.assertAlmostEqual(policy_hz_rt(6, 0.30), 54.0)
        self.assertAlmostEqual(policy_hz_rt(15, 0.30), 135.0)
        self.assertAlmostEqual(policy_hz_rt(10, 0.30), 90.0)
        self.assertAlmostEqual(policy_hz_rt(6, 0.15), 27.0)
        self.assertAlmostEqual(policy_hz_rt(10, 0.15), 45.0)
        self.assertAlmostEqual(policy_hz_rt(15, 0.15), 67.5)
        self.assertAlmostEqual(policy_hz_rt(10, 0.15) - 27.0, 18.0)
        with self.assertRaises(ValueError):
            policy_hz_rt(0, 0.3)

    def test_registry_reference_keeps_registry_port(self):
        self.assertEqual(registry_repository("registry.example/policy:v1"), "registry.example/policy")
        self.assertEqual(registry_repository("localhost:5000/ns/policy:v1"), "localhost:5000/ns/policy")
        self.assertEqual(registry_repository("ns/policy@sha256:" + "a" * 64), "ns/policy")

    def test_push_reference_requires_registry_repository_and_unique_tag(self):
        self.assertEqual(validate_push_reference("registry.example/ns/policy:r22"),
                         "registry.example/ns/policy:r22")
        self.assertEqual(validate_push_reference("localhost:5000/policy:r22"), "localhost:5000/policy:r22")
        for bad in ("ubuntu", "namespace/policy:r22", "registry.example/policy", "registry.example/policy:latest",
                    "registry.example/policy@sha256:" + "a" * 64):
            with self.assertRaises(ValueError, msg=bad):
                validate_push_reference(bad)

    def test_digest_must_come_from_registry_label(self):
        digest = "sha256:" + "a" * 64
        self.assertEqual(parse_registry_digest(f"Name: x\nDigest: {digest}\n"), digest)
        self.assertEqual(parse_registry_digest(f"v1: digest: {digest} size: 1234\n"), digest)
        with self.assertRaises(ValueError):
            parse_registry_digest(f"image ID: {digest}\n")

    def test_image_metrics(self):
        payload = [{"Id": "sha256:" + "b" * 64, "Size": 1234,
                    "RootFS": {"Layers": ["sha256:x", "sha256:y"]}}]
        self.assertEqual(image_metrics(payload), ("sha256:" + "b" * 64, 1234, 2))

    def test_urdf_must_be_nonempty_regular_file_not_symlink(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            real = ROOT / "third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
            self.assertEqual(require_urdf(real),
                             "8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735")
            link = root / "link.urdf"
            link.symlink_to(real)
            with self.assertRaises(ValueError):
                require_urdf(link)
            wrong = root / "robot.urdf"
            wrong.write_text("<robot/>")
            with self.assertRaises(ValueError):
                require_urdf(wrong)
            with self.assertRaises(ValueError):
                require_urdf(root / "missing.urdf")

    def test_handin_record_contains_required_measurements(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            flags = root / "flags.json"
            local = root / "local.json"
            out = root / "HANDIN.json"
            flags.write_text(json.dumps({"profile": "sim", "flags": {"x": {"on": False}}}))
            local.write_text(json.dumps({"cold_start_s": 1.25, "first_action_latency_ms": 4.5,
                                         "policy_hz_wall": 2222.2, "pass": True}))
            args = argparse.Namespace(
                output=str(out), git_sha="1" * 40,
                urdf_sha256="8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735",
                platform="linux/amd64", multi_arch=False,
                registry_reference="registry.example/g1:v1", digest="sha256:" + "c" * 64,
                image_id="sha256:" + "d" * 64, size=1048576, layers=7, flags=str(flags),
                local_smoke=str(local), pulled_smoke=None, pulled_image_id=None, pulled_size=None,
                pulled_layers=None, pulled_tag=None,
            )
            write_handin_record(args)
            record = json.loads(out.read_text())
            self.assertEqual(record["git_sha"], "1" * 40)
            self.assertEqual(record["urdf_sha256"],
                             "8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735")
            self.assertEqual(record["digest_reference"],
                             "registry.example/g1@sha256:" + "c" * 64)
            self.assertEqual((record["size_bytes"], record["layer_count"]), (1048576, 7))
            self.assertEqual((record["cold_start_s"], record["first_action_latency_ms"]), (1.25, 4.5))
            self.assertIn("flags_dump", record)

    def test_shell_syntax_and_help_do_not_need_docker(self):
        script = ROOT / "tools" / "handin.sh"
        subprocess.run(["bash", "-n", str(script)], check=True)
        help_run = subprocess.run(["bash", str(script), "--help"], check=True, capture_output=True, text=True)
        self.assertIn("--multi-arch", help_run.stdout)


if __name__ == "__main__":
    unittest.main()
