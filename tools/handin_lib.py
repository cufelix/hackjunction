#!/usr/bin/env python3
"""Small, testable helpers for tools/handin.sh (standard library only)."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path


DIGEST_RE = re.compile(r"(?:^|\s)digest:\s*(sha256:[0-9a-f]{64})(?:\s|$)", re.IGNORECASE)
EXPECTED_URDF_SHA256 = "8bbf006633fc50b616f665c7a970780cc296577a0adfd7d28b049e751c238735"


def policy_hz_rt(ticks_per_reply, rtf, runner_hz=30.0):
    """Slow-sim command-rate model used by the runner gate."""
    ticks, factor, hz = float(ticks_per_reply), float(rtf), float(runner_hz)
    if ticks <= 0 or factor < 0 or hz <= 0:
        raise ValueError("ticks and runner_hz must be positive; rtf must be non-negative")
    return ticks * factor * hz


def registry_repository(reference):
    """Remove a tag or digest from a Docker reference while preserving a registry port."""
    ref = reference.strip()
    if not ref or any(c.isspace() for c in ref):
        raise ValueError("registry reference must be a non-empty Docker image reference")
    ref = ref.split("@", 1)[0]
    slash, colon = ref.rfind("/"), ref.rfind(":")
    repo = ref[:colon] if colon > slash else ref
    if not repo or repo.endswith("/"):
        raise ValueError(f"invalid registry reference {reference!r}")
    return repo


def validate_push_reference(reference):
    """Require an explicit registry host, repository, and release tag; digest destinations are not mutable push tags."""
    ref = reference.strip()
    if "@" in ref:
        raise ValueError("push destination must be a tag, not a digest reference")
    if "/" not in ref:
        raise ValueError("push destination must include an explicit registry host and repository")
    host, remainder = ref.split("/", 1)
    if not remainder or not (host == "localhost" or "." in host or ":" in host):
        raise ValueError("push destination must start with an explicit registry host")
    slash, colon = ref.rfind("/"), ref.rfind(":")
    if colon <= slash or not ref[colon + 1:]:
        raise ValueError("push destination must have an explicit tag")
    if ref[colon + 1:] == "latest":
        raise ValueError("refusing mutable ':latest'; use a unique release tag")
    registry_repository(ref)
    return ref


def parse_registry_digest(output):
    """Read the manifest digest reported by a registry-facing Docker command."""
    matches = DIGEST_RE.findall(output)
    if not matches:
        raise ValueError("registry output did not contain a 'Digest: sha256:…' line")
    return matches[0].lower()


def image_metrics(inspect_payload):
    data = json.loads(inspect_payload) if isinstance(inspect_payload, str) else inspect_payload
    if not isinstance(data, list) or len(data) != 1:
        raise ValueError("expected docker image inspect JSON for exactly one image")
    image = data[0]
    layers = (image.get("RootFS") or {}).get("Layers")
    if not isinstance(layers, list):
        raise ValueError("docker image inspect JSON has no RootFS.Layers list")
    size = image.get("Size")
    if not isinstance(size, int) or size < 0:
        raise ValueError("docker image inspect JSON has no non-negative integer Size")
    image_id = image.get("Id")
    if not isinstance(image_id, str) or not image_id.startswith("sha256:"):
        raise ValueError("docker image inspect JSON has no sha256 image ID")
    return image_id, size, len(layers)


def require_urdf(path):
    p = Path(path)
    if p.is_symlink():
        raise ValueError(f"required URDF must be a real file in the build context, not a symlink: {p}")
    if not p.is_file():
        raise ValueError(f"required URDF is missing from the build context: {p}")
    if p.stat().st_size == 0:
        raise ValueError(f"required URDF is empty: {p}")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    if digest != EXPECTED_URDF_SHA256:
        raise ValueError(f"required URDF checksum mismatch: {p} has sha256:{digest}")
    return digest


def write_smoke_record(path, cold_start_s, action_path, ready_line):
    action = json.loads(Path(action_path).read_text())
    record = {
        "cold_start_s": round(float(cold_start_s), 3),
        "ready_line": ready_line,
        **action,
    }
    Path(path).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")


def write_handin_record(args):
    flags = json.loads(Path(args.flags).read_text())
    local = json.loads(Path(args.local_smoke).read_text())
    pulled = json.loads(Path(args.pulled_smoke).read_text()) if args.pulled_smoke else None
    pulled_image = None
    if args.pulled_image_id:
        pulled_image = {
            "tag": args.pulled_tag,
            "image_id": args.pulled_image_id,
            "size_bytes": args.pulled_size,
            "layer_count": args.pulled_layers,
        }
    digest = args.digest or None
    record = {
        "schema_version": 1,
        "generated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "git_sha": args.git_sha,
        "urdf_sha256": args.urdf_sha256,
        "platform": args.platform,
        "multi_arch": args.multi_arch,
        "registry_reference": args.registry_reference or None,
        "digest": digest,
        "digest_reference": f"{registry_repository(args.registry_reference)}@{digest}"
        if digest and args.registry_reference else None,
        "image_id": args.image_id,
        "size_bytes": args.size,
        "size_mib": round(args.size / (1024 * 1024), 2),
        "layer_count": args.layers,
        "cold_start_s": local["cold_start_s"],
        "first_action_latency_ms": local["first_action_latency_ms"],
        "first_action_policy_hz": local["policy_hz_wall"],
        "flags_dump": flags,
        "verification": {"local": local, "pulled": pulled, "pulled_image": pulled_image},
    }
    target = Path(args.output)
    temp = target.with_name(f".{target.name}.tmp-{os.getpid()}")
    temp.write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    os.replace(temp, target)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("now-ns")
    p = sub.add_parser("require-urdf")
    p.add_argument("path")
    sub.add_parser("image-metrics")
    sub.add_parser("registry-digest")
    p = sub.add_parser("registry-repository")
    p.add_argument("reference")
    p = sub.add_parser("validate-push-reference")
    p.add_argument("reference")
    p = sub.add_parser("smoke-record")
    p.add_argument("--output", required=True)
    p.add_argument("--cold-start-s", required=True, type=float)
    p.add_argument("--action", required=True)
    p.add_argument("--ready-line", required=True)
    p = sub.add_parser("write-record")
    p.add_argument("--output", required=True)
    p.add_argument("--git-sha", required=True)
    p.add_argument("--urdf-sha256", required=True)
    p.add_argument("--platform", required=True)
    p.add_argument("--multi-arch", action="store_true")
    p.add_argument("--registry-reference", default="")
    p.add_argument("--digest", default="")
    p.add_argument("--image-id", required=True)
    p.add_argument("--size", required=True, type=int)
    p.add_argument("--layers", required=True, type=int)
    p.add_argument("--flags", required=True)
    p.add_argument("--local-smoke", required=True)
    p.add_argument("--pulled-smoke")
    p.add_argument("--pulled-image-id")
    p.add_argument("--pulled-tag")
    p.add_argument("--pulled-size", type=int)
    p.add_argument("--pulled-layers", type=int)
    args = parser.parse_args(argv)

    if args.command == "now-ns":
        import time
        print(time.monotonic_ns())
    elif args.command == "require-urdf":
        print(require_urdf(args.path))
    elif args.command == "image-metrics":
        print("\t".join(map(str, image_metrics(sys.stdin.read()))))
    elif args.command == "registry-digest":
        print(parse_registry_digest(sys.stdin.read()))
    elif args.command == "registry-repository":
        print(registry_repository(args.reference))
    elif args.command == "validate-push-reference":
        print(validate_push_reference(args.reference))
    elif args.command == "smoke-record":
        write_smoke_record(args.output, args.cold_start_s, args.action, args.ready_line)
    elif args.command == "write-record":
        write_handin_record(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
