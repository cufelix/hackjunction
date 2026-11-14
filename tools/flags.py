#!/usr/bin/env python3
"""Print the effective feature-flag set of the classical policy (config.yaml + profile + overlay).

    python3 tools/flags.py                          # what tests / run.py / sim_eval.py get (no overlay in the repo)
    python3 tools/flags.py --overlay submission     # what the submission image gets (overlays/submission.yaml)
    python3 tools/flags.py --overlay path/to/x.yaml --json
    docker run --rm --entrypoint python3 <image> tools/flags.py     # inside the image (baked overlay.yaml)

Exit 0; with --expect-off exit 1 if any flag is on (CI / build guard for the baseline image).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaltonomous.baselines.classical.config import effective_flags, load_config, resolve_overlay  # noqa: E402


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--overlay", default=None, help='overlay name/path; "none" = no overlay (default: as load_config)')
    ap.add_argument("--profile", default=None, help="config profile (default: as load_config, i.e. sim)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--expect-off", action="store_true", help="exit 1 if any flag is on")
    a = ap.parse_args(argv)
    cfg = load_config(profile=a.profile, overlay=a.overlay)
    rows = effective_flags(cfg)
    ov = resolve_overlay(a.overlay)
    if a.json:
        print(json.dumps({"profile": cfg["profile"], "overlay": str(ov) if ov else None,
                          "flags": {k: {"value": v, "on": on, "owner": o} for k, v, on, o, _ in rows}}, indent=2))
    else:
        print(f"profile={cfg['profile']}  overlay={ov if ov else 'none'}")
        w = max(len(k) for k, *_ in rows)
        for k, v, on, owner, what in rows:
            print(f"  {'ON ' if on else 'off'}  {k:<{w}}  {json.dumps(v):<6}  {owner:<13} {what}")
        n = sum(on for _, _, on, _, _ in rows)
        print(f"{n} of {len(rows)} flags on" + ("  (= stock baseline behaviour)" if n == 0 else ""))
    return 1 if a.expect_off and any(on for _, _, on, _, _ in rows) else 0


if __name__ == "__main__":
    sys.exit(main())
