#!/usr/bin/env python3
"""Flattened EFFECTIVE sim-profile config (profile sim + overlay + profile_auto.sim_overrides) as sorted key=value lines.
Run in two trees (PYTHONPATH=<tree>) and diff:  tools/dump_sim_cfg.py finals_r23 > a.txt ; ... finals_r24 > b.txt"""
import hashlib
import json
import sys

from aaltonomous.baselines.classical.config import load_config


def flat(d, p=""):
    for k, v in d.items():
        if isinstance(v, dict) and v:
            yield from flat(v, f"{p}{k}.")
        else:
            yield f"{p}{k}", v


def main(overlay):
    probe = load_config(profile="none", overlay=overlay)
    so = dict(probe["profile_auto"].get("sim_overrides") or {})
    cfg = load_config(profile="sim", overlay=overlay, overrides=so)
    lines = sorted(f"{k}={json.dumps(v, sort_keys=True, default=str)}" for k, v in flat(cfg))
    print("\n".join(lines))
    print("# sha256", hashlib.sha256("\n".join(lines).encode()).hexdigest()[:16], file=sys.stderr)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
