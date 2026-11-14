#!/usr/bin/env python3
"""Bake PROFILE_MODE (auto | sim | real) into an overlay file: sets flags['profile_auto.force'].

auto -> null (auto-detect, unchanged), sim -> 'sim', real -> 'none'. Used by the Dockerfiles at build time.
    python3 tools/set_profile_mode.py aaltonomous/baselines/classical/overlay.yaml real
"""
import sys

import yaml

MODES = {"auto": None, "sim": "sim", "real": "none"}


def main(path, mode):
    if mode not in MODES:
        raise SystemExit(f"PROFILE_MODE={mode!r}: want auto | sim | real")
    d = yaml.safe_load(open(path))
    d.setdefault("flags", {})["profile_auto.force"] = MODES[mode]
    yaml.safe_dump(d, open(path, "w"), sort_keys=False)
    print(f"profile_auto.force = {MODES[mode]!r} (PROFILE_MODE={mode}) in {path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    main(*sys.argv[1:])
