#!/usr/bin/env python3
"""Release gate: the baked overlay (overlays/submission.yaml) must carry exactly the flags of the chosen rcN.

    python3 tools/release_gate.py rc1      # exit 0 = PASS, 1 = FAIL (differing flags printed)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaltonomous.baselines.classical.config import (  # noqa: E402
    EXPECTED_FLAGS, FLAGS, _get, check_expected_flags, flags_dump, hand_cal_conflicts, load_config, pitch_cal_conflicts)


def diff_flags(a, b):
    """[(flag, a_value, b_value)] for every config.FLAGS key whose value differs (overlay name ignored)."""
    return [(k, _get(a, k), _get(b, k)) for k in FLAGS if _get(a, k) != _get(b, k)]


def gate(rc, out=print, expected=None):
    sub, rcfg = load_config(overlay="submission"), load_config(overlay=rc)
    diffs = diff_flags(sub, rcfg)
    for k, s, r in diffs:
        out(f"{k}: submission={s} rc={r}")
    conflicts = pitch_cal_conflicts(rcfg) + hand_cal_conflicts(rcfg)
    for c in conflicts:
        out(f"CONFLICT: {c}")
    # EXPECTED_FLAGS.json (written by promote.sh, read by the server at startup) must also equal the overlay.
    stale = [f"EXPECTED_FLAGS.json: {d}" for d in check_expected_flags(rcfg, expected)]
    for d in stale:
        out(d)
    ok = not diffs and not conflicts and not stale
    out(f"RELEASE GATE PASS: submission == {rc}" if ok else "RELEASE GATE FAIL")
    return ok


def write_expected(rc, path=EXPECTED_FLAGS):
    Path(path).write_text(json.dumps({"overlay": rc, "flags": flags_dump(load_config(overlay=rc))}, indent=1, sort_keys=True) + "\n")


if __name__ == "__main__":
    a = sys.argv[1:]
    if len(a) == 3 and a[1] == "--write-expected":   # promote.sh: dump the overlay's flags, then gate
        write_expected(a[0], a[2])
        a = a[:1]
    if len(a) != 1:
        sys.exit("usage: release_gate.py <rc_overlay_name> [--write-expected <path>]")
    sys.exit(0 if gate(a[0], expected=sys.argv[3] if len(sys.argv) == 4 else None) else 1)
