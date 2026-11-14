#!/usr/bin/env python3
"""Release gate: the baked overlay (overlays/submission.yaml) must carry exactly the flags of the chosen rcN.

    python3 tools/release_gate.py rc1      # exit 0 = PASS, 1 = FAIL (differing flags printed)
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaltonomous.baselines.classical.config import FLAGS, _get, load_config  # noqa: E402


def diff_flags(a, b):
    """[(flag, a_value, b_value)] for every config.FLAGS key whose value differs (overlay name ignored)."""
    return [(k, _get(a, k), _get(b, k)) for k in FLAGS if _get(a, k) != _get(b, k)]


def gate(rc, out=print):
    diffs = diff_flags(load_config(overlay="submission"), load_config(overlay=rc))
    for k, s, r in diffs:
        out(f"{k}: submission={s} rc={r}")
    out(f"RELEASE GATE PASS: submission == {rc}" if not diffs else "RELEASE GATE FAIL")
    return not diffs


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: release_gate.py <rc_overlay_name>")
    sys.exit(0 if gate(sys.argv[1]) else 1)
