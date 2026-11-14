#!/usr/bin/env python3
"""Markdown table of every config.FLAGS entry: config default vs finals_r21 vs the submission overlay.

    PYTHONPATH=. python3 tools/gen_flag_table.py > docs/flag_table.md

Rows whose finals_r21 value differs from the config default are bold. Warns on stderr when a FLAGS "off" value
disagrees with the config.yaml default (the registry and the file should not drift apart).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaltonomous.baselines.classical.config import FLAGS, STOCK, _get, load_config  # noqa: E402


def fmt(v):
    # IP literals are redacted: the doc this feeds must not carry hosts or addresses.
    s = re.sub(r"\d+\.\d+\.\d+\.\d+", "<ip>", json.dumps(v))
    return s.replace("|", "\\|")


def main():
    base, fin, sub = load_config(overlay="none"), load_config(overlay="finals_r21"), load_config(overlay="submission")
    print("| flag | what | default | finals_r21 | submission |")
    print("|---|---|---|---|---|")
    for k, (_owner, off, what) in FLAGS.items():
        d, f, s = _get(base, k), _get(fin, k), _get(sub, k)
        if off != STOCK and json.dumps(off) != json.dumps(d):
            print(f"warn: {k}: FLAGS off={off!r} but config default={d!r}", file=sys.stderr)
        cells = [f"`{k}`", what.replace("|", "\\|"), f"`{fmt(d)}`", f"`{fmt(f)}`", f"`{fmt(s)}`"]
        if json.dumps(f) != json.dumps(d):
            cells = [f"**{c}**" for c in cells]
        print("| " + " | ".join(cells) + " |")
    print(f"{len(FLAGS)} flags", file=sys.stderr)


if __name__ == "__main__":
    main()
