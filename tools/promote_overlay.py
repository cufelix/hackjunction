#!/usr/bin/env python3
"""Make overlays/submission.yaml carry the effective FLAGS values of another overlay, in place.

    python3 tools/promote_overlay.py rc1 aaltonomous/baselines/classical/overlays/submission.yaml

Compares effective values (config.yaml + profile + overlay, as release_gate.py does) and rewrites only the value
token of each differing `  key: value   # comment` line. Comments, order and every other line stay as they were.
Prints `CHANGED key: old -> new` per change. The new file is verified in a temp file before it replaces the target.
Exit 0 (also when nothing differs), 2 bad input, 3 verification failed (target untouched).
"""
from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from aaltonomous.baselines.classical.config import FLAGS, OVERLAY_DIR, _get, load_config  # noqa: E402

PLAIN = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
RESERVED = {"true", "false", "null", "yes", "no", "on", "off", "y", "n"}


def fmt(v):
    """YAML text of a flag value, in the style submission.yaml already uses."""
    if v is None:
        return "null"
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    if isinstance(v, (int, float)):
        return repr(v)
    s = str(v)
    return s if PLAIN.fullmatch(s) and s.lower() not in RESERVED else json.dumps(s)


def replace_line(line, key, new):
    """`  key: old   # c` -> `  key: new   # c`, comment kept in its column when the value still fits."""
    m = re.match(r"^([ \t]+)" + re.escape(key) + r":([ \t]*)(.*?)([ \t]+#.*)?$", line)
    if m is None:
        return None
    ind, sp, _old, cm = m.groups()
    head = f"{ind}{key}:{sp}{fmt(new)}"
    if cm is None:
        return head
    comment = cm.lstrip(" \t")
    column = len(line) - len(comment)          # where the original '#' sat
    return head + " " * max(1, column - len(head)) + comment


def fail(code, msg):
    print(f"promote_overlay: {msg}", file=sys.stderr)
    return code


def main(argv):
    if len(argv) != 3:
        return fail(2, "usage: promote_overlay.py <overlay-name> <submission.yaml>")
    name, target = argv[1], Path(argv[2])
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.-]*", name) or ".." in name:
        return fail(2, f"bad overlay name {name!r}")
    src = OVERLAY_DIR / f"{name}.yaml"
    if not src.is_file():
        return fail(2, f"no overlay {src}")
    want = load_config(overlay=str(src))       # raises on placeholder / unknown flag keys
    have = load_config(overlay=str(target))
    diffs = [k for k in FLAGS if _get(have, k) != _get(want, k)]
    if not diffs:
        print("NO CHANGES")
        return 0
    lines = target.read_text().split("\n")
    for k in diffs:
        idx = [i for i, line in enumerate(lines) if re.match(r"^[ \t]+" + re.escape(k) + r":", line)]
        if len(idx) != 1:
            return fail(2, f"{k}: {len(idx)} line(s) in {target}, need exactly one (add the key first)")
        new_line = replace_line(lines[idx[0]], k, _get(want, k))
        if new_line is None:
            return fail(2, f"{k}: cannot rewrite line {idx[0] + 1}")
        lines[idx[0]] = new_line
    tmp = target.with_name(target.name + ".new")
    tmp.write_text("\n".join(lines))
    check = load_config(overlay=str(tmp))
    bad = [k for k in FLAGS if _get(check, k) != _get(want, k)]
    if bad:
        tmp.unlink()
        return fail(3, f"verification failed, target untouched: {bad}")
    os.replace(tmp, target)
    for k in diffs:
        print(f"CHANGED {k}: {fmt(_get(have, k))} -> {fmt(_get(want, k))}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
