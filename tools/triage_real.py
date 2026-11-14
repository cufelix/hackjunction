#!/usr/bin/env python3
"""Real-test triage: policy logs (+ optional hand-made video tags) -> one failure tag per pick.

    python3 tools/triage_real.py run1.log [run2.log ...] [--video tags.csv] [--json out.json] [--md out.md]

Stdlib only, so it runs anywhere the organisers' logs land. One pick = one `execute <colour> <side> arm -> <bin> bin`
line; every tagged line after it (until the next execute / episode end) is evidence for that pick.

Tags, first match wins:
  wrong_bin     [landing] ... wrong_bin                              (-1 on the real score)
  missed_grasp  [held_abort], [grasp_diag] held=no, [landing] grasp_failed
  carry_drop    held=yes, then [landing] lost / bounce (object not in the target bin)
  release_stuck [release_confirm] ... still not empty after the re-open
  clip          plan safety_clips > 0 or a [nan_guard] line during the pick
  timeout       "episode time limit reached" while the pick was open
  ok            [landing] ok or "verified <colour> in <bin>"
  unknown       none of the above (no landing review on, log cut, ...)

Video tags (CSV header: episode,pick,tag,note; episode and pick are 1-based as printed) override the log tag, so
the organisers' video can correct what the log could not see (e.g. a carry drop with landing review off).
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import Counter, defaultdict

TAGS = ("wrong_bin", "missed_grasp", "carry_drop", "release_stuck", "clip", "timeout", "ok", "unknown")

RE_T = re.compile(r"\[classical t=\s*([0-9.]+)\]")
RE_EXEC = re.compile(r"execute (\S+)\s+(left|right)\s+arm -> (\S+) bin(.*)")
RE_XYZ = re.compile(r"(grasp|drop)=\(([-+0-9.]+),([-+0-9.]+),([-+0-9.]+)\)")
RE_CLIPS = re.compile(r"safety_clips=(\d+)")
RE_DIAG = re.compile(r"\[grasp_diag\] .*?held=(yes|no|n/a)")
RE_LANDING = re.compile(r"\[landing\] t=[0-9.]+ (\S+)->(\S+): (\w+)")
RE_LABEL = re.compile(r"\[retry_vary\] (\S+) label=(\S+)")
RE_POSTURE = re.compile(r"\[grasp_posture\] t=[0-9.]+ (\S+) (left|right) label=(\S+)")


def _new_pick(ep, idx, m, t, line_no, src, labels):
    color, side, target, rest = m.group(1), m.group(2), m.group(3), m.group(4)
    xyz = {k: (float(x), float(y), float(z)) for k, x, y, z in RE_XYZ.findall(rest)}
    clips = RE_CLIPS.search(rest)
    return {"src": src, "episode": ep, "pick": idx, "t": t, "line": line_no, "color": color, "side": side,
            "target": target, "label": labels.get(color), "grasp": xyz.get("grasp"), "drop": xyz.get("drop"),
            "safety_clips": int(clips.group(1)) if clips else 0, "held": None, "landing": None,
            "held_abort": False, "release_stuck": False, "nan_guard": False, "verified": False, "timeout": False,
            "evidence": []}


def classify(p):
    if p["landing"] == "wrong_bin":
        return "wrong_bin"
    if p["held_abort"] or p["held"] == "no" or p["landing"] == "grasp_failed":
        return "missed_grasp"
    if p["landing"] in ("lost", "bounce"):
        return "carry_drop" if p["held"] == "yes" or p["held"] is None else "missed_grasp"
    if p["release_stuck"]:
        return "release_stuck"
    if p["safety_clips"] > 0 or p["nan_guard"]:
        return "clip"
    if p["timeout"]:
        return "timeout"
    if p["landing"] == "ok" or p["verified"]:
        return "ok"
    return "unknown"


def parse(lines, src="log"):
    picks, ep, idx, cur, labels, t = [], 1, 0, None, {}, 0.0
    ep_seen = False
    for n, line in enumerate(lines, 1):
        mt = RE_T.search(line)
        if mt:
            t = float(mt.group(1))
        ml = RE_LABEL.search(line) or RE_POSTURE.search(line)
        if ml:
            labels[ml.group(1)] = ml.group(ml.lastindex)
            if cur is not None and cur["color"] == ml.group(1):
                cur["label"] = ml.group(ml.lastindex)
        me = RE_EXEC.search(line)
        if me:
            if cur is not None:
                picks.append(cur)
            idx += 1
            cur = _new_pick(ep, idx, me, t, n, src, labels)
            ep_seen = True
            continue
        if "[classical] episode end" in line:
            if cur is not None:
                picks.append(cur)
                cur = None
            if ep_seen:
                ep, idx, ep_seen = ep + 1, 0, False
            labels = {}
            continue
        if cur is None:
            continue
        hit = True
        md = RE_DIAG.search(line)
        mland = RE_LANDING.search(line)
        if md and md.group(1) != "n/a":
            cur["held"] = md.group(1)
        elif "[held_abort]" in line:
            cur["held_abort"] = True
        elif mland and mland.group(1) == cur["color"]:
            cur["landing"] = mland.group(3)
        elif "[release_confirm]" in line and "still not empty after the re-open" in line:
            cur["release_stuck"] = True
        elif "[nan_guard]" in line:
            cur["nan_guard"] = True
        elif re.search(rf"verified {re.escape(cur['color'])} in ", line):
            cur["verified"] = True
        elif "episode time limit reached" in line:
            cur["timeout"] = True
        else:
            hit = False
        if hit:
            cur["evidence"].append(f"{n}: {line.strip()[:160]}")
    if cur is not None:
        picks.append(cur)
    for p in picks:
        p["tag"] = classify(p)
        p["tag_source"] = "log"
    return picks


def apply_video(picks, rows):
    by_key = {(p["episode"], p["pick"]): p for p in picks}
    unmatched = []
    for r in rows:
        tag = (r.get("tag") or "").strip()
        if tag not in TAGS:
            raise ValueError(f"video tag {tag!r} not in {TAGS}")
        key = (int(r["episode"]), int(r["pick"]))
        if key in by_key:
            by_key[key]["tag"], by_key[key]["tag_source"] = tag, "video"
            by_key[key]["video_note"] = (r.get("note") or "").strip()
        else:
            unmatched.append(r)
    return unmatched


def summary(picks):
    out = {"n": len(picks), "by_tag": Counter(p["tag"] for p in picks)}
    grid = defaultdict(Counter)
    for p in picks:
        grid[f"{p['side']}/{p['label'] or p['color']}"][p["tag"]] += 1
    out["by_side_object"] = {k: dict(v) for k, v in sorted(grid.items())}
    return out


def markdown(picks, s):
    lines = [f"# Real-test triage: {s['n']} picks", "", "| tag | n |", "|---|---|"]
    lines += [f"| {t} | {s['by_tag'].get(t, 0)} |" for t in TAGS if s["by_tag"].get(t)]
    lines += ["", "## By hand / object", "", "| hand/object | " + " | ".join(TAGS) + " |",
              "|---|" + "---|" * len(TAGS)]
    for k, c in s["by_side_object"].items():
        lines.append(f"| {k} | " + " | ".join(str(c.get(t, 0)) for t in TAGS) + " |")
    lines += ["", "## Picks", "", "| ep | # | t | colour | label | hand | target | tag | src | grasp xyz |",
              "|---|---|---|---|---|---|---|---|---|---|"]
    for p in picks:
        g = "" if p["grasp"] is None else "(%+.2f,%+.2f,%+.2f)" % p["grasp"]
        lines.append(f"| {p['episode']} | {p['pick']} | {p['t']:.1f} | {p['color']} | {p['label'] or ''} | "
                     f"{p['side']} | {p['target']} | **{p['tag']}** | {p['tag_source']} | {g} |")
    return "\n".join(lines) + "\n"


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--video", help="CSV episode,pick,tag,note from watching the video; overrides log tags")
    ap.add_argument("--json")
    ap.add_argument("--md")
    a = ap.parse_args(argv)
    picks = []
    for path in a.logs:
        with open(path, errors="replace") as f:
            got = parse(f, src=path)
        off = max((p["episode"] for p in picks), default=0)
        for p in got:
            p["episode"] += off
        picks += got
    unmatched = []
    if a.video:
        with open(a.video, newline="") as f:
            unmatched = apply_video(picks, list(csv.DictReader(f)))
    s = summary(picks)
    md = markdown(picks, s)
    if a.md:
        with open(a.md, "w") as f:
            f.write(md)
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"summary": {**s, "by_tag": dict(s["by_tag"])}, "picks": picks}, f, indent=1)
    print(md if not a.md else md.split("## Picks")[0])
    if unmatched:
        print(f"WARNING: {len(unmatched)} video tag row(s) matched no logged pick: {unmatched}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
