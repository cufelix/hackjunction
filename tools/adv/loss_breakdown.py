#!/usr/bin/env python3
"""loss_breakdown.py: why was every object NOT sorted correctly? Per profile table + example episodes with log evidence.

    loss_breakdown.py [--examples 3] [--out report.md] <profile_label>=<run_dir>[,<run_dir>...] ...
    e.g. loss_breakdown.py real_hard=runs/real_hard__*s1000n5,runs/real_hard__*s1005n5 real_mild=runs/real_mild__*

Inputs per run dir (run_adv.sh output): eval.json (per-object outcomes, layout, final positions, dr_params) and
policy_server.log ([classical] skip/execute lines with pelvis-frame positions, [grasp_diag], 'held=no', 'nothing left').
Run from a checkout with the repo importable (needs aaltonomous.scene: numpy, yaml).

Each lost object (outcome != correct) gets ONE category, first matching rule (P = its true start position, pelvis frame,
incl. the profile's object_region remap; a "detection" = a `skip`/`execute` log line of that colour with a position):
  e  time ran out        the episode hit the time limit (no 'nothing left to sort') and the object was never attempted
  d  wrong colour        a detection of ANOTHER colour within 3 cm of P and none of its own colour within 6 cm
  b  wrong position      detections of its colour exist, but the nearest is 3-10 cm from P (execute there = grasp miss)
  f1 grasp failed        executed within 3 cm of P: 'held=no after the close' (closed on nothing / slipped)
  f2 dropped in carry    executed within 3 cm, held=yes at the close, object still ended on the counter / wrong bin
  f3 skipped, no plan    detected within 3 cm (skip only), never executed: IK / bin-rim / no-joint-path infeasible
  c  false object        no detection near P and a phantom (a detection of that colour > 10 cm from every real object,
                         e.g. orange bin rim / tape read as 'red') was executed or skipped in the episode
  a  not detected        no detection of its colour within 10 cm of P, no phantom either
Episode level: phantom detections per episode are reported separately (they cost time even when no object is lost to them).
"""
import argparse
import glob
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from aaltonomous.scene.scene import Scene  # noqa: E402

SC = Scene.load()
O0 = np.asarray(SC.origin, float)
CATS = [("a", "not detected"), ("b", "detected at wrong position"), ("c", "false object / tape picked"),
        ("d", "wrong colour"), ("e", "time ran out"), ("f1", "grasp failed (held=no)"),
        ("f2", "dropped in carry / wrong place"), ("f3", "detected, no feasible plan")]
LINE = re.compile(r"^(\S+ \S+) INFO \[classical(?: t=\s*([\d.]+))?\] (skip|execute)\s+(\w+)\s+(?:(left|right) arm -> \w+ bin\s+grasp=\(([^)]*)\)|top=\(([^)]*)\))(.*)$")


def vec(s):
    return np.array([float(v) for v in s.split(",")][:2])


def segment_log(path):
    """policy_server.log -> {seed: [lines]}"""
    out, cur = defaultdict(list), None
    for ln in Path(path).read_text(errors="replace").splitlines():
        m = re.search(r"reset task=\S+ seed=(\d+)", ln)
        if m:
            cur = int(m.group(1))
        if cur is not None:
            out[cur].append(ln)
    return out


def parse_dets(lines):
    """-> list of dict(kind, colour, pos, line, held) in order; `held=no after the close` attaches to the last execute."""
    dets = []
    for ln in lines:
        m = LINE.match(ln)
        if m:
            kind, col = m.group(3), m.group(4)
            pos = vec(m.group(6) or m.group(7))
            dets.append({"kind": kind, "colour": col, "pos": pos, "line": ln, "held": None})
        elif dets and dets[-1]["kind"] == "execute":
            if "held=no after the close" in ln:
                dets[-1]["held"] = False
                dets[-1]["line2"] = ln
            elif "[grasp_diag]" in ln and dets[-1]["held"] is None:
                dets[-1]["held"] = "held=yes" in ln
                dets[-1]["line2"] = ln
    return dets


def start_positions(trial):
    """name -> (colour, shape, true start xy pelvis) incl. the sim's object_region remap."""
    R = (((trial.get("dr_params") or {}).get("adv")) or {}).get("object_region")
    names = [o["name"] for o in trial.get("final_objects", [])]
    pool = defaultdict(list)
    for n in names:
        _, c, s, _ = n.split("_")
        pool[(c, s)].append(n)
    out = {}
    for o in trial["layout"]:
        n = pool[(o["color"], o.get("shape", "cube"))].pop(0)
        px, py = O0[0] + o["y"], O0[1] + o["x"]
        if R:
            px = R["x_c"] + R["kx"] * (px - 0.295)
            py = R["y_c"] + R["ky"] * py
        out[n] = (o["color"], o.get("shape"), np.array([px, py]))
    return out


def classify(trial, lines):
    dets = parse_dets(lines)
    starts = start_positions(trial)
    allp = [v[2] for v in starts.values()]
    ended_full = any("nothing left to sort" in l for l in lines)
    phantom = [d for d in dets if min(np.linalg.norm(d["pos"] - p) for p in allp) > 0.10]
    res = []
    for o in trial["outcomes"]:
        if o["outcome"] == "correct":
            continue
        col, shape, P = starts[o["name"]]
        mine = [d for d in dets if d["colour"] == col]
        near = [(np.linalg.norm(d["pos"] - P), d) for d in mine]
        near3 = [d for dist, d in near if dist <= 0.03]
        ph_same = [d for d in phantom if d["colour"] == col]
        other3 = [d for d in dets if d["colour"] != col and np.linalg.norm(d["pos"] - P) <= 0.03]
        cat, ev = None, []
        ex3 = [d for d in near3 if d["kind"] == "execute"]
        if ex3:
            d = ex3[-1]
            cat = "f1" if d["held"] is False else "f2"
            ev = [d["line"], d.get("line2", "")]
        elif near3:
            cat, ev = "f3", [near3[-1]["line"]]
        elif other3 and not [1 for dist, _ in near if dist <= 0.06]:
            cat, ev = "d", [other3[0]["line"]]
        elif near and 0.03 < min(x[0] for x in near) <= 0.10:
            d = min(near, key=lambda x: x[0])[1]
            cat, ev = "b", [d["line"], f"(true start ({P[0]:+.3f},{P[1]:+.3f}), nearest detection {min(x[0] for x in near)*100:.1f} cm away)"]
        elif not ended_full and not mine:
            cat, ev = "e", ["(no 'nothing left to sort' line: episode ran to the time limit)"]
        elif ph_same or phantom:
            d = (ph_same or phantom)[0]
            cat, ev = "c", [d["line"]]
        else:
            cat, ev = "a", [f"(no {col} detection within 10 cm of true start ({P[0]:+.3f},{P[1]:+.3f}); {len(dets)} detections in episode)"]
        res.append({"seed": trial["seed"], "name": o["name"], "outcome": o["outcome"], "cat": cat,
                    "evidence": [e for e in ev if e], "start": [round(float(P[0]), 3), round(float(P[1]), 3)]})
    return res, len(phantom), len(dets)


def load_run(d):
    ev = sorted(glob.glob(f"{d}/*/eval.json") + glob.glob(f"{d}/eval.json"))[-1]
    trials = json.load(open(ev))["trials"]
    logs = segment_log(Path(d) / "policy_server.log")
    return trials, logs


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--examples", type=int, default=3)
    ap.add_argument("--out")
    a = ap.parse_args()
    md = []
    for spec in a.runs:
        label, dirs = spec.split("=", 1)
        rows, n_eps, n_obj, ph, nd = [], 0, 0, 0, 0
        for d in sum([sorted(glob.glob(x)) for x in dirs.split(",")], []):
            trials, logs = load_run(d)
            for t in trials:
                n_eps += 1
                n_obj += t["n_objects"]
                r, p, nde = classify(t, logs.get(t["seed"], []))
                rows += r
                ph += p
                nd += nde
        c = Counter(r["cat"] for r in rows)
        lost = len(rows)
        md.append(f"### {label}: {n_eps} episodes, {n_obj} objects, {lost} lost ({100*lost/max(n_obj,1):.0f} %); "
                  f"phantom detections {ph} of {nd} ({100*ph/max(nd,1):.0f} %)\n")
        md.append("| cat | reason | count | % of lost |\n|---|---|---|---|")
        for k, nm in CATS:
            md.append(f"| {k} | {nm} | {c[k]} | {100*c[k]/max(lost,1):.0f} % |")
        md.append("")
        for k, nm in CATS:
            ex = [r for r in rows if r["cat"] == k]
            seen, shown = set(), 0
            for r in ex:
                if r["seed"] in seen or shown >= a.examples:
                    continue
                seen.add(r["seed"])
                shown += 1
                md.append(f"- **{k} {nm}**, seed {r['seed']} {r['name']} (outcome {r['outcome']}, start {r['start']}):")
                md += [f"    `{e[:330]}`" for e in r["evidence"]]
        md.append("")
    txt = "\n".join(md)
    print(txt)
    if a.out:
        Path(a.out).write_text(txt + "\n")


if __name__ == "__main__":
    main()
