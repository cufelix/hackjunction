#!/usr/bin/env python3
"""Real-frame recall A/B on drop_2 (ep 0-39 ONLY; 40-47 are held out and never opened).

Two steps:

  detect  run one policy tree + overlay (+ optional flag overrides) over the drop_2 head frames, robot-path
          (include_bins=False), with the recorded arm/hand pose as robot_q (so color_gate.fk_mask works as on the
          robot). drop_2 has NO head depth, so depth=None, exactly as in the E1 audit. Writes <out>/detections_e1.jsonl
          in the E1 schema (accepted_false = what the policy would see as pickable).
      PYTHONPATH=<tree> python tools/recall_ab.py detect --tree <tree> --overlay finals_r21 [--set k=v ...] --out <dir>

  score   visible-only recall per class via tools/e1_fair_recall.py (headline = conservative notes + current-pose FK,
          the same contract as the 0.597 baseline in FINAL-RECALL.md) + an FP census of the unmatched accepted
          detections: centroid inside a GT bin_tape / bin_wall / hand / other bbox, else "phantom".
      PYTHONPATH=. python tools/recall_ab.py score <dir> [<dir> ...] --md <report.md>

  --colour-aware  second mode: detection must agree in colour (orange<->red, purple<->blue, yellow) to match; a
          wrong-colour detection near a GT is an FP and tallied as wrong_colour. Prints the sanity check that the
          colour-agnostic reimplementation reproduces the E1 headline and FP census exactly.

Matching = the E1 contract (greedy, 25 px nearest centroid, colour-agnostic, file order, one used-set per frame).
Tentative detections (perception.partial_ok_near_hand) are counted as accepted and also reported separately.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import time

E1_DIR = "results/1148-opencode-e1"
FRAMES = "/var/tmp/hf_drop2/frames"
PARQUET = "/var/tmp/hf_drop2/data/chunk-000/file-000.parquet"
LABELS = f"{E1_DIR}/labels_clean.jsonl"
HELD_OUT_FROM = 40
TAGS = ("a", "b", "c", "z")
MATCH_PX = 25.0
HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
DATA = os.environ.get("E1_ROOT", ROOT)      # worktree holding results/1148-opencode-e1 (untracked there)


def parse_set(items):
    out = {}
    for it in items or []:
        k, v = it.split("=", 1)
        try:
            out[k] = json.loads(v)
        except json.JSONDecodeError:
            out[k] = v
    return out


def cmd_detect(a):
    sys.path.insert(0, os.path.abspath(a.tree))
    import cv2
    import numpy as np
    import pandas as pd
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.perception import Perception
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene

    cwd = os.getcwd()
    os.chdir(a.tree)                                   # tree-relative paths (URDF, joint_spec) resolve in the arm's tree
    try:
        cfg = load_config(overlay=a.overlay, overrides=parse_set(a.set))
        p = Perception(cfg, Scene.load(), Calibration.load())
    finally:
        os.chdir(cwd)
    flags_on = {k: v for k, v in parse_set(a.set).items()}
    print(f"[ab] tree={a.tree} overlay={cfg.get('overlay')} overrides={flags_on}", flush=True)

    import inspect
    has_rq = "robot_q" in inspect.signature(p.detect).parameters
    fidx = json.load(open(os.path.join(DATA, E1_DIR, "frame_index.json")))
    df = pd.read_parquet(PARQUET, columns=["index", "observation.state"])
    states = {int(i): np.asarray(s, float) for i, s in zip(df["index"], df["observation.state"])}
    os.makedirs(a.out, exist_ok=True)
    for f in ("frame_index.json", "label_verify.json"):
        if os.path.exists(os.path.join(DATA, E1_DIR, f)):
            shutil.copy(os.path.join(DATA, E1_DIR, f), a.out)
    t0, n, nq = time.time(), 0, 0
    with open(os.path.join(a.out, "detections_e1.jsonl"), "w") as fh:
        for ep in range(HELD_OUT_FROM):               # 40-47 never touched
            for tag in TAGS:
                name = f"e{ep:02d}{tag}"
                path = os.path.join(FRAMES, f"{name}.png")
                if name not in fidx or not os.path.exists(path):
                    continue
                gi = int(fidx[name])
                s = states.get(gi)
                rq = (s[:14], s[14:28]) if s is not None and len(s) >= 28 else None
                nq += rq is not None
                rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
                os.chdir(a.tree)
                try:
                    kw = {"robot_q": rq} if has_rq else {}        # older trees have no robot_q (no FK mask)
                    dets, rej = p.detect(rgb, None, debug=True, include_bins=False, **kw)
                finally:
                    os.chdir(cwd)
                acc = [dict(color=d.color, uv=[round(float(d.uv[0]), 2), round(float(d.uv[1]), 2)], n_px=int(d.n_px),
                            shape=getattr(d, "shape", None), tentative=bool(getattr(d, "tentative", False)))
                       for d in dets]
                fh.write(json.dumps(dict(frame=name, ep=ep, tag=tag, frame_idx=gi, held_out=False,
                                         accepted_false=acc, n_rejected=len(rej))) + "\n")
                n += 1
    meta = dict(tree=os.path.abspath(a.tree), overlay=cfg.get("overlay"), overrides=flags_on, frames=n, robot_q_supported=has_rq,
                frames_with_robot_q=nq, seconds=round(time.time() - t0, 1))
    try:
        meta["tree_sha"] = subprocess.check_output(["git", "-C", a.tree, "rev-parse", "--short", "HEAD"], text=True).strip()
    except Exception:
        pass
    json.dump(meta, open(os.path.join(a.out, "arm.json"), "w"), indent=1)
    print(f"[ab] {n} frames ({nq} with robot_q) in {meta['seconds']} s -> {a.out}", flush=True)


def _bbox(r):
    b = r.get("bbox_px")
    return b if b and len(b) == 4 else None


def _inb(uv, b):
    return b[0] <= uv[0] <= b[2] and b[1] <= uv[1] <= b[3]


def fp_census(d, labels):
    """Unmatched accepted detections by what they sit on. Same greedy match as e1_real_audit.match_gt."""
    dets = {(r["ep"], r["frame_idx"]): r for r in map(json.loads, open(os.path.join(d, "detections_e1.jsonl")))}
    by_frame = {}
    for r in labels:
        ep = int(r.get("ep", -1))
        if ep < 0 or ep >= HELD_OUT_FROM or r.get("frame_idx") is None:
            continue
        by_frame.setdefault((ep, int(r["frame_idx"])), []).append(r)
    c = {"tape": 0, "bin_wall": 0, "hand": 0, "other": 0, "phantom": 0}
    by_col = {}
    tent = tent_matched = 0
    for key, rows in by_frame.items():
        fr = dets.get(key)
        if fr is None:
            continue
        acc = fr["accepted_false"]
        used = set()
        for g in rows:
            if str(g.get("kind", "")).lower() != "object" or not g.get("centroid_px"):
                continue
            cx, cy = g["centroid_px"]
            best, bd = None, MATCH_PX
            for i, x in enumerate(acc):
                if i in used:
                    continue
                dist = math.hypot(x["uv"][0] - cx, x["uv"][1] - cy)
                if dist <= bd:
                    best, bd = i, dist
            if best is not None:
                used.add(best)
        tent += sum(x.get("tentative", False) for x in acc)
        tent_matched += sum(acc[i].get("tentative", False) for i in used)
        for i, x in enumerate(acc):
            if i in used:
                continue
            k = "phantom"
            for kind, lab in (("bin_tape", "tape"), ("bin_wall", "bin_wall"), ("hand", "hand"), ("other", "other")):
                if any(_inb(x["uv"], b) for b in (_bbox(g) for g in rows if g.get("kind") == kind) if b):
                    k = lab
                    break
            c[k] += 1
            by_col[x["color"]] = by_col.get(x["color"], 0) + 1
    return {"fp": c, "fp_total": sum(c.values()), "fp_by_colour": by_col, "tentative": tent, "tentative_matched": tent_matched}


# ---- colour-aware scoring ------------------------------------------------------------------------------------------
# GT colours are real names (orange/purple/yellow); the policy emits sim names (red/blue/yellow). Alias both ways.
COLOUR_CANON = {"orange": "red", "red": "red", "purple": "blue", "blue": "blue", "yellow": "yellow"}


def canon_colour(c):
    c = str(c).lower()
    return COLOUR_CANON.get(c, c)


def colours_agree(gt_cls, det_colour):
    return canon_colour(str(gt_cls).split("-")[0]) == canon_colour(det_colour)


def score_colour(d, fr, colour_aware):
    """Visible-only recall per class (+ FP census) with optional colour constraint.

    Visibility = e1_fair_recall's headline `locus == visible_on_mat` per GT row (read from its fair_recall.json
    per_gt, so the filter is exactly the E1 one). Greedy order = all GT object rows of the frame in file order
    (non-visible GT still consume detections, as in the E1 match). With colour_aware=False this reproduces the
    colour-agnostic headline exactly. With colour_aware=True a detection is only a candidate for a GT of the
    same (aliased) colour; same-position wrong-colour detections stay unmatched and count as FP + wrong_colour.
    """
    dets = {(r["ep"], r["frame_idx"]): r for r in map(json.loads, open(os.path.join(d, "detections_e1.jsonl")))}
    labels = [json.loads(ln) for ln in open(os.path.join(DATA, LABELS)) if ln.strip()]
    boxes = {}
    for r in labels:
        ep = int(r.get("ep", -1))
        if ep < 0 or ep >= HELD_OUT_FROM or r.get("frame_idx") is None:
            continue
        boxes.setdefault((ep, int(r["frame_idx"])), []).append(r)
    by_frame = {}
    for g in fr["per_gt"]:
        if g["ep"] >= HELD_OUT_FROM:
            continue
        by_frame.setdefault((g["ep"], g["frame_idx"]), []).append(g)
    per_class, matched_total, n_total = {}, 0, 0
    fp = {"tape": 0, "bin_wall": 0, "hand": 0, "other": 0, "phantom": 0}
    wrong, confusion, fp_by_col = 0, {}, {}
    for key, gts in by_frame.items():
        fd = dets.get(key)
        acc = fd["accepted_false"] if fd else []
        used = set()
        for g in gts:
            hit = False
            c = g["centroid"]
            if c is not None:
                best, bd = None, MATCH_PX
                for i, x in enumerate(acc):
                    if i in used or (colour_aware and not colours_agree(g["cls"], x["color"])):
                        continue
                    dist = math.hypot(x["uv"][0] - c[0], x["uv"][1] - c[1])
                    if dist <= bd:
                        best, bd = i, dist
                if best is not None:
                    used.add(best)
                    hit = True
            if g["locus"] == "visible_on_mat":
                st = per_class.setdefault(g["cls"], {"n": 0, "matched": 0})
                st["n"] += 1
                st["matched"] += int(hit)
                n_total += 1
                matched_total += int(hit)
        if not fd:
            continue
        rows = boxes.get(key, [])
        for i, x in enumerate(acc):
            if i in used:
                continue
            k = "phantom"
            for kind, lab in (("bin_tape", "tape"), ("bin_wall", "bin_wall"), ("hand", "hand"), ("other", "other")):
                if any(_inb(x["uv"], b) for b in (_bbox(r) for r in rows if r.get("kind") == kind) if b):
                    k = lab
                    break
            fp[k] += 1
            fp_by_col[x["color"]] = fp_by_col.get(x["color"], 0) + 1
            if colour_aware:   # unmatched, within 25 px of a GT object of another colour -> wrong colour
                near = [g for g in gts if g["centroid"] is not None and not colours_agree(g["cls"], x["color"])
                        and math.hypot(x["uv"][0] - g["centroid"][0], x["uv"][1] - g["centroid"][1]) <= MATCH_PX]
                if near:
                    g = min(near, key=lambda g: math.hypot(x["uv"][0] - g["centroid"][0], x["uv"][1] - g["centroid"][1]))
                    wrong += 1
                    kk = f"{g['cls']}->{x['color']}"
                    confusion[kk] = confusion.get(kk, 0) + 1
    pc = {k: dict(n=v["n"], matched=v["matched"], recall=round(v["matched"] / v["n"], 3) if v["n"] else None)
          for k, v in sorted(per_class.items())}
    return {"n": n_total, "matched": matched_total, "recall": round(matched_total / n_total, 3) if n_total else 0.0,
            "per_class": pc, "fp": fp, "fp_total": sum(fp.values()), "wrong_colour": wrong,
            "wrong_colour_pairs": dict(sorted(confusion.items(), key=lambda kv: -kv[1])), "fp_by_colour": fp_by_col}


def detection_colours(dirs):
    s = {}
    for d in dirs:
        for r in map(json.loads, open(os.path.join(d, "detections_e1.jsonl"))):
            for x in r["accepted_false"]:
                s[x["color"]] = s.get(x["color"], 0) + 1
    return s


def cmd_score(a):
    labels = [json.loads(ln) for ln in open(os.path.join(DATA, LABELS)) if ln.strip()]
    rows = []
    for d in a.dirs:
        if not (a.reuse_fair and os.path.exists(os.path.join(d, "fair_recall.json"))):
            subprocess.run([sys.executable, os.path.join(HERE, "e1_fair_recall.py"), "--outdir", d,
                            "--labels", os.path.join(DATA, LABELS), "--counter-z", "0.15"],
                           check=True, cwd=ROOT, stdout=subprocess.DEVNULL, env={**os.environ, "PYTHONPATH": ROOT})
        fr = json.load(open(os.path.join(d, "fair_recall.json")))
        arm = json.load(open(os.path.join(d, "arm.json")))
        rows.append((d, arm, fr, fp_census(d, labels)))
    if a.colour_aware:
        return _score_colour_report(a, rows)
    L = ["| arm | overlay / overrides | recall (visible) | orange-cube | orange-prism | purple-cube | yellow-cube | "
         "yellow-cyl | FP total | tape | wall | hand | other | phantom | tentative (matched) |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    out = []
    for d, arm, fr, fp in rows:
        h = _headline(fr)
        cls = h["per_class"]
        cell = lambda k: f"{cls[k]['matched']}/{cls[k]['n']} = {cls[k]['recall']:.3f}" if k in cls else "-"
        ov = arm["overlay"] + ("" if not arm["overrides"] else " + " + ", ".join(f"{k.split('.')[-1]}={v}" for k, v in arm["overrides"].items()))
        f = fp["fp"]
        L.append(f"| {os.path.basename(d.rstrip('/'))} | {ov} @{arm.get('tree_sha', '?')} | **{h['matched']}/{h['n']} = {h['recall']:.3f}** | "
                 + " | ".join(cell(k) for k in ("orange-cube", "orange-prism", "purple-cube", "yellow-cube", "yellow-cylinder"))
                 + f" | {fp['fp_total']} | {f['tape']} | {f['bin_wall']} | {f['hand']} | {f['other']} | {f['phantom']} | {fp['tentative']} ({fp['tentative_matched']}) |")
        out.append(dict(dir=d, arm=arm, headline=h, fp=fp))
    txt = "\n".join(L)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")
        json.dump(out, open(os.path.splitext(a.md)[0] + ".json", "w"), indent=1)


def _score_colour_report(a, rows):
    print("detection colour strings found:", detection_colours([r[0] for r in rows]), flush=True)
    out, L = [], ["| arm | agnostic | colour-aware | drop | orange-cube | orange-prism | purple-cube | yellow-cube | yellow-cyl "
                  "| wrong_colour | FP agn -> col |", "|---|---|---|---|---|---|---|---|---|---|---|"]
    ok = True
    for d, arm, fr, fp in rows:
        h = _headline(fr)
        ag = score_colour(d, fr, False)
        ca = score_colour(d, fr, True)
        same = (ag["matched"], ag["n"]) == (h["matched"], h["n"]) and all(
            ag["per_class"].get(k, {}).get("matched") == v["matched"] for k, v in h["per_class"].items())
        fp_same = ag["fp"] == fp["fp"]
        ok &= same and fp_same
        name = "/".join(d.rstrip("/").split("/")[-2:])
        print(f"[sanity] {name}: agnostic reimpl {ag['matched']}/{ag['n']} vs e1_fair_recall {h['matched']}/{h['n']} "
              f"per-class {'OK' if same else 'MISMATCH'}; FP census {'OK' if fp_same else 'MISMATCH'}", flush=True)
        cls = ca["per_class"]
        cell = lambda k: f"{cls[k]['matched']}/{cls[k]['n']} = {cls[k]['recall']:.3f}" if k in cls else "-"
        L.append(f"| {name} | {ag['matched']}/{ag['n']} = {ag['recall']:.3f} | **{ca['matched']}/{ca['n']} = {ca['recall']:.3f}** | "
                 f"{ca['recall'] - ag['recall']:+.3f} | " + " | ".join(cell(k) for k in ("orange-cube", "orange-prism", "purple-cube", "yellow-cube", "yellow-cylinder"))
                 + f" | {ca['wrong_colour']} | {ag['fp_total']} -> {ca['fp_total']} |")
        out.append(dict(dir=d, arm=arm, agnostic=ag, colour_aware=ca, drop=round(ca["recall"] - ag["recall"], 3),
                        sanity_agnostic_reproduced=same and fp_same))
    print("SANITY (colour constraint off reproduces e1_fair_recall headline + FP census):", "PASS" if ok else "FAIL")
    txt = "\n".join(L)
    print(txt)
    if a.md:
        open(a.md, "w").write(txt + "\n")
        json.dump(out, open(os.path.splitext(a.md)[0] + ".json", "w"), indent=1)


def _headline(fr):
    """The conservative headline (notes + current-pose FK), visible-only: overall + per class."""
    h = fr["headline"]["visible_only"]
    ov = h["overall"]
    return {"n": ov["n"], "matched": ov["matched"], "recall": ov["recall"] or 0.0, "per_class": h["per_class"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("detect")
    d.add_argument("--tree", required=True)
    d.add_argument("--overlay", required=True)
    d.add_argument("--set", action="append", help="flag override k=v (JSON value)")
    d.add_argument("--out", required=True)
    s = sp.add_parser("score")
    s.add_argument("dirs", nargs="+")
    s.add_argument("--md")
    s.add_argument("--colour-aware", action="store_true",
                   help="a detection may only match a GT of the same (aliased) colour; also reports wrong_colour")
    s.add_argument("--reuse-fair", action="store_true", help="reuse an existing fair_recall.json instead of re-running it")
    a = ap.parse_args()
    (cmd_detect if a.cmd == "detect" else cmd_score)(a)


if __name__ == "__main__":
    main()
