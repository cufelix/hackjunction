#!/usr/bin/env python3
"""E1 offline perception audit of the submission detector on real HF drop_2 frames.

MEASUREMENT ONLY: runs the EXISTING submission perception (overlay=submission:
hsv_set=robust, split_blobs=true, classify_shape=true) on 192 RGB frames
(48 episodes x 4 tags a/b/c/z = 20/50/80/97% of the episode) with depth=None,
exactly as the robot would see it without a depth stream. No thresholds are
tuned here; every number is a property of the shipped detector.

Per frame we run detect() twice:
  - include_bins=True  ("audit" view: bin-hull pixels survive to _reject, so
                        in_bin:* reasons become visible)
  - include_bins=False ("robot" path: the projected bin hulls are cut from the
                        mask before grouping - what the policy actually gets)

Outputs (results/1148-opencode-e1/):
  detections_e1.jsonl  one row per frame (both modes, accepted + rejected)
  audit_e1.json        aggregate statistics (+ GT metrics if labels_e1.jsonl
                       exists; otherwise "pending")
  AUDIT.md             human report

Run from the worktree root:
  PYTHONPATH=. <venv>/bin/python tools/e1_real_audit.py
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene
from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.perception import Perception

FRAMES = "/var/tmp/hf_drop2/frames"
ROOT = "/var/tmp/hf_drop2"
OUTDIR = "results/1148-opencode-e1"
TAGS = (("a", 0.2), ("b", 0.5), ("c", 0.8), ("z", 0.97))   # drop2_extract.py fractions
FPS = 30                                                    # meta/info.json
BIN_ZONE_V = 0.38 * 480      # < 182.4 px: orange tape / bin rim zone
MAT_ZONE_V = 0.45 * 480      # < 216.0 px: above the mat top edge
SMALL_PX = 1000              # small-blob FP census cutoff
MATCH_PX = 25.0              # GT centroid match radius
HELD_OUT_FROM = 40           # episodes 40..47 are held out (report, never tune)


# --------------------------------------------------------------------------
# serialisation
# --------------------------------------------------------------------------

def det_row(d):
    return dict(color=d.color, uv=[round(float(d.uv[0]), 2), round(float(d.uv[1]), 2)],
                n_px=int(d.n_px), shape=d.shape, method=d.method,
                top=[round(float(x), 4) for x in d.top])


def rej_row(d, reason):
    return dict(color=d.color, uv=[round(float(d.uv[0]), 2), round(float(d.uv[1]), 2)],
                n_px=int(d.n_px), reason=reason)


def reason_key(reason):
    """Histogram bucket: keep in_bin:<name> whole, collapse 'height ...' details."""
    return reason if reason.startswith("in_bin") else reason.split()[0]


# --------------------------------------------------------------------------
# yellow shape diagnostics (pre-split blobs, robot-mode mask)
# --------------------------------------------------------------------------

def yellow_diag_for_frame(p, rgb, dets_false):
    """For every accepted yellow detection (robot mode) compute bbox aspect,
    circularity 4*pi*A/P^2 and extent on its PRE-SPLIT connected blob."""
    ydets = [d for d in dets_false if d.color == "yellow"]
    if not ydets:
        return []
    c = p.cfg
    m = p.masks(rgb)["yellow"] & p._keep_mask(None, exclude_bins=True)
    k = int(c["open_px"])
    if k > 0:
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    out = []
    for d in ydets:
        v, u = int(round(d.uv[1])), int(round(d.uv[0]))
        li = 0
        for dv in (-1, 0, 1):                      # centroid may sit on a split seam
            for du in (-1, 0, 1):
                vv, uu = min(max(v + dv, 0), 479), min(max(u + du, 0), 639)
                if lab[vv, uu] > 0:
                    li = int(lab[vv, uu])
                    break
            if li:
                break
        if li == 0:
            out.append(dict(uv=[float(d.uv[0]), float(d.uv[1])], n_px=int(d.n_px),
                            shape=d.shape, blob_found=False))
            continue
        blob = (lab == li).astype(np.uint8)
        A = int(stats[li, cv2.CC_STAT_AREA])
        x, y, w, h = cv2.boundingRect(blob)
        cnts, _ = cv2.findContours(blob, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        P = max(cv2.arcLength(cn, True) for cn in cnts) if cnts else float("nan")
        circ = 4.0 * math.pi * A / (P * P) if P and math.isfinite(P) else float("nan")
        out.append(dict(uv=[float(d.uv[0]), float(d.uv[1])], n_px=int(d.n_px),
                        shape=d.shape, blob_found=True, blob_px=A,
                        bbox=[int(x), int(y), int(w), int(h)],
                        aspect=round(w / h, 3) if h else None,
                        circularity=round(float(circ), 3) if math.isfinite(circ) else None,
                        extent=round(A / float(w * h), 3) if w * h else None))
    return out


# --------------------------------------------------------------------------
# GT matching hook (labels_e1.jsonl, produced by another agent, ep 0..39)
# --------------------------------------------------------------------------

GT_COLOR = {"orange": "red", "purple": "blue", "violet": "blue", "yellow": "yellow"}


def gt_centroid(row):
    for key in ("centroid_px", "centroid", "uv", "center", "centre"):
        if isinstance(row.get(key), (list, tuple)) and len(row[key]) >= 2:
            return float(row[key][0]), float(row[key][1])
    bb = gt_bbox(row)
    if bb is not None:
        return (bb[0] + bb[2]) / 2.0, (bb[1] + bb[3]) / 2.0
    return None


def gt_bbox(row):
    bb = row.get("bbox_px", row.get("bbox"))
    if not (isinstance(bb, (list, tuple)) and len(bb) == 4):
        return None
    x0, y0, a, b = (float(v) for v in bb)
    if a > x0 and b > y0:            # [x0,y0,x1,y1]
        return x0, y0, a, b
    return x0, y0, x0 + a, y0 + b    # [x,y,w,h]


def gt_object_class(row):
    """Per-class recall key. Felix K1/K2 format: '<colour>-<shape>' (orange-cube, orange-prism,
    purple-cube, yellow-cube, yellow-cylinder). Legacy object/class/name keys are kept for the
    tag-keyed bootstrap format."""
    colour = str(row.get("colour", "")).lower()
    if colour:
        shape = row.get("shape")
        shape = str(shape).lower() if shape is not None else "unknown"
        return f"{colour}-{shape}"
    for key in ("object", "object_class", "class", "name", "label_detail"):
        v = row.get(key)
        if isinstance(v, str) and v:
            return v.lower()
    return str(row.get("kind", "object")).lower()


def in_bbox(c, bb, pad=0.0):
    return bb is not None and bb[0] - pad <= c[0] <= bb[2] + pad and bb[1] - pad <= c[1] <= bb[3] + pad


def match_gt(labels_path, frames_rows):
    """Match robot-mode accepted detections to GT labels. Returns a metrics dict.

    Authoritative label format (Felix, K1/K2): JSONL rows keyed by (ep, frame_idx) with
    centroid_px / bbox_px / colour / kind / shape (NO tag field). Audit frames carry the
    authoritative frame_idx (frame_index.json / extractor ceil, see F1), so the (ep, frame_idx)
    join IS the frame_index.json mapping. Legacy tag-keyed bootstrap rows are still accepted.
    """
    rows = []
    with open(labels_path) as f:
        for ln in f:
            ln = ln.strip()
            if ln:
                rows.append(json.loads(ln))

    # detection lookups: authoritative (ep, frame_idx) and legacy (ep, tag)
    det_by_fidx = {(fr["ep"], fr["frame_idx"]): fr for fr in frames_rows}
    det_by_tag = {(fr["ep"], fr["tag"]): fr for fr in frames_rows}

    by_frame = {}
    n_unkeyed = 0
    for r in rows:
        ep = r.get("ep", r.get("episode"))
        if ep is None:
            n_unkeyed += 1
            continue
        ep = int(ep)
        if r.get("frame_idx") is not None:
            key = ("fidx", ep, int(r["frame_idx"]))
        elif r.get("tag", r.get("frame_tag")) is not None:
            key = ("tag", ep, r.get("tag", r.get("frame_tag")))
        else:
            n_unkeyed += 1
            continue
        by_frame.setdefault(key, []).append(r)

    per_class = {}
    per_class_held = {}
    per_colour = {}
    per_colour_held = {}
    colour_hits = colour_matches = 0
    bin_false_picks = 0
    bin_false_pick_frames = set()
    unmatched_areas = []
    unmatched_by_colour = {}
    n_frames_matched = 0
    missing_frames = 0

    for key, gt_rows in sorted(by_frame.items()):
        fr = (det_by_fidx.get((key[1], key[2])) if key[0] == "fidx"
              else det_by_tag.get((key[1], key[2])))
        if fr is None:
            missing_frames += 1
            continue
        n_frames_matched += 1
        dets = fr["accepted_false"]
        ep = fr["ep"]
        used = set()
        # 1) kind=object rows -> per-class / per-colour recall + colour accuracy
        for g in gt_rows:
            kind = str(g.get("kind", g.get("label", g.get("type", "")))).lower()
            if kind != "object":
                continue
            c = gt_centroid(g)
            if c is None:
                continue
            cls = gt_object_class(g)
            gt_colour = str(g.get("colour", "")).lower()
            bucket = per_class_held if ep >= HELD_OUT_FROM else per_class
            st = bucket.setdefault(cls, dict(n=0, matched=0))
            st["n"] += 1
            cbucket = per_colour_held if ep >= HELD_OUT_FROM else per_colour
            cst = cbucket.setdefault(gt_colour or "unknown", dict(n=0, matched=0))
            cst["n"] += 1
            best, bd = None, MATCH_PX
            for i, d in enumerate(dets):
                if i in used:
                    continue
                dist = math.hypot(d["uv"][0] - c[0], d["uv"][1] - c[1])
                if dist <= bd:
                    best, bd = i, dist
            if best is not None:
                used.add(best)
                st["matched"] += 1
                cst["matched"] += 1
                want = GT_COLOR.get(gt_colour)
                if want is None:                     # legacy: infer colour from the class string
                    for name, col in GT_COLOR.items():
                        if name in cls:
                            want = col
                            break
                if want is not None:
                    colour_matches += 1
                    colour_hits += int(dets[best]["color"] == want)
        # 2) unmatched accepted detections: tape/rim false pick (centroid inside a GT
        #    bin_tape/bin_wall bbox_px) vs other FP candidate
        bin_bboxes = [gt_bbox(g) for g in gt_rows
                      if str(g.get("kind", g.get("label", g.get("type", "")))).lower()
                      in ("bin_tape", "bin_wall")]
        bin_bboxes = [bb for bb in bin_bboxes if bb is not None]
        for i, d in enumerate(dets):
            if i in used:
                continue
            unmatched_areas.append(int(d["n_px"]))
            unmatched_by_colour[d["color"]] = unmatched_by_colour.get(d["color"], 0) + 1
            if any(in_bbox((d["uv"][0], d["uv"][1]), bb) for bb in bin_bboxes):
                bin_false_picks += 1
                bin_false_pick_frames.add(fr["frame"])

    def summarise(bucket):
        out = {}
        for cls, st in sorted(bucket.items()):
            out[cls] = dict(n=st["n"], matched=st["matched"],
                            recall=round(st["matched"] / st["n"], 3) if st["n"] else None)
        return out

    def area_dist(a):
        if not a:
            return dict(n=0)
        q = np.percentile(a, [5, 25, 50, 75, 95])
        return dict(n=len(a), min=int(min(a)), p5=round(float(q[0]), 1), p25=round(float(q[1]), 1),
                    p50=round(float(q[2]), 1), p75=round(float(q[3]), 1), p95=round(float(q[4]), 1),
                    max=int(max(a)))

    unmatched_total = len(unmatched_areas)
    return dict(
        status="ok", labels_file=labels_path, n_label_rows=len(rows),
        n_label_rows_unkeyed=n_unkeyed,
        n_frames_with_labels=len(by_frame), n_frames_matched=n_frames_matched,
        n_frames_missing_detections=missing_frames, match_radius_px=MATCH_PX,
        per_class_recall=summarise(per_class),
        per_class_recall_held_out=summarise(per_class_held),
        per_colour_recall=summarise(per_colour),
        per_colour_recall_held_out=summarise(per_colour_held),
        colour_accuracy=(dict(hits=colour_hits, matched=colour_matches,
                              acc=round(colour_hits / colour_matches, 3))
                         if colour_matches else None),
        false_picks_on_bin_regions=bin_false_picks,
        n_frames_with_bin_false_picks=len(bin_false_pick_frames),
        bin_false_pick_frames=sorted(bin_false_pick_frames),
        unmatched_accepted_total=unmatched_total,
        unmatched_accepted_area=area_dist(unmatched_areas),
        unmatched_accepted_by_colour=dict(sorted(unmatched_by_colour.items())),
        other_unmatched_accepted=unmatched_total - bin_false_picks,
    )


def in_bin_large_rejects(rows, min_px=2000):
    """F3 caveat data: audit-view in_bin rejects >= min_px are NOT all tape/wall - they include
    real objects on the mat in front of bins that the depth=None robot hull-cut drops."""
    bycol = {}
    frames_by_col = {}
    allframes = set()
    for r in rows:
        for d in r["rejected_true"]:
            if d["reason"].startswith("in_bin") and d["n_px"] >= min_px:
                c = d["color"]
                bycol[c] = bycol.get(c, 0) + 1
                frames_by_col.setdefault(c, set()).add(r["frame"])
                allframes.add(r["frame"])
    return dict(min_px=min_px, total=sum(bycol.values()), n_frames=len(allframes),
                by_colour={c: dict(n=bycol[c], n_frames=len(frames_by_col[c]))
                           for c in sorted(bycol)})


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=FRAMES)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--outdir", default=OUTDIR)
    args = ap.parse_args()

    t_start = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    manifest = json.load(open(f"{args.root}/manifest.json"))
    info_fps = FPS
    try:
        info_fps = int(json.load(open(f"{args.root}/meta/info.json")).get("fps", FPS))
    except Exception:
        pass

    # F1: authoritative (ep,tag)->frame_idx map from the label lane (frame_index.json, ep 0-39).
    # For frames absent from it (held-out ep 40-47) reproduce drop2_extract.py exactly: it saved
    # the FIRST decoded frame with pts >= ts-0.02, i.e. ceil((ts-0.02)*fps) at constant fps.
    frame_index = {}
    frame_index_path = f"{args.outdir}/frame_index.json"
    if os.path.exists(frame_index_path):
        try:
            frame_index = json.load(open(frame_index_path))
        except Exception:
            frame_index = {}

    cfg = load_config(overlay="submission")
    p = Perception(cfg, Scene.load(), Calibration.load())
    print(f"[e1] overlay={cfg.get('overlay')} profile={cfg.get('profile')} "
          f"hsv_set={cfg['perception'].get('hsv_set')} split_blobs={cfg['perception'].get('split_blobs')} "
          f"classify_shape={cfg['perception'].get('classify_shape')}", flush=True)

    det_path = f"{args.outdir}/detections_e1.jsonl"
    frames_rows = []
    sanity = None
    n_missing = 0
    with open(det_path, "w") as fh:
        for m in manifest:
            ep, task, t0, t1 = int(m["episode"]), m["task"], float(m["t0"]), float(m["t1"])
            held_out = ep >= HELD_OUT_FROM
            for tag, frac in TAGS:
                name = f"e{ep:02d}{tag}"
                path = f"{args.frames}/{name}.png"
                if not os.path.exists(path):
                    n_missing += 1
                    continue
                # F1: authoritative frame index in the concatenated video. frame_index.json is the
                # source of truth where present; otherwise reproduce drop2_extract.py exactly:
                # first decoded frame with pts >= ts-0.02  =>  ceil((ts-0.02)*fps).
                ts = round(t0 + (t1 - t0) * frac, 3)
                if name in frame_index:
                    frame_idx = int(frame_index[name])
                else:
                    frame_idx = int(math.ceil((ts - 0.02) * info_fps))
                frame_idx_ep = int(round((ts - t0) * info_fps))  # index inside the episode
                bgr = cv2.imread(path)
                rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                dets_t, rej_t = p.detect(rgb, None, debug=True, include_bins=True)
                dets_f, rej_f = p.detect(rgb, None, debug=True, include_bins=False)
                ydiag = yellow_diag_for_frame(p, rgb, dets_f)

                if ep == 0 and tag == "a":       # sanity vs the probed e00a baseline
                    reds = sorted([d for d in dets_t if d.color == "red"], key=lambda d: -d.n_px)
                    yellows = sorted([d for d in dets_t if d.color == "yellow"], key=lambda d: -d.n_px)
                    blues = [d for d in dets_t if d.color == "blue"]
                    sanity = dict(
                        accepted=len(dets_t), rejected=len(rej_t),
                        expected_accepted=20, expected_rejected=14,
                        pass_counts=(len(dets_t) == 20 and len(rej_t) == 14),
                        n_red=len(reds), n_yellow=len(yellows), n_blue=len(blues),
                        red_largest_px=int(reds[0].n_px) if reds else None,
                        red_largest_uv=[round(float(v), 1) for v in reds[0].uv] if reds else None,
                        yellow_px=[int(d.n_px) for d in yellows],
                        yellow_uv=[[round(float(v), 1) for v in d.uv] for d in yellows],
                        blue_px_range=[int(min(d.n_px for d in blues)), int(max(d.n_px for d in blues))] if blues else None,
                    )
                    print(f"[e1] sanity e00a include_bins=True: {sanity}", flush=True)

                row = dict(
                    frame=name, ep=ep, tag=tag, frame_idx=frame_idx, frame_idx_ep=frame_idx_ep,
                    ts_video_s=ts, held_out=held_out, task=task,
                    accepted_true=[det_row(d) for d in dets_t],
                    rejected_true=[rej_row(d, r) for d, r in rej_t],
                    accepted_false=[det_row(d) for d in dets_f],
                    rejected_false=[rej_row(d, r) for d, r in rej_f],
                    yellow_diag=ydiag,
                )
                fh.write(json.dumps(row) + "\n")
                frames_rows.append(row)
                if len(frames_rows) % 24 == 0:
                    print(f"[e1] {len(frames_rows)} frames, {time.time() - t_start:.0f}s", flush=True)

    detect_s = time.time() - t_start
    print(f"[e1] detection done: {len(frames_rows)} frames in {detect_s:.1f}s "
          f"({1000 * detect_s / max(len(frames_rows), 1):.0f} ms/frame x2 modes), missing={n_missing}",
          flush=True)

    # ---------------------------------------------------------------- stats
    def colour_counts(rows, key):
        out = {"red": 0, "yellow": 0, "blue": 0, "other": 0}
        for r in rows:
            for d in r[key]:
                out[d["color"] if d["color"] in out else "other"] += 1
        return out

    def split(rows):
        return ([r for r in rows if not r["held_out"]], [r for r in rows if r["held_out"]])

    def area_percentiles(rows, key, color):
        a = [d["n_px"] for r in rows for d in r[key] if d["color"] == color]
        if not a:
            return dict(n=0)
        q = np.percentile(a, [5, 25, 50, 75, 95])
        return dict(n=len(a), min=int(min(a)), p5=round(float(q[0]), 1), p25=round(float(q[1]), 1),
                    p50=round(float(q[2]), 1), p75=round(float(q[3]), 1), p95=round(float(q[4]), 1),
                    max=int(max(a)))

    def bin_zone(rows, key):
        z38 = {"red": 0, "yellow": 0, "blue": 0}
        z45 = {"red": 0, "yellow": 0, "blue": 0}
        per_frame38 = []
        for r in rows:
            c38 = 0
            for d in r[key]:
                if d["color"] not in z38:
                    continue
                if d["uv"][1] < BIN_ZONE_V:
                    z38[d["color"]] += 1
                    c38 += 1
                if d["uv"][1] < MAT_ZONE_V:
                    z45[d["color"]] += 1
            per_frame38.append(c38)
        return dict(v_lt_0p38=z38, v_lt_0p45=z45,
                    total_v_lt_0p38=sum(z38.values()), total_v_lt_0p45=sum(z45.values()),
                    frames_with_any_v_lt_0p38=sum(1 for c in per_frame38 if c > 0))

    def reason_hist(rows, key):
        h = {}
        for r in rows:
            for d in r[key]:
                k = reason_key(d["reason"])
                h[k] = h.get(k, 0) + 1
        return dict(sorted(h.items(), key=lambda kv: -kv[1]))

    def small_census(rows):
        out = {}
        for key in ("accepted_false", "rejected_false", "accepted_true"):
            per = {}
            for r in rows:
                for d in r[key]:
                    if d["n_px"] < SMALL_PX:
                        k = d["color"] if key != "rejected_false" else f'{d["color"]}:{reason_key(d["reason"])}'
                        per[k] = per.get(k, 0) + 1
            out[key] = dict(sorted(per.items(), key=lambda kv: -kv[1]))
        return out

    tr, ho = split(frames_rows)
    stats = {}
    for label, rows in (("all", frames_rows), ("train_ep0_39", tr), ("held_out_ep40_47", ho)):
        if label != "all" and not rows:
            continue
        s = dict(
            n_frames=len(rows),
            accepted_true_total=sum(len(r["accepted_true"]) for r in rows),
            accepted_false_total=sum(len(r["accepted_false"]) for r in rows),
            rejected_true_total=sum(len(r["rejected_true"]) for r in rows),
            rejected_false_total=sum(len(r["rejected_false"]) for r in rows),
            accepted_true_by_colour=colour_counts(rows, "accepted_true"),
            accepted_false_by_colour=colour_counts(rows, "accepted_false"),
            accepted_per_frame_false=dict(
                mean=round(float(np.mean([len(r["accepted_false"]) for r in rows])), 2),
                median=float(np.median([len(r["accepted_false"]) for r in rows])),
                min=int(min(len(r["accepted_false"]) for r in rows)),
                max=int(max(len(r["accepted_false"]) for r in rows)),
                hist={str(k): int(v) for k, v in
                      zip(*np.unique([len(r["accepted_false"]) for r in rows], return_counts=True))}),
            area_percentiles_false={c: area_percentiles(rows, "accepted_false", c)
                                    for c in ("red", "yellow", "blue")},
            area_percentiles_true={c: area_percentiles(rows, "accepted_true", c)
                                   for c in ("red", "yellow", "blue")},
            bin_zone_robot=bin_zone(rows, "accepted_false"),
            bin_zone_audit=bin_zone(rows, "accepted_true"),
            rejected_reasons_true=reason_hist(rows, "rejected_true"),
            rejected_reasons_false=reason_hist(rows, "rejected_false"),
            small_blob_census=small_census(rows),
        )
        # include_bins delta: per frame accepted(True) - accepted(False)
        delta = np.array([len(r["accepted_true"]) - len(r["accepted_false"]) for r in rows])
        s["include_bins_delta"] = dict(
            accepted_total_true_minus_false=int(delta.sum()),
            frames_with_delta=int((delta != 0).sum()),
            delta_mean=round(float(delta.mean()), 3), delta_max=int(delta.max()),
            note=("audit view keeps bin-hull pixels to the in_bin reject; robot view cuts them "
                  "before grouping, so blobs touching a bin can differ"),
        )
        stats[label] = s

    # yellow shape separability
    yd = [y for r in frames_rows for y in r["yellow_diag"] if y.get("blob_found")]
    ytrain = [y for r in tr for y in r["yellow_diag"] if y.get("blob_found")]

    def ystats(lst):
        if not lst:
            return dict(n=0)
        asp = np.array([y["aspect"] for y in lst if y["aspect"]])
        cir = np.array([y["circularity"] for y in lst if y["circularity"]])
        ext = np.array([y["extent"] for y in lst if y["extent"]])
        return dict(n=len(lst),
                    aspect=dict(p10=round(float(np.percentile(asp, 10)), 3), p50=round(float(np.percentile(asp, 50)), 3), p90=round(float(np.percentile(asp, 90)), 3)),
                    circularity=dict(p10=round(float(np.percentile(cir, 10)), 3), p50=round(float(np.percentile(cir, 50)), 3), p90=round(float(np.percentile(cir, 90)), 3)),
                    extent=dict(p10=round(float(np.percentile(ext, 10)), 3), p50=round(float(np.percentile(ext, 50)), 3), p90=round(float(np.percentile(ext, 90)), 3)),
                    by_shape_label={k: sum(1 for y in lst if str(y.get("shape")) == k)
                                    for k in sorted({str(y.get("shape")) for y in lst})})

    # per-label distributions + IQR overlap between the two dominant labels
    def per_label(lst):
        out = {}
        for lab in sorted({str(y.get("shape")) for y in lst}):
            sub = [y for y in lst if str(y.get("shape")) == lab]
            cir = np.array([y["circularity"] for y in sub if y.get("circularity")])
            ext = np.array([y["extent"] for y in sub if y.get("extent")])
            out[lab] = dict(
                n=len(sub),
                circularity_p10_50_90=[round(float(v), 3) for v in np.percentile(cir, [10, 50, 90])] if len(cir) else None,
                extent_p10_50_90=[round(float(v), 3) for v in np.percentile(ext, [10, 50, 90])] if len(ext) else None,
            )
        return out

    def overlap_verdict(lst):
        labs = {k: v for k, v in per_label(lst).items() if v["n"] >= 10 and v.get("circularity_p10_50_90")}
        keys = sorted(labs, key=lambda k: -labs[k]["n"])[:2]
        if len(keys) < 2:
            return "insufficient data per label for an overlap test", None
        a, b = (np.array(labs[k]["circularity_p10_50_90"])[[0, 2]] for k in keys)
        ov = max(0.0, min(a[1], b[1]) - max(a[0], b[0])) / max(a[1] - a[0], b[1] - b[0], 1e-9)
        verdict = ("circularity p10-p90 ranges of the two dominant labels overlap "
                   f"{ov:.0%} -> NOT cleanly separable from these 2D features alone"
                   if ov > 0.5 else
                   f"overlap {ov:.0%} -> partially separable" if ov > 0.15 else
                   f"overlap {ov:.0%} -> separable")
        return verdict, dict(labels=keys, circularity_overlap_frac=round(float(ov), 3))

    yv, yv_detail = overlap_verdict(ytrain or yd)
    yellow_separability = dict(
        all_frames=ystats(yd), train_only=ystats(ytrain),
        per_label_train=per_label(ytrain),
        verdict=yv, verdict_detail=yv_detail,
        note=("aspect/circularity/extent of accepted yellow blobs (pre-split robot-mode mask). "
              "Shape labels come from the shipped classify_shape (metric path with depth=None "
              "falls back to the 2D blob). No threshold is proposed here - measurement only."),
        samples=yd[:40],
    )

    # F3 caveat data: audit-view in_bin rejects >= 2000px include real objects, not just tape
    in_bin_large = in_bin_large_rejects(frames_rows)

    # GT hook
    labels_path = f"{args.outdir}/labels_e1.jsonl"
    if os.path.exists(labels_path):
        try:
            gt_metrics = match_gt(labels_path, frames_rows)
        except Exception as e:
            gt_metrics = dict(status="error", error=f"{type(e).__name__}: {e}")
    else:
        gt_metrics = dict(status="pending",
                          note=f"{labels_path} not present at run time; re-run this script "
                               "after the labelling agent publishes it (Felix K1 format, ep 0-39, "
                               "keyed (ep, frame_idx)).")

    audit = dict(
        meta=dict(
            tool="tools/e1_real_audit.py", date=time.strftime("%Y-%m-%d %H:%M:%S"),
            commit=os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
            overlay=cfg.get("overlay"), profile=cfg.get("profile"),
            flags=dict(hsv_set=cfg["perception"].get("hsv_set"),
                       split_blobs=cfg["perception"].get("split_blobs"),
                       classify_shape=cfg["perception"].get("classify_shape"),
                       min_area_px=cfg["perception"].get("min_area_px"),
                       open_px=cfg["perception"].get("open_px")),
            n_frames=len(frames_rows), n_missing=n_missing, fps=info_fps,
            runtime_s=dict(detect=round(detect_s, 1), total=round(time.time() - t_start, 1)),
            bin_zone_v_px=BIN_ZONE_V, mat_zone_v_px=MAT_ZONE_V,
        ),
        sanity_e00a=sanity,
        stats=stats,
        yellow_shape_separability=yellow_separability,
        in_bin_large_rejects=in_bin_large,
        gt_metrics=gt_metrics,
    )
    with open(f"{args.outdir}/audit_e1.json", "w") as fh:
        json.dump(audit, fh, indent=1)

    write_report(args.outdir, audit)
    print(f"[e1] wrote {args.outdir}/detections_e1.jsonl, audit_e1.json, AUDIT.md "
          f"(total {time.time() - t_start:.1f}s)", flush=True)
    return 0


def write_report(outdir, a):
    s_all = a["stats"]["all"]
    san = a["sanity_e00a"] or {}
    gt = a["gt_metrics"]
    L = []
    L.append("# E1 offline perception audit — submission detector on real HF drop_2 frames\n")
    L.append(f"_Generated by `tools/e1_real_audit.py` at {a['meta']['date']} "
             f"(commit {a['meta']['commit']}). Measurement only: no thresholds tuned, "
             "no policy code touched._\n")
    L.append("## Setup\n")
    L.append(f"- Detector: submission overlay (`hsv_set={a['meta']['flags']['hsv_set']}`, "
             f"`split_blobs={a['meta']['flags']['split_blobs']}`, "
             f"`classify_shape={a['meta']['flags']['classify_shape']}`, "
             f"`min_area_px={a['meta']['flags']['min_area_px']}`), depth=None on every frame "
             "(all 3D outputs are ray-plane assumptions, method='plane').")
    L.append(f"- Frames: {a['meta']['n_frames']} = 48 episodes x 4 tags (a/b/c/z = 20/50/80/97%), "
             "640x480 RGB cam_high. Episodes 40-47 **held out** (reported separately, never tuned on).")
    L.append("- Two passes per frame: `include_bins=True` (audit view: bin pixels reach the "
             "`in_bin:*` reject) and `include_bins=False` (robot path: projected bin hulls cut "
             "from the mask before grouping).\n")
    L.append("## Sanity check (e00a, include_bins=True)\n")
    if san:
        L.append(f"- accepted {san.get('accepted')} / rejected {san.get('rejected')} "
                 f"(expected 20/14): **{'PASS' if san.get('pass_counts') else 'FAIL'}**")
        L.append(f"- red {san.get('n_red')} (largest {san.get('red_largest_px')} px @ "
                 f"uv {san.get('red_largest_uv')} = bin-wall blob), yellow {san.get('n_yellow')} "
                 f"{san.get('yellow_px')}, blue {san.get('n_blue')} "
                 f"(px range {san.get('blue_px_range')} — background FPs).\n")
    L.append("## Totals\n")
    L.append("| split | frames | acc(robot) | acc(audit) | rej(robot) | rej(audit) |")
    L.append("|---|---|---|---|---|---|")
    for k in ("all", "train_ep0_39", "held_out_ep40_47"):
        if k not in a["stats"]:
            continue
        s = a["stats"][k]
        L.append(f"| {k} | {s['n_frames']} | {s['accepted_false_total']} | "
                 f"{s['accepted_true_total']} | {s['rejected_false_total']} | "
                 f"{s['rejected_true_total']} |")
    L.append("")
    L.append("Accepted by colour (robot path / audit view), all frames: "
             f"red {s_all['accepted_false_by_colour']['red']}/{s_all['accepted_true_by_colour']['red']}, "
             f"yellow {s_all['accepted_false_by_colour']['yellow']}/{s_all['accepted_true_by_colour']['yellow']}, "
             f"blue {s_all['accepted_false_by_colour']['blue']}/{s_all['accepted_true_by_colour']['blue']}\n")
    apf = s_all["accepted_per_frame_false"]
    L.append(f"Accepted per frame (robot): mean {apf['mean']}, median {apf['median']}, "
             f"min {apf['min']}, max {apf['max']}; histogram {apf['hist']}\n")
    L.append("## Area percentiles (accepted, robot path, all frames)\n")
    L.append("| colour | n | p5 | p25 | p50 | p75 | p95 | max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for c in ("red", "yellow", "blue"):
        q = s_all["area_percentiles_false"][c]
        if q.get("n"):
            L.append(f"| {c} | {q['n']} | {q['p5']} | {q['p25']} | {q['p50']} | {q['p75']} | "
                     f"{q['p95']} | {q['max']} |")
    L.append("")
    L.append("## Bin-zone false-pick candidates (accepted detections)\n")
    L.append(f"Zone A: centroid v < {a['meta']['bin_zone_v_px']:.0f} px (tape/rim); "
             f"Zone B: v < {a['meta']['mat_zone_v_px']:.0f} px (above mat edge).\n")
    L.append("| split | view | zone A red/yellow/blue | zone A total | frames with any | zone B total |")
    L.append("|---|---|---|---|---|---|")
    for k in ("all", "train_ep0_39", "held_out_ep40_47"):
        if k not in a["stats"]:
            continue
        for view, key in (("robot", "bin_zone_robot"), ("audit", "bin_zone_audit")):
            z = a["stats"][k][key]
            zz = z["v_lt_0p38"]
            L.append(f"| {k} | {view} | {zz['red']}/{zz['yellow']}/{zz['blue']} | "
                     f"{z['total_v_lt_0p38']} | {z['frames_with_any_v_lt_0p38']} | "
                     f"{z['total_v_lt_0p45']} |")
    L.append("")
    L.append("## Rejected-reason histogram\n")
    L.append("| split | view | reasons |")
    L.append("|---|---|---|")
    for k in ("all", "train_ep0_39", "held_out_ep40_47"):
        if k not in a["stats"]:
            continue
        for view, key in (("robot", "rejected_reasons_false"), ("audit", "rejected_reasons_true")):
            L.append(f"| {k} | {view} | {a['stats'][k][key]} |")
    L.append("")
    L.append("## include_bins=True vs False delta\n")
    for k in ("all", "train_ep0_39", "held_out_ep40_47"):
        if k not in a["stats"]:
            continue
        d = a["stats"][k]["include_bins_delta"]
        L.append(f"- {k}: accepted audit-robot = {d['accepted_total_true_minus_false']} "
                 f"(mean/frame {d['delta_mean']}, max {d['delta_max']}), "
                 f"frames differing {d['frames_with_delta']}/{a['stats'][k]['n_frames']}")
    L.append("")
    L.append("## Yellow shape separability (cube vs cylinder)\n")
    ys = a["yellow_shape_separability"]
    for scope in ("train_only", "all_frames"):
        st = ys.get(scope) or {}
        if st.get("n"):
            L.append(f"- {scope}: n={st['n']}, aspect p10/50/90 = "
                     f"{st['aspect']['p10']}/{st['aspect']['p50']}/{st['aspect']['p90']}, "
                     f"circularity {st['circularity']['p10']}/{st['circularity']['p50']}/{st['circularity']['p90']}, "
                     f"extent {st['extent']['p10']}/{st['extent']['p50']}/{st['extent']['p90']}, "
                     f"shape labels {st['by_shape_label']}")
    for lab, st in (ys.get("per_label_train") or {}).items():
        L.append(f"- label '{lab}' (train): n={st['n']}, circularity p10/50/90 = "
                 f"{st['circularity_p10_50_90']}, extent p10/50/90 = {st['extent_p10_50_90']}")
    L.append(f"- verdict: **{ys.get('verdict')}**"
             + (f" ({json.dumps(ys.get('verdict_detail'))})" if ys.get("verdict_detail") else ""))
    L.append(f"- {ys.get('note')}\n")
    L.append(f"## Small-blob FP census (n_px < {1000})\n")
    L.append("| split | view | counts |")
    L.append("|---|---|---|")
    for k in ("all", "train_ep0_39", "held_out_ep40_47"):
        if k not in a["stats"]:
            continue
        for view, key in (("robot accepted", "accepted_false"),
                          ("robot rejected (colour:reason)", "rejected_false"),
                          ("audit accepted", "accepted_true")):
            L.append(f"| {k} | {view} | {a['stats'][k]['small_blob_census'][key]} |")
    L.append("")
    L.append("## GT metrics\n")
    if gt.get("status") == "pending":
        L.append(f"**PENDING** — {gt.get('note')}\n")
    elif gt.get("status") == "ok":
        L.append(f"- labels: {gt['n_label_rows']} rows, {gt['n_frames_matched']} frames matched "
                 f"(match radius {gt['match_radius_px']} px, robot-mode accepted detections; "
                 "keyed (ep, frame_idx) via frame_index.json)")
        L.append(f"- per-class recall (colour-shape, ep 0-39): {json.dumps(gt['per_class_recall'])}")
        L.append(f"- per-colour recall: {json.dumps(gt.get('per_colour_recall', {}))}")
        if gt.get("colour_accuracy"):
            ca = gt["colour_accuracy"]
            L.append(f"- colour accuracy (orange→red, purple→blue, yellow→yellow): "
                     f"{ca['acc']} ({ca['hits']}/{ca['matched']})")
        L.append(f"- tape/rim false picks (accepted centroid inside GT bin_tape/bin_wall bbox): "
                 f"{gt['false_picks_on_bin_regions']} in "
                 f"{gt.get('n_frames_with_bin_false_picks', 0)} frames")
        L.append(f"- unmatched accepted detections (FP candidates): {gt['unmatched_accepted_total']} "
                 f"(area px {json.dumps(gt.get('unmatched_accepted_area', {}))}); on bin regions: "
                 f"{gt['false_picks_on_bin_regions']}, other: "
                 f"{gt.get('other_unmatched_accepted', gt['unmatched_accepted_total'] - gt['false_picks_on_bin_regions'])}\n")
    else:
        L.append(f"status={gt.get('status')}: {gt.get('error') or gt.get('note')}\n")
    L.append("## Caveats\n")
    L.append("- **No depth, no robot pose**: every `top` (pelvis xyz), the `in_bin:*`, `off_mat` "
             "and `height` rejects are ray-plane assumptions from the FIXED scene geometry, not "
             "measurements. Frame crops where the real camera pose differs from the calibration "
             "pose will shift all 3D quantities.")
    L.append("- **Orange bins match robust red** (bins H 2-7, red window H 0-16): bin tape/wall "
             "pixels are only suppressed by the projected fixed-pose bin hull + in_bin reject; "
             "there is no RGB bin detector. Anything bin-coloured outside the projected hulls "
             "can be accepted as a red object.")
    # F6: corrected small-blue wording (robot accepts MORE small blue than audit; height never fires)
    sb = s_all["small_blob_census"]
    rb = sb["accepted_false"].get("blue", 0)
    ab = sb["accepted_true"].get("blue", 0)
    L.append(f"- **Small blue background FPs** are common (H 100-145 window). The robot path "
             f"accepts MORE small blue (<1000 px: {rb}) than the audit view ({ab}); the audit "
             "view's in_bin hull rejects remove some that the robot path keeps. The `height` "
             "reject never fires on the plane path (h=1.0 exactly), so `off_mat` is the only "
             "reject that removes small blue.")
    # F3: audit-view in_bin rejects are not all tape/wall (real objects hull-cut under depth=None)
    ibl = a.get("in_bin_large_rejects", {})
    bc = ibl.get("by_colour", {})

    def _fmt(c):
        v = bc.get(c, {})
        return f"{v.get('n', 0)} {c} ({v.get('n_frames', 0)} frames)"

    L.append(f"- **Audit-view `in_bin` rejects are NOT all tape/wall**: among them "
             f"{_fmt('blue')} / {_fmt('red')} / {_fmt('yellow')} are >= {ibl.get('min_px', 2000)} px "
             f"({ibl.get('total', 0)} total over {ibl.get('n_frames', 0)} frames). The blue/purple "
             "and yellow ones are real objects on the mat in front of bins (e.g. e00a purple cube "
             "5437 px, orange prism 4307 px) that the depth=None robot hull-cut drops; with real "
             "depth the 3D keep-mask would keep them. This depresses apparent recall in the audit "
             "view - the robot view is the relevant one.")
    # F4: big blue accepts are arm/base hardware, so an area floor alone is not a sufficient gate
    maxblue = s_all["area_percentiles_false"]["blue"].get("max", 0)
    L.append(f"- **Big blue accepts include robot arm/base hardware**, not only speckle (max "
             f"{maxblue} px, e.g. e33b arm 14914 px, e35c arm band 5336 px, e00a metal wrist "
             "3795 px). An area floor >= 1000 px alone will NOT remove these; relevant to "
             "color.gate design.")
    L.append("- Episodes 40-47 are held out: numbers above are reported but were not used for "
             "any decision.")
    L.append("")
    L.append("## Review fixes (independent review PASS-WITH-FINDINGS, 2026-11-14)\n")
    L.append("Targeted fixes only; no thresholds tuned and no measurement logic changed "
             "(overlay=submission, depth=None, both modes, detection calls identical).")
    L.append("- **F1 (frame_idx)**: `frame_idx` now loaded from `frame_index.json` (authoritative, "
             "ep 0-39); held-out frames absent from it use the exact extractor computation "
             "`ceil((ts-0.02)*fps)`. Was `round(ts*fps)`, +1 on 32/192 rows (29 train, 3 held-out). "
             "All 160 train rows now match frame_index.json; the 32 held-out rows use the extractor "
             "formula. Fixes the K1 label join.")
    L.append("- **F2 (yellow shape histogram)**: `by_shape_label` now compares `str(shape)` to the "
             "string key, so the 'None' bucket is counted (was always 0). Bucket sums now equal n.")
    L.append("- **F5 (GT hook)**: `match_gt` rewritten for the Felix K1/K2 label format - rows keyed "
             "(ep, frame_idx) and joined to audit frames via frame_index.json; reads "
             "centroid_px/bbox_px/colour/kind/shape; per-class recall by colour-shape (orange-cube, "
             "orange-prism, purple-cube, yellow-cube, yellow-cylinder) plus per-colour; "
             "colour_accuracy vs the mapped colour; tape/rim false picks by centroid-in-bbox "
             "(count + frames); unmatched accepted FP candidates (count + area distribution). "
             "Legacy tag-keyed bootstrap rows still accepted.")
    L.append("- **F3 (report)**: added caveat that audit-view in_bin rejects include real objects "
             "hull-cut under depth=None (not all tape); see Caveats.")
    L.append("- **F4 (report)**: added caveat that big blue FPs are arm/base hardware, so an area "
             "floor alone is insufficient; see Caveats.")
    L.append("- **F6 (report)**: corrected small-blue wording - robot path accepts MORE small blue "
             "than the audit view; the height reject never fires on the plane path (h=1.0).")
    with open(f"{outdir}/AUDIT.md", "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
