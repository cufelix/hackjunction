#!/usr/bin/env python3
"""E1 FAIR visible-only object recall (measurement only).

The raw recall in audit_e1.json (0.33-0.85 per colour-shape) uses ALL kind=object
GT rows as the denominator, including objects the robot-path detector cannot
possibly see: held in the gripper, occluded by the arm/wrist, or already inside
a bin. This tool stratifies the GT by locus/visibility and reports recall on
detectable (visible-on-mat) objects only.

THREE classification modes (all reported; the first is the headline):

  HEADLINE (calibrated, conservative): notes + CURRENT-POSE FK.
    1. note says held/grasped                       -> in_hand
    2. note says occluded/behind/split by wrist...  -> occluded
    3. note says bin (incl. rim)                    -> in_bin
    4. centroid inside CURRENT-POSE FK hand mask    -> in_hand
    5. centroid inside CURRENT-POSE FK arm/base mask-> occluded
    6. otherwise                                    -> visible_on_mat
    Calibration findings (crops + FK-source probe, see FAIR-RECALL.md):
    - FKMask.frame_mask() ORs in the FULL NEUTRAL-POSE body (base_mask). 61/101
      naive FK hits land on neutral-pose ARM pixels where the arm is NOT in the
      current frame (49 of them were detected). Those are artifacts, not
      occlusion -> kept visible (conservative).
    - Unsure rows stay visible => headline fair recall is a LOWER bound.
    - No geometry: robust to the counter_z placeholder (review F1).

  VARIANT literal-2D-FK (the uncalibrated prescription): notes + the public
    combined FK mask (incl. neutral-pose body), hand region = current-pose hand
    capsules, everything else arm. Shows the cost of the neutral-pose artifact.

  VARIANT geom (prescribed rule, approximate): literal-2D-FK + ray-plane bin
    test at counter_z=0.01 m NOMINAL (sim placeholder, z_plane=0.0325 m):
    centroid xy inside a bin footprint -> in_bin, UNLESS the note says 'on mat'.
    At the placeholder height the plane is too low and demonstrably projects
    visible mat objects into the bin footprints (e00a purple cube / orange
    prism; 32 tag-a rows before any placement), so this variant OVER-excludes
    and inflates recall: the approximate upper end of the fair interval.
    Corrected counter_z re-run pending (another agent).

Matching: the EXACT greedy 25 px nearest-centroid contract of
tools/e1_real_audit.py match_gt (imported read-only), robot-path accepted
detections (accepted_false) from detections_e1.jsonl, one shared used-set per
frame over ALL object rows in file order, so raw totals reproduce match_gt
bit-for-bit and the locus strata partition them.

Episodes 40-47 are HELD OUT and never opened (labels only cover ep 0-39 anyway).
No thresholds tuned, no policy code touched, inputs read-only.

Outputs (results/1148-opencode-e1/):
  fair_recall.json   locus counts, raw vs visible-only recall, per-frame/per-GT detail
  FAIR-RECALL.md     human report

Run from the worktree root:
  PYTHONPATH=. <venv>/bin/python tools/e1_fair_recall.py [--crops]
  (--counter-z / --suffix for the pending corrected-counter_z re-run; they only
   affect the geom variant, never the headline)
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import shutil
import sys
import time

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene
from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.perception import Perception

OUTDIR = "results/1148-opencode-e1"
FRAMES = "/var/tmp/hf_drop2/frames"
SCRATCH = ".scratch-e1-fairrecall"
COUNTER_Z_NOMINAL = 0.01      # sim placeholder (review F1); NOT a measured table height
HELD_OUT_FROM = 40
MATCH_PX = 25.0               # same contract as e1_real_audit.match_gt

# --- note patterns (calibrated on the labels_e1.jsonl note histogram + crops) ---
HELD_RE = re.compile(r"\bheld\b|\bin (?:left |right )?hand\b|under .*fingers")
OCC_RE = re.compile(r"occluded|\bbehind\b|split by (?:wrist|gripper|arm|finger)")
BIN_RE = re.compile(r"\bbin\b")
ON_MAT_RE = re.compile(r"\bon mat\b|\bon the mat\b")
RIM_RE = re.compile(r"\brim\b")
EDGE_RE = re.compile(r"frame edge|out of frame|(?:bottom|left|right|top) edge")

RULE_HEADLINE = (
    "HEADLINE (calibrated, conservative): note held->in_hand; note occluded/behind/"
    "split-by-wrist/gripper->occluded; note bin(incl rim)->in_bin; else centroid in "
    "CURRENT-POSE FK hand mask->in_hand; else centroid in CURRENT-POSE FK arm/base "
    "mask->occluded; else visible_on_mat. Neutral-pose-only FK hits are artifacts "
    "(arm not there in this frame) and do NOT exclude. No geometry: unsure rows stay "
    "visible, so the headline is a LOWER bound, robust to the counter_z placeholder."
)
RULE_LITERAL_FK = (
    "VARIANT literal-2D-FK (uncalibrated prescription): same note rules, then the "
    "PUBLIC combined FK mask (current pose OR full neutral-pose body): current-pose "
    "hand capsules->in_hand, any other mask pixel->occluded. Includes the "
    "neutral-pose arm artifact."
)
RULE_GEOM = (
    "VARIANT geom (approximate): literal-2D-FK plus ray-plane xy inside a bin "
    "footprint (GateGeometry, counter_z=0.01 nominal = sim placeholder) -> in_bin, "
    "unless the note says 'on mat'. Over-excludes at the placeholder height; "
    "approximate upper end, not the headline."
)


# --------------------------------------------------------------------------
# read-only imports of sibling tools (another agent may be editing colorgate)
# --------------------------------------------------------------------------

def _import_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load_real_audit():
    return _import_file("e1_real_audit_readonly",
                        os.path.join(os.path.dirname(__file__), "e1_real_audit.py"))


def load_colorgate():
    """FKMask + GateGeometry, read-only. If the live file is mid-edit (SyntaxError),
    fall back to a private copy in .scratch-e1-fairrecall/."""
    path = os.path.join(os.path.dirname(__file__), "e1_colorgate_audit.py")
    try:
        return _import_file("e1_colorgate_audit_readonly", path), path
    except SyntaxError:
        os.makedirs(SCRATCH, exist_ok=True)
        copy = os.path.join(SCRATCH, "e1_colorgate_audit_copy.py")
        shutil.copyfile(path, copy)
        return _import_file("e1_colorgate_audit_copy", copy), copy


# --------------------------------------------------------------------------
# FK masks. The public frame_mask() is the source of truth for the combined
# silhouette; it ORs in FKMask.base_mask = the FULL neutral-pose body. We
# partition the capsules (layout: 3 base + 16/side = 8 arm incl shoulder +
# 8 hand/fingers) into current-pose hand, current-pose arm+base, and the
# neutral-pose arms, and sanity-check hand|arm|neutral == public combined.
# If the layout ever changes, fall back to ok_nosplit (any public FK hit ->
# occluded; conservative, never inflates the visible bucket).
# --------------------------------------------------------------------------

def fk_submasks(fk, ep, frame_idx_ep):
    """dict(hand, arm, neutral, combined, status). arm = current arm + static base."""
    combined, status = fk.frame_mask(ep, frame_idx_ep)
    if status != "ok":
        return dict(hand=None, arm=None, neutral=None, combined=None, status=status)
    gi = fk.global_index(ep, frame_idx_ep)
    st = fk.states.get(gi)
    q = fk.q0.copy()
    for n, v in zip(fk.joint_names, st):
        q[fk.iq[n]] = v
    caps = fk._capsules(q)
    caps0 = fk._capsules(fk.q0)
    if len(caps) == 35 and len(caps0) == 35:
        hand = fk._mask(caps[11:19] + caps[27:35])
        arm = fk._mask(caps[0:11] + caps[19:27])
        neutral = fk._mask(caps0[3:])
        if np.array_equal(hand | arm | neutral, combined):
            return dict(hand=hand, arm=arm, neutral=neutral, combined=combined, status="ok")
    return dict(hand=None, arm=None, neutral=None, combined=combined, status="ok_nosplit")


def mask_hit(mask, c):
    if mask is None or c is None:
        return False
    u = int(min(max(round(c[0]), 0), 639))
    v = int(min(max(round(c[1]), 0), 479))
    return bool(mask[v, u])


def note_class(note):
    n = (note or "").lower()
    if HELD_RE.search(n):
        return "held"
    if OCC_RE.search(n):
        return "occluded"
    if BIN_RE.search(n):
        return "bin"
    if ON_MAT_RE.search(n):
        return "on_mat"
    return "no_locus_info" if not n else "other"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--labels", default=None, help="default: <outdir>/labels_e1.jsonl")
    ap.add_argument("--counter-z", type=float, default=COUNTER_Z_NOMINAL,
                    help="ray-plane counter height (m, pelvis) for the GEOM VARIANT. "
                         "Default 0.01 = sim placeholder (review F1); corrected value "
                         "pending another agent. Never affects the headline.")
    ap.add_argument("--suffix", default="", help="output suffix for variant re-runs")
    ap.add_argument("--crops", action="store_true",
                    help="write calibration crops to .scratch-e1-fairrecall/crops/")
    args = ap.parse_args()

    t_start = time.time()
    os.makedirs(SCRATCH, exist_ok=True)
    labels_path = args.labels or f"{args.outdir}/labels_e1.jsonl"
    real = load_real_audit()
    cg, cg_path = load_colorgate()
    assert real.MATCH_PX == MATCH_PX

    rows = [json.loads(ln) for ln in open(labels_path) if ln.strip()]
    dets = [json.loads(ln) for ln in open(f"{args.outdir}/detections_e1.jsonl") if ln.strip()]
    det_by_fidx = {(d["ep"], d["frame_idx"]): d for d in dets}

    cfg = load_config(overlay="submission")
    p = Perception(cfg, Scene.load(), Calibration.load())
    geom = cg.GateGeometry(p, counter_z=args.counter_z)
    fk = cg.FKMask(p.cal, enabled=True)
    print(f"[fair] labels={labels_path} rows={len(rows)} counter_z={geom.counter_z} "
          f"(z_plane={geom.z_plane}) FK={fk.status} colorgate={cg_path}", flush=True)
    if fk.status != "ok":
        print("[fair] WARNING: FK unavailable; locus falls back to notes only", flush=True)

    # ---- group GT object rows exactly like match_gt (file order within frame) ----
    by_frame = {}
    n_obj = n_heldout = 0
    for r in rows:
        if str(r.get("kind", "")).lower() != "object":
            continue
        ep = int(r.get("ep", r.get("episode")))
        if ep >= HELD_OUT_FROM:               # held-out: never processed
            n_heldout += 1
            continue
        if r.get("frame_idx") is None:
            continue
        n_obj += 1
        by_frame.setdefault((ep, int(r["frame_idx"])), []).append(r)

    # known label FPs from the independent visual verification (annotation only;
    # rows are NOT removed from the primary numbers - measurement only)
    known_fp = set()
    lv_path = f"{args.outdir}/label_verify.json"
    if os.path.exists(lv_path):
        try:
            lv = json.load(open(lv_path))
            for e in lv.get("aggregate", {}).get("errors_and_misses", []):
                if e.get("verdict") == "not-an-object":
                    known_fp.add((e["frame"], int(e["i"])))
        except Exception:
            pass

    # ---- classify + match, one greedy pass per frame (match_gt contract) ----
    per_gt = []
    per_frame = []
    fk_status_counts = {}
    mask_cache = {}
    crop_wanted = {}                          # category -> first row (for --crops)
    missing_frames = 0

    for key, gt_rows in sorted(by_frame.items()):
        fr = det_by_fidx.get(key)
        if fr is None:
            missing_frames += 1
            continue
        ep, fidx = key
        name = fr["frame"]
        fk_ep = fr.get("frame_idx_ep")
        if fk_ep is None:
            fk_ep = fidx - int(fk.offsets.get(ep, 0))
        if key not in mask_cache:
            mask_cache[key] = fk_submasks(fk, ep, fk_ep)
        fm = mask_cache[key]
        fk_status = fm["status"]
        fk_status_counts[fk_status] = fk_status_counts.get(fk_status, 0) + 1

        dets_list = fr["accepted_false"]
        used = set()
        frame_recs = []
        for i, g in enumerate(gt_rows):
            c = real.gt_centroid(g)
            note = g.get("note") or ""
            nc = note_class(note)
            # geometry trace for every row (variant + caveat data)
            xy, _ = geom.ray_plane(c) if c is not None else (None, None)
            verdict, bdist, bname = geom.bin_zone(xy)
            # FK hit source: current hand / current arm+base / neutral-pose artifact
            if fk_status == "ok":
                fk_hit = ("hand" if mask_hit(fm["hand"], c) else
                          "arm" if mask_hit(fm["arm"], c) else
                          "neutral_artifact" if mask_hit(fm["neutral"], c) else "none")
            elif fk_status == "ok_nosplit":
                fk_hit = "robot_nosplit" if mask_hit(fm["combined"], c) else "none"
            else:
                fk_hit = fk_status

            # --- headline locus: notes, then CURRENT-POSE FK; no geometry ---
            if nc == "held":
                locus, ev = "in_hand", "note:held"
            elif nc == "occluded":
                locus, ev = "occluded", "note:occluded"
            elif nc == "bin":
                locus, ev = "in_bin", "note:bin" + (":rim" if RIM_RE.search(note.lower()) else "")
            elif fk_hit == "hand":
                locus, ev = "in_hand", "fk:hand"
            elif fk_hit in ("arm", "robot_nosplit"):
                locus, ev = "occluded", "fk:" + fk_hit
            else:
                locus, ev = "visible_on_mat", ("note:on_mat" if nc == "on_mat"
                                               else "default:visible")
            # --- variant literal-2D-FK: public combined mask (incl neutral body) ---
            if locus == "visible_on_mat" and fk_hit == "neutral_artifact":
                locus_l, ev_l = "occluded", "fk:neutral_artifact"
            else:
                locus_l, ev_l = locus, ev
            # --- variant geom: literal FK + nominal-z bin footprint (on_mat wins) ---
            if locus_l == "visible_on_mat" and nc != "on_mat" and verdict == "in_bin":
                locus_g, ev_g = "in_bin", f"geom:in_bin:{bname}@z{geom.counter_z:g}"
            else:
                locus_g, ev_g = locus_l, ev_l

            # --- greedy 25px match, identical to match_gt ---
            matched, mdist, duv = False, None, None
            if c is not None:
                best, bd = None, MATCH_PX
                for j, d in enumerate(dets_list):
                    if j in used:
                        continue
                    dist = math.hypot(d["uv"][0] - c[0], d["uv"][1] - c[1])
                    if dist <= bd:
                        best, bd = j, dist
                if best is not None:
                    used.add(best)
                    matched, mdist = True, round(bd, 2)
                    duv = dets_list[best]["uv"]

            rec = dict(
                ep=ep, frame=name, frame_idx=fidx, i=i,
                colour=str(g.get("colour", "")).lower(),
                shape=str(g.get("shape")).lower() if g.get("shape") is not None else "unknown",
                cls=real.gt_object_class(g),
                centroid=[round(c[0], 1), round(c[1], 1)] if c else None,
                note=note, note_class=nc,
                locus=locus, evidence=ev,
                locus_literal_fk=locus_l, evidence_literal_fk=ev_l,
                locus_geom_variant=locus_g, evidence_geom_variant=ev_g,
                fk_hit=fk_hit, bin_verdict=verdict,
                bin_dist_m=round(bdist, 4) if bdist is not None else None, bin_name=bname,
                note_fk_conflict=(nc == "on_mat" and locus != "visible_on_mat"),
                frame_edge=bool(EDGE_RE.search(note.lower())),
                known_fp=(name, i) in known_fp,
                matched=matched, match_dist_px=mdist, det_uv=duv,
            )
            per_gt.append(rec)
            frame_recs.append(rec)
            # crop calibration selection (first row per category, file order)
            if args.crops:
                cats = []
                if nc == "occluded" and "gripper" in note.lower():
                    cats.append("occ_gripper")
                if nc in ("no_locus_info", "other") and fk_hit == "hand":
                    cats.append("silent_fk_hand")
                if nc in ("no_locus_info", "other") and fk_hit == "arm":
                    cats.append("silent_fk_arm")
                if fk_hit == "neutral_artifact" and nc not in ("held", "occluded", "bin"):
                    cats.append("neutral_artifact")
                if ev_g.startswith("geom:") and name.endswith("a") and c[1] >= 216:
                    cats.append("geom_mat_object_tag_a")
                if "mask missed" in note.lower():
                    cats.append("mask_missed")
                if RIM_RE.search(note.lower()):
                    cats.append("bin_rim")
                if note.startswith("merged comp split"):
                    cats.append("merged_split")
                if rec["known_fp"]:
                    cats.append("known_fp")
                if nc == "on_mat" and fk_hit in ("hand", "arm"):
                    cats.append("on_mat_fk_conflict")
                for cat in cats:
                    crop_wanted.setdefault(cat, rec)

        def vis_stats(locus_key):
            vis = [r for r in frame_recs if r[locus_key] == "visible_on_mat"]
            return len(vis), sum(1 for r in vis if r["matched"])

        vn, vm = vis_stats("locus")
        ln, lm = vis_stats("locus_literal_fk")
        gn, gm = vis_stats("locus_geom_variant")
        per_frame.append(dict(
            frame=name, ep=ep, tag=fr["tag"], frame_idx=fidx, fk_status=fk_status,
            n_gt=len(frame_recs), n_accepted=len(dets_list),
            locus_counts={L: sum(1 for r in frame_recs if r["locus"] == L)
                          for L in ("in_hand", "occluded", "in_bin", "visible_on_mat")},
            raw_matched=sum(1 for r in frame_recs if r["matched"]),
            visible_n=vn, visible_matched=vm,
            literal_fk_visible_n=ln, literal_fk_visible_matched=lm,
            geom_variant_visible_n=gn, geom_variant_visible_matched=gm,
        ))

    # ---- aggregate ----
    def summarise(recs, key="cls"):
        out = {}
        for r in recs:
            st = out.setdefault(r[key] or "unknown", dict(n=0, matched=0))
            st["n"] += 1
            st["matched"] += int(r["matched"])
        return {k: dict(n=v["n"], matched=v["matched"],
                        recall=round(v["matched"] / v["n"], 3) if v["n"] else None)
                for k, v in sorted(out.items())}

    def overall(recs):
        n, m = len(recs), sum(1 for r in recs if r["matched"])
        return dict(n=n, matched=m, recall=round(m / n, 3) if n else None)

    locus_order = ("in_hand", "occluded", "in_bin", "visible_on_mat")

    def mode_block(locus_key, ev_key):
        vis = [r for r in per_gt if r[locus_key] == "visible_on_mat"]
        ev_hist = {}
        for r in per_gt:
            k = ":".join(r[ev_key].split(":")[:2])
            ev_hist[k] = ev_hist.get(k, 0) + 1
        return dict(
            locus_counts={L: sum(1 for r in per_gt if r[locus_key] == L) for L in locus_order},
            recall_by_locus={L: overall([r for r in per_gt if r[locus_key] == L])
                             for L in locus_order},
            evidence_histogram=dict(sorted(ev_hist.items())),
            visible_only=dict(overall=overall(vis), per_class=summarise(vis),
                              per_colour=summarise(vis, "colour")),
        )

    headline = mode_block("locus", "evidence")
    literal = mode_block("locus_literal_fk", "evidence_literal_fk")
    variant = mode_block("locus_geom_variant", "evidence_geom_variant")
    raw = dict(overall=overall(per_gt), per_class=summarise(per_gt),
               per_colour=summarise(per_gt, "colour"))

    # cross-check 1: my raw totals vs the imported match_gt run on the same inputs
    mg = real.match_gt(labels_path, dets)
    cross = dict(
        match_gt_per_class=mg.get("per_class_recall"),
        my_raw_per_class=raw["per_class"],
        per_class_identical=(mg.get("per_class_recall") == raw["per_class"]),
        per_colour_identical=(mg.get("per_colour_recall") == raw["per_colour"]),
        match_gt_matched_total=sum(v["matched"] for v in mg.get("per_class_recall", {}).values()),
        my_matched_total=sum(1 for r in per_gt if r["matched"]),
    )
    # cross-check 2: vs the published audit_e1.json (same detections file)
    audit_path = f"{args.outdir}/audit_e1.json"
    if os.path.exists(audit_path):
        pub = json.load(open(audit_path)).get("gt_metrics", {})
        cross["published_audit_per_class_identical"] = (pub.get("per_class_recall") == raw["per_class"])

    # ambiguity / evidence breakdown
    nc_counts = {}
    for r in per_gt:
        nc_counts[r["note_class"]] = nc_counts.get(r["note_class"], 0) + 1
    n_no_info = nc_counts.get("no_locus_info", 0) + nc_counts.get("other", 0)
    artifact_rows = [r for r in per_gt if r["fk_hit"] == "neutral_artifact"]
    n_artifact_note_silent = sum(1 for r in artifact_rows
                                 if r["note_class"] not in ("held", "occluded", "bin"))
    ambiguity = dict(
        note_class_counts=dict(sorted(nc_counts.items())),
        n_note_locus_decisive=sum(1 for r in per_gt if r["note_class"] in ("held", "occluded", "bin")),
        n_fk_decided=sum(1 for r in per_gt if r["evidence"].startswith("fk:")),
        n_default_visible=sum(1 for r in per_gt if r["evidence"] == "default:visible"),
        n_geom_decided_variant=sum(1 for r in per_gt
                                   if r["evidence_geom_variant"].startswith("geom:")),
        n_no_locus_info_in_note=n_no_info,
        ambiguity_rate_no_locus_info=round(n_no_info / max(len(per_gt), 1), 3),
        n_note_on_mat_overruled_by_fk=sum(1 for r in per_gt if r["note_fk_conflict"]),
        n_note_on_mat_would_be_overruled_by_geom=sum(
            1 for r in per_gt if r["note_class"] == "on_mat" and r["bin_verdict"] == "in_bin"
            and r["locus"] == "visible_on_mat"),
        n_frame_edge_annotated=sum(1 for r in per_gt if r["frame_edge"]),
        n_bin_rim_notes=sum(1 for r in per_gt if ":rim" in r["evidence"]),
        definition=("ambiguity_rate_no_locus_info = fraction of GT rows whose note carries no "
                    "locus information (empty, shape-check, merged-split, mask-missed, edge "
                    "notes); these are decided by current-pose FK or kept visible by the "
                    "conservative default. Ambiguous rows are NEVER excluded on geometry in "
                    "the headline, and neutral-pose FK artifacts never exclude."),
    )

    # FK calibration census (the neutral-pose artifact finding)
    fk_calibration = dict(
        n_public_fk_hits=sum(1 for r in per_gt if r["fk_hit"] in
                             ("hand", "arm", "neutral_artifact", "robot_nosplit")),
        n_current_pose_hits=sum(1 for r in per_gt if r["fk_hit"] in ("hand", "arm", "robot_nosplit")),
        n_neutral_artifact=len(artifact_rows),
        n_neutral_artifact_matched=sum(1 for r in artifact_rows if r["matched"]),
        n_neutral_artifact_note_silent=n_artifact_note_silent,
        neutral_artifact_examples=[dict(frame=r["frame"], i=r["i"], cls=r["cls"],
                                        note=r["note"], matched=r["matched"])
                                   for r in artifact_rows[:5]],
        finding=("FKMask.frame_mask ORs in the full neutral-pose body (base_mask). "
                 f"{len(artifact_rows)} GT centroids ({n_artifact_note_silent} without decisive "
                 "notes) land on neutral-pose ARM pixels where the arm is not in the current "
                 f"frame; {sum(1 for r in artifact_rows if r['matched'])} of them were detected. "
                 "Crops confirm visible objects (e.g. e06a O3). The headline keeps them "
                 "visible; the literal-2D-FK variant shows the uncalibrated prescription."),
        depth_ordering_probe=("A ray-capsule depth-ordering probe (scratch) was inconclusive: "
                              "the sign of (object depth - robot depth) flips with the unknown "
                              "table height (26/39 rows flip between counter_z 0.01 and 0.15), "
                              "so depth ordering is NOT used. Current-pose 2D FK exclusion is "
                              "kept per the audit rule; crops show a few of those objects sit "
                              "in front of the hand (e09a/e02a/e14c) - bounded by the "
                              "notes-only sensitivity below."),
    )

    # why the geom variant is not the headline: tag-a census + worked example
    geom_rows = [r for r in per_gt if r["evidence_geom_variant"].startswith("geom:")]
    geom_tag_a = [r for r in geom_rows if r["frame"].endswith("a")]
    geom_evidence = dict(
        n_geom_decided=len(geom_rows),
        n_geom_decided_tag_a=len(geom_tag_a),
        n_geom_decided_centroid_v_ge_216=sum(1 for r in geom_rows if r["centroid"][1] >= 216),
        n_geom_decided_matched=sum(1 for r in geom_rows if r["matched"]),
        tag_a_example=[dict(frame=r["frame"], i=r["i"], cls=r["cls"], centroid=r["centroid"],
                            note=r["note"]) for r in geom_tag_a[:4]],
        explanation=("At tag a (20% of each episode) no object has been placed yet, so tag-a "
                     "objects cannot be in a bin; e00a O0/O1 (purple cube, orange prism) are "
                     "documented in AUDIT.md F3 as real mat objects in front of the bins. The "
                     "nominal counter_z=0.01 plane is too low (real table +5-14 cm higher, "
                     "review F1/R9), so centroid rays overshoot into the bin footprints. The "
                     "geom variant therefore over-excludes visible mat objects and inflates "
                     "recall; the headline drops geometry entirely (conservative lower bound)."),
    )

    # sensitivity: ep bands (label verification: ep0-29 100% accurate, ep30-39 error cluster)
    bands = {}
    for label, lo, hi in (("ep0_29", 0, 29), ("ep30_39", 30, 39)):
        sub = [r for r in per_gt if lo <= r["ep"] <= hi]
        bands[label] = dict(
            raw=overall(sub),
            visible_only=overall([r for r in sub if r["locus"] == "visible_on_mat"]),
            visible_only_per_class=summarise([r for r in sub if r["locus"] == "visible_on_mat"]),
            notes_only=overall([r for r in sub if r["locus"] == "visible_on_mat"
                                or r["evidence"].startswith("fk:")]),
            geom_variant_visible_only=overall(
                [r for r in sub if r["locus_geom_variant"] == "visible_on_mat"]))
    # sensitivity: notes only (no FK at all) - bounds the current-pose FK approximation
    notes_only_recs = [r for r in per_gt if r["locus"] == "visible_on_mat"
                       or r["evidence"].startswith("fk:")]
    # sensitivity: drop the independently-verified FP rows (annotation only)
    fp_rows = [r for r in per_gt if r["known_fp"]]
    nofp_vis = [r for r in per_gt if not r["known_fp"] and r["locus"] == "visible_on_mat"]
    sensitivity = dict(
        notes_only_no_fk=dict(overall=overall(notes_only_recs),
                              per_class=summarise(notes_only_recs),
                              note=("drops ALL FK exclusions (current-pose too); upper-bounds "
                                    "the effect of FK mask approximation on the headline")),
        known_fp_rows=[dict(frame=r["frame"], i=r["i"], cls=r["cls"], note=r["note"],
                            locus=r["locus"], matched=r["matched"]) for r in fp_rows],
        visible_only_excluding_known_fp=overall(nofp_vis),
        visible_only_excluding_known_fp_per_class=summarise(nofp_vis),
        note2=("known FPs from label_verify.json (independent visual audit); primary numbers "
               "keep them (measurement only). Dropping them is a sensitivity, not the headline."),
    )

    fk_census = dict(status=fk.status, frame_status_counts=fk_status_counts,
                     method=("FKMask public frame_mask (pinocchio FK from parquet joint state, "
                             "simplified link capsules + static neutral-pose body) projected "
                             "into cam_high; partitioned into current-pose hand, current-pose "
                             "arm+base, neutral-pose arms; sanity hand|arm|neutral == public "
                             "combined; fallback ok_nosplit -> any FK hit counts as occluded"),
                     source=cg_path)

    meta = dict(
        tool="tools/e1_fair_recall.py", date=time.strftime("%Y-%m-%d %H:%M:%S"),
        commit=os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
        labels_file=labels_path, n_label_rows=len(rows), n_gt_objects=n_obj,
        n_gt_held_out_skipped=n_heldout, n_frames=len(per_frame),
        missing_detection_frames=missing_frames,
        match_radius_px=MATCH_PX,
        match_contract=("e1_real_audit.match_gt greedy nearest-centroid, robot-path "
                        "accepted_false, shared used-set per frame, file order"),
        headline_mode="notes + current-pose FK (calibrated, conservative, geometry-free)",
        counter_z_m=geom.counter_z, counter_z_is_sim_placeholder=(args.counter_z == COUNTER_Z_NOMINAL),
        gate_height_m=cg.GATE_HEIGHT_M, z_plane_m=geom.z_plane,
        held_out_from=HELD_OUT_FROM, overlay=cfg.get("overlay"),
        runtime_s=round(time.time() - t_start, 1),
    )
    caveats = [
        "counter_z=0.01 m is the SIM PLACEHOLDER (review F1), not a measured table height. "
        "The geom variant's in_bin stratification is approximate and demonstrably over-excludes "
        "at this height (see geom_variant_evidence). A corrected counter_z re-run is pending "
        "(another agent); use --counter-z/--suffix then. The HEADLINE (notes+current-pose FK) "
        "is robust to this: it uses no geometry.",
        "Headline fair recall is a LOWER bound: unsure rows stay visible, including silent-note "
        "objects that may already be inside bins, and neutral-pose FK artifacts. The geom "
        "variant is the approximate upper end; the corrected counter_z will tighten the interval.",
        "FK masks are simplified link capsules from the recorded joint state projected with the "
        "CALIBRATION camera pose; frames whose real pose differs shift the mask. The public "
        "combined mask includes the static neutral-pose body; 64 GT centroids land on "
        "neutral-pose ARM pixels (arm not there in-frame); the 61 without decisive notes are "
        "kept visible in the headline. "
        "Current-pose 2D FK exclusion can still over-exclude objects sitting IN FRONT of the "
        "hand (crops e09a/e02a/e14c); the notes-only sensitivity bounds this.",
        "labels_e1.jsonl is the v1 draft (the colorgate lane treats labels_e1_v2.jsonl as "
        "authoritative); this run uses the task-specified v1. Independent verification "
        "(LABEL-VERIFY.md): ep0-29 100% accurate in sample; ep30-39 has 10 FP rows + 5 missed "
        "objects. Known FPs are annotated and a drop-them sensitivity is reported; the 5 missed "
        "visible objects mean even the fair number slightly UNDERSTATES detectable recall.",
        "Note evidence outranks FK/geometry for held/occluded/bin: 'on mat right, occluded by "
        "right gripper' is occluded (free object, gripper in front), not in_hand. 'on mat' notes "
        "outrank the approximate geom variant but NOT FK occlusion at the centroid. in_hand and "
        "occluded are both excluded, so the split never changes the fair recall itself.",
        "Bin-RIM notes ('on left bin rim', 'at right bin rim, being placed') are counted in_bin "
        "per the note-mentions-bin rule; the robot path hull-cuts bin regions, so rim objects "
        "are not reliably detectable as free mat objects. Rim count is reported separately.",
        "Episodes 40-47 held out: never opened or processed.",
    ]

    out = dict(
        meta=meta, rule_headline=RULE_HEADLINE, rule_literal_fk=RULE_LITERAL_FK,
        rule_geom_variant=RULE_GEOM,
        headline=headline, variant_literal_fk=literal, geom_variant=variant,
        fk_calibration=fk_calibration, geom_variant_evidence=geom_evidence,
        raw=raw, ambiguity=ambiguity, fk_mask=fk_census, crosscheck=cross,
        ep_band_sensitivity=bands, sensitivity=sensitivity,
        caveats=caveats, per_frame=per_frame, per_gt=per_gt,
    )
    suf = args.suffix
    json_name = f"fair_recall{suf}.json"
    md_name = "FAIR-RECALL.md" if not suf else f"FAIR-RECALL-{suf.strip('_').upper()}.md"
    with open(f"{args.outdir}/{json_name}", "w") as fh:
        json.dump(out, fh, indent=1)
    write_report(args.outdir, out, md_name)
    print(f"[fair] headline locus counts: {headline['locus_counts']}", flush=True)
    print(f"[fair] raw overall: {raw['overall']}", flush=True)
    print(f"[fair] VISIBLE-ONLY headline: {headline['visible_only']['overall']}", flush=True)
    print(f"[fair] literal-2D-FK variant: {literal['visible_only']['overall']}; "
          f"geom variant: {variant['visible_only']['overall']} "
          f"(approximate, counter_z={geom.counter_z})", flush=True)
    print(f"[fair] FK neutral-pose artifacts kept visible: "
          f"{fk_calibration['n_neutral_artifact']} "
          f"({fk_calibration['n_neutral_artifact_matched']} matched)", flush=True)
    print(f"[fair] crosscheck: per_class_identical={cross['per_class_identical']} "
          f"per_colour_identical={cross['per_colour_identical']} "
          f"published_identical={cross.get('published_audit_per_class_identical')}", flush=True)
    print(f"[fair] wrote {json_name}, {md_name} ({time.time() - t_start:.1f}s)", flush=True)

    # ---- calibration crops ----
    if args.crops and crop_wanted:
        cdir = os.path.join(SCRATCH, "crops")
        os.makedirs(cdir, exist_ok=True)
        sel = []
        for cat, rec in sorted(crop_wanted.items()):
            bgr = cv2.imread(f"{FRAMES}/{rec['frame']}.png")
            if bgr is None:
                continue
            key = (rec["ep"], rec["frame_idx"])
            fm = mask_cache.get(key, {})
            comb_m, hand_m, neutral_m = fm.get("combined"), fm.get("hand"), fm.get("neutral")
            ov = bgr.copy()
            if comb_m is not None:
                ov[comb_m > 0] = (0.45 * ov[comb_m > 0] + 0.55 * np.array([0, 0, 200])).astype(np.uint8)
            if hand_m is not None:
                cnts, _ = cv2.findContours(hand_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ov, cnts, -1, (0, 200, 255), 1)
            if neutral_m is not None:
                cnts, _ = cv2.findContours(neutral_m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(ov, cnts, -1, (255, 0, 255), 1)
            x0, y0, x1, y1 = 0, 0, 640, 480
            bb = None
            for g in by_frame[key]:
                cc = real.gt_centroid(g)
                if cc and abs(cc[0] - rec["centroid"][0]) < 0.6 and abs(cc[1] - rec["centroid"][1]) < 0.6:
                    bb = real.gt_bbox(g)
                    break
            if bb:
                x0, y0 = max(0, int(bb[0]) - 60), max(0, int(bb[1]) - 60)
                x1, y1 = min(640, int(bb[2]) + 60), min(480, int(bb[3]) + 60)
                cv2.rectangle(ov, (int(bb[0]), int(bb[1])), (int(bb[2]), int(bb[3])), (0, 255, 0), 1)
            cx, cy = int(rec["centroid"][0]), int(rec["centroid"][1])
            cv2.drawMarker(ov, (cx, cy), (0, 255, 0), cv2.MARKER_CROSS, 12, 1)
            crop = ov[y0:y1, x0:x1]
            crop = cv2.resize(crop, None, fx=3, fy=3, interpolation=cv2.INTER_NEAREST)
            path = os.path.join(cdir, f"{cat}_{rec['frame']}_O{rec['i']}.png")
            cv2.imwrite(path, crop)
            sel.append(dict(category=cat, path=path, **{k: rec[k] for k in
                       ("frame", "i", "cls", "note", "locus", "evidence",
                        "locus_geom_variant", "fk_hit", "bin_verdict", "matched")}))
        with open(os.path.join(SCRATCH, "crop_selection.json"), "w") as fh:
            json.dump(sel, fh, indent=1)
        print(f"[fair] wrote {len(sel)} calibration crops to {cdir}", flush=True)
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def write_report(outdir, a, md_name):
    m, raw = a["meta"], a["raw"]
    h, lit, v = a["headline"], a["variant_literal_fk"], a["geom_variant"]
    L = []
    L.append("# E1 FAIR visible-only object recall — GT stratified by locus/visibility\n")
    L.append(f"_Generated by `tools/e1_fair_recall.py` at {m['date']} (commit {m['commit']}). "
             "MEASUREMENT ONLY: no thresholds tuned, no policy code touched, inputs read-only. "
             "Episodes 40-47 held out (never opened)._\n")
    L.append("## Why\n")
    L.append("Raw recall (audit_e1.json) counts ALL GT objects in the denominator, including "
             "objects the robot-path detector cannot possibly see: held in the gripper, occluded "
             "by the arm/wrist, or already inside a bin. The fair number restricts the denominator "
             "to **visible-on-mat** objects.\n")
    L.append("## Method\n")
    L.append(f"- Labels: `{m['labels_file']}` — {m['n_label_rows']} rows, "
             f"**{m['n_gt_objects']} kind=object GT rows** over {m['n_frames']} frames (ep 0-39).")
    L.append(f"- Matching: `{m['match_contract']}`, radius {m['match_radius_px']} px. "
             "One shared greedy pass over ALL object rows per frame, so the locus strata "
             "partition the raw totals exactly (cross-checked against the imported `match_gt` "
             "and the published audit_e1.json).")
    L.append(f"- **Headline rule**: {a['rule_headline']}")
    L.append(f"- **Literal-2D-FK variant**: {a['rule_literal_fk']}")
    L.append(f"- **Geom variant rule**: {a['rule_geom_variant']}")
    L.append(f"- FK: {a['fk_mask']['method']} (status {a['fk_mask']['status']}, "
             f"frame status {a['fk_mask']['frame_status_counts']}).")
    L.append(f"- Bin geometry (geom variant only): GateGeometry ray-plane at z={m['z_plane_m']} m "
             f"(counter_z={m['counter_z_m']} m + {m['gate_height_m']*100:.2f} cm). "
             "**counter_z=0.01 m is the sim placeholder (review F1)** — see below.\n")
    L.append("## GT denominator breakdown (headline locus)\n")
    L.append("| locus | n | share | matched | recall | meaning |")
    L.append("|---|---|---|---|---|---|")
    meaning = dict(in_hand="held/grasped (note or current-pose hand mask) — not a free object",
                   occluded="behind arm/wrist/gripper (note or current-pose arm mask)",
                   in_bin="inside bin or on rim (note evidence) — hull-cut by robot path",
                   visible_on_mat="free, visible on the mat — DETECTABLE")
    tot = sum(h["locus_counts"].values())
    for loc in ("in_hand", "occluded", "in_bin", "visible_on_mat"):
        st = h["recall_by_locus"][loc]
        L.append(f"| {loc} | {st['n']} | {st['n']/tot:.1%} | {st['matched']} | "
                 f"{st['recall'] if st['recall'] is not None else '—'} | {meaning[loc]} |")
    L.append(f"| **total** | **{tot}** | 100% | {raw['overall']['matched']} | "
             f"{raw['overall']['recall']} | |")
    L.append("")
    L.append(f"Evidence histogram (headline): {h['evidence_histogram']}")
    L.append(f"Bin-rim notes inside in_bin: {a['ambiguity']['n_bin_rim_notes']}; "
             f"frame-edge annotations (kept visible): {a['ambiguity']['n_frame_edge_annotated']}.\n")
    L.append("## Raw vs FAIR (visible-only) recall — ep 0-39, HEADLINE\n")
    L.append("| class | raw n | raw matched | raw recall | visible n | visible matched | "
             "**fair recall** |")
    L.append("|---|---|---|---|---|---|---|")
    for cls in sorted(set(raw["per_class"]) | set(h["visible_only"]["per_class"])):
        r = raw["per_class"].get(cls, dict(n=0, matched=0, recall=None))
        vv = h["visible_only"]["per_class"].get(cls, dict(n=0, matched=0, recall=None))
        L.append(f"| {cls} | {r['n']} | {r['matched']} | {r['recall']} | {vv['n']} | "
                 f"{vv['matched']} | **{vv['recall'] if vv['recall'] is not None else '—'}** |")
    ro, vo = raw["overall"], h["visible_only"]["overall"]
    L.append(f"| **overall** | {ro['n']} | {ro['matched']} | {ro['recall']} | {vo['n']} | "
             f"{vo['matched']} | **{vo['recall']}** |")
    L.append("")
    L.append("Per colour (headline visible-only): " + "; ".join(
        f"{c}: {x['matched']}/{x['n']} = {x['recall']}"
        for c, x in h["visible_only"]["per_colour"].items()))
    L.append("")
    lo, go = lit["visible_only"]["overall"], v["visible_only"]["overall"]
    no = a["sensitivity"]["notes_only_no_fk"]["overall"]
    L.append(f"**The honest number: visible-only recall = {vo['matched']}/{vo['n']} = "
             f"{vo['recall']:.1%}** (raw {ro['recall']:.1%}). Conservative LOWER bound. "
             f"Mode comparison: literal-2D-FK {lo['matched']}/{lo['n']} = {lo['recall']:.3f}; "
             f"notes-only (no FK) {no['matched']}/{no['n']} = {no['recall']:.3f}; "
             f"geom variant {go['matched']}/{go['n']} = {go['recall']:.3f} (approximate, "
             f"over-excludes at placeholder counter_z). Fair interval ≈ "
             f"**[{vo['recall']:.3f}, ~{go['recall']:.3f}]**, to be tightened by the pending "
             "corrected-counter_z re-run.\n")
    L.append("## FK calibration — the neutral-pose artifact\n")
    fc = a["fk_calibration"]
    L.append(f"- Public FK mask hits: {fc['n_public_fk_hits']}; current-pose hits: "
             f"{fc['n_current_pose_hits']}; **neutral-pose arm artifacts: "
             f"{fc['n_neutral_artifact']}** ({fc['n_neutral_artifact_note_silent']} without "
             f"decisive notes; {fc['n_neutral_artifact_matched']} of all artifact rows "
             "matched a robot detection).")
    L.append(f"- {fc['finding']}")
    L.append(f"- Examples: {fc['neutral_artifact_examples']}")
    L.append(f"- {fc['depth_ordering_probe']}\n")
    L.append("## Geom variant (approximate) — why it is NOT the headline\n")
    ge = a["geom_variant_evidence"]
    L.append(f"- Variant locus counts: {v['locus_counts']}; evidence: {v['evidence_histogram']}")
    L.append(f"- Geometry decided {ge['n_geom_decided']} rows as in_bin; "
             f"**{ge['n_geom_decided_tag_a']} of them at tag a** (20% of the episode, before any "
             f"placement — objects cannot be in bins yet), "
             f"{ge['n_geom_decided_centroid_v_ge_216']} with centroid v >= 216 px (mat zone), "
             f"and {ge['n_geom_decided_matched']} matched any robot detection.")
    L.append(f"- Worked example: {ge['tag_a_example']}. AUDIT.md F3 documents the e00a purple "
             "cube (5437 px) and orange prism (4307 px) as REAL mat objects in front of the "
             "bins that the depth=None robot hull-cut drops — visible objects, i.e. legitimate "
             "denominator members and real detector failures, not in_bin objects.")
    L.append(f"- {ge['explanation']}")
    L.append("- Variant visible-only per class: " + "; ".join(
        f"{c}: {x['matched']}/{x['n']} = {x['recall']}"
        for c, x in v["visible_only"]["per_class"].items()) + "\n")
    L.append("## Classification ambiguity\n")
    amb = a["ambiguity"]
    L.append(f"- Note class counts: {amb['note_class_counts']}")
    L.append(f"- Decided by note alone: {amb['n_note_locus_decisive']}; by current-pose FK: "
             f"{amb['n_fk_decided']}; conservative visible default: {amb['n_default_visible']}; "
             f"by geometry (variant only): {amb['n_geom_decided_variant']}.")
    L.append(f"- Ambiguity rate (notes with no locus info): "
             f"{amb['n_no_locus_info_in_note']}/{tot} = {amb['ambiguity_rate_no_locus_info']:.1%}. "
             + amb["definition"])
    L.append(f"- 'on mat' notes overruled by FK occlusion (kept excluded): "
             f"{amb['n_note_on_mat_overruled_by_fk']}; 'on mat' notes the geom variant would "
             f"overrule (protected in the variant): "
             f"{amb['n_note_on_mat_would_be_overruled_by_geom']}.\n")
    L.append("## Sensitivities\n")
    for band, st in a["ep_band_sensitivity"].items():
        L.append(f"- {band}: raw {st['raw']['matched']}/{st['raw']['n']} = {st['raw']['recall']}; "
                 f"visible-only {st['visible_only']['matched']}/{st['visible_only']['n']} = "
                 f"{st['visible_only']['recall']} "
                 f"(geom variant {st['geom_variant_visible_only']['recall']})")
    kf = a["sensitivity"]
    L.append(f"- Known label FPs (independent visual verification, label_verify.json): "
             f"{len(kf['known_fp_rows'])} rows annotated; dropping them: visible-only "
             f"{kf['visible_only_excluding_known_fp']['matched']}/{kf['visible_only_excluding_known_fp']['n']}"
             f" = {kf['visible_only_excluding_known_fp']['recall']} (sensitivity only; primary "
             "numbers keep all rows).")
    L.append("")
    L.append("## Cross-checks\n")
    cc = a["crosscheck"]
    L.append(f"- My raw per-class == imported match_gt: **{cc['per_class_identical']}**; "
             f"per-colour: **{cc['per_colour_identical']}**; == published audit_e1.json: "
             f"**{cc.get('published_audit_per_class_identical')}** "
             f"(matched total {cc['my_matched_total']} vs {cc['match_gt_matched_total']}).\n")
    L.append("## Caveats\n")
    for c in a["caveats"]:
        L.append(f"- {c}")
    L.append("")
    L.append("## Files\n")
    L.append(f"- Machine-readable: `{outdir}/fair_recall.json` (per-frame + per-GT detail, "
             "locus evidence for all three modes, FK/bin trace, match distances).")
    L.append("- Calibration crops: `.scratch-e1-fairrecall/crops/` + `crop_selection.json` "
             "(red = public FK mask, yellow = current hand, magenta = neutral-pose arms, "
             "green = GT bbox/centroid).")
    L.append("")
    with open(f"{outdir}/{md_name}", "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
