#!/usr/bin/env python
"""Offline replay of the classical policy's colour perception on real venue frames (no GPU, no simulator, no depth).

Runs the REAL `Perception.detect()` code path (HSV masks -> blob extraction -> bin_size_guard -> split_blobs ->
`_detect_blob` -> reject -> color_gate) on RGB frames cut from the finals recordings, and compares every candidate
blob with hand-labelled ground truth (object + bin boxes, `test_data/venue_20261115/*.json`).

DEPTH: the recordings contain no depth. We pass depth=None. Stages that need depth are therefore NOT exercised:
  - bin_size_guard 3D checks (top height > 7.5 cm, footprint > 13 cm): skipped; only its AREA rule (> 4x expected object
    area, expected area from the camera ray at the object-top plane) is evaluated.
  - depth top-face localisation / height estimate / depth_recover / 3D color_gate rules: skipped, detections are located
    by the plane fallback (pixel ray x counter plane), so `xy` is only a plane projection.
Everything else (colour thresholds, open_px, min_area, split, color_gate no-Z rules, sat floor) is the shipped code.

Usage (from the repo root):
  uv run -q --with numpy --with opencv-python-headless --with pyyaml --with scipy python tools/venue_replay.py
  ... tools/venue_replay.py --frames r1_t12 r2_t12 --configs legacy guard guard+bin_image --save-overlays /tmp/ov
`legacy` = 1063951 behaviour (finals_r25_gen with perception.bin_size_guard off), `guard` = hotfix 77ba4e0,
`guard+bin_image` = hotfix plus the default-off perception.bin_image_guard rule.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

DATA = os.path.join(ROOT, "aaltonomous", "baselines", "classical", "test_data", "venue_20261115")
# policy colour class of each physical object colour (orange cube / cylinder are the policy's "red" class)
POLICY_CLASS = {"blue": "blue", "orange": "red", "yellow": "yellow"}
CONFIGS = {
    "legacy": {"perception.bin_size_guard": False},
    "guard": {"perception.bin_size_guard": True},
    "guard+bin_image": {"perception.bin_size_guard": True, "perception.bin_image_guard": True},
    # data point only: widen the gate_below_bins first-run band (venue right-bin front wall starts at row ~220 > 0.45*480)
    "guard+bin_image+first_run_0.55": {"perception.bin_size_guard": True, "perception.bin_image_guard": True,
                                       "perception.gate_below_bins_params.first_run_top_frac": 0.55},
    "guard+first_run_0.55+rowmed61": {"perception.bin_size_guard": True, "perception.gate_below_bins_params.first_run_top_frac": 0.55,
                                      "perception.gate_below_bins_params.row_median_px": 61},
    "guard+rowmed61_only": {"perception.bin_size_guard": True, "perception.gate_below_bins_params.row_median_px": 61},
    "guard+first_run_0.55": {"perception.bin_size_guard": True, "perception.gate_below_bins_params.first_run_top_frac": 0.55},
}


def load_frames(names=None):
    out = []
    for jp in sorted(glob.glob(os.path.join(DATA, "*.json"))):
        with open(jp) as fh:
            lab = json.load(fh)
        if names and lab["frame"][:-4] not in names:
            continue
        out.append((lab, os.path.join(DATA, lab["frame"])))
    return out


def make_perception(overrides, overlay="finals_r25_gen", log=lambda *_: None):
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.perception import Perception
    from aaltonomous.interface.calibration import Calibration
    from aaltonomous.scene.scene import Scene
    cfg = load_config(overlay=overlay, overrides=dict(overrides))
    return Perception(cfg, Scene.load(), Calibration.load(), log=log)


def _box_mask(shape, boxes):
    m = np.zeros(shape, bool)
    for x0, y0, x1, y1 in boxes:
        m[max(0, y0):y1, max(0, x0):x1] = True
    return m


def run_frame(perc, rgb):
    """-> list of candidate dicts: accepted detections and rejected ones, each with colour, area, uv, mask (if known),
    status ('accepted' | 'rejected:<why>')."""
    import cv2
    masks = {}      # id(det) -> blob mask
    guard_masks = []  # (colour-less) masks of blobs rejected by the size guard (rejected before _detect_blob)
    orig_blob, orig_guard = perc._detect_blob, perc._bin_sized_blob

    def det_blob(color, blob, area, centroid, depth, include_bins, dets, rejected, *a, **k):
        n_d, n_r = len(dets), len(rejected)
        orig_blob(color, blob, area, centroid, depth, include_bins, dets, rejected, *a, **k)
        for d in dets[n_d:]:
            masks[id(d)] = blob
        for d, _ in rejected[n_r:]:
            masks[id(d)] = blob

    def guard(blob, area, centroid, depth):
        why = orig_guard(blob, area, centroid, depth)
        if why:
            guard_masks.append((tuple(centroid), blob))
        return why

    perc._detect_blob, perc._bin_sized_blob = det_blob, guard
    try:
        dets, rejected = perc.detect(rgb, None, debug=True)
    finally:
        perc._detect_blob, perc._bin_sized_blob = orig_blob, orig_guard
    cands = []
    for d in dets:
        cands.append(dict(colour=d.color, area=int(d.n_px), uv=tuple(d.uv) if d.uv is not None else None,
                          xy=tuple(float(v) for v in d.top[:2]), status="accepted", mask=masks.get(id(d)), det=d))
    for d, why in rejected:
        m = masks.get(id(d))
        if m is None and d.uv is not None:   # guard rejection: match the mask by centroid
            for c, bm in guard_masks:
                if abs(c[0] - d.uv[0]) < 1.0 and abs(c[1] - d.uv[1]) < 1.0:
                    m = bm
                    break
        cands.append(dict(colour=d.color, area=int(d.n_px), uv=tuple(d.uv) if d.uv is not None else None,
                          xy=tuple(float(v) for v in d.top[:2]), status="rejected:" + str(why), mask=m, det=d))
    return cands


def classify(cand, lab, shape):
    """Where does this blob sit w.r.t. the labels? -> (kind, name, bin_frac, obj_frac, matched_object_index)."""
    objs = [o["box"] for o in lab["objects"]]
    bins = [b["box"] for b in lab["bins"]]
    om = [_box_mask(shape, [b]) for b in objs]
    obj_union = _box_mask(shape, objs)
    bin_only = _box_mask(shape, bins) & ~obj_union
    m = cand["mask"]
    if m is None:
        if cand["uv"] is None:
            return "unknown", "", 0.0, 0.0, None
        m = np.zeros(shape, bool)
        u, v = int(round(cand["uv"][0])), int(round(cand["uv"][1]))
        m[max(0, v - 2):v + 3, max(0, u - 2):u + 3] = True
    m = m > 0
    n = max(1, int(m.sum()))
    bf = float((m & bin_only).sum()) / n
    of = float((m & obj_union).sum()) / n
    ov = [float((m & x).sum()) for x in om]
    j = int(np.argmax(ov)) if ov and max(ov) > 0 else None
    if bf >= 0.5 and of < 0.5:
        bn = max(lab["bins"], key=lambda b: float((m & _box_mask(shape, [b["box"]])).sum()))["name"]
        return "bin", bn, bf, of, None
    if of >= 0.5:
        return "object", f'{lab["objects"][j]["colour"]} {lab["objects"][j]["shape"]}', bf, of, j
    return ("mixed" if of > 0.1 and bf > 0.1 else "other"), "", bf, of, j


def rank_red(perc, cands):
    """Order the planner would try the accepted red candidates: policy.order=nearest_bin -> distance (plane xy) to the
    mapped bin (policy.fixed_mapping red->right). Returns the sorted candidate list."""
    from aaltonomous.baselines.classical.planner import lookup_detection_bin
    import aaltonomous.baselines.classical.planner as pl
    pol = perc.cfg.get("policy") or {}
    mapping = pol.get("fixed_mapping") or {}
    out = []
    for c in cands:
        if c["status"] != "accepted" or c["colour"] != "red":
            continue
        b = lookup_detection_bin(perc.cfg, mapping, c["det"])
        if b is None or not np.all(np.isfinite(c["xy"])):
            key = float("inf")
        else:
            bxy = perc.scene.mat_to_pelvis(np.r_[perc.scene.cfg["bins"]["centers"][b], 0.0])[:2]
            key = float(np.linalg.norm(np.asarray(c["xy"]) - bxy))
        out.append((key, c))
    out.sort(key=lambda t: t[0])
    return [c for _, c in out]


def evaluate(names=None, configs=None, overlay="finals_r25_gen", save_overlays=None, verbose=True):
    """-> {config: {frame: result}} with result = dict(cands, accepted_on_bin, found, missed, first_red)."""
    import cv2
    res = {}
    for cname in (configs or list(CONFIGS)):
        perc = make_perception(CONFIGS[cname], overlay=overlay)
        res[cname] = {}
        for lab, path in load_frames(names):
            rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            cands = run_frame(perc, rgb)
            for c in cands:
                c["kind"], c["name"], c["bin_frac"], c["obj_frac"], c["obj_idx"] = classify(c, lab, rgb.shape[:2])
            acc = [c for c in cands if c["status"] == "accepted"]
            on_bin = [c for c in acc if c["kind"] in ("bin", "mixed") and c["colour"] in ("red", "yellow", "blue")]
            found, missed = [], []
            for j, o in enumerate(lab["objects"]):
                hit = [c for c in acc if c["obj_idx"] == j and c["kind"] == "object"
                       and c["colour"] == POLICY_CLASS[o["colour"]]]
                (found if hit else missed).append(f'{o["colour"]} {o["shape"]}')
            ranked = rank_red(perc, cands)
            res[cname][lab["frame"][:-4]] = dict(cands=cands, on_bin=on_bin, found=found, missed=missed,
                                                 first_red=ranked[0] if ranked else None, ranked=ranked, lab=lab)
            if save_overlays:
                _save_overlay(save_overlays, cname, lab, rgb, cands)
    if verbose:
        report(res)
    return res


def _save_overlay(outdir, cname, lab, rgb, cands):
    import cv2
    os.makedirs(outdir, exist_ok=True)
    im = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    for b in lab["bins"]:
        x0, y0, x1, y1 = b["box"]
        cv2.rectangle(im, (x0, y0), (x1, y1), (255, 0, 255), 1)
    for o in lab["objects"]:
        x0, y0, x1, y1 = o["box"]
        cv2.rectangle(im, (x0, y0), (x1, y1), (255, 255, 255), 1)
    for c in cands:
        if c["uv"] is None:
            continue
        ok = c["status"] == "accepted"
        col = (0, 200, 0) if ok and c["kind"] == "object" else (0, 0, 255) if ok else (128, 128, 128)
        u, v = int(c["uv"][0]), int(c["uv"][1])
        cv2.circle(im, (u, v), 5, col, -1)
        cv2.putText(im, f'{c["colour"][:3]}{c["area"]}', (u + 6, v), 0, 0.4, col, 1)
    cv2.imwrite(os.path.join(outdir, f'{lab["frame"][:-4]}__{cname.replace("+", "_")}.png'), im)


def report(res):
    for cname, frames in res.items():
        print(f"\n=== config: {cname}  ({CONFIGS[cname]})")
        print("frame   arm  blobs(r/y/b acc)  on-bin acc  | objects found / 5          | first red -> what it is")
        tot_bin = tot_found = tot_n = 0
        for f, r in frames.items():
            acc = [c for c in r["cands"] if c["status"] == "accepted"]
            cnt = {k: sum(1 for c in acc if c["colour"] == k) for k in ("red", "yellow", "blue")}
            fr = r["first_red"]
            br = [i + 1 for i, c in enumerate(r["ranked"]) if c["kind"] in ("bin", "mixed")]
            br_s = f" [bin red ranks: {br} of {len(r['ranked'])}]" if br else ""
            fr_s = "-" if fr is None else f'{fr["kind"]}{(":" + fr["name"]) if fr["name"] else ""} area={fr["area"]} uv=({fr["uv"][0]:.0f},{fr["uv"][1]:.0f})'
            tot_bin += len(r["on_bin"])
            tot_found += len(r["found"])
            tot_n += len(r["lab"]["objects"])
            print(f'{f:7s} {"Y" if r["lab"]["arm_visible"] else "n"}   {cnt["red"]}/{cnt["yellow"]}/{cnt["blue"]}'
                  f'{"":13s}{len(r["on_bin"]):>2}           | {len(r["found"])}/5 miss={r["missed"] or "-"}'
                  f' | {fr_s}{br_s}')
            for c in r["on_bin"]:
                print(f'          ON-BIN accepted {c["colour"]} area={c["area"]} uv=({c["uv"][0]:.0f},{c["uv"][1]:.0f}) '
                      f'{c["name"]} bin_frac={c["bin_frac"]:.2f}')
            why = {}
            for c in r["cands"]:
                if c["status"] != "accepted" and c["kind"] in ("bin", "mixed"):
                    why[c["status"]] = why.get(c["status"], 0) + 1
            if why:
                print(f'          bin-region blobs rejected: {why}')
        print(f"TOTAL {cname}: accepted candidates on a bin = {tot_bin}; objects found = {tot_found}/{tot_n}")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", nargs="*", help="frame names (default: all), e.g. r1_t12")
    ap.add_argument("--configs", nargs="*", default=list(CONFIGS), choices=list(CONFIGS))
    ap.add_argument("--overlay", default="finals_r25_gen")
    ap.add_argument("--save-overlays", help="directory for annotated PNGs")
    a = ap.parse_args(argv)
    evaluate(a.frames, a.configs, a.overlay, a.save_overlays)


if __name__ == "__main__":
    main()
