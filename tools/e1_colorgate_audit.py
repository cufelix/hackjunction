#!/usr/bin/env python3
"""E1 color.gate post-filter audit (Timi's PROPOSED rules, messages f369bb04 + cb364bfb).

MEASUREMENT ONLY: re-runs the EXISTING submission detector (overlay=submission,
hsv_set=robust, split_blobs=true, classify_shape=true) with depth=None on all 192
real HF drop_2 frames (include_bins=True audit view: blobs are whole), then applies
the proposed gate rules as POST-FILTERS on the candidate pool = accepted + ALL
rejected detections. No rule is tuned; no policy code or baseline audit is touched.

Rules (verbatim from Timi):
  Rule 1 (area floor): expected area of a 45mm object = (fx*0.045/Z)^2; accept
    >= 0.4x expected. Z unknown (no depth) -> floor 1000 px. PRIMARY = worst-case
    1000px floor. SECONDARY variant: distance-scaled using Z from ray-plane at
    h=2.25cm (both reported).
  Rule 2 (3D bin zone, NO 2D image hull): object position = centroid ray-plane at
    height 2.25 cm above counter (no depth). xy INSIDE a bin footprint (scene.yaml
    centers, outer_size 0.21, no +3cm margin) -> REJECT. In FRONT of the bin ->
    KEEP. Near the edge (<= 2 cm from the footprint) -> only with valid depth AND
    height above plane 3-7 cm; without depth -> REJECT.
  Rule 3 (count sanity): >6 candidates survive per frame -> flag DEGRADED (census
    only; the real tier logic is not implemented here).
  Rule 4 (shape): 2D 'triangle' without depth -> shape_unknown (annotate + count).
  FK mask: arm+hand link capsules + static base from the parquet joint state at
    each frame's frame_idx (pinocchio FK, URDF g1_body29_hand14), projected into
    cam_high; candidates whose centroid lies in the mask are dropped.

Gate funnel order: rule1 -> rule2 -> rule2b (workspace, Timi b3084182) -> FK mask.
Rule 3/4 are census/annotation on the survivors. Both rule-1 variants run the same
rule2+rule2b+FK pipeline.

Outputs (results/1148-opencode-e1/):
  colorgate_e1.json            aggregates + per-frame survivors with rule_trace
  colorgate_candidates_e1.jsonl  one row per pool candidate (full trace)
  COLOR-GATE.md                human report

GT hook: same contract as tools/e1_real_audit.py match_gt (imported read-only):
labels keyed (ep, frame_idx) via frame_index.json. Prefers labels_e1_v2.jsonl
(authoritative) when present; labels_e1.jsonl is the DEPRECATED 2057-row draft
and any metrics from it are marked DRAFT everywhere. PLUS stratified recall:
GT objects IN FRONT of bins vs inside/near (GT centroid ray-plane at 2.25cm).

Run from the worktree root:
  PYTHONPATH=. <venv>/bin/python tools/e1_colorgate_audit.py [--no-fk]
"""
from __future__ import annotations

import argparse
import importlib.util
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
PARQUET = "/var/tmp/hf_drop2/data/chunk-000/file-000.parquet"
URDF = "third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf"
TAGS = (("a", 0.2), ("b", 0.5), ("c", 0.8), ("z", 0.97))
FPS = 30
HELD_OUT_FROM = 40

# --- gate parameters (Timi's rules, verbatim; NOT tuned here) ---
GATE_HEIGHT_M = 0.0225        # rule 2 ray-plane height above counter
OBJECT_SIZE_M = 0.045         # rule 1 expected-area object size
AREA_FLOOR_PX = 1000          # rule 1 PRIMARY worst-case floor
AREA_COEF = 0.4               # rule 1 SECONDARY: accept >= 0.4 x expected
NEAR_EDGE_M = 0.02            # rule 2 near-edge band (no depth -> reject)
DEGRADED_THRESHOLD = 6        # rule 3: >6 survivors -> DEGRADED
BIN_ZONE_V = 0.38 * 480       # bin-zone FP definition (a)
MATCH_PX = 25.0               # GT match radius (same contract as e1_real_audit)
# rule 2b (Timi b3084182): positive WORKSPACE mask, pelvis frame, config-key style
WS_PELVIS_X_MIN = 0.12        # m
WS_PELVIS_X_MAX = 0.45        # m
WS_PELVIS_ABS_Y_MAX = 0.40    # m
WS_FRONT_MARGIN_M = 0.01      # must be in front of the bin-row near edge minus this

RULES_VERBATIM = {
    "rule1": ("expected area of a 45mm object = (fx*0.045/Z)^2; accept >= 0.4x expected. "
              "Z unknown (no depth) -> floor 1000 px. PRIMARY = worst-case 1000px floor. "
              "SECONDARY variant: distance-scaled using Z from ray-plane at h=2.25cm."),
    "rule2": ("object position = ray-plane at height 2.25 cm above counter (no depth). "
              "xy INSIDE a bin footprint (scene.yaml centers, outer_size 0.21, no +3cm margin) "
              "-> REJECT. In FRONT of the bin -> KEEP. Near the edge (<= 2 cm from the footprint) "
              "-> only with valid depth AND height above plane 3-7 cm; without depth -> REJECT."),
    "rule2b": ("positive WORKSPACE mask (Timi b3084182): accept only candidates whose xy (depth, or "
               "ray-plane at 2.25cm) lies in the workspace in front of the bins: pelvis x 0.12-0.45 m, "
               "|y| <= 0.40 m, and in front of the near edge of the bin row (live footprint) minus 1 cm. "
               "Everything behind the bin row / outside the table = reject. Objects never spawn there."),
    "rule3": "if >6 candidates survive per frame -> flag frame DEGRADED (census only).",
    "rule4": "2D 'triangle' without depth -> shape_unknown (annotate; report counts).",
    "fk_mask": ("FK from parquet joint state at each frame's frame_idx; project arm+hand link "
                "geometries (simplified link capsules) + static robot base into cam_high; drop "
                "candidates whose centroid lies in the mask."),
}


# --------------------------------------------------------------------------
# ray-plane geometry (same math as perception._from_plane, re-run at 2.25 cm)
# --------------------------------------------------------------------------

class GateGeometry:
    def __init__(self, p: Perception, counter_z=None):
        self.p = p
        self.T = p.cal._T(p.cam)
        self.K = p.cal.cams[p.cam].K
        self.R = self.T[:3, :3]
        self.o = self.T[:3, 3]
        # counter_z override (sensitivity variant); default = scene.yaml value (unchanged behaviour)
        self.counter_z = float(p.scene.counter_z if counter_z is None else counter_z)
        self.z_plane = self.counter_z + GATE_HEIGHT_M
        self.bins = {name: np.asarray(c, float) for name, c in p.scene.cfg["bins"]["centers"].items()}
        self.half = np.asarray(p.scene.cfg["bins"]["outer_size"], float) / 2.0   # NO margin
        # rule 2b: near edge of the bin row in the PELVIS frame (live footprint, no margin)
        o = p.scene.origin
        self.ws_near_edge_pelvis_x = float(min(o[0] + c[1] - self.half[1] for c in self.bins.values()))

    def workspace(self, xy):
        """rule 2b (Timi b3084182): (pass, reason) for a MAT-frame xy at the 2.25cm plane."""
        if xy is None:
            return False, "no_plane_hit"
        pp = self.p.scene.mat_to_pelvis(np.array([xy[0], xy[1], 0.0]))
        if pp[0] < WS_PELVIS_X_MIN:
            return False, "x_low"
        if pp[0] > WS_PELVIS_X_MAX:
            return False, "x_high"
        if abs(pp[1]) > WS_PELVIS_ABS_Y_MAX:
            return False, "y_out"
        if pp[0] >= self.ws_near_edge_pelvis_x - WS_FRONT_MARGIN_M:
            return False, "behind_bins"
        return True, "ok"

    def ray_plane(self, uv):
        """Centroid ray -> (mat xy, optical Z) at the 2.25cm plane, or (None, None)."""
        ray_opt = np.array([(uv[0] - self.K[0, 2]) / self.K[0, 0],
                            (uv[1] - self.K[1, 2]) / self.K[1, 1], 1.0])
        ray = self.R @ ray_opt
        if ray[2] >= -1e-6:
            return None, None
        s = (self.z_plane - self.o[2]) / ray[2]     # optical depth (ray_opt z = 1)
        if s <= 0:
            return None, None
        pt = self.o + s * ray
        pm = self.p.scene.pelvis_to_mat(pt)
        return pm[:2], float(s)

    def bin_zone(self, xy):
        """(verdict, distance-to-footprint m, bin name): inside / near_edge / front."""
        if xy is None:
            return "no_plane_hit", None, None
        best_d, best_name = float("inf"), None
        for name, ctr in self.bins.items():
            d = np.abs(xy - ctr)
            if np.all(d <= self.half):
                return "in_bin", 0.0, name
            out = np.maximum(d - self.half, 0.0)
            dist = float(np.hypot(out[0], out[1]))
            if dist < best_d:
                best_d, best_name = dist, name
        if best_d <= NEAR_EDGE_M:
            return "near_edge", best_d, best_name
        return "front", best_d, best_name


# --------------------------------------------------------------------------
# FK mask: simplified link capsules from the parquet joint state (no STL meshes
# are shipped in this worktree, so collision meshes are approximated by capsules
# between link-frame origins; Timi explicitly allowed simplified link cylinders).
# --------------------------------------------------------------------------

ARM_CHAIN = ["shoulder_pitch_link", "shoulder_roll_link", "shoulder_yaw_link", "elbow_link",
             "wrist_roll_link", "wrist_pitch_link", "wrist_yaw_link", "hand_palm_link"]
ARM_R = [0.05, 0.05, 0.045, 0.04, 0.038, 0.032, 0.032, 0.035]
FINGERS = [("hand_thumb_0_link", "hand_thumb_1_link", "hand_thumb_2_link"),
           ("hand_middle_0_link", "hand_middle_1_link"),
           ("hand_index_0_link", "hand_index_1_link")]
LINKS = (["pelvis", "waist_yaw_link", "waist_roll_link", "torso_link", "head_link"] +
         [f"{s}_{l}" for s in ("left", "right") for l in
          ARM_CHAIN + [f for fl in FINGERS for f in fl]])


def capsule_pts(a, b, r, n_ang=12):
    ax = b - a
    L = float(np.linalg.norm(ax))
    t = ax / L if L > 1e-6 else np.array([1.0, 0, 0])
    tmp = np.array([0.0, 0, 1]) if abs(t[2]) < 0.9 else np.array([1.0, 0, 0])
    n1 = np.cross(t, tmp); n1 /= np.linalg.norm(n1)
    n2 = np.cross(t, n1)
    ang = np.linspace(0, 2 * np.pi, n_ang, endpoint=False)
    circ = np.outer(np.cos(ang), n1) + np.outer(np.sin(ang), n2)
    pts = [a + f * ax + r * circ for f in np.linspace(0, 1, 5)]
    for end, sgn in ((a, -1), (b, 1)):
        for pol in (np.pi / 6, np.pi / 3):
            pts.append(end + sgn * r * np.cos(pol) * t + r * np.sin(pol) * circ)
    return np.vstack(pts)


class FKMask:
    """Per-frame robot silhouette mask in cam_high from the recorded joint state."""

    def __init__(self, cal, enabled=True):
        self.cal = cal
        self.enabled = enabled
        self.status = "disabled"
        self.states = {}
        self.offsets = {}
        if not enabled:
            return
        import pandas as pd
        import pinocchio as pin
        import yaml
        self.pin = pin
        self.model = pin.buildModelFromUrdf(URDF)
        self.data = self.model.createData()
        spec = yaml.safe_load(open("aaltonomous/interface/joint_spec.yaml"))
        self.joint_names = [j["name"] for j in spec["arm"]] + [j["name"] for j in spec["hand"]]
        self.iq = {n: self.model.joints[self.model.getJointId(n)].idx_q for n in self.joint_names}
        self.q0 = pin.neutral(self.model)          # legs/waist at 0 = the stiff pose
        self.fid = {n: self.model.getFrameId(n) for n in LINKS}
        df = pd.read_parquet(PARQUET)
        # parquet `index` = global concatenated frame index = offset(ep) + frame_index;
        # frame_index.json (train) and the extractor ceil formula (held-out, manifest ts
        # are global) both live in that space.
        self.states = {int(i): np.asarray(s, float) for i, s in zip(df["index"], df["observation.state"])}
        cnt = df.groupby("episode_index").size()
        self.offsets = cnt.cumsum().shift(1, fill_value=0).to_dict()
        self.base_mask = self._mask(self._capsules(self.q0))   # waist stiff -> base is static
        self.status = "ok"

    def global_index(self, ep, frame_idx_ep):
        return int(self.offsets.get(ep, 0)) + int(frame_idx_ep)

    def _capsules(self, q):
        self.pin.framesForwardKinematics(self.model, self.data, q)
        P = {n: self.data.oMf[f].translation.copy() for n, f in self.fid.items()}
        caps = [(P["pelvis"], P["waist_roll_link"], 0.11),
                (P["waist_roll_link"], P["torso_link"], 0.12),
                (P["torso_link"], P["head_link"], 0.08)]
        for s in ("left", "right"):
            ch = [f"{s}_{l}" for l in ARM_CHAIN]
            caps.append((P["torso_link"], P[ch[0]], 0.055))
            caps += [(P[ch[i]], P[ch[i + 1]], ARM_R[i]) for i in range(len(ch) - 1)]
            caps.append((P[f"{s}_hand_palm_link"], P[f"{s}_hand_middle_0_link"], 0.03))
            for fl in FINGERS:
                names = [f"{s}_{l}" for l in fl]
                caps += [(P[names[i]], P[names[i + 1]], 0.014) for i in range(len(names) - 1)]
                M = self.data.oMf[self.fid[names[-1]]]
                tip = M.translation + M.rotation @ np.array([0.052, 0, 0])
                caps.append((P[names[-1]], tip, 0.012))
        return caps

    def _mask(self, caps):
        m = np.zeros((480, 640), np.uint8)
        for a, b, r in caps:
            uvz = self.cal.project("head", capsule_pts(a, b, r))
            ok = uvz[:, 2] > 0.05
            if ok.sum() < 3:
                continue
            hull = cv2.convexHull(np.round(uvz[ok, :2]).astype(np.int32))
            cv2.fillConvexPoly(m, hull, 255)
        return m

    def frame_mask(self, ep, frame_idx_ep):
        """(mask, status): mask=None when the joint state is unavailable."""
        if not self.enabled:
            return None, "disabled"
        gi = self.global_index(ep, frame_idx_ep)
        st = self.states.get(gi)
        if st is None:
            return None, "no_state"
        q = self.q0.copy()
        for n, v in zip(self.joint_names, st):
            q[self.iq[n]] = v
        m = self._mask(self._capsules(q))
        m |= self.base_mask
        return m, "ok"


# --------------------------------------------------------------------------
# gate pipeline
# --------------------------------------------------------------------------

def gate_candidate(cand, geom, fk_mask):
    """Apply rules 1+2+FK to one pool candidate. Returns a trace dict (both rule-1 variants)."""
    uv, n_px = cand["uv"], cand["n_px"]
    xy, z_opt = geom.ray_plane(uv)
    r2, r2_dist, r2_bin = geom.bin_zone(xy)
    # rule 1 PRIMARY: worst-case 1000 px floor
    r1p = n_px >= AREA_FLOOR_PX
    # rule 1 SECONDARY: 0.4 x (fx*0.045/Z)^2 at the 2.25cm plane; no plane hit -> primary floor
    if z_opt is not None:
        expected = (geom.K[0, 0] * OBJECT_SIZE_M / z_opt) ** 2
        thr = AREA_COEF * expected
        r1s = n_px >= thr
    else:
        expected, thr = None, float(AREA_FLOOR_PX)
        r1s = r1p
    # rule 2: depth=None worst case -> near-edge rejects; inside rejects; front keeps
    r2_pass = r2 == "front"
    # rule 2b (Timi b3084182): positive workspace mask in the pelvis frame
    r2b_pass, r2b_reason = geom.workspace(xy)
    # FK mask: centroid in the robot silhouette -> drop
    fk_hit = False
    if fk_mask is not None:
        u = int(min(max(round(uv[0]), 0), 639))
        v = int(min(max(round(uv[1]), 0), 479))
        fk_hit = bool(fk_mask[v, u])
    trace = []
    trace.append(f"r1:{'pass' if r1p else 'floor'}({n_px}px)")
    trace.append(f"r2:{r2}" + (f":{r2_bin}" if r2_bin else "") +
                 (f"(d={r2_dist:.3f}m)" if r2_dist is not None else ""))
    trace.append(f"r2b:{r2b_reason}")
    trace.append("fk:" + ("hit" if fk_hit else "pass"))
    surv_p = r1p and r2_pass and r2b_pass and not fk_hit
    surv_s = r1s and r2_pass and r2b_pass and not fk_hit
    drop = None
    if not surv_p:
        drop = ("rule1" if not r1p else
                ("rule2:" + r2 if not r2_pass else
                 ("rule2b:" + r2b_reason if not r2b_pass else "fk_mask")))
    return dict(
        xy_gate=[round(float(xy[0]), 4), round(float(xy[1]), 4)] if xy is not None else None,
        z_optical=round(z_opt, 4) if z_opt is not None else None,
        r1_primary=bool(r1p), r1_secondary=bool(r1s),
        r1s_expected_px=round(expected, 1) if expected is not None else None,
        r1s_threshold_px=round(thr, 1),
        r2=r2, r2_dist_m=round(r2_dist, 4) if r2_dist is not None else None, r2_bin=r2_bin,
        r2b_pass=bool(r2b_pass), r2b_reason=r2b_reason,
        fk_hit=fk_hit, survives_primary=bool(surv_p), survives_secondary=bool(surv_s),
        drop_reason_primary=drop, rule_trace=trace,
    )


def survivor_row(cand, tr):
    return dict(color=cand["color"], uv=cand["uv"], n_px=cand["n_px"], shape=cand["shape"],
                shape_unknown=bool(cand["shape"] == "triangle"),    # rule 4 (depth=None)
                orig=cand["orig"], rule_trace=tr["rule_trace"],
                xy_gate=tr["xy_gate"], r2=tr["r2"])


# --------------------------------------------------------------------------
# GT hook (same contract as e1_real_audit.match_gt, imported read-only)
# --------------------------------------------------------------------------

def load_real_audit_module():
    path = os.path.join(os.path.dirname(__file__), "e1_real_audit.py")
    spec = importlib.util.spec_from_file_location("e1_real_audit_readonly", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def stratified_recall(labels_rows, frames_rows, geom, survivors_key):
    """Recall stratified by GT-centroid ray-plane position at 2.25cm:
    front (outside footprint, >2cm) vs near_edge (<=2cm) vs inside vs no_plane_hit.
    Same greedy 25px nearest-centroid contract as match_gt."""
    det_by_fidx = {(fr["ep"], fr["frame_idx"]): fr for fr in frames_rows}
    # group + sort exactly like match_gt so greedy matching order (and totals) agree
    by_frame = {}
    for g in labels_rows:
        if str(g.get("kind", "")).lower() != "object":
            continue
        ep = g.get("ep", g.get("episode"))
        if ep is None or g.get("frame_idx") is None:
            continue
        by_frame.setdefault((int(ep), int(g["frame_idx"])), []).append(g)
    strata = {}
    miss = {}
    for key, gt_rows in sorted(by_frame.items()):
        fr = det_by_fidx.get(key)
        if fr is None:
            continue
        for g in gt_rows:
            c = None
            for k in ("centroid_px", "centroid", "uv", "center", "centre"):
                if isinstance(g.get(k), (list, tuple)) and len(g[k]) >= 2:
                    c = (float(g[k][0]), float(g[k][1]))
                    break
            if c is None:
                continue
            xy, _ = geom.ray_plane(c)
            verdict, dist, _ = geom.bin_zone(xy)
            if verdict == "front":
                ws_pass, ws_reason = geom.workspace(xy)
                if not ws_pass:
                    verdict = "front_outside_workspace"
            split = "held_out" if fr["held_out"] else "train"
            st = strata.setdefault(split, {}).setdefault(verdict, dict(n=0, matched=0))
            st["n"] += 1
            dets = fr[survivors_key]
            best, bd = None, MATCH_PX
            for i, d in enumerate(dets):
                if i in fr.get("_used", set()):
                    continue
                dd = math.hypot(d["uv"][0] - c[0], d["uv"][1] - c[1])
                if dd <= bd:
                    best, bd = i, dd
            if best is not None:
                fr.setdefault("_used", set()).add(best)
                st["matched"] += 1
            else:
                # miss attribution: nearest FULL-POOL candidate within 25px -> which rule dropped it
                bp, bpd = None, MATCH_PX
                for x in fr.get("candidates", []):
                    dd = math.hypot(x["uv"][0] - c[0], x["uv"][1] - c[1])
                    if dd <= bpd:
                        bp, bpd = x, dd
                if bp is None:
                    reason = "no_candidate_within_25px"
                elif not bp["r1_primary"]:
                    reason = "rule1_floor"
                elif bp["r2"] != "front":
                    reason = "rule2_" + bp["r2"]
                elif not bp.get("r2b_pass", True):
                    reason = "rule2b_" + bp.get("r2b_reason", "?")
                elif bp["fk_hit"]:
                    reason = "fk_mask"
                else:
                    reason = "other"
                ms = miss.setdefault(split, {}).setdefault(verdict, {})
                ms[reason] = ms.get(reason, 0) + 1
    out = {}
    for split, ss in sorted(strata.items()):
        out[split] = {k: dict(n=v["n"], matched=v["matched"],
                              recall=round(v["matched"] / v["n"], 3) if v["n"] else None)
                      for k, v in sorted(ss.items())}
    return out, miss


def gt_hook(outdir, frames_rows, geom):
    """Returns (gt_metrics dict, labels_rows or None). Prefers authoritative v2 labels;
    labels_e1.jsonl is the DEPRECATED draft -> every metric marked DRAFT."""
    v2 = os.path.join(outdir, "labels_e1_v2.jsonl")
    draft = os.path.join(outdir, "labels_e1.jsonl")
    if os.path.exists(v2):
        path, status, banner = v2, "ok", None
    elif os.path.exists(draft):
        path, status = draft, "DRAFT"
        banner = ("DRAFT — deprecated label source (labels_e1.jsonl, 2057-row draft). "
                  "Authoritative v2 labels still in production; orchestrator will re-run.")
    else:
        return dict(status="pending",
                    note="no labels_e1_v2.jsonl or labels_e1.jsonl at run time; re-run after "
                         "the labelling agent publishes (Felix format, keyed (ep, frame_idx))."), None
    rows = [json.loads(ln) for ln in open(path) if ln.strip()]
    real = load_real_audit_module()
    # match_gt contract: robot-mode accepted list lives under 'accepted_false'; here it is
    # the PRIMARY gate survivors (and secondarily the secondary-variant survivors).
    for fr in frames_rows:
        fr["accepted_false"] = fr["survivors_primary"]
    try:
        metrics = real.match_gt(path, frames_rows)
    except Exception as e:
        return dict(status="error", error=f"{type(e).__name__}: {e}", labels_file=path), rows
    metrics["status"] = status
    metrics["labels_file"] = path
    if banner:
        metrics["draft_banner"] = banner
    # stratified recall (primary survivors); cross-check totals vs match_gt
    strat, miss = stratified_recall(rows, frames_rows, geom, "survivors_primary")
    metrics["stratified_recall_by_bin_position"] = strat
    metrics["miss_attribution_by_stratum"] = miss
    matched_sum = sum(v["matched"] for ss in strat.values() for v in ss.values())
    pc = metrics.get("per_colour_recall", {})
    pch = metrics.get("per_colour_recall_held_out", {})
    gt_matched = sum(v["matched"] for v in pc.values()) + sum(v["matched"] for v in pch.values())
    metrics["stratified_crosscheck"] = dict(stratified_matched=matched_sum, match_gt_matched=gt_matched,
                                            consistent=(matched_sum == gt_matched))
    # secondary variant comparison (same contract)
    for fr in frames_rows:
        fr["accepted_false"] = fr["survivors_secondary"]
        fr.pop("_used", None)
    try:
        m2 = real.match_gt(path, frames_rows)
        metrics["secondary_variant"] = dict(
            per_colour_recall=m2.get("per_colour_recall"),
            per_colour_recall_held_out=m2.get("per_colour_recall_held_out"),
            colour_accuracy=m2.get("colour_accuracy"),
            false_picks_on_bin_regions=m2.get("false_picks_on_bin_regions"),
            unmatched_accepted_total=m2.get("unmatched_accepted_total"))
    except Exception as e:
        metrics["secondary_variant"] = dict(error=f"{type(e).__name__}: {e}")
    for fr in frames_rows:
        fr.pop("_used", None)
        fr.pop("accepted_false", None)
    return metrics, rows


# --------------------------------------------------------------------------
# aggregation helpers
# --------------------------------------------------------------------------

def split_rows(rows):
    return ([r for r in rows if not r["held_out"]], [r for r in rows if r["held_out"]])


def per_frame_stats(rows, key):
    counts = [len(r[key]) for r in rows]
    by_colour = {}
    for c in ("red", "yellow", "blue"):
        cc = [sum(1 for d in r[key] if d["color"] == c) for r in rows]
        by_colour[c] = dict(total=sum(cc), mean=round(float(np.mean(cc)), 3), max=int(max(cc)))
    return dict(mean=round(float(np.mean(counts)), 3), median=float(np.median(counts)),
                min=int(min(counts)), max=int(max(counts)),
                hist={str(k): int(v) for k, v in zip(*np.unique(counts, return_counts=True))},
                by_colour=by_colour)


def funnel_stats(rows):
    f = dict(pool_total=0, orig_accepted=0, orig_rejected=0,
             rule1_primary_rejects=0, rule1_secondary_rejects=0,
             rule2_rejects=dict(in_bin=0, near_edge=0, no_plane_hit=0),
             rule2b_rejects=dict(x_low=0, x_high=0, y_out=0, behind_bins=0, no_plane_hit=0),
             rule2b_rejects_by_colour=dict(red=0, yellow=0, blue=0),
             fk_drops=0, survivors_primary=0, survivors_secondary=0,
             rule2_rejects_by_colour=dict(red=0, yellow=0, blue=0),
             fk_drops_by_colour=dict(red=0, yellow=0, blue=0))
    for r in rows:
        for t in r["candidates"]:
            f["pool_total"] += 1
            f["orig_accepted" if t["orig"] == "accepted" else "orig_rejected"] += 1
            if not t["r1_primary"]:
                f["rule1_primary_rejects"] += 1
            if not t["r1_secondary"]:
                f["rule1_secondary_rejects"] += 1
            # sequential funnel attribution: rule1 -> rule2 -> rule2b -> fk (primary)
            if t["r1_primary"]:
                if t["r2"] != "front":
                    f["rule2_rejects"][t["r2"]] += 1
                    if t["color"] in f["rule2_rejects_by_colour"]:
                        f["rule2_rejects_by_colour"][t["color"]] += 1
                elif not t.get("r2b_pass", True):
                    reason = t.get("r2b_reason", "no_plane_hit")
                    f["rule2b_rejects"][reason] = f["rule2b_rejects"].get(reason, 0) + 1
                    if t["color"] in f["rule2b_rejects_by_colour"]:
                        f["rule2b_rejects_by_colour"][t["color"]] += 1
                elif t["fk_hit"]:
                    f["fk_drops"] += 1
                    if t["color"] in f["fk_drops_by_colour"]:
                        f["fk_drops_by_colour"][t["color"]] += 1
            f["survivors_primary"] += int(t["survives_primary"])
            f["survivors_secondary"] += int(t["survives_secondary"])
    return f


def bin_zone_fp(rows, key):
    a = dict(red=0, yellow=0, blue=0)
    b = dict(red=0, yellow=0, blue=0)
    frames_a, frames_b = set(), set()
    # where are the def-(a) survivors relative to the bin row? (mat y; footprint y in [0.26, 0.47])
    pos = dict(behind_bin_row=0, in_front_of_row=0, beside_bins=0, no_xy=0)
    for r in rows:
        for d in r[key]:
            if d["uv"][1] < BIN_ZONE_V:
                a[d["color"]] += 1
                frames_a.add(r["frame"])
                xy = d.get("xy_gate")
                if xy is None:
                    pos["no_xy"] += 1
                elif xy[1] > 0.47:
                    pos["behind_bin_row"] += 1
                elif xy[1] < 0.26:
                    pos["in_front_of_row"] += 1
                else:
                    pos["beside_bins"] += 1
            if d["r2"] in ("in_bin", "near_edge"):
                b[d["color"]] += 1
                frames_b.add(r["frame"])
    return dict(def_a_v_lt_0p38=dict(by_colour=a, total=sum(a.values()), n_frames=len(frames_a),
                                    frames=sorted(frames_a), position_vs_bin_row=pos),
                def_b_inside_or_near=dict(by_colour=b, total=sum(b.values()), n_frames=len(frames_b),
                                          frames=sorted(frames_b)))


def shape_stats(rows, key):
    hist, unknown = {}, dict(red=0, yellow=0, blue=0)
    for r in rows:
        for d in r[key]:
            s = str(d["shape"])
            hist[s] = hist.get(s, 0) + 1
            if d["shape_unknown"]:
                unknown[d["color"]] += 1
    return dict(shape_hist=dict(sorted(hist.items())),
                rule4_triangle_to_shape_unknown=dict(by_colour=unknown, total=sum(unknown.values())))


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames", default=FRAMES)
    ap.add_argument("--root", default=ROOT)
    ap.add_argument("--outdir", default=OUTDIR)
    ap.add_argument("--no-fk", action="store_true", help="skip the FK mask (report 'without FK mask')")
    ap.add_argument("--counter-z", type=float, default=None,
                    help="override counter (mat) surface height in pelvis m for the ray-plane "
                         "(sensitivity variant; default = scene.yaml value, nominal outputs unchanged)")
    ap.add_argument("--suffix", default="",
                    help="output filename suffix so a variant writes SEPARATE files "
                         "(e.g. _realz -> colorgate_e1_realz.json / COLOR-GATE-REALZ.md); nominal untouched")
    args = ap.parse_args()

    t_start = time.time()
    os.makedirs(args.outdir, exist_ok=True)
    manifest = json.load(open(f"{args.root}/manifest.json"))
    try:
        info_fps = int(json.load(open(f"{args.root}/meta/info.json")).get("fps", FPS))
    except Exception:
        info_fps = FPS

    frame_index = {}
    fi_path = f"{args.outdir}/frame_index.json"
    if os.path.exists(fi_path):
        try:
            frame_index = json.load(open(fi_path))
        except Exception:
            frame_index = {}

    cfg = load_config(overlay="submission")
    p = Perception(cfg, Scene.load(), Calibration.load())
    geom = GateGeometry(p, counter_z=args.counter_z)
    print(f"[cg] overlay={cfg.get('overlay')} hsv_set={cfg['perception'].get('hsv_set')} "
          f"split_blobs={cfg['perception'].get('split_blobs')} gate_h={GATE_HEIGHT_M}m "
          f"counter_z={geom.counter_z}m "
          f"shipped_object_height={p.height}m floor={AREA_FLOOR_PX}px", flush=True)

    t_fk0 = time.time()
    fk = FKMask(p.cal, enabled=not args.no_fk)
    fk_init_s = time.time() - t_fk0
    print(f"[cg] FK mask: {fk.status} (init {fk_init_s:.1f}s, {len(fk.states)} joint states)", flush=True)

    frames_rows, cand_rows = [], []
    sanity = None
    n_missing = 0
    fk_status_counts = {}
    t_det0 = time.time()
    for m in manifest:
        ep, task, t0, t1 = int(m["episode"]), m["task"], float(m["t0"]), float(m["t1"])
        held_out = ep >= HELD_OUT_FROM
        for tag, frac in TAGS:
            name = f"e{ep:02d}{tag}"
            path = f"{args.frames}/{name}.png"
            if not os.path.exists(path):
                n_missing += 1
                continue
            ts = round(t0 + (t1 - t0) * frac, 3)
            if name in frame_index:
                frame_idx = int(frame_index[name])          # global concatenated index
                frame_idx_ep = frame_idx - int(fk.offsets.get(ep, 0)) if fk.enabled else None
            else:
                # held-out: manifest t0/t1 are GLOBAL video times, so the extractor formula
                # ceil((ts-0.02)*fps) already yields the global concatenated index
                frame_idx = int(math.ceil((ts - 0.02) * info_fps))
                frame_idx_ep = frame_idx - int(fk.offsets.get(ep, 0)) if fk.enabled else None
            rgb = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB)
            dets, rej = p.detect(rgb, None, debug=True, include_bins=True)

            if ep == 0 and tag == "a":
                sanity = dict(accepted=len(dets), rejected=len(rej),
                              expected_accepted=20, expected_rejected=14,
                              pass_counts=(len(dets) == 20 and len(rej) == 14))
                print(f"[cg] sanity e00a include_bins=True: {sanity}", flush=True)

            fk_mask, fk_status = fk.frame_mask(ep, frame_idx_ep if fk.enabled else 0)
            fk_status_counts[fk_status] = fk_status_counts.get(fk_status, 0) + 1

            pool = ([dict(color=d.color, uv=[round(float(d.uv[0]), 2), round(float(d.uv[1]), 2)],
                          n_px=int(d.n_px), shape=d.shape, orig="accepted",
                          top_shipped=[round(float(x), 4) for x in d.top]) for d in dets] +
                    [dict(color=d.color, uv=[round(float(d.uv[0]), 2), round(float(d.uv[1]), 2)],
                          n_px=int(d.n_px), shape=d.shape, orig=reason,
                          top_shipped=[round(float(x), 4) for x in d.top]) for d, reason in rej])
            candidates = []
            for cand in pool:
                tr = gate_candidate(cand, geom, fk_mask)
                candidates.append(dict(cand, **tr))
                cand_rows.append(dict(frame=name, ep=ep, tag=tag, held_out=held_out, **cand, **tr))
            surv_p = [survivor_row(c, c) for c in candidates if c["survives_primary"]]
            surv_s = [survivor_row(c, c) for c in candidates if c["survives_secondary"]]
            frames_rows.append(dict(
                frame=name, ep=ep, tag=tag, frame_idx=frame_idx, frame_idx_ep=frame_idx_ep,
                held_out=held_out, task=task, fk_status=fk_status,
                pool_n=len(pool), orig_accepted_n=len(dets), orig_rejected_n=len(rej),
                candidates=candidates,
                survivors_primary=surv_p, survivors_secondary=surv_s,
                degraded_primary=len(surv_p) > DEGRADED_THRESHOLD,
                degraded_secondary=len(surv_s) > DEGRADED_THRESHOLD,
            ))
            if len(frames_rows) % 24 == 0:
                print(f"[cg] {len(frames_rows)} frames, {time.time() - t_start:.0f}s", flush=True)
    detect_s = time.time() - t_det0
    print(f"[cg] detection+gate done: {len(frames_rows)} frames in {detect_s:.1f}s, missing={n_missing}",
          flush=True)

    # ---------------------------------------------------------------- stats
    tr, ho = split_rows(frames_rows)
    stats = {}
    for label, rows in (("all", frames_rows), ("train_ep0_39", tr), ("held_out_ep40_47", ho)):
        if not rows:
            continue
        deg_p = [r["frame"] for r in rows if r["degraded_primary"]]
        deg_s = [r["frame"] for r in rows if r["degraded_secondary"]]
        stats[label] = dict(
            n_frames=len(rows),
            funnel=funnel_stats(rows),
            accepted_per_frame_primary=per_frame_stats(rows, "survivors_primary"),
            accepted_per_frame_secondary=per_frame_stats(rows, "survivors_secondary"),
            degraded_primary=dict(n=len(deg_p), frames=deg_p,
                                  counts={r["frame"]: len(r["survivors_primary"]) for r in rows
                                          if r["degraded_primary"]}),
            degraded_secondary=dict(n=len(deg_s), frames=deg_s),
            bin_zone_fp_primary=bin_zone_fp(rows, "survivors_primary"),
            bin_zone_fp_secondary=bin_zone_fp(rows, "survivors_secondary"),
            bin_zone_pre_gate_accepted=None,   # filled below (definition a, pre-gate accepted pool)
            shape_primary=shape_stats(rows, "survivors_primary"),
            shape_secondary=shape_stats(rows, "survivors_secondary"),
        )
        # pre-gate accepted pool bin-zone (definition a) for baseline comparison
        a = dict(red=0, yellow=0, blue=0)
        nfr = 0
        for r in rows:
            hit = False
            for c in r["candidates"]:
                if c["orig"] == "accepted" and c["uv"][1] < BIN_ZONE_V and c["color"] in a:
                    a[c["color"]] += 1
                    hit = True
            nfr += int(hit)
        stats[label]["bin_zone_pre_gate_accepted"] = dict(
            def_a_v_lt_0p38=dict(by_colour=a, total=sum(a.values()), n_frames=nfr))

    # target check: <=6 accepted/frame and 0 bin-zone FP on TRAIN (primary variant)
    st = stats.get("train_ep0_39", {})
    target = dict(
        variant="primary (1000px floor)",
        train_max_accepted_per_frame=st.get("accepted_per_frame_primary", {}).get("max"),
        train_mean_accepted_per_frame=st.get("accepted_per_frame_primary", {}).get("mean"),
        pass_le6_per_frame=(st.get("accepted_per_frame_primary", {}).get("max", 99) <= DEGRADED_THRESHOLD),
        train_bin_zone_fp_def_a=st.get("bin_zone_fp_primary", {}).get("def_a_v_lt_0p38", {}).get("total"),
        train_bin_zone_fp_def_b=st.get("bin_zone_fp_primary", {}).get("def_b_inside_or_near", {}).get("total"),
        pass_zero_bin_zone_def_a=(st.get("bin_zone_fp_primary", {}).get("def_a_v_lt_0p38", {}).get("total", -1) == 0),
        pass_zero_bin_zone_def_b=(st.get("bin_zone_fp_primary", {}).get("def_b_inside_or_near", {}).get("total", -1) == 0),
        train_degraded_frames=st.get("degraded_primary", {}).get("n"),
    )

    # FK drop census (sequential funnel attribution already in funnel_stats; here raw hits)
    fk_census = dict(status=fk.status, method=("simplified link capsules (no STL meshes shipped) + "
                     "static base at waist=0; pinocchio FK from parquet observation.state "
                     "(arm 14 + hand 14, joint_spec order); centroid-in-mask drop") if fk.enabled else "skipped",
                     init_s=round(fk_init_s, 1), frame_status_counts=fk_status_counts,
                     raw_centroid_hits=dict(red=0, yellow=0, blue=0), raw_hits_total=0,
                     frames_with_hits=set())
    for r in frames_rows:
        for c in r["candidates"]:
            if c["fk_hit"]:
                fk_census["raw_centroid_hits"][c["color"]] += 1
                fk_census["raw_hits_total"] += 1
                fk_census["frames_with_hits"].add(r["frame"])
    fk_census["frames_with_hits"] = sorted(fk_census["frames_with_hits"])
    fk_census["n_frames_with_hits"] = len(fk_census["frames_with_hits"])

    # GT hook
    t_gt0 = time.time()
    gt_metrics, _ = gt_hook(args.outdir, frames_rows, geom)
    gt_s = time.time() - t_gt0

    # ---------------------------------------------------------------- write
    per_frame_out = []
    for r in frames_rows:
        per_frame_out.append(dict(
            frame=r["frame"], ep=r["ep"], tag=r["tag"], frame_idx=r["frame_idx"],
            held_out=r["held_out"], fk_status=r["fk_status"],
            pool_n=r["pool_n"], orig_accepted_n=r["orig_accepted_n"],
            n_survivors_primary=len(r["survivors_primary"]),
            n_survivors_secondary=len(r["survivors_secondary"]),
            degraded_primary=r["degraded_primary"], degraded_secondary=r["degraded_secondary"],
            survivors_primary=r["survivors_primary"],
            survivors_secondary_n_by_colour={c: sum(1 for d in r["survivors_secondary"] if d["color"] == c)
                                             for c in ("red", "yellow", "blue")},
            rule_rejects={**{k: sum(1 for c in r["candidates"] if c["drop_reason_primary"] == k)
                             for k in ("rule1", "rule2:in_bin", "rule2:near_edge", "rule2:no_plane_hit")},
                          **{f"rule2b:{reason}": sum(1 for c in r["candidates"]
                                                     if c["drop_reason_primary"] == f"rule2b:{reason}")
                             for reason in ("x_low", "x_high", "y_out", "behind_bins", "no_plane_hit")},
                          "rule2b_total": sum(1 for c in r["candidates"]
                                              if str(c["drop_reason_primary"]).startswith("rule2b")),
                          "fk_mask": sum(1 for c in r["candidates"] if c["drop_reason_primary"] == "fk_mask")},
        ))
    audit = dict(
        meta=dict(tool="tools/e1_colorgate_audit.py", date=time.strftime("%Y-%m-%d %H:%M:%S"),
                  commit=os.popen("git rev-parse --short HEAD 2>/dev/null").read().strip(),
                  overlay=cfg.get("overlay"), profile=cfg.get("profile"),
                  flags=dict(hsv_set=cfg["perception"].get("hsv_set"),
                             split_blobs=cfg["perception"].get("split_blobs"),
                             classify_shape=cfg["perception"].get("classify_shape")),
                  n_frames=len(frames_rows), n_missing=n_missing, fps=info_fps,
                  depth=None, include_bins=True,
                  gate_height_m=GATE_HEIGHT_M, counter_z_m=geom.counter_z,
                   counter_z_override=args.counter_z, output_suffix=args.suffix,
                   shipped_object_height_m=p.height,
                  area_floor_px=AREA_FLOOR_PX, area_coef=AREA_COEF, near_edge_m=NEAR_EDGE_M,
                  degraded_threshold=DEGRADED_THRESHOLD, bin_zone_v_px=BIN_ZONE_V,
                  rules_verbatim=RULES_VERBATIM,
                  funnel_order="rule1 -> rule2 -> rule2b (workspace) -> fk_mask; rule3/rule4 census on survivors",
                  runtime_s=dict(detect_gate=round(detect_s, 1), fk_init=round(fk_init_s, 1),
                                 gt=round(gt_s, 1), total=round(time.time() - t_start, 1))),
        sanity_e00a=sanity,
        target_check_train=target,
        stats=stats,
        fk_mask=fk_census,
        gt_metrics=gt_metrics,
        per_frame=per_frame_out,
    )
    suf = args.suffix
    json_name = f"colorgate_e1{suf}.json"
    jsonl_name = f"colorgate_candidates_e1{suf}.jsonl"
    md_name = "COLOR-GATE.md" if not suf else f"COLOR-GATE-{suf.strip('_').upper()}.md"
    audit["meta"]["output_json"] = json_name
    with open(f"{args.outdir}/{json_name}", "w") as fh:
        json.dump(audit, fh, indent=1)
    with open(f"{args.outdir}/{jsonl_name}", "w") as fh:
        for c in cand_rows:
            fh.write(json.dumps(c) + "\n")
    write_report(args.outdir, audit, md_name)
    print(f"[cg] wrote {json_name}, {jsonl_name}, {md_name} "
          f"(total {time.time() - t_start:.1f}s)", flush=True)
    return 0


# --------------------------------------------------------------------------
# report
# --------------------------------------------------------------------------

def write_report(outdir, a, md_name="COLOR-GATE.md"):
    L = []
    L.append("# E1 color.gate post-filter audit — Timi's proposed rules on real HF drop_2 frames\n")
    L.append(f"_Generated by `tools/e1_colorgate_audit.py` at {a['meta']['date']} "
             f"(commit {a['meta']['commit']}). MEASUREMENT ONLY: rules applied verbatim as post-filters, "
             "nothing tuned, no policy code or baseline audit touched._\n")
    L.append("## Setup\n")
    m = a["meta"]
    L.append(f"- Detector: submission overlay (`hsv_set={m['flags']['hsv_set']}`, "
             f"`split_blobs={m['flags']['split_blobs']}`, `classify_shape={m['flags']['classify_shape']}`), "
             "**depth=None** on every frame (worst case), `include_bins=True` audit view (blobs whole).")
    L.append(f"- Candidate pool per frame = accepted + ALL rejected detections "
             f"({a['sanity_e00a']['accepted']}/{a['sanity_e00a']['rejected']} on e00a, expected 20/14: "
             f"**{'PASS' if a['sanity_e00a']['pass_counts'] else 'FAIL'}**).")
    L.append(f"- Frames: {m['n_frames']} = 48 episodes x 4 tags (a/b/c/z = 20/50/80/97%), 640x480 cam_high. "
             "Episodes 40-47 **held out** (reported separately, never tuned on).")
    L.append(f"- Gate ray-plane height: **{m['gate_height_m']*100:.2f} cm** above counter "
             f"(counter_z={m.get('counter_z_m', 0.01):.3f} m pelvis"
             + (f", OVERRIDDEN from nominal 0.01 for this sensitivity variant" if m.get('counter_z_override') is not None else "")
             + f"). Note: the shipped detector's object_height is "
             f"**{m['shipped_object_height_m']*100:.1f} cm** (top z=0.055 m pelvis), not 5.5 cm; "
             "the gate re-computes ray-plane at 2.25 cm itself, independent of the shipped `top`.")
    L.append(f"- Funnel order: rule1 -> rule2 -> rule2b (workspace) -> FK mask; rule3 (>{m['degraded_threshold']} survivors) "
             "and rule4 (triangle->shape_unknown) are census/annotation on survivors.")
    L.append(f"- Runtime: detection+gate {m['runtime_s']['detect_gate']} s, FK init {m['runtime_s']['fk_init']} s, "
             f"GT {m['runtime_s']['gt']} s, total **{m['runtime_s']['total']} s**.\n")
    L.append("## Rules (verbatim, Timi messages f369bb04 + cb364bfb)\n")
    for k, v in m["rules_verbatim"].items():
        L.append(f"- **{k}**: {v}")
    L.append("")
    L.append("## Rejection funnel (sequential: rule1 -> rule2 -> rule2b -> FK)\n")
    L.append("| split | pool | rule1 floor rej (primary) | rule1 rej (secondary) | rule2 in_bin | rule2 near_edge | rule2 no_plane | rule2b rej | FK drops | survivors primary | survivors secondary |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for k in ("all", "train_ep0_39", "held_out_ep40_47"):
        if k not in a["stats"]:
            continue
        f = a["stats"][k]["funnel"]
        r2 = f["rule2_rejects"]
        r2b_total = sum(f.get("rule2b_rejects", {}).values())
        L.append(f"| {k} | {f['pool_total']} | {f['rule1_primary_rejects']} | {f['rule1_secondary_rejects']} | "
                 f"{r2['in_bin']} | {r2['near_edge']} | {r2['no_plane_hit']} | {r2b_total} | {f['fk_drops']} | "
                 f"{f['survivors_primary']} | {f['survivors_secondary']} |")
    L.append("")
    L.append("Rule-2b (workspace) rejects by reason: " + "; ".join(
        f"{k}: {a['stats'][k]['funnel'].get('rule2b_rejects', {})}" for k in ("train_ep0_39", "held_out_ep40_47")
        if k in a["stats"]))
    L.append("")
    L.append("Rule-2 rejects by colour (primary funnel): " + "; ".join(
        f"{k}: {a['stats'][k]['funnel']['rule2_rejects_by_colour']}" for k in ("train_ep0_39", "held_out_ep40_47")
        if k in a["stats"]))
    L.append("")
    # F4: secondary-vs-primary note computed from the CURRENT aggregates (no hardcoded counts).
    if "train_ep0_39" in a["stats"]:
        ft = a["stats"]["train_ep0_39"]["funnel"]
        sp, ss = ft["survivors_primary"], ft["survivors_secondary"]
        zap = a["stats"]["train_ep0_39"]["bin_zone_fp_primary"]["def_a_v_lt_0p38"]["total"]
        zas = a["stats"]["train_ep0_39"]["bin_zone_fp_secondary"]["def_a_v_lt_0p38"]["total"]
        rel = "MORE" if ss > sp else ("FEWER" if ss < sp else "the SAME")
        cmp = "looser" if ss > sp else ("tighter" if ss < sp else "identical")
        note = (f"Secondary rule-1 variant note (computed from this run's train funnel): the distance-scaled "
                f"0.4x-expected floor yields {rel} survivors than the 1000px primary floor "
                f"({ss} vs {sp}), i.e. it is {cmp} overall at this plane; bin-zone FP def(a) "
                f"{zas} (secondary) vs {zap} (primary).")
        gt = a.get("gt_metrics", {})
        sv = gt.get("secondary_variant") or {}
        if sv.get("per_colour_recall") and gt.get("per_colour_recall"):
            note += (f" GT per-colour recall primary={json.dumps(gt['per_colour_recall'])} vs "
                     f"secondary={json.dumps(sv['per_colour_recall'])}.")
        elif gt.get("status") == "pending":
            note += " GT recall comparison unavailable (labels pending)."
        L.append(note)
    L.append("")
    L.append("## Accepted per frame (survivors)\n")
    L.append("| split | variant | mean | median | max | red mean/max | yellow mean/max | blue mean/max |")
    L.append("|---|---|---|---|---|---|---|---|")
    for k in ("train_ep0_39", "held_out_ep40_47", "all"):
        if k not in a["stats"]:
            continue
        for var, key in (("primary", "accepted_per_frame_primary"), ("secondary", "accepted_per_frame_secondary")):
            s = a["stats"][k][key]
            bc = s["by_colour"]
            L.append(f"| {k} | {var} | {s['mean']} | {s['median']} | {s['max']} | "
                     f"{bc['red']['mean']}/{bc['red']['max']} | {bc['yellow']['mean']}/{bc['yellow']['max']} | "
                     f"{bc['blue']['mean']}/{bc['blue']['max']} |")
    L.append("")
    for k in ("train_ep0_39", "held_out_ep40_47"):
        if k in a["stats"]:
            L.append(f"- {k} primary survivor-count histogram: {a['stats'][k]['accepted_per_frame_primary']['hist']}")
    L.append("")
    L.append("## Rule 3 — DEGRADED census (>6 survivors/frame)\n")
    for k in ("train_ep0_39", "held_out_ep40_47", "all"):
        if k not in a["stats"]:
            continue
        d = a["stats"][k]["degraded_primary"]
        ds = a["stats"][k]["degraded_secondary"]
        counts = d['counts'] if (d['n'] and k != "all") else ""
        L.append(f"- {k}: primary **{d['n']}** frames {counts}; secondary {ds['n']} frames")
    L.append("")
    L.append("## Rule 4 — shape annotation (survivors, depth=None)\n")
    for k in ("train_ep0_39", "held_out_ep40_47"):
        if k not in a["stats"]:
            continue
        s = a["stats"][k]["shape_primary"]
        L.append(f"- {k}: shape histogram {s['shape_hist']}; triangle->shape_unknown "
                 f"{s['rule4_triangle_to_shape_unknown']['total']} {s['rule4_triangle_to_shape_unknown']['by_colour']}")
    L.append("")
    L.append("## Bin-zone FP on survivors (two definitions)\n")
    L.append(f"(a) centroid v < {m['bin_zone_v_px']:.1f} px; (b) ray-plane xy at 2.25cm inside/near "
             f"(<= {m['near_edge_m']*100:.0f} cm) a bin footprint — (b) must be ~0 by construction (rule 2).\n")
    L.append("| split | variant | def(a) red/yellow/blue | def(a) total (frames) | def(b) total | pre-gate accepted def(a) |")
    L.append("|---|---|---|---|---|---|")
    for k in ("train_ep0_39", "held_out_ep40_47", "all"):
        if k not in a["stats"]:
            continue
        for var, key in (("primary", "bin_zone_fp_primary"), ("secondary", "bin_zone_fp_secondary")):
            z = a["stats"][k][key]
            za, zb = z["def_a_v_lt_0p38"], z["def_b_inside_or_near"]
            ca = za["by_colour"]
            pre = a["stats"][k]["bin_zone_pre_gate_accepted"]["def_a_v_lt_0p38"]["total"]
            L.append(f"| {k} | {var} | {ca['red']}/{ca['yellow']}/{ca['blue']} | "
                     f"{za['total']} ({za['n_frames']}) | {zb['total']} | {pre} |")
    L.append("")
    tc = a["target_check_train"]
    L.append("### Target check (TRAIN, primary variant)\n")
    L.append(f"- <=6 accepted/frame: max = **{tc['train_max_accepted_per_frame']}** "
             f"(mean {tc['train_mean_accepted_per_frame']}) -> **{'PASS' if tc['pass_le6_per_frame'] else 'FAIL'}**")
    L.append(f"- 0 bin-zone FP def(a): **{tc['train_bin_zone_fp_def_a']}** -> "
             f"**{'PASS' if tc['pass_zero_bin_zone_def_a'] else 'FAIL'}**")
    L.append(f"- 0 bin-zone FP def(b): **{tc['train_bin_zone_fp_def_b']}** -> "
             f"**{'PASS' if tc['pass_zero_bin_zone_def_b'] else 'FAIL'}** (0 expected by construction)")
    L.append(f"- DEGRADED frames on train: {tc['train_degraded_frames']}\n")
    if not tc["pass_zero_bin_zone_def_a"]:
        za = a["stats"]["train_ep0_39"]["bin_zone_fp_primary"]["def_a_v_lt_0p38"]
        L.append(f"def(a) survivors appear in {za['n_frames']}/160 train frames "
                 f"(e.g. {', '.join(za['frames'][:6])}...; full list in {m.get('output_json', 'colorgate_e1.json')}).")
        L.append(f"**Position of the def(a) survivors vs the bin row (mat y; footprint y 0.26-0.47): "
                 f"{za['position_vs_bin_row']}** — the 3D-only rule keeps everything outside the "
                 "footprint+2cm band, including bin walls/rim whose ray-plane xy lands BEHIND the bin "
                 "row; 'in FRONT -> KEEP' has no behind-the-bin counterpart. Measurement only: no rule "
                 "change is proposed here.\n")
    L.append("## FK mask\n")
    fk = a["fk_mask"]
    L.append(f"- status: **{fk['status']}**; {fk['method']}")
    L.append(f"- frame status counts: {fk['frame_status_counts']}; init {fk['init_s']} s")
    L.append(f"- raw centroid-in-mask hits (any pool candidate): {fk['raw_hits_total']} "
             f"{fk['raw_centroid_hits']} over {fk['n_frames_with_hits']} frames")
    for k in ("train_ep0_39", "held_out_ep40_47"):
        if k in a["stats"]:
            f = a["stats"][k]["funnel"]
            L.append(f"- {k}: FK drops in the sequential funnel (after rule1+rule2+rule2b): {f['fk_drops']} "
                     f"{f['fk_drops_by_colour']}")
    L.append("- Validation: known arm/base FPs fall in the mask (e33b 14914 px arm, e35c 5336 px arm band, "
             "e00a 3795 px metal wrist); mat objects (yellow cylinder, blue cube) are not covered. "
             "Capsule radii are engineering approximations (arm 3.2-5.5 cm, fingers 1.2-1.4 cm); no STL "
             "meshes are shipped in the worktree, so this is the simplified-cylinder variant Timi allowed.")
    L.append("- Caveat: legs/waist held at the stiff pose (0); a real waist deviation would shift the mask.\n")
    L.append("## GT metrics\n")
    gt = a["gt_metrics"]
    if gt.get("status") == "pending":
        L.append(f"**PENDING** — {gt.get('note')}\n")
    elif gt.get("status") in ("ok", "DRAFT"):
        if gt.get("draft_banner"):
            L.append(f"> **{gt['draft_banner']}**\n")
        L.append(f"- labels: {gt['labels_file']} ({gt['n_label_rows']} rows, status={gt['status']}), "
                 f"{gt['n_frames_matched']} frames matched, radius {gt['match_radius_px']} px, "
                 "matched against PRIMARY gate survivors (keyed (ep, frame_idx) via frame_index.json).")
        L.append(f"- per-class recall (train): {json.dumps(gt.get('per_class_recall', {}))}")
        L.append(f"- per-class recall (held-out): {json.dumps(gt.get('per_class_recall_held_out', {}))}")
        L.append(f"- per-colour recall (train): {json.dumps(gt.get('per_colour_recall', {}))}")
        L.append(f"- per-colour recall (held-out): {json.dumps(gt.get('per_colour_recall_held_out', {}))}")
        if gt.get("colour_accuracy"):
            ca = gt["colour_accuracy"]
            L.append(f"- colour accuracy (orange->red, purple->blue, yellow->yellow): {ca['acc']} "
                     f"({ca['hits']}/{ca['matched']})")
        L.append(f"- tape/rim false picks (survivor centroid in GT bin_tape/bin_wall bbox): "
                 f"{gt['false_picks_on_bin_regions']} in {gt.get('n_frames_with_bin_false_picks', 0)} frames")
        L.append(f"- unmatched survivors (FP candidates): {gt['unmatched_accepted_total']} "
                 f"(on bin regions {gt['false_picks_on_bin_regions']}, other "
                 f"{gt.get('other_unmatched_accepted', '?')})")
        strat = gt.get("stratified_recall_by_bin_position", {})
        L.append("- **stratified recall by GT ray-plane position at 2.25cm** "
                 "(front = outside footprint >2cm; near_edge <=2cm; inside):")
        for split, ss in strat.items():
            L.append(f"  - {split}: {json.dumps(ss)}")
        cc = gt.get("stratified_crosscheck", {})
        L.append(f"  - cross-check vs match_gt matched totals: {cc}")
        miss = gt.get("miss_attribution_by_stratum", {})
        if miss:
            L.append("- miss attribution (unmatched GT objects -> nearest pool candidate <=25px, "
                     "which rule dropped it):")
            for split, ss in miss.items():
                for stratum, rr in ss.items():
                    L.append(f"  - {split}/{stratum}: {json.dumps(rr)}")
        if gt.get("secondary_variant"):
            sv = gt["secondary_variant"]
            L.append(f"- secondary variant (distance-scaled floor): per-colour recall "
                     f"{json.dumps(sv.get('per_colour_recall', {}))}, colour accuracy "
                     f"{json.dumps(sv.get('colour_accuracy'))}, tape FP {sv.get('false_picks_on_bin_regions')}")
        L.append("")
    else:
        L.append(f"status={gt.get('status')}: {gt.get('error') or gt.get('note')}\n")
    L.append("## Caveats\n")
    L.append("- **No depth anywhere**: every gate position is the centroid ray intersected with the plane at "
             "counter_z + 2.25 cm under the FIXED calibration pose. Blobs whose visible centroid is not at "
             "2.25 cm (objects in the gripper, tall stacks, bin walls) are mis-positioned by that assumption; "
             "the near-edge rule rejects without depth exactly because of this.")
    L.append("- **Worst-case area floor**: 1000 px is the no-depth floor; the secondary variant uses the "
             "ray-plane Z at 2.25 cm, which is itself an assumption. Expected area uses fx=604.827 and a "
             "45 mm object, coefficient 0.4, verbatim.")
    L.append("- **Audit-view pool**: include_bins=True keeps bin pixels, so in_bin:* rejects are whole blobs "
             "and can be re-judged by the gate; the robot path (include_bins=False) cuts hulls pre-grouping "
             "and can split blobs differently. Gate survivors here are not identical to what the robot "
             "pipeline would produce with the same gate.")
    L.append("- **Orange bins match robust red**: bin tape/wall pixels outside the 3D footprint test (e.g. "
             "tape on the mat >2 cm in front of a bin) can survive rule 2; definition-(a) bin-zone FP "
             "counts them. The shipped 2D hull cut is NOT part of this gate (rule 2 is 3D-only by design).")
    L.append("- **FK mask is approximate**: capsule silhouettes, not collision meshes; centroid-in-mask test "
             "only (a blob mostly on the arm with its centroid off the mask survives). Joint state is the "
             "RECORDED teleop state at the frame index; camera extrinsics assume waist=0.")
    L.append("- **Rule 4** only annotates: 'triangle' without depth becomes shape_unknown for downstream "
             "shape-keyed instructions; counts above, no detection is dropped for shape.")
    L.append("- **Recall interpretation**: the label source annotates ALL objects, including ones already "
             "sorted into bins, parked near a bin edge, or held in the gripper. The gate deliberately "
             "rejects inside/near-bin positions (rule 2) and centroids on the robot (FK mask), so total "
             "recall is bounded well below 1 by design; the front-of-bins stratum isolates the mat case. "
             "Objects in the gripper are elevated, so their 2.25cm ray-plane position is an assumption.")
    L.append("- Episodes 40-47 are held out: reported, never used for any decision.")
    if gt.get("draft_banner"):
        L.append("- **GT numbers are DRAFT** (deprecated 2057-row label source); authoritative v2 labels "
                 "pending, orchestrator re-runs.")
    L.append("")
    with open(f"{outdir}/{md_name}", "w") as fh:
        fh.write("\n".join(L) + "\n")


if __name__ == "__main__":
    sys.exit(main())
