"""Load config.yaml, build the 14-vectors of the Dex3 postures (left/right mirrored) and check them against
joint_spec.yaml so a bad posture fails at load time, not on the robot."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import yaml

from aaltonomous.interface.safety import JointSpec

HERE = Path(__file__).resolve().parent
DEFAULT_CONFIG = HERE / "config.yaml"
# Flag overlay baked into the submission image (submission.Dockerfile copies overlays/submission.yaml here). The file
# is absent in the repo (gitignored), so tests / run.py / sim_eval.py run the stock config unless asked otherwise.
BAKED_OVERLAY = HERE / "overlay.yaml"
OVERLAY_DIR = HERE / "overlays"
POSTURES = ("open", "preshape", "closed")

STOCK = "<stock>"   # FLAGS off-value for a tuning param: OFF = the stock config value (base + profile, no overlay)
# Limits an overlay may never exceed (hard rule: nothing may move faster than the stock sim profile).
LIMITS = {"motion.carry_scale": (0.0, 1.0)}
CHOICES = {"grasp.drop_slot_overflow": ("counter", "stack"), "grasp.close_posture": ("closed", "closed_tight"),
           "policy.bin_lookup": ("colour", "resolve_bin"), "perception.hsv_set": ("stock", "robust")}
VECTORS = {"grasp.point_in_palm": 3, "hand.closed_tight": 7, "perception.y_affine": 2}

# Every opt-in feature of the integration branch: dotted config key -> (owner, value that means OFF, what it does).
# All defaults in config.yaml are the OFF value, so with no overlay the policy is the stock classical baseline.
FLAGS = {
    "instruction.resolve": ("Felix #1", False, "instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)"),
    "policy.insert_guard": ("insert", False, "rung 3 (insert_shapes task id or the hole/lid sentence): pick nothing (0 pts, never -1 on the lid)"),
    "policy.bin_lookup": ("round 8", "colour", "colour (stock) | resolve_bin: (colour,shape)->colour->shape->default"),
    "perception.y_affine": ("Timi 23 §13", None, "y -= a*y + b (perceived - true y model, pelvis frame)"),
    "perception.self_calibration.enabled": ("Timi calib", False,
                                            "per-episode head-camera xy offset from the bin walls (depth)"),
    "perception.classify_shape": ("Oliver #5/#7", False, "Detection.shape from the top face (shape_id)"),
    "perception.triangle_pose": ("Oliver #5", False, "triangles: apex yaw + robust xy centre"),
    "perception.near_bin_guard": ("Oliver #7", False, "classify_shape abstains next to a bin (cut top face)"),
    "perception.no_blob_shape": ("Oliver #5/#7", False, "no metric top face -> shape None, never the 2D blob ratio"),
    "perception.shape_diag": ("Oliver #5/#7", False, "passive [shape_diag] log line per detection (no action change)"),
    "perception.hsv_set": ("real colour", "stock", "stock | robust: one HSV set for sim + the real head cam (hsv_robust)"),
    "perception.split_blobs": ("real colour", False, "split touching same-colour objects at distance-transform peaks"),
    "grasp.yaw_align": ("Oliver #5", False, "triangle grasp: finger direction from apex yaw (needs shape+pose)"),
    "grasp.yaw_phi_deg": ("Oliver #5", STOCK, "yaw_align: finger heading = apex + phi (stock 105)"),
    "grasp.yaw_tri_shift": ("Oliver #5", STOCK, "yaw_align: grasp point to the pinch midpoint (stock true)"),
    "policy.time_triage": ("Oliver #3", False, "skip plans that do not fit the time left, try the next object"),
    "policy.max_total_attempts": ("Oliver #3", None, "global per-episode cap on started picks"),
    "policy.max_attempts": ("stock param", 2, "tries per object before it is skipped (stock 2)"),
    "policy.retry_pass": ("Oliver #3", False, "once per episode: max_attempts-skipped objects get retry_extra_attempts"),
    "policy.time_margin_s": ("Oliver #3", 0.0, "plans must end this long before the time limit"),
    "policy.episode_summary_log": ("Oliver #3", False, "one log line per episode end"),
    "policy.verify_placement": ("Timi #6", False, "verify each release in the bins, correct wrong-bin, drop slots"),
    "policy.grasp_diag_log": ("Felix grasp", False, "passive [grasp_diag] line per pick: stall vs closed, held"),
    "policy.verify_yellow_unknown": ("Felix #6 rev", False, "#6: yellow in-bin detection = unknown, never mismatch"),
    "policy.max_corrections_per_episode": ("Felix #6 rev", None, "#6: cap corrections; counter picks go first"),
    "grasp.drop_slots": ("Timi 23", False, "standalone drop slots: next object in a bin -> next slot"),
    "grasp.drop_slot_overflow": ("Felix #6 rev", "counter", "slots used up: counter (leave it) | stack (slot 0)"),
    "motion.carry_scale": ("Felix grasp", 1.0, "<= 1: slower lift + carry only"),
    "motion.close_hold_s": ("Felix grasp", STOCK, "hold after closing the hand (sim profile 0.25 s)"),
    "motion.lift_on_stall.enabled": ("round 9", False, "lift as soon as the fingers stall (>= close_hold_s, <= timeout)"),
    "motion.lift_on_stall.timeout_s": ("round 9", STOCK, "lift_on_stall: longest close hold (stock 1.2 s)"),
    "grasp.point_in_palm": ("pinch fix", STOCK, "object centre in the LEFT palm frame (right = [1,-1,1] mirror)"),
    "grasp.close_posture_by_shape": ("hybrid", False, "per pick: demo posture only for a confident metric triangle, else the configured one"),
    "grasp.posture_by_shape_min_conf": ("hybrid", 0.5, "close_posture_by_shape: min shape confidence (0..1) for the demo posture"),
    "grasp.posture_apex_gate": ("apex gate", False, "close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)"),
    "grasp.generic_fallback": ("generic grasp", False, "unknown/low-confidence shape: geometry-only top-down grasp, colour-only bin resolution"),
    "grasp.close_posture": ("pinch fix", "closed", "grasp posture: closed | closed_tight"),
    "hand.closed_tight": ("pinch fix", STOCK, "the closed_tight posture (LEFT DDS order, right mirrored)"),
}


def mirror_left_to_right(left7):
    """Left DDS order (thumb0,1,2, middle0,1, index0,1) -> right DDS order (thumb0,1,2, index0,1, middle0,1),
    signs flipped (joint_spec.yaml: the hands are mirrored, the same motion has opposite signs)."""
    l = np.asarray(left7, float)
    return -np.array([l[0], l[1], l[2], l[5], l[6], l[3], l[4]])


def _set(cfg, k, v):  # "motion.arm_max_vel": 0.3
    d = cfg
    keys = k.split(".")
    for kk in keys[:-1]:
        d = d[kk]
    d[keys[-1]] = v


def _get(cfg, k):
    d = cfg
    for kk in k.split("."):
        if not isinstance(d, dict) or kk not in d:
            raise KeyError(k)
        d = d[kk]
    return d


def resolve_overlay(overlay=None):
    """Overlay file to apply: argument > $AALTO_OVERLAY > the baked-in overlay.yaml (if present) > none.
    A bare name (e.g. "submission") means overlays/<name>.yaml; "none" / "" = no overlay."""
    if overlay is None:
        overlay = os.environ.get("AALTO_OVERLAY")
    if overlay is None:
        return BAKED_OVERLAY if BAKED_OVERLAY.exists() else None
    if overlay in ("", "none"):
        return None
    p = Path(overlay)
    if not p.suffix and not p.exists():
        p = OVERLAY_DIR / f"{overlay}.yaml"
    return p


def load_overlay(path):
    """{name, flags} of an overlay file. flags = dotted keys (like profiles); only FLAGS keys may be set, so an overlay
    cannot silently retune anything else and a typo fails at load time (= at docker build), not in the finals."""
    data = yaml.safe_load(Path(path).read_text()) or {}
    if data.get("placeholder"):
        raise ValueError(f"overlay {path} is a placeholder (values TBD): fill it in and drop `placeholder: true`")
    flags = data.get("flags") or {}
    bad = sorted(set(flags) - set(FLAGS))
    if bad:
        raise ValueError(f"overlay {path}: unknown flag(s) {bad} (known: {sorted(FLAGS)})")
    return {"name": str(data.get("name") or Path(path).stem), "flags": dict(flags)}


def load_config(path=None, overrides=None, profile=None, overlay=None):
    """config.yaml + the selected profile's overrides (profiles.<name>, dotted keys) + the flag overlay + `overrides`.
    Profile: argument > $AALTO_PROFILE > the file's `profile:` key; "none" / "" = base values only.
    Overlay: see resolve_overlay (the submission image bakes overlays/submission.yaml in as overlay.yaml)."""
    cfg = yaml.safe_load(Path(path or DEFAULT_CONFIG).read_text())
    if profile is None:
        profile = os.environ.get("AALTO_PROFILE", cfg.get("profile"))
    profiles = cfg.pop("profiles", None) or {}
    if profile and profile != "none":
        if profile not in profiles:
            raise ValueError(f"unknown config profile {profile!r} (have {sorted(profiles)})")
        for k, v in profiles[profile].items():
            _set(cfg, k, v)
    cfg["profile"] = profile or "none"
    ov_path = resolve_overlay(overlay)
    if ov_path is not None:
        ov = load_overlay(ov_path)
        for k, v in ov["flags"].items():
            _get(cfg, k)  # must exist in config.yaml
            _set(cfg, k, v)
        cfg["overlay"] = ov["name"]
    for k, v in (overrides or {}).items():
        _set(cfg, k, v)
    check_flag_values(cfg)
    return cfg


def check_flag_values(cfg):
    """Hard limits on flag values (fail at load = at docker build): carry_scale in (0, 1], known overflow modes."""
    for k, (lo, hi) in LIMITS.items():
        v = float(_get(cfg, k))
        if not lo < v <= hi:
            raise ValueError(f"{k} = {v} outside ({lo}, {hi}]: never faster than the stock profile")
    for k, ok in CHOICES.items():
        if _get(cfg, k) not in ok:
            raise ValueError(f"{k} = {_get(cfg, k)!r}, want one of {ok}")
    for k, n in VECTORS.items():
        v = _get(cfg, k)
        if k == "grasp.point_in_palm" and v == "auto":
            continue
        if k == "perception.y_affine" and v is None:
            continue
        if not (isinstance(v, (list, tuple)) and len(v) == n and all(isinstance(x, (int, float)) for x in v)):
            raise ValueError(f"{k} = {v!r}: want {n} numbers")
    g = cfg.get("grasp") or {}
    for key in ("generic_shape_min_conf", "generic_min_confidence"):
        if not 0.0 <= float(g.get(key, 0.5)) <= 1.0:
            raise ValueError(f"grasp.{key} must be in [0, 1]")
    narrow = float(g.get("generic_narrow_width", 0.035))
    widest = float(g.get("generic_max_width", 0.060))
    if not 0.005 <= narrow <= widest <= 0.060:
        raise ValueError("generic grasp widths must satisfy 0.005 <= narrow <= max <= 0.060 m")
    if not 0.0 <= float(g.get("generic_yaw_tol_deg", 15.0)) <= 90.0:
        raise ValueError("grasp.generic_yaw_tol_deg must be in [0, 90]")
    if not 0.0 <= float(g.get("generic_yaw_max_dev_deg", 45.0)) <= 90.0:
        raise ValueError("grasp.generic_yaw_max_dev_deg must be in [0, 90]")
    if not float(g.get("generic_yaw_hint_weight", 3.0)) > 0.0:
        raise ValueError("grasp.generic_yaw_hint_weight must be positive")
    # Landmine guard (Oliver a44dd79): mirror_left_to_right flips thumb0's sign, the exact mirror keeps it (right
    # fingertips ~99 mm off for the demo preshape). Until the mirror is fixed, active postures must keep thumb0 = 0.
    h = cfg.get("hand") or {}
    right = h.get("right") or {}
    for name in ("preshape", "closed", "closed_tight"):
        if name in h and name not in right and abs(float(h[name][0])) > 1e-12:
            raise ValueError(f"hand.{name} thumb0={h[name][0]}: thumb0≠0 requires fixing mirror_left_to_right "
                             f"(keep thumb0 sign)")


def effective_flags(cfg):
    """[(key, value, on, owner, what)] for every FLAGS entry of a loaded config."""
    out, stock = [], None
    for k, (owner, off, what) in FLAGS.items():
        v = _get(cfg, k)
        if off == STOCK:
            if stock is None:
                stock = load_config(profile=cfg.get("profile"), overlay="none")
            off = _get(stock, k)
        out.append((k, v, v != off, owner, what))
    return out


def hand_postures(cfg, spec: JointSpec | None = None):
    """{name: (14,) hand vector} for open / preshape / closed (+ extra postures), validated against the URDF limits."""
    spec = spec or JointSpec()
    h = cfg["hand"]
    missing = [n for n in POSTURES if n not in h]
    if missing:
        raise ValueError(f"hand postures missing: {missing}")
    right = h.get("right") or {}
    out = {}
    for name in [k for k in h if k != "right"]:  # open, preshape, closed (required) + any extra (e.g. release)
        left = np.asarray(h[name], float)
        r = np.asarray(right[name], float) if name in right else mirror_left_to_right(left)
        v = np.r_[left, r]
        bad = [spec.hand_names[i] for i in np.flatnonzero((v < spec.hand_lower - 1e-9) | (v > spec.hand_upper + 1e-9))]
        if bad:
            raise ValueError(f"hand posture {name!r} outside joint_spec limits: {bad}")
        out[name] = v
    return out


def home_arm(cfg, spec: JointSpec | None = None):
    spec = spec or JointSpec()
    q = np.asarray(cfg["home_arm_q"], float)
    if q.shape != (14,):
        raise ValueError("home_arm_q must have 14 values")
    m = float(cfg["ik"]["limit_margin"])
    bad = [spec.arm_names[i] for i in np.flatnonzero((q < spec.arm_lower + m) | (q > spec.arm_upper - m))]
    if bad:
        raise ValueError(f"home_arm_q too close to the joint limits: {bad}")
    return q

