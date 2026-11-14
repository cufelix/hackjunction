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
           "grasp.held_detector": ("stall", "finger_sum"),
           "policy.bin_lookup": ("colour", "resolve_bin"), "perception.hsv_set": ("stock", "robust", "robust_v2"),
           "perception.principal_point": ("file", "center"),
           "perception.online_pitch_cal": (False, True, "table"), "brain.backend": ("stub", "llamacpp")}
VECTORS = {"grasp.point_in_palm": 3, "hand.closed_tight": 7, "perception.y_affine": 2}

# Every opt-in feature of the integration branch: dotted config key -> (owner, value that means OFF, what it does).
# All defaults in config.yaml are the OFF value, so with no overlay the policy is the stock classical baseline.
FLAGS = {
    "brain.backend": ("brain", "stub", "stub (= classical choice) | llamacpp (bounded async HTTP + GBNF)"),
    "brain.url": ("brain", "http://127.0.0.1:8080/completion", "llama.cpp server for brain.backend=llamacpp"),
    "brain.find": ("brain", False, "brain tool find(query): VLM grounding of a phrase onto our detections (information only)"),
    "brain.enabled": ("brain", False, "closed-set tool choice by a planner backend (stub = classical choice), EV-gated"),
    "calib.plane.apply_height": ("P0 table", False, "runtime counter_z from the head-depth counter-plane fit (median of the first calib.plane.looks looks, clamped to +-max_dz_m, nominal on bad depth / tilt > max_tilt_deg); every consumer reads scene.counter_z"),
    "calib.fk_xcheck": ("Timi B2", False, "log-only RGB head-pose cross-check: residual pitch/yaw of the pose in use vs Dex3 FK hand blob"),
    "skills.memory": ("skill memory", False, "persist grasp-variant outcomes and reorder retry-vary arms"),
    "skills.memory_path": ("skill memory", None, "SQLite path, unset = in-memory (AALTO_SKILLS_DB overrides)"),
    "instruction.resolve": ("Felix #1", False, "instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)"),
    "instruction.max_chars": ("fuzz fix", 0, "cap the instruction text at N chars, cut at the last sentence/clause boundary (resolver is quadratic: 50k chars = 19 s); 0 = off"),
    "safety.nan_guard": ("fuzz fix", False, "non-finite arm_q/hand_q: skip the planner, hold the last emitted action (zero hold before any); non-finite head_depth -> 0 (missing)"),
    "instruction.normalize": ("Felix lang", False, "typo/synonym normaliser before the resolver: resolve(normalize(text)) (needs instruction.resolve)"),
    "instruction.strict": ("Felix strict", False, "never guess: unparsed text abstains all; clauses with negation/condition/placement/relation/swap/quantity abstain their keys; unnamed keys dropped (needs instruction.resolve)"),
    "policy.safety_shell": ("safety", False, "act(): hold-last on exception / deadline miss, all heavy work in the look thread; plans slowed to (1-policy.safety_shell_vel_reserve) x the filter velocity only if that is > 0"),
    "policy.chunk_ticks": ("finals rate", None, "override motion.chunk; null keeps the profile value"),
    "policy.hold_ticks": ("finals rate", None, "override motion.hold_ticks; null keeps the profile value"),
    "policy.fixed_mapping": ("organisers P0", None, "colour->bin fallback when the instruction is empty/missing (null = sim FIXED_MAPPING; real robot: yellow left, blue center, red right)"),
    "policy.insert_guard": ("insert", False, "rung 3 (insert_shapes task id or the hole/lid sentence): pick nothing (0 pts, never -1 on the lid)"),
    "policy.bin_lookup": ("round 8", "colour", "colour (stock) | resolve_bin: (colour,shape)->colour->shape->default"),
    "perception.y_affine": ("Timi 23 §13", None, "y -= a*y + b (perceived - true y model, pelvis frame)"),
    "perception.self_calibration.enabled": ("Timi calib", False,
                                            "per-episode head-camera xy offset from the bin walls (depth)"),
    "perception.online_pitch_cal": ("Timi pitch", False,
                                    "per-episode camera pitch: true=orange bin rims | table=counter plane"),
    "perception.hand_calibration.enabled": ("Oliver hand-calib", False,
                                            "per-episode head-camera 6-DOF pose from the arms in depth (ICP vs FK)"),
    "perception.classify_shape": ("Oliver #5/#7", False, "Detection.shape from the top face (shape_id)"),
    "perception.triangle_pose": ("Oliver #5", False, "triangles: apex yaw + robust xy centre"),
    "perception.near_bin_guard": ("Oliver #7", False, "classify_shape abstains next to a bin (cut top face)"),
    "perception.no_blob_shape": ("Oliver #5/#7", False, "no metric top face -> shape None, never the 2D blob ratio"),
    "perception.shape_diag": ("Oliver #5/#7", False, "passive [shape_diag] log line per detection (no action change)"),
    "perception.shape_2d_v2": ("r17", False, "yellow: 2D cube/cylinder from the blob top edge (quad_rms + n_vert) where the shape would be None / the blob fallback"),
    "perception.hsv_set": ("real colour", "stock", "stock | robust: one HSV set for sim + the real head cam (hsv_robust) | robust_v2: robust with red H 0-22, yellow H 24-52 (hsv_robust_v2)"),
    "perception.hue_guard.enabled": ("hue guard", False, "robust_v2 band 22-24 + per-frame: adjacent red|yellow pair closer than min_gap_deg -> abstain (color_conf 0.3, off the bins)"),
    "perception.colour_kmeans": ("EV gate", False, "per-episode k=3 HSV object-pixel clusters re-centre robust hue ranges"),
    "perception.principal_point": ("Oliver lane 3", "file", "file | center: Isaac renders cx,cy at the image centre"),
    "perception.split_blobs": ("real colour", False, "split touching same-colour objects at distance-transform peaks"),
    "perception.color_gate.live_bins": ("Felix gate", False, "bin footprint = the policy's current bin estimate (bin.vote shift / calibrate_bins); none -> nominal + near_margin_m towards the robot"),
    "perception.color_gate.fk_mask": ("Felix gate", False, "mask the FK-posed arms + hands (dilated) out of the image before blob grouping"),
    "perception.color_gate.sat_floor": ("Felix gate", False, "reject blue (purple) blobs with median HSV S < sat_floor_params.blue and yellow < .yellow (shadow / sheen); red and orange unchanged"),
    "perception.color_gate.enabled": ("Felix gate", False, "post-filter: area floor + ceiling by distance, workspace mask, bin footprint/rim rejects, degraded when > max_cands (no truncation), shape None for non-depth triangles"),
    "perception.inject_cam_error": ("inject", None, "TEST ONLY: [dpitch_deg, dyaw_deg, dx, dy, dz] perturbs the head "
                                                    "T_pelvis_optical perception uses (simulates a mis-calibrated head)"),
    "grasp.yaw_align": ("Oliver #5", False, "triangle grasp: finger direction from apex yaw (needs shape+pose)"),
    "grasp.yaw_phi_deg": ("Oliver #5", STOCK, "yaw_align: finger heading = apex + phi (stock 105)"),
    "grasp.yaw_tri_shift": ("Oliver #5", STOCK, "yaw_align: grasp point to the pinch midpoint (stock true)"),
    "policy.time_triage": ("Oliver #3", False, "skip plans that do not fit the time left, try the next object"),
    "policy.max_total_attempts": ("Oliver #3", None, "global per-episode cap on started picks"),
    "policy.max_attempts": ("stock param", 2, "tries per object before it is skipped (stock 2)"),
    "policy.retry_pass": ("Oliver #3", False, "once per episode: max_attempts-skipped objects get retry_extra_attempts"),
    "policy.time_margin_s": ("Oliver #3", 0.0, "plans must end this long before the time limit"),
    "policy.episode_summary_log": ("Oliver #3", False, "one log line per episode end"),
    "policy.retry_vary_grasp": ("retry-vary", False, "retry after the object remains in the next frame with another "
                                                     "grasp (posture/heading), never an identical one"),
    "policy.max_attempts_vary": ("retry-vary", None, "with retry_vary_grasp: tries per object up to this, only with time left"),
    "policy.first_object_speed_scale": ("finals safe", None,
                                          "null | (0,1]: arm speed x this for every plan made before the first pick starts"),
    "policy.verify_placement": ("Timi #6", False, "verify each release in the bins, correct wrong-bin, drop slots"),
    "policy.place_gate": ("EV gate", False, "require colour and target-bin confidence before planning a placement"),
    "policy.place_gate_colour_min": ("EV gate", 0.6, "minimum HSV/blob colour confidence (used only by place_gate)"),
    "policy.place_gate_bin_min": ("EV gate", 0.5, "minimum residual-derived bin confidence (used only by place_gate)"),
    "policy.object_time_budget_s": ("EV gate", None, "cumulative planned motion budget per tracked object; null = off"),
    "policy.grasp_diag_log": ("Felix grasp", False, "passive [grasp_diag] line per pick: stall vs closed, held"),
    "grasp.held_detector": ("real held", "stall", "stall (legacy max error) | finger_sum (bias-corrected Dex3 fingers)"),
    "policy.verify_yellow_unknown": ("Felix #6 rev", False, "#6: yellow in-bin detection = unknown, never mismatch"),
    "policy.max_corrections_per_episode": ("Felix #6 rev", None, "#6: cap corrections; counter picks go first"),
    "grasp.drop_slots": ("Timi 23", False, "standalone drop slots: next object in a bin -> next slot"),
    "grasp.drop_slot_overflow": ("Felix #6 rev", "counter", "slots used up: counter (leave it) | stack (slot 0)"),
    "grasp.drop_height_extra_m": ("finals safe", 0.0,
                                    "extra bin-release height (capped at the carry height); IK failure uses the stock target"),
    "grasp.drop_lateral_margin_m": ("finals safe", 0.0,
                                      "move the bin release toward the bin centre (y only, footprint-checked); IK failure uses the stock target"),
    "motion.carry_scale": ("Felix grasp", 1.0, "<= 1: slower lift + carry only"),
    "motion.arm_vel_cap": ("Timi vel", None, "rad/s cap on motion.arm_max_vel (arm_max_acc scaled by the same ratio); null = profile"),
    "motion.close_hold_s": ("Felix grasp", STOCK, "hold after closing the hand (sim profile 0.25 s)"),
    "motion.lift_on_stall.enabled": ("round 9", False, "lift as soon as the fingers stall (>= close_hold_s, <= timeout)"),
    "motion.lift_on_stall.timeout_s": ("round 9", STOCK, "lift_on_stall: longest close hold (stock 1.2 s)"),
    "grasp.point_in_palm": ("pinch fix", STOCK, "object centre in the LEFT palm frame (right = [1,-1,1] mirror)"),
    "grasp.close_posture_by_shape": ("hybrid", False, "per pick: demo posture only for a confident metric triangle, else the configured one"),
    "grasp.posture_by_shape_min_conf": ("hybrid", 0.5, "close_posture_by_shape: min shape confidence (0..1) for the demo posture"),
    "grasp.posture_apex_gate": ("apex gate", False, "close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)"),
    "grasp.generic_fallback": ("generic grasp", False, "unknown/low-confidence shape: geometry-only top-down grasp, colour-only bin resolution"),
    "grasp.close_posture": ("pinch fix", "closed", "grasp posture: closed | closed_tight"),
    "grasp.border_posture": ("retry-vary", False, "label None from abstain_border -> demo posture on the first try"),
    "grasp.cylinder_no_yaw_vary": ("cyl yaw", False, "retry on a detected cylinder varies the posture only, never the finger yaw (rotationally symmetric)"),
    "bin.geometry_runtime": ("bin geo", False, "every bin consumer (perception masks/hulls/near_bin/in_bin/gate, planner drop/slots/collision) follows bin_geometry(): nominal + the voted shift"),
    "bin.vote.enabled": ("bin vote", False, "per-bin position from 3 votes (depth rim, tape, prior) over the first frames"),
    "hand.closed_tight": ("pinch fix", STOCK, "the closed_tight posture (LEFT DDS order, right mirrored)"),
    "debug.log_joint_offset": ("finals diag", False,
                                 "log cmd-minus-measured during steady holds in the first second (no control change)"),
    "policy.confidence_order": ("D1/D2", False, "order by 2*p/est_time (p_colour*p_bin*p_grasp), never the hardest first"),
    "policy.object_time_cap": ("D2", False, "per-object cap (remaining-15)/left 15..35 s, T-20 s no new object, T-8 s release/retract"),
    "policy.speed_tiers": ("D3", False, "1st object 0.7 + 1.5 cm clearance, clean 0.85/0.9, failure 0.6 + centre bin only"),
    "policy.confidence_tiers": ("D4", False, "hold 3-5 s, calibration C -> CONFIDENT / DEGRADED / PRIOR-ONLY"),
    "decision.isolated_first": ("isolated-first", False, "candidates within decision.isolated_min_m of another detection go after the isolated ones"),
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
    _apply_chunk_ticks(cfg)
    _apply_arm_vel_cap(cfg)
    return cfg


def _apply_chunk_ticks(cfg):
    """Apply opt-in reply-size overrides. Null leaves both legacy motion values untouched."""
    pol, motion = cfg["policy"], cfg["motion"]
    if pol.get("chunk_ticks") is not None:
        motion["chunk"] = pol["chunk_ticks"]
    if pol.get("hold_ticks") is not None:
        motion["hold_ticks"] = pol["hold_ticks"]


def _apply_arm_vel_cap(cfg):
    """motion.arm_vel_cap: arm_max_vel = min(profile value, cap), arm_max_acc x the same ratio. Never faster."""
    mo = cfg.get("motion") or {}
    cap = mo.get("arm_vel_cap")
    if cap is None:
        return
    cap = float(cap)
    if not 0.0 < cap <= 1.0:
        raise ValueError(f"motion.arm_vel_cap = {cap} outside (0, 1] rad/s")
    v = float(mo["arm_max_vel"])
    if cap < v:
        mo["arm_max_vel"], mo["arm_max_acc"] = cap, float(mo["arm_max_acc"]) * cap / v


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
    fm = _get(cfg, "policy.fixed_mapping")
    if fm is not None and not (isinstance(fm, dict) and set(fm) == {"red", "yellow", "blue"}
                               and sorted(map(str, fm.values())) == ["center", "left", "right"]):
        raise ValueError(f"policy.fixed_mapping = {fm!r}: want null or exactly red/yellow/blue -> a permutation of left/center/right")
    mc = _get(cfg, "instruction.max_chars")
    if not isinstance(mc, int) or isinstance(mc, bool) or mc < 0:
        raise ValueError(f"instruction.max_chars = {mc!r}: want an int >= 0 (0 = off)")
    if not isinstance(_get(cfg, "safety.nan_guard"), bool):
        raise ValueError("safety.nan_guard: want true or false")
    for k in ("policy.chunk_ticks", "policy.hold_ticks"):
        v = _get(cfg, k)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            raise ValueError(f"{k} = {v!r}: want null or an int >= 1")
    for k in ("policy.place_gate_colour_min", "policy.place_gate_bin_min"):
        v = float(_get(cfg, k))
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{k} = {v}: want a confidence in [0, 1]")
    budget = _get(cfg, "policy.object_time_budget_s")
    if budget is not None:
        budget = float(budget)
        if not np.isfinite(budget) or budget <= 0.0:
            raise ValueError(f"policy.object_time_budget_s = {budget}: want null or a finite positive number of seconds")
    ice = _get(cfg, "perception.inject_cam_error")
    if ice is not None and not (isinstance(ice, (list, tuple)) and len(ice) == 5
                                and all(isinstance(x, (int, float)) for x in ice)):
        raise ValueError(f"perception.inject_cam_error = {ice!r}: want null or [dpitch_deg, dyaw_deg, dx, dy, dz]")
    first_scale = _get(cfg, "policy.first_object_speed_scale")
    if first_scale is not None and not 0.0 < float(first_scale) <= 1.0:
        raise ValueError("policy.first_object_speed_scale must be null or in (0, 1]")
    for k in ("grasp.drop_height_extra_m", "grasp.drop_lateral_margin_m"):
        v = float(_get(cfg, k))
        if not np.isfinite(v) or v < 0.0:
            raise ValueError(f"{k} must be a finite non-negative distance")
    pol = cfg["policy"]
    if first_scale is not None and pol.get("safety_shell") and float(pol.get("safety_shell_vel_reserve", 0.0)) > 0.0:
        # the shell wraps planner.plan with closures bound to the stock planner: the first-pick scale would not reach it
        raise ValueError("policy.first_object_speed_scale cannot be combined with policy.safety_shell_vel_reserve > 0")
    _check_color_gate(cfg)
    hg = cfg.get("perception", {}).get("hue_guard") or {}
    if not 0.0 <= float(hg.get("red_min_s", 110.0)) <= 255.0:
        raise ValueError(f"perception.hue_guard.red_min_s = {hg.get('red_min_s')}: want 0-255")
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


def _check_color_gate(cfg):
    """perception.color_gate tuning: fail at load, never inside detect() (Codex M8)."""
    g = (cfg.get("perception") or {}).get("color_gate") or {}
    if not g.get("enabled"):
        return

    def num(k, lo=0.0, opt=False):
        v = g.get(k)
        if v is None and opt:
            return
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < lo:
            raise ValueError(f"perception.color_gate.{k} = {v!r}: want a finite number >= {lo}")
    for k in ("area_frac", "min_px_no_z", "edge_m", "near_margin_m"):
        num(k)
    for k in ("max_area_frac", "max_px_no_z", "ws_abs_y", "ws_edge_m"):
        num(k, opt=True)
    num("max_cands", lo=1)
    for k in ("edge_h", "ws_x"):
        v = g.get(k)
        if v is None and k == "ws_x":
            continue
        if not (isinstance(v, (list, tuple)) and len(v) == 2 and all(isinstance(x, (int, float)) and np.isfinite(x)
                                                                     for x in v) and v[0] <= v[1]):
            raise ValueError(f"perception.color_gate.{k} = {v!r}: want [lo, hi] finite, lo <= hi")
    if g.get("ws_x") is not None and g.get("ws_abs_y") is None:
        raise ValueError("perception.color_gate.ws_x needs ws_abs_y")


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



def hand_cal_conflicts(cfg):
    """Release-gate rule: perception.hand_calibration ON + a non-zero y_affine / xy_offset (or self_calibration) is a
    FAIL: all correct the same head-camera error, stacking double-corrects. hand_calibration + online_pitch_cal is
    ALLOWED (B1 chains after the pitch fit)."""
    p = cfg.get("perception") or {}
    if not (p.get("hand_calibration") or {}).get("enabled", False):
        return []
    out = []
    for k in ("y_affine", "xy_offset"):
        v = p.get(k)
        if v is not None and any(abs(float(x)) > 0.0 for x in v):
            out.append(f"perception.hand_calibration is ON together with {k}={list(v)} (double correction)")
    if (p.get("self_calibration") or {}).get("enabled", False):
        out.append("perception.hand_calibration is ON together with perception.self_calibration (double correction)")
    return out


def pitch_cal_conflicts(cfg):
    """Release-gate rule: online_pitch_cal ON + a non-zero perception.y_affine / perception.xy_offset is a FAIL.
    Those offsets were fitted against the NOMINAL extrinsics; correcting the camera on top double-corrects."""
    p = cfg.get("perception") or {}
    if not p.get("online_pitch_cal", False):
        return []
    out = []
    ya = p.get("y_affine")
    if ya is not None and any(abs(float(x)) > 0.0 for x in ya):
        out.append(f"perception.online_pitch_cal is ON together with y_affine={list(ya)} (fitted against nominal extrinsics)")
    xo = p.get("xy_offset")
    if xo is not None and any(abs(float(x)) > 0.0 for x in xo):
        out.append(f"perception.online_pitch_cal is ON together with xy_offset={list(xo)} (fitted against nominal extrinsics)")
    return out
