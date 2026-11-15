#!/usr/bin/env python3
"""Fail unless the server's effective real-robot config is the reviewed finals allow-list."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from aaltonomous.baselines.classical.config import effective_flags  # noqa: E402
from aaltonomous.baselines.classical.policy_server import ClassicalServerPolicy  # noqa: E402


EXPECTED_FINALS_FLAGS = frozenset({   # finals_r22 (rc7): reviewed release set
    "bin.geometry_runtime",
    "bin.vote.async_extract",
    "bin.vote.enabled",
    "bin.vote.frame_transform_snapshot",
    "bin.vote.signed_rim_fit",
    "calib.plane.apply_height",
    "grasp.close_posture",
    "grasp.close_posture_by_shape",
    "grasp.drop_slot_grid",
    "grasp.drop_slots",
    "grasp.held_calibration.enabled",
    "grasp.held_detector",
    "grasp.point_in_palm",
    "grasp.posture_apex_gate",
    "grasp.release_confirm",
    "grasp.release_wait_s",
    "hand.closed_tight",
    "hand.mirror_index_middle",
    "instruction.max_chars",
    "instruction.multilingual",
    "instruction.normalize",
    "instruction.resolve",
    "instruction.strict",
    "motion.arm_vel_cap",
    "motion.close_hold_s",
    "motion.output_caps",
    "perception.classify_shape",
    "perception.color_gate.enabled",
    "perception.color_gate.fk_dilate_px",
    "perception.color_gate.fk_mask",
    "perception.color_gate.live_bins",
    "perception.color_gate.near_margin_m",
    "perception.color_gate.sat_floor",
    "perception.hsv_set",
    "perception.online_pitch_cal",
    "perception.split_blobs",
    "policy.bin_lookup",
    "policy.fixed_mapping",
    "policy.generation_workers",
    "policy.grasp_diag_log",
    "policy.insert_guard",
    "policy.max_attempts_vary",
    "policy.max_total_attempts",
    "policy.retry_vary_grasp",
    "safety.nan_guard",
    "scene.counter_z_prior",
})

# r23 layer on top of r22 (finals_r23*, finals_r24*): reviewed 2026-11-14/15
R23_FLAGS = frozenset({
    "bin.vote.max_shift_m", "instruction.hazard_guard", "instruction.structured", "instruction.venue_colour_aliases",
    "perception.gate_below_bins", "planner.lift_from_contact", "policy.fixed_instruction_first", "policy.safety_shell",
    "profile_auto.auto_detect", "profile_auto.sim_overrides",
})
# brain/universal layer: tuning values (present, non-default, in every universal-derived overlay incl. *_classic) ...
BRAIN_PARAMS = frozenset({"brain.backend", "brain.call_timeout_s", "brain.find_n_predict", "brain.image_max_side_px",
                          "brain.plan_timeout_s", "perception.novel.max_area_px",
                          "perception.novel.fp_filter"})   # fp_filter: stock ON, shipped OFF (SHA2)
# ... and the switches (ON only in finals_r24 / finals_r23_universal)
BRAIN_SWITCHES = frozenset({"brain.closed_loop", "brain.enabled", "brain.ev_gate", "brain.find",
                            "brain.find_to_detection", "brain.grounded_picks", "brain.planner", "grasp.generic",
                            "instruction.novel_objects", "perception.novel_objects", "policy.novel_obstacles",
                            "grasp.novel_near_rim_retry"})
# real-only perception recall flags (r24; false in profile_auto.sim_overrides)
REALPERC_FLAGS = frozenset({"perception.color_gate.fk_silhouette", "perception.prism_speckle_floor",
                            "perception.real_merge_split"})


# real-demo (drop_3) real-only values: release height + slow final approach (point_in_palm / closed_tight values were already ON)
REAL_DEMO_FLAGS = frozenset({"grasp.drop_height_extra_m", "grasp.approach_slow_mps",
                             "perception.table_pitch_calibration.max_abs_pitch_deg"})   # real fix: table pitch guard 15 -> 18


# final gen layer (reviewed 2026-11-15): realgp offsets, chaos hold-on-late, generality parser, arm tau ff, real perception rescues
GEN_FINAL_FLAGS = frozenset({"grasp.arm_xy_offset", "motion.output_hold_on_late", "instruction.generality", "instruction.venue_veto",
                             "policy.shape_rule_guard", "motion.arm_tau_ff", "motion.sag_comp", "perception.bin_bottom_first_run",
                             "perception.fk_colour_rescue", "calib.plane.robust_height", "perception.plan_frame",
                             "calib.plane.robust_agree_m", "perception.height_range", "bin.fallback_centre_x", "bin.fallback_drop_z", "bin.reach_x_max", "bin.reach_drop_z", "planner.budget_escalate", "insert.reach_gated"})
R25_FLAGS = frozenset({"brain.known_task_hard_off", "planner.candidate_budget", "profile_auto.second_cue"})
# fsm layer (real-only dead-time / failure-mode fixes; false in profile_auto.sim_overrides), finals_r25_gen only
FSM_FLAGS = frozenset({"policy.zero_det_relook", "policy.startup_overlap", "policy.contact_start_unblock", "perception.look_summary"})


def expected_flags(overlay_name):
    """Reviewed ON-set for the baked overlay; any other overlay keeps the r22 allow-list."""
    if overlay_name.startswith("finals_r25"):   # r25* = finals_r24 set + the r25 layer (reviewed 2026-11-15)
        return expected_flags("finals_r24") | R25_FLAGS | (
            {"grasp.arm_xy_offset"} if overlay_name == "finals_r25_xy" else set()) | (
            FSM_FLAGS | GEN_FINAL_FLAGS if overlay_name == "finals_r25_gen" else set())
    if overlay_name in ("finals_r24", "finals_r24_classic", "finals_r23_universal"):
        exp = EXPECTED_FINALS_FLAGS | R23_FLAGS | BRAIN_PARAMS
        if overlay_name != "finals_r24_classic":
            exp |= BRAIN_SWITCHES
            if overlay_name == "finals_r24":
                exp -= {"perception.novel.fp_filter"}   # fp_filter TRUE = stock: not reported as ON (classic/universal ship it false)
        if overlay_name != "finals_r23_universal":
            exp |= REALPERC_FLAGS | REAL_DEMO_FLAGS
            exp -= {"grasp.close_posture_by_shape"}   # real: one pinch_demo-family posture for all shapes
        return exp
    return EXPECTED_FINALS_FLAGS


EXPECTED_OUTPUT_CAPS = {"arm_vel": 0.85, "arm_acc": 4.0, "hand_vel": 1.5, "hand_acc": 8.0}


def check(overlay=None, out=print):
    # Construct the same class as the server entry point. Its snapshot includes profile selection, profile overrides,
    # the baked/explicit overlay, flag validation, and derived motion clamps.
    server = ClassicalServerPolicy(profile="none", overlay=overlay)
    cfg = server.config_snapshot
    on = {key for key, _, enabled, _, _ in effective_flags(cfg) if enabled}
    on.discard("profile_auto.force")   # PROFILE_MODE image marker (null | sim | none); behaviour covered by test_profile_force
    exp = expected_flags(str(cfg.get("overlay", "none")))
    unexpected, missing = sorted(on - exp), sorted(exp - on)
    if unexpected:
        out(f"unexpected finals flags ON: {unexpected}")
    if missing:
        out(f"expected finals flags missing: {missing}")
    caps = cfg["motion"].get("output_caps")
    if caps != EXPECTED_OUTPUT_CAPS:
        out(f"motion.output_caps={caps!r}, expected {EXPECTED_OUTPUT_CAPS!r}")
    motion = cfg["motion"]
    limits_ok = (float(motion["arm_max_vel"]) <= 0.85 and float(motion["arm_max_acc"]) <= 4.0
                 and float(motion["hand_max_vel"]) <= 1.5)
    if not limits_ok:
        out("effective finals motion exceeds arm_vel=0.85, arm_acc=4.0 or hand_vel=1.5")
    ok = cfg.get("profile") == "none" and not unexpected and not missing and caps == EXPECTED_OUTPUT_CAPS and limits_ok
    if cfg.get("profile") != "none":
        out(f"effective profile={cfg.get('profile')!r}, expected 'none'")
    out("RELEASE FLAGS PASS" if ok else "RELEASE FLAGS FAIL")
    return ok


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlay", default=None, help="explicit overlay; default uses the server's baked overlay")
    args = ap.parse_args(argv)
    return 0 if check(args.overlay) else 1


if __name__ == "__main__":
    raise SystemExit(main())
