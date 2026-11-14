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

EXPECTED_OUTPUT_CAPS = {"arm_vel": 0.85, "arm_acc": 4.0, "hand_vel": 1.5, "hand_acc": 8.0}


def check(overlay=None, out=print):
    # Construct the same class as the server entry point. Its snapshot includes profile selection, profile overrides,
    # the baked/explicit overlay, flag validation, and derived motion clamps.
    server = ClassicalServerPolicy(profile="none", overlay=overlay)
    cfg = server.config_snapshot
    on = {key for key, _, enabled, _, _ in effective_flags(cfg) if enabled}
    unexpected, missing = sorted(on - EXPECTED_FINALS_FLAGS), sorted(EXPECTED_FINALS_FLAGS - on)
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
