"""Feature flags + the submission overlay (integration branch). Run:
    python -m unittest aaltonomous.baselines.classical.test_flags -v
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np

from .config import BAKED_OVERLAY, FLAGS, OVERLAY_DIR, STOCK, _get, effective_flags, load_config, load_overlay


class TestFlags(unittest.TestCase):
    def test_every_flag_defaults_off(self):
        cfg = load_config(overlay="none")
        for k, (_, off, _) in FLAGS.items():
            if off != STOCK:
                self.assertEqual(_get(cfg, k), off, k)
        self.assertFalse(any(on for _, _, on, _, _ in effective_flags(cfg)))

    def test_skill_memory_flags_default_off(self):
        cfg = load_config(overlay="none")
        self.assertFalse(cfg["skills"]["memory"])
        self.assertIsNone(cfg["skills"]["memory_path"])   # unset: in-memory, no /tmp file

    def test_h7_m3_flags_default_off(self):
        cfg = load_config(overlay="none")
        self.assertFalse(cfg["policy"]["generation_workers"])
        self.assertFalse(cfg["grasp"]["held_calibration"]["enabled"])

    def test_calibration_and_empty_recovery_remain_existing_opt_ins(self):
        """C5/H6, H3, and C4 fixes must not change the flags-off policy."""
        stock = load_config(overlay="none")
        finals = load_config(overlay="finals_r21")
        self.assertIn("calib.plane.apply_height", FLAGS)
        self.assertIn("perception.online_pitch_cal", FLAGS)
        self.assertIn("grasp.held_detector", FLAGS)
        self.assertFalse(stock["calib"]["plane"]["apply_height"])
        self.assertFalse(stock["perception"]["online_pitch_cal"])
        self.assertEqual(stock["grasp"]["held_detector"], "stall")
        self.assertTrue(finals["calib"]["plane"]["apply_height"])
        self.assertEqual(finals["perception"]["online_pitch_cal"], "table")
        self.assertEqual(finals["grasp"]["held_detector"], "finger_sum")

    def test_depth_blob_recover_flag_defaults_off(self):
        cfg = load_config(overlay="none")
        self.assertFalse(cfg["perception"]["depth_blob_recover"])
        self.assertIn("perception.depth_blob_recover", FLAGS)
        self.assertFalse(load_config(overlay="submission")["perception"]["depth_blob_recover"])

    def test_finals_r22_enables_only_the_grasp_reliability_flags_over_r22(self):
        base = load_config(overlay="finals_r22")
        got = load_config(overlay="finals_r22_grasp")
        base["overlay"] = "finals_r22_grasp"
        base["grasp"]["staggered_release"] = True
        base["policy"]["retry_jitter_m"] = 0.004
        base["perception"]["assoc_radius_m"] = 0.10
        base["held"]["read_through_carry"] = True
        self.assertEqual(got, base)

    def test_finals_r22_softgrip_adds_only_the_three_soft_flags_over_r22(self):
        base = load_config(overlay="finals_r22")
        got = load_config(overlay="finals_r22_softgrip")
        base["overlay"] = "finals_r22_softgrip"
        base["grasp"]["soft_hand"] = base["grasp"]["soft_squeeze"] = True
        base["held"]["soft_detector"] = True
        self.assertEqual(got, base)

    def test_soft_flags_default_off_and_need_finger_sum(self):
        cfg = load_config(overlay="none")
        self.assertEqual((cfg["grasp"]["soft_squeeze"], cfg["grasp"]["soft_hand"], cfg["held"]["soft_detector"]),
                         (False, False, False))
        with self.assertRaises(ValueError):
            load_config(overlay="none", overrides={"held.soft_detector": True})
        with self.assertRaises(ValueError):
            load_config(overlay="none", overrides={"grasp.soft_overshoot": [0, 0, 0, 3.0, 0, 0, 0]})

    def test_no_baked_overlay_in_the_repo(self):
        self.assertFalse(BAKED_OVERLAY.exists(), "overlay.yaml is only created inside the submission image")

    def test_shipped_submission_overlay_is_stock_plus_the_ab_proven_flags(self):
        stock = load_config(overlay="none")
        sub = load_config(overlay="submission")
        self.assertEqual(sub.pop("overlay"), "submission")
        on = {k for k, _, is_on, _, _ in effective_flags(load_config(overlay="submission")) if is_on}
        self.assertEqual(on, {"instruction.resolve", "policy.bin_lookup", "grasp.close_posture", "hand.closed_tight",
                              "grasp.point_in_palm", "motion.close_hold_s", "perception.classify_shape",
                              "perception.hsv_set", "perception.split_blobs", "policy.grasp_diag_log",
                              "grasp.close_posture_by_shape", "grasp.posture_apex_gate",
                              "policy.insert_guard"})        # A/B: sort_language 54/74 vs 1/74 (box A, twice)
        sub["instruction"]["resolve"] = False
        sub["policy"]["bin_lookup"] = "colour"
        sub["policy"]["insert_guard"] = stock["policy"]["insert_guard"]   # rung 3 only: abstain (0 instead of -1)
        for k in ("close_posture", "point_in_palm"):                       # pinch_v1 grasp (Modal 60/82 vs 52/82)
            sub["grasp"][k] = stock["grasp"][k]
        sub["hand"]["closed_tight"] = stock["hand"]["closed_tight"]
        sub["motion"]["close_hold_s"] = stock["motion"]["close_hold_s"]
        for sec, k in (("perception", "classify_shape"), ("perception", "hsv_set"), ("perception", "split_blobs"),
                       ("policy", "grasp_diag_log"), ("grasp", "close_posture_by_shape"),
                       ("grasp", "posture_apex_gate")):                     # final = cand_r14 (r14: 32/38, 28/36)
            sub[sec][k] = stock[sec][k]
        self.assertEqual(sub, stock)
        # The frozen submission is unchanged; only the newly registered, default-off overrides may be absent.
        sub_keys = set(load_overlay(OVERLAY_DIR / "submission.yaml")["flags"])
        self.assertEqual(set(FLAGS) - sub_keys, {"policy.chunk_ticks", "policy.hold_ticks"})

    def test_ab_safety_is_cand_r17_plus_the_safety_shell(self):
        def on(name):
            return {k for k, _, is_on, _, _ in effective_flags(load_config(overlay=name)) if is_on}
        self.assertEqual(on("ab_safety"), on("cand_r17") | {"policy.safety_shell"})
        self.assertNotIn("policy.safety_shell", on("cand_r17"))
        self.assertEqual(load_config(overlay="ab_safety")["policy"]["safety_shell_vel_reserve"], 0.0)

    def test_overlay_flips_only_its_flags(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "x.yaml"
            p.write_text("name: x\nflags:\n  policy.verify_placement: true\n  policy.max_total_attempts: 12\n")
            cfg = load_config(overlay=str(p))
        on = {k for k, _, is_on, _, _ in effective_flags(cfg) if is_on}
        self.assertEqual(on, {"policy.verify_placement", "policy.max_total_attempts"})
        self.assertEqual(cfg["profile"], "sim")      # applied on top of the sim profile
        self.assertEqual(cfg["motion"]["arm_max_vel"], load_config(overlay="none")["motion"]["arm_max_vel"])

    def test_ab_overlays_flip_exactly_their_flags(self):
        expect = {"ab_retry": {"policy.retry_pass"}, "ab_maxatt4": {"policy.max_attempts"},
                  "ab_resolver": {"instruction.resolve"}, "ab_verify": {"policy.verify_placement"},
                  "ab_triage": {"policy.time_triage", "policy.max_total_attempts"},
                  "ab_hold06": {"motion.close_hold_s"}, "ab_hold12": {"motion.close_hold_s"},
                  "ab_slots": {"grasp.drop_slots"}, "ab_hold12_slots": {"motion.close_hold_s", "grasp.drop_slots"},
                  "ab_carry05": {"motion.carry_scale"}, "ab_calib": {"perception.self_calibration.enabled"},
                  "ab_pitch_cal": {"perception.online_pitch_cal"},
                  "ab_pitch_cal_table": {"instruction.resolve", "policy.insert_guard", "policy.bin_lookup",
                                           "perception.online_pitch_cal", "perception.classify_shape",
                                           "perception.hsv_set", "perception.split_blobs",
                                           "policy.max_total_attempts", "grasp.drop_slots",
                                           "grasp.close_posture", "hand.closed_tight", "grasp.point_in_palm",
                                           "motion.close_hold_s", "grasp.close_posture_by_shape",
                                           "grasp.posture_apex_gate"},
                  "ab_yaff": {"perception.y_affine"},
                  "ab_pinch_hybrid": {"grasp.close_posture", "hand.closed_tight", "grasp.point_in_palm",
                                      "motion.close_hold_s", "perception.classify_shape",
                                      "grasp.close_posture_by_shape"},
                  "ab_pinch_hybrid_apex": {"grasp.close_posture", "hand.closed_tight", "grasp.point_in_palm",
                                           "motion.close_hold_s", "perception.classify_shape",
                                           "grasp.close_posture_by_shape", "grasp.posture_apex_gate"},
                  "ab_pinch_hybrid_yaff": {"grasp.close_posture", "hand.closed_tight", "grasp.point_in_palm",
                                           "motion.close_hold_s", "perception.classify_shape",
                                           "grasp.close_posture_by_shape",
                                           "perception.y_affine"},
                  "ab_pinch_demo_stall": {"grasp.close_posture", "hand.closed_tight", "grasp.point_in_palm",
                                          "motion.lift_on_stall.enabled"},
                  "ab_lang_full": {"instruction.resolve", "perception.classify_shape", "perception.no_blob_shape",
                                   "policy.bin_lookup"},
                  "ab_verify_lang": {"policy.verify_placement", "policy.verify_yellow_unknown",
                                     "policy.max_corrections_per_episode"},
                  "ab_realcolour": {"instruction.resolve", "policy.bin_lookup", "grasp.point_in_palm",
                                    "grasp.close_posture", "hand.closed_tight", "motion.close_hold_s",
                                    "perception.hsv_set", "perception.split_blobs"}}
        r14 = {k for k, _, is_on, _, _ in effective_flags(load_config(overlay="cand_r14")) if is_on}
        expect["ab_retry_vary"] = r14 | {"policy.insert_guard", "policy.retry_vary_grasp",
                                         "policy.max_attempts_vary", "policy.max_total_attempts"}
        expect["ab_retry_vary_slots"] = expect["ab_retry_vary"] | {"grasp.drop_slots"}
        nopp = {k for k, _, is_on, _, _ in effective_flags(load_config(overlay="cand_r17")) if is_on}
        expect["ab_shape2d"] = nopp | {"perception.shape_2d_v2"}
        expect["ab_skills"] = nopp | {"skills.memory"}   # A/B arm = cand_r17 + the flag
        expect["ab_place_gate"] = nopp | {"policy.place_gate"}
        expect["ab_object_time_budget_s"] = nopp | {"policy.object_time_budget_s"}
        expect["ab_colour_kmeans"] = nopp | {"perception.colour_kmeans"}
        expect["ab_cgate"] = nopp | {"perception.color_gate.enabled"}
        expect["ab_sat_floor"] = nopp | {"perception.color_gate.enabled", "perception.color_gate.sat_floor"}
        expect["ab_plane_cgate"] = nopp | {"perception.color_gate.enabled", "perception.online_pitch_cal"}
        expect["ab_cgate_fk"] = nopp | {"perception.color_gate.enabled", "perception.color_gate.fk_mask"}
        expect["ab_cgate_live"] = nopp | {"perception.color_gate.enabled", "perception.color_gate.live_bins"}
        expect["ab_plane_cgate_live"] = expect["ab_plane_cgate"] | {"perception.color_gate.live_bins"}
        expect["ab_plane_cgate_live_fk"] = expect["ab_plane_cgate_live"] | {"perception.color_gate.fk_mask"}
        expect["ab_decision"] = nopp | {"policy.confidence_order", "policy.object_time_cap", "policy.speed_tiers",
                                        "policy.confidence_tiers"}
        expect["ab_hand_calib"] = nopp | {"perception.hand_calibration.enabled"}   # A/B arm = cand_r17 + the flag
        expect["ab_binvote"] = nopp | {"bin.vote.enabled"}   # rebased on cand_r17
        expect["ab_bingeo"] = nopp | {"bin.vote.enabled", "bin.geometry_runtime"}   # cand_r17 + vote + runtime geometry
        expect["ab_plane_cgate_live_vote"] = expect["ab_plane_cgate_live"] | {"bin.vote.enabled"}
        expect["ab_isolated_first"] = nopp | {"decision.isolated_first"}   # rebased on cand_r17
        expect["ab_held_detector"] = nopp | {"grasp.held_detector"}   # rebased on cand_r17
        expect["ab_first_object_speed_scale"] = nopp | {"policy.first_object_speed_scale"}
        expect["ab_drop_height_extra_m"] = nopp | {"grasp.drop_height_extra_m"}
        expect["ab_drop_lateral_margin_m"] = nopp | {"grasp.drop_lateral_margin_m"}
        expect["ab_log_joint_offset"] = nopp | {"debug.log_joint_offset"}
        expect["ab_robust_input"] = nopp | {"instruction.max_chars", "safety.nan_guard"}   # cand_r17 + the fuzz fixes
        expect["ab_hue_v2"] = nopp   # hsv_set robust_v2 is a value change of a flag already on in cand_r17
        expect["ab_hue_guard"] = nopp | {"perception.hue_guard.enabled"}
        expect["ab_generic_fallback"] = nopp | {"grasp.generic_fallback"}   # rebased on cand_r17
        expect["ab_apply_height"] = nopp | {"calib.plane.apply_height"}
        expect["ab_fk_xcheck"] = nopp | {"calib.fk_xcheck"}   # rebased on cand_r17: log-only B2 arm
        expect["ab_cylinder"] = nopp | {"grasp.cylinder_no_yaw_vary"}   # cand_r17 + the cylinder yaw flag
        expect["cand_r18"] = expect["ab_plane_cgate_live_fk"] | {"motion.arm_vel_cap"}   # candidate, not ON
        expect["finals_r18"] = expect["cand_r18"] | {"instruction.normalize", "grasp.release_wait_s", "grasp.release_confirm"}   # finals candidate + lag R44/R49
        expect["finals_r19"] = expect["finals_r18"] | {"perception.hue_guard.enabled", "instruction.max_chars", "safety.nan_guard",
                                                        "instruction.strict"}   # r18 + hue + fuzz + strict
        expect["finals_r19b"] = expect["finals_r19"] | {"policy.safety_shell", "perception.color_gate.sat_floor"}   # P0 Hz
        expect["finals_r19c"] = (expect["finals_r19"] | {"policy.safety_shell", "grasp.drop_slot_grid"}) - {"perception.online_pitch_cal", "perception.color_gate.enabled",
                                                                              "perception.color_gate.live_bins", "perception.color_gate.fk_mask"}
        expect["finals_r20"] = expect["finals_r19b"] | {"bin.geometry_runtime", "grasp.held_detector",
                                                          "calib.plane.apply_height", "grasp.drop_slot_grid"}
        expect["finals_r21"] = expect["finals_r20"] | {"perception.color_gate.fk_dilate_px",
                                                         "perception.color_gate.near_margin_m",
                                                         "policy.generation_workers", "grasp.held_calibration.enabled",
                                                         "motion.output_caps"}   # R29 + fixfk + H7/M3 + M4
        expect["finals_r22"] = (expect["finals_r21"] | {"bin.vote.enabled", "bin.vote.signed_rim_fit", "bin.vote.async_extract",
                                                       "bin.vote.frame_transform_snapshot", "instruction.multilingual"}) \
            - {"perception.hue_guard.enabled", "policy.safety_shell"}   # r22: fixvote ON, bisect rc1->r19c
        expect["finals_r22_recall"] = expect["finals_r22"] | {
            "perception.sat_floor_adaptive", "perception.area_floor_geometric", "perception.topface_mode",
            "perception.partial_ok_near_hand", "perception.relook_before_done"}   # recall A/B on top of r22
        expect["finals_r22_cal"] = expect["finals_r22"] | {"calib.hand_icp_primary", "calib.counter_after_hand",
                                                       "calib.refit_per_look", "perception.principal_point"}   # handcal A/B
        expect["finals_r22_grasp"] = expect["finals_r22"] | {"perception.assoc_radius_m", "policy.retry_jitter_m",
                                                         "grasp.staggered_release", "held.read_through_carry"}   # release A/B
        expect["finals_r22_gate"] = expect["finals_r22"] | {
            "perception.reject_log", "perception.gate_below_bins", "perception.tape_value_cut",
            "perception.split_by_solidity", "perception.color_gate.sat_floor_params.blue",
            "perception.color_gate.fk_silhouette", "perception.sim_center_principal_point", "perception.aim_z_frac",
            "perception.still_vote"}   # gatefix A/B
        expect["finals_r22_chunk"] = expect["finals_r22"] | {"policy.chunk_ticks", "policy.hold_ticks"}   # handin rate A/B
        expect["finals_r22_novel"] = expect["finals_r22"] | {
            "perception.novel_objects", "grasp.generic", "instruction.novel_objects", "policy.novel_obstacles"}   # novel R22 on r22
        expect["rc1_strict"] = {k for k, _, is_on, _, _ in effective_flags(load_config(overlay="rc1")) if is_on} \
            | {"instruction.strict"}   # rc1 + the strict flag and nothing else
        inj = {"perception.inject_cam_error"}
        expect["ab_inj_p7"] = expect["ab_inj_p4"] = expect["ab_inj_p7_yl"] = expect["ab_inj_p7_yr"] = nopp | inj
        expect["ab_plane_inj_p7"] = nopp | inj | {"perception.online_pitch_cal"}
        expect["ab_plane_b1_inj_p7"] = nopp | inj | {"perception.online_pitch_cal", "perception.hand_calibration.enabled"}
        expect["ab_b1_inj_p7"] = nopp | inj | {"perception.hand_calibration.enabled"}
        for n in ("finals_r18", "finals_r19", "finals_r19b", "finals_r19c", "finals_r20", "finals_r21", "finals_r22",
                  "finals_r22_recall", "finals_r22_cal", "finals_r22_grasp", "finals_r22_gate", "finals_r22_chunk",
                  "finals_r22_novel"):
            expect[n] = expect[n] | {"policy.fixed_mapping"}   # real-robot fixed fallback
        for n in ("finals_r20", "finals_r21", "finals_r22", "finals_r22_recall", "finals_r22_cal", "finals_r22_grasp", "finals_r22_gate", "finals_r22_chunk", "finals_r22_novel"):   # only where apply_height measures depth; r18-r19c/real would apply the prior even WITH depth
            expect[n] = expect[n] | {"scene.counter_z_prior"}
        for n in [k for k in expect if k.startswith("finals_")]:
            expect[n] = expect[n] | {"hand.mirror_index_middle"}   # R42a
        expect["finals_r22_softgrip"] = expect["finals_r22"] | {"grasp.soft_hand", "grasp.soft_squeeze",
                                                                 "held.soft_detector"}   # softgrip A/B: r22 + its 3 flags
        for name, keys in expect.items():
            keys = keys | {"policy.grasp_diag_log"}     # passive diagnostics ride along in every A/B arm
            cfg = load_config(overlay=name)
            self.assertEqual(cfg["overlay"], name)
            on = {k for k, _, is_on, _, _ in effective_flags(cfg) if is_on}
            self.assertEqual(on, keys, name)
        self.assertEqual(load_config(overlay="ab_maxatt4")["policy"]["max_attempts"], 4)
        self.assertEqual(load_config(overlay="ab_triage")["policy"]["max_total_attempts"], 10)
        self.assertEqual(load_config(overlay="ab_pitch_cal_table")["perception"]["online_pitch_cal"], "table")

    def test_finals_real_loads_as_cand_r16_with_calibrated_intrinsics(self):
        from aaltonomous.interface.calibration import Calibration
        from aaltonomous.scene.scene import Scene
        from .perception import Perception

        sim = load_config(overlay="cand_r16")
        real = load_config(overlay="finals_real")
        self.assertEqual((sim["perception"]["principal_point"], real["perception"]["principal_point"]),
                         ("center", "file"))
        self.assertEqual(real["policy"]["max_total_attempts"], 18)
        sim["overlay"] = real["overlay"]
        sim["perception"]["principal_point"] = "file"
        sim["policy"]["fixed_mapping"] = real["policy"]["fixed_mapping"]   # organisers P0
        self.assertTrue(real["hand"]["mirror_index_middle"])
        sim["hand"]["mirror_index_middle"] = True                          # R42a
        sim["grasp"].update(release_wait_s=0.5, release_confirm=True)   # lag R44/R49 (finals only)
        self.assertEqual(real, sim)

        calibration = Calibration.load()
        perception = Perception(real, Scene.load(), calibration, log=None)
        np.testing.assert_array_equal(perception.cal.cams[perception.cam].K,
                                      calibration.cams[perception.cam].K)

    def test_finals_r22_cal_is_r22_plus_calibration_and_sim_principal_point(self):
        r21, r22 = load_config(overlay="finals_r22"), load_config(overlay="finals_r22_cal")
        self.assertEqual(r22["profile"], "sim")
        self.assertEqual(r22["perception"]["principal_point"], "center")
        for key in ("hand_icp_primary", "counter_after_hand", "refit_per_look"):
            self.assertTrue(r22["calib"][key])
            r22["calib"][key] = r21["calib"][key]
        r22["perception"]["principal_point"] = r21["perception"]["principal_point"]
        r22["overlay"] = r21["overlay"]
        self.assertEqual(r22, r21)

    def test_every_retry_vary_overlay_has_the_global_safety_cap(self):
        retry = {f.stem: load_config(overlay=str(f)) for f in OVERLAY_DIR.glob("*.yaml")
                 if load_config(overlay=str(f))["policy"]["retry_vary_grasp"]}
        self.assertEqual(set(retry), {"ab_apply_height", "ab_b1_inj_p7", "ab_bingeo", "ab_binvote",
                                      "ab_cgate", "ab_cgate_fk", "ab_cgate_live",
                                      "ab_colour_kmeans", "ab_cylinder", "ab_decision",
                                      "ab_drop_height_extra_m", "ab_drop_lateral_margin_m",
                                      "ab_first_object_speed_scale", "ab_fk_xcheck",
                                      "ab_generic_fallback", "ab_hand_calib", "ab_held_detector",
                                      "ab_hue_guard", "ab_hue_v2", "ab_inj_p4", "ab_inj_p7",
                                      "ab_inj_p7_yl", "ab_inj_p7_yr", "ab_isolated_first",
                                      "ab_log_joint_offset", "ab_object_time_budget_s",
                                      "ab_place_gate", "ab_plane_b1_inj_p7", "ab_plane_cgate",
                                      "ab_plane_cgate_live", "ab_plane_cgate_live_fk",
                                      "ab_plane_cgate_live_vote", "ab_plane_inj_p7",
                                      "ab_retry_vary", "ab_retry_vary_slots", "ab_robust_input",
                                      "ab_safety", "ab_sat_floor", "ab_shape2d", "ab_skills",
                                      "ab_vel80", "ab_vel85", "ab_vel90", "brain_llm", "cand_r16",
                                      "cand_r16_nopp", "cand_r17", "cand_r18", "finals_r18",
                                      "finals_r19", "finals_r19b", "finals_r19c", "finals_r20",
                                      "finals_r21", "finals_r22", "finals_r22_cal",
                                      "finals_r22_chunk", "finals_r22_gate", "finals_r22_grasp",
                                      "finals_r22_novel", "finals_r22_recall",
                                      "finals_r22_softgrip", "finals_real", "rc1", "rc1_strict",
                                      "safe_profile"})
        self.assertTrue(all(cfg["policy"]["max_total_attempts"] == 18 for n, cfg in retry.items()
                            if n != "cand_r16_nopp"))   # cand_r16_nopp = HISTORICAL, pre-fix (None)

    def test_instruction_normalize_flag_is_off_by_default_and_rc1_adds_only_it(self):
        self.assertFalse(load_config(overlay="none")["instruction"]["normalize"])
        self.assertFalse(load_config(overlay="none")["instruction"]["multilingual"])
        self.assertFalse(load_config(overlay="submission")["instruction"]["multilingual"])
        description = FLAGS["instruction.multilingual"][2]
        for language in ("FI", "CS", "DE", "SV", "ES", "FR", "IT", "PL", "PT", "UK", "RU", "SK"):
            self.assertIn(language, description)
        self.assertFalse(load_config(overlay="cand_r17")["instruction"]["normalize"])
        base = load_config(overlay="cand_r17")
        rc1 = load_config(overlay="rc1")
        self.assertTrue(rc1["instruction"]["normalize"])
        base["overlay"], base["instruction"]["normalize"] = "rc1", True
        self.assertEqual(rc1, base)

    def test_recall_flags_default_off_and_r22_adds_exactly_them(self):
        keys = {"perception.sat_floor_adaptive", "perception.area_floor_geometric",
                "perception.topface_mode", "perception.partial_ok_near_hand",
                "perception.relook_before_done"}
        stock = load_config(overlay="none")
        self.assertFalse(stock["perception"]["sat_floor_adaptive"])
        self.assertFalse(stock["perception"]["area_floor_geometric"])
        self.assertEqual(stock["perception"]["topface_mode"], "top35")
        self.assertFalse(stock["perception"]["partial_ok_near_hand"])
        self.assertFalse(stock["perception"]["relook_before_done"])
        base_on = {k for k, _, on, _, _ in effective_flags(load_config(overlay="finals_r22")) if on}
        r22 = {k for k, _, on, _, _ in effective_flags(load_config(overlay="finals_r22_recall")) if on}
        self.assertEqual(r22, base_on | keys)
        base = load_config(overlay="finals_r22")
        recall = load_config(overlay="finals_r22_recall")
        base.pop("overlay")
        recall.pop("overlay")
        for key, off in (("sat_floor_adaptive", False), ("area_floor_geometric", False),
                         ("topface_mode", "top35"), ("partial_ok_near_hand", False),
                         ("relook_before_done", False)):
            recall["perception"][key] = off
        self.assertEqual(recall, base)

    def test_cand_r17_differs_from_historical_r16_nopp_only_in_max_total_attempts(self):
        r16, r17 = load_config(overlay="cand_r16_nopp"), load_config(overlay="cand_r17")
        self.assertEqual((r16["overlay"], r17["overlay"]), ("cand_r16_nopp", "cand_r17"))
        self.assertIsNone(r16["policy"]["max_total_attempts"])
        self.assertEqual(r17["policy"]["max_total_attempts"], 18)
        for c in (r16, r17):
            c.pop("overlay")
            c["policy"].pop("max_total_attempts")
        self.assertEqual(r16, r17)

    def test_normalize_flag_changes_the_resolved_mapping_only_when_on(self):
        from .policy import ClassicalPolicy
        text = "Put the cans in the right receptacle and the bricks in the left receptacle."
        for flag in (False, True):
            cfg = load_config(overlay="rc1")
            cfg["instruction"]["normalize"] = flag
            pol = ClassicalPolicy(config=cfg, log=None)
            m = pol._instruction_mapping({"instruction": text})
            self.assertEqual((m or {}).get("cube") == "left" and (m or {}).get("cylinder") == "right", flag, (flag, m))

    def test_normalize_selfcheck(self):
        from . import normalize
        normalize._selfcheck()

    def test_safe_profile_is_cand_r17_plus_conservative_values(self):
        base = load_config(overlay="cand_r17")
        safe = load_config(overlay="safe_profile")
        base["overlay"] = "safe_profile"
        base["policy"]["first_object_speed_scale"] = 0.7
        base["grasp"]["drop_height_extra_m"] = 0.04
        base["grasp"]["drop_lateral_margin_m"] = 0.02
        base["debug"]["log_joint_offset"] = False
        for key in ("online_pitch_cal", "colour_kmeans"):
            base["perception"][key] = False
        base["perception"]["hand_calibration"]["enabled"] = False
        base["bin"]["vote"]["enabled"] = False
        self.assertEqual(safe, base)
        self.assertFalse(safe["perception"]["self_calibration"]["enabled"])
        self.assertIsNone(safe["perception"]["y_affine"])

    def test_conservative_ab_overlays_are_cand_r17_plus_one_flag(self):
        changes = {
            "ab_first_object_speed_scale": ("policy", "first_object_speed_scale", 0.7),
            "ab_drop_height_extra_m": ("grasp", "drop_height_extra_m", 0.04),
            "ab_drop_lateral_margin_m": ("grasp", "drop_lateral_margin_m", 0.02),
            "ab_log_joint_offset": ("debug", "log_joint_offset", True),
        }
        for overlay, (section, key, value) in changes.items():
            expected = load_config(overlay="cand_r17")
            expected["overlay"] = overlay
            expected[section][key] = value
            self.assertEqual(load_config(overlay=overlay), expected, overlay)

    def test_every_overlay_loads_and_its_postures_are_in_urdf_limits(self):
        from .config import hand_postures, home_arm
        from .kinematics import ArmKinematics
        for f in sorted(OVERLAY_DIR.glob("*.yaml")):
            cfg = load_config(overlay=str(f))
            hand_postures(cfg)                            # raises outside joint_spec (URDF) limits
            home_arm(cfg)
        for name, pip in (("ab_pinch_v1", [0.0862, -0.0864, 0.0143]), ("ab_pinch_demo", [0.1094, -0.0694, 0.0142]),
                          ("ab_pinch_v1_yaff", [0.0862, -0.0864, 0.0143]),
                          ("ab_pinch_demo_yaff", [0.1094, -0.0694, 0.0142]),
                          ("ab_pinch_demo_yaw", [0.1094, -0.0694, 0.0142])):
            cfg = load_config(overlay=name)
            self.assertEqual(cfg["grasp"]["close_posture"], "closed_tight")
            kin = ArmKinematics(cfg)
            self.assertEqual(list(kin.offset["left"]), pip)
            self.assertEqual(list(kin.offset["right"]), [pip[0], -pip[1], pip[2]])   # right = [x, +y, z]

    def test_ab_held_detector_is_cand_r17_plus_finger_sum(self):
        cand = load_config(overlay="cand_r17")
        ab = load_config(overlay="ab_held_detector")
        self.assertEqual(cand.pop("overlay"), "cand_r17")
        self.assertEqual(ab.pop("overlay"), "ab_held_detector")
        self.assertEqual(cand["grasp"]["held_detector"], "stall")
        ab["grasp"]["held_detector"] = "stall"
        self.assertEqual(ab, cand)
        self.assertEqual(load_config(overlay="ab_held_detector")["grasp"]["held_threshold_rad"], 0.045)

    def test_thumb0_nonzero_fails_until_the_mirror_is_fixed(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "t0.yaml"
            p.write_text("flags:\n  grasp.close_posture: closed_tight\n"
                         "  hand.closed_tight: [-0.67, 0.92, 0.15, -0.28, -0.02, -0.19, -0.05]\n")
            with self.assertRaisesRegex(ValueError, "thumb0≠0 requires fixing mirror_left_to_right"):
                load_config(overlay=str(p))
        with self.assertRaisesRegex(ValueError, "keep thumb0 sign"):
            load_config(overrides={"hand.preshape": [0.3, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4]})

    def test_hard_limits(self):
        """Overlays can never make anything faster than the stock sim profile."""
        self.assertFalse([k for k in FLAGS if k.startswith("motion.") and k not in
                          ("motion.carry_scale", "motion.close_hold_s", "motion.lift_on_stall.enabled",
                           "motion.lift_on_stall.timeout_s", "motion.arm_vel_cap",
                           "motion.output_caps")])   # cap flags only reduce motion
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "fast.yaml"
            for bad in ("motion.carry_scale: 1.5", "motion.carry_scale: 0.0", "grasp.drop_slot_overflow: pile",
                        "perception.online_pitch_cal: bins", "motion.arm_vel_cap: 1.5", "motion.arm_vel_cap: 0.0"):
                p.write_text("flags:\n  " + bad + "\n")
                with self.assertRaises(ValueError, msg=bad):
                    load_config(overlay=str(p))
        self.assertEqual(load_config(overlay="ab_hold12")["motion"]["close_hold_s"], 1.2)
        self.assertEqual(load_config(overlay="none")["motion"]["close_hold_s"], 0.25)   # sim profile
        for overrides in ({"policy.first_object_speed_scale": 0.0}, {"policy.first_object_speed_scale": 1.01},
                          {"policy.first_object_speed_scale": float("nan")},
                          {"grasp.drop_slot_grid": 1},
                          {"grasp.drop_height_extra_m": -0.01}, {"grasp.drop_height_extra_m": float("inf")},
                          {"grasp.drop_lateral_margin_m": -0.01},
                          {"policy.first_object_speed_scale": 0.7, "policy.safety_shell": True,
                           "policy.safety_shell_vel_reserve": 0.2}):
            with self.assertRaises(ValueError, msg=overrides):
                load_config(overrides=overrides)
        for key, value in (("policy.place_gate_colour_min", -0.1), ("policy.place_gate_bin_min", 1.1),
                           ("policy.object_time_budget_s", 0.0), ("policy.object_time_budget_s", float("inf"))):
            with self.assertRaises(ValueError, msg=f"{key}={value}"):
                load_config(overrides={key: value})

    def test_chunk_tick_flags_override_motion_only_when_enabled(self):
        stock = load_config(overlay="none")
        self.assertEqual((stock["policy"]["chunk_ticks"], stock["policy"]["hold_ticks"]), (None, None))
        self.assertEqual((stock["motion"]["chunk"], stock["motion"]["hold_ticks"]), (10, 6))
        tuned = load_config(overlay="finals_r22_chunk")
        self.assertEqual((tuned["policy"]["chunk_ticks"], tuned["policy"]["hold_ticks"]), (15, 10))
        self.assertEqual((tuned["motion"]["chunk"], tuned["motion"]["hold_ticks"]), (15, 10))
        base = load_config(overlay="finals_r22")
        base.pop("overlay")
        tuned.pop("overlay")
        tuned["policy"]["chunk_ticks"] = tuned["policy"]["hold_ticks"] = None
        tuned["motion"]["chunk"], tuned["motion"]["hold_ticks"] = 10, 6
        self.assertEqual(tuned, base)   # the new overlay changes only the two reply-size flags
        for key in ("policy.chunk_ticks", "policy.hold_ticks"):
            for bad in (0, -1, 1.5, True):
                with self.assertRaises(ValueError, msg=f"{key}={bad!r}"):
                    load_config(overrides={key: bad})

    def test_chunk_flags_off_preserve_pre_change_action_bytes(self):
        from tools.equivalence_check import episode_digest
        self.assertEqual(episode_digest(0, 25.0),
                         "episode seed=0 tl=25 acts=65 end=done clips=0 "
                         "outcomes=['correct', 'counter', 'counter', 'counter', 'correct'] plans=3 "
                         "sha=8a38277f5cddb04c")   # repinned: 40f4c0d (pre-gatefix, pre-handin) already gives this; 7a17757d was the stale older-base pin

    def test_placeholder_overlay_is_refused(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "tbd.yaml"
            p.write_text("placeholder: true\nflags:\n  grasp.close_posture: closed_tight\n")
            with self.assertRaises(ValueError):
                load_config(overlay=str(p))                # values TBD: must not run by accident
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "pinch.yaml"
            p.write_text("flags:\n  grasp.close_posture: closed_tight\n  grasp.point_in_palm: [0.099, -0.084, 0.0]\n"
                         "  hand.closed_tight: [0.0, -0.1, 1.2, -1.1, -0.7, -1.1, -0.7]\n")
            cfg = load_config(overlay=str(p))
            self.assertEqual(cfg["grasp"]["close_posture"], "closed_tight")
            p.write_text("flags:\n  grasp.point_in_palm: [0.1, 0.0]\n")
            with self.assertRaises(ValueError):
                load_config(overlay=str(p))

    def test_unknown_overlay_key_fails_loudly(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "typo.yaml"
            p.write_text("flags:\n  policy.verify_placment: true\n")
            with self.assertRaises(ValueError):
                load_config(overlay=str(p))
            p.write_text("flags:\n  motion.arm_max_vel: 2.0\n")   # not a flag: overlays cannot retune
            with self.assertRaises(ValueError):
                load_config(overlay=str(p))


if __name__ == "__main__":
    unittest.main()
