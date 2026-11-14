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
           "perception.topface_mode": ("top35", "whole_blob"),
           "perception.principal_point": ("file", "center"),
           "perception.online_pitch_cal": (False, True, "table"), "brain.backend": ("stub", "llamacpp")}
VECTORS = {"grasp.point_in_palm": 3, "grasp.cage_point_in_palm": 3, "grasp.cage_finger_posture": 7,
           "hand.closed_tight": 7, "perception.y_affine": 2, "grasp.soft_overshoot": 7}

# Every opt-in feature of the integration branch: dotted config key -> (owner, value that means OFF, what it does).
# All defaults in config.yaml are the OFF value, so with no overlay the policy is the stock classical baseline.
FLAGS = {
    "profile_auto.auto_detect": ("profile auto", False, "no AALTO_PROFILE/arg: pick sim | none (real) from the first frames (profile_detect.py); sim if a majority of frames pass; vote = depth holes only (plane std and palette are log-only)"),
    "profile_auto.auto_frames": ("profile auto", 5, "frames inspected before the profile is chosen (zero-row holds meanwhile), >= 1; sim needs a majority of frames; vote = depth holes only (plane std and palette are log-only)"),
    "profile_auto.sim_overrides": ("profile auto", {}, "auto-detect=sim only: dotted key -> value applied as load_config overrides (sim-breaking finals flags back to rc1); real robot keeps the overlay values"),
    "brain.backend": ("brain", "stub", "stub (= classical choice) | llamacpp (bounded async HTTP + GBNF)"),
    "brain.url": ("brain", "http://127.0.0.1:8080/completion", "llama.cpp server for brain.backend=llamacpp"),
    "brain.find": ("brain", False, "brain tool find(query): VLM grounding of a phrase onto our detections (information only)"),
    "brain.enabled": ("brain", False, "closed-set tool choice by a planner backend (stub = classical choice), EV-gated"),
    "calib.plane.apply_height": ("P0 table", False, "pre-motion runtime counter_z from N mutually consistent head-depth plane fits (spread/MAD gated, once per episode, clamped to +-max_dz_m); atomically shared by scene and planner keepouts before a checked staging route"),
    "scene.counter_z_prior": ("prior", None, "null | table height in the pelvis frame (m, 0..0.3) used while no depth plane fit is available (apply_height overrides); null = nominal scene.yaml z"),
    "calib.hand_icp_primary": ("finals calibration", False,
                               "hold at the episode start and fit the full head-camera extrinsic from FK arm surfaces before motion"),
    "calib.counter_after_hand": ("finals calibration", False,
                                 "estimate counter height only after the camera pose is selected; require a MAD-consistent sample set"),
    "calib.refit_per_look": ("finals calibration", False,
                             "freshness-guarded bounded counter-plane update on every look, with a plane-tilt hand-fit cross-check"),
    "calib.wide_counter_window": ("counter-z", False, "counter-z estimation window nominal -8 .. +22 cm (dominant-layer seed, then the usual +-3 cm plane fit with tilt/inlier gates; estimate outside = nominal), runtime height estimated even without calib.plane.apply_height, mat z follows the runtime counter; see COUNTERZ.md"),
    "planner.bin_wall_extra_m": ("counter-z", 0.0, "extra bin wall height (m) on the release/carry height and the rim keepout (planner)"),
    "calib.fk_xcheck": ("Timi B2", False, "log-only RGB head-pose cross-check: residual pitch/yaw of the pose in use vs Dex3 FK hand blob"),
    "skills.memory": ("skill memory", False, "persist grasp-variant outcomes and reorder retry-vary arms"),
    "skills.memory_path": ("skill memory", None, "SQLite path, unset = in-memory (AALTO_SKILLS_DB overrides)"),
    "brain.planner": ("brain", False, "whole-instruction closed-set primitive plan for resolver abstention/relations"),
    "brain.image_max_side_px": ("brain speed", 0, "downscale the head RGB to this longest side before encoding it for find()/decide (boxes are scaled back to original pixels); 0 = full size"),
    "brain.find_n_predict": ("brain speed", 400, "find() output token cap (box JSON); 400 = the old hard-coded cap"),
    "brain.find_to_detection": ("brain agent", False, "find() point that matches no detection -> a novel Detection from the depth blob there (region grow above the counter plane, generic_grasp, ungraspable -> skip); needs brain.find"),
    "brain.timeout_s": ("brain", 2.0, "decision deadline (s): later decisions are dropped, the classical planner decides (stock config.yaml 2.0)"),
    "brain.request_timeout_s": ("brain", 1.5, "absolute llamacpp HTTP deadline (s) (stock config.yaml 1.5)"),
    "brain.closed_loop": ("brain agent", False, "agent loop: re-plan (async, capped) after every pick/place/find outcome with fresh world JSON + step history; last valid plan keeps running; 2 failed grasps -> other variant or skip; needs brain.planner"),
    "instruction.resolve": ("Felix #1", False, "instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)"),
    "instruction.max_chars": ("fuzz fix", 0, "cap the instruction text at N chars, cut at the last sentence/clause boundary (resolver is quadratic: 50k chars = 19 s); 0 = off"),
    "safety.nan_guard": ("fuzz fix", False, "non-finite arm_q/hand_q: skip the planner and hold the last trusted target (before trust, return no action until one complete finite encoder frame); non-finite head_depth -> 0 (missing)"),
    "instruction.normalize": ("Felix lang", False, "typo/synonym normaliser before the resolver: resolve(normalize(text)) (needs instruction.resolve)"),
    "instruction.multilingual": ("multilingual", False, "closed-grammar FI/CS/DE/SV/ES/FR/IT/PL/PT/UK/RU/SK instructions -> cached English canonical form before resolve; unsupported foreign input stays unparsed (needs instruction.resolve)"),
    "instruction.strict": ("Felix strict", False, "never guess: unparsed text abstains all; clauses with negation/condition/placement/relation/swap/quantity abstain their keys; unnamed keys dropped (needs instruction.resolve)"),
    "instruction.structured": ("langzip", False, "complete deterministic zip/colon/positional/conditional lists before the strict filter (needs instruction.resolve + strict)"),
    "instruction.hazard_guard": ("bugsweep", False, "abstain on contradictory destinations, unsupported conditionals, scoped complements and misspelled control words (needs instruction.structured)"),
    "instruction.venue_colour_aliases": ("T10 coloralias", False, "venue colour words (purple/violet/lilac/lavender/magenta, orange/amber/tangerine, yellow/gold, FI/CS red+blue and inflections) named before normalize; amber = orange (needs instruction.resolve)"),
    "policy.safety_shell": ("safety", False, "act(): hold-last on exception / deadline miss, all heavy work in the look thread; plans slowed to (1-policy.safety_shell_vel_reserve) x the filter velocity only if that is > 0"),
    "policy.generation_workers": ("H7", False, "generation-tagged look workers compute on an episode snapshot; reset invalidates immediately and never waits for stale work"),
    "policy.fixed_mapping": ("organisers P0", None, "explicit sort_fixed scene metadata passed by server reset (null = shared FIXED_MAPPING; finals venue: yellow left, blue center, red right); never a language fallback"),
    "policy.chunk_ticks": ("finals rate", None, "override motion.chunk; null keeps the profile value"),
    "policy.hold_ticks": ("finals rate", None, "override motion.hold_ticks; null keeps the profile value"),
    "policy.done_confirm": ("finals done", False, "finish only after configured consecutive empty looks spanning the confirmation window; failed looks retry until the episode limit"),
    "perception.look_timeout_s": ("finals done", None, "wall-clock deadline for one asynchronous detector call; timeout is a failed look, never evidence that the scene is clear"),
    "planner.per_target_guard": ("finals done", False, "contain a planning exception or infeasible result to that target, cool it down, and continue with other detections"),
    "policy.frozen_rgb_guard": ("frozen rgb", False, "head RGB bit-identical to the previous look's although a pick started (or the episode reset) since then: the look is frozen (log [frozen_rgb]), yields no picks, homes and retries; max_consecutive frozen looks stop picking (0 pts); last hash persists across episodes"),
    "policy.frozen_rgb_max_consecutive": ("frozen rgb", 3, "frozen_rgb_guard: consecutive frozen looks after which the episode stops picking (needs policy.frozen_rgb_guard)"),
    "policy.insert_guard": ("insert", False, "rung 3 (insert_shapes task id or the hole/lid sentence): pick nothing (0 pts, never -1 on the lid)"),
    "policy.fixed_instruction_first": ("policy", False, "sort_fixed: when the instruction resolver maps EVERY venue colour to a bin (no abstain/None), the instruction beats the venue map; else venue"),
    "policy.bin_lookup": ("round 8", "colour", "colour (stock) | resolve_bin: (colour,shape)->colour->shape->default"),
    "perception.y_affine": ("Timi 23 §13", None, "y -= a*y + b (perceived - true y model, pelvis frame)"),
    "perception.self_calibration.enabled": ("Timi calib", False,
                                            "per-episode head-camera xy offset from the bin walls (depth)"),
    "perception.online_pitch_cal": ("Timi pitch", False,
                                    "hold-only pre-motion camera calibration: true=orange bin rims | table=counter plane; one immutable transform/version per frame"),
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
    "perception.glare_reject.enabled": ("bare table", False, "drop small specular highlights (V >= v_min, S <= s_max) on the counter plane, and their low-S halo, from the colour masks (glossy bare table: glare must not read as yellow/white objects)"),
    "perception.reflection_reject.enabled": ("bare table", False, "drop a colour blob directly below a stronger same-colour blob with mirrored shape, lower S and table-plane / no depth (reflection in a glossy table)"),
    "perception.local_bg.enabled": ("bare table", False, "drop a colour blob whose median H/S equals the ring of pixels around it (local table colour; no global mat / background statistic)"),
    "perception.colour_kmeans": ("EV gate", False, "per-episode k=3 HSV object-pixel clusters re-centre robust hue ranges"),
    "perception.principal_point": ("Oliver lane 3", "file", "file | center: Isaac renders cx,cy at the image centre"),
    "perception.sim_center_principal_point": ("gate fix", False, "use centred camera intrinsics only under the sim profile"),
    "perception.aim_z_frac": ("gate fix", 0.25, "no-depth centroid ray aims this object-height fraction below z_top"),
    "perception.split_blobs": ("real colour", False, "split touching same-colour objects at distance-transform peaks"),
    "perception.assoc_radius_m": ("finals grasp", 0.06, "same-colour detection-to-retry association radius across frames"),
    "perception.color_gate.live_bins": ("Felix gate", False, "bin footprint = tight confident per-bin vote; prior/disagree/missing confidence -> nominal-to-near-margin union"),
    "perception.color_gate.fk_mask": ("Felix gate", False, "mask the FK-posed arms + hands (dilated) out of the image before blob grouping"),
    "perception.color_gate.fk_dilate_px": ("New H2", 0, "FK arm/hand per-link silhouette dilation in pixels; 0 = stock sparse-mask tuning"),
    "perception.color_gate.near_margin_m": ("New H2/M1", 0.0, "low-confidence live-bin hedge towards the robot; finals uses bin.vote.max_shift_m"),
    "perception.rim_z_from_depth": ("rim z", False, "measure each bin's rim height from head depth over its wall footprint (p90 of wall points, median of the first looks, clamped to nominal +-max_delta_m, nominal on bad depth); planner rim clearance, drop height, carry height and wall collision follow it"),
    "perception.color_gate.sat_floor": ("Felix gate", False, "reject blue (purple) blobs with median HSV S < sat_floor_params.blue and yellow < .yellow (shadow / sheen); red and orange unchanged"),
    "perception.sat_floor_adaptive": ("recall P0", False, "blue/purple S floor = max(abs_min, counter-background p90 + margin), retaining the configured hue band"),
    "perception.area_floor_geometric": ("recall P0", False, "area floor from projected prism top x min_frac and ceiling from projected cube top x max_factor"),
    "perception.topface_mode": ("recall P0", "top35", "top35 | whole_blob: whole-blob hue vote when the brightest-35% subset is too small"),
    "perception.partial_ok_near_hand": ("recall P0", False, "retain sub-floor blobs touching the FK arm/hand-mask boundary as tentative; policy must re-observe before picking"),
    "perception.relook_before_done": ("recall P0", False, "before nothing-left, retract to home, wait for fresh frames, and run one final relaxed detection"),
    "perception.split_by_solidity": ("gate fix", False, "classify compact blobs as single objects and force-split merged geometry"),
    "perception.tape_value_cut": ("gate fix", False, "reject thin orange candidates without a strong S/V core"),
    "perception.bins_from_image": ("gate fix", False, "bin masks from the taped rims found in the image (no depth / untrusted bin poses)"),
    "perception.gate_below_bins": ("gate fix", False, "when live bins are untrusted, gate candidates below observed bin mass in image space"),
    "perception.reject_log": ("gate fix", False, "log and retain per-colour rejection-stage counts"),
    "perception.still_vote": ("gate fix", False, "non-blocking N-frame vote over policy-declared still frames"),
    "perception.color_gate.fk_silhouette": ("gate fix", False, "fill one projected polygon per FK arm/hand link before dilation"),
    "perception.color_gate.sat_floor_params.blue": ("gate fix", 110, "blue/purple saturation floor variant"),
    "perception.color_gate.workspace_fail_open": ("recall", False, "workspace rule (2b) fails open on the nominal-counter-z bias: while the counter height is unmeasured (no depth / plane fit fallback) a plane-method top is re-projected along its ray to workspace_assumed_counter_z before the test; measured / depth tops: unchanged"),
    "perception.hue_guard.yellow_fix": ("recall", False, "hue_guard: a red-labelled candidate whose own top-face hue is in the yellow range is re-labelled yellow before the low_s_red floor"),
    "perception.neutral_obstacles": ("Oliver neutral", False, "dark / grey / colourless non-target blobs (median V < neutral_params.v_dark 55, or S < s_grey 30; sizeable >= min_area_px 300; above the counter when depth exists) are never detections; they become Perception.neutral_obstacles (3D footprint from depth when present, else image-only and ignored) and the planner rejects hand points within margin_m 0.02 of a footprint (any phase); search budget plan_timeout_s 1.0 then fail open to the stock plan. No new colour class"),
    "perception.color_gate.enabled": ("Felix gate", False, "post-filter: area floor + ceiling by distance, workspace mask, bin footprint/rim rejects, degraded when > max_cands (no truncation), shape None for non-depth triangles"),
    "perception.depth_blob_recover": ("P0 perception", False, "recover colour-path misses from above-counter depth blobs; relaxed-HSV known colours only, unknowns abstain"),
    "perception.novel_objects": ("novel R22", False, "depth connected components above the counter -> conservative open-vocabulary object candidates"),
    "perception.inject_cam_error": ("inject", None, "TEST ONLY: [dpitch_deg, dyaw_deg, dx, dy, dz] perturbs the head "
                                                    "T_pelvis_optical perception uses (simulates a mis-calibrated head)"),
    "perception.prism_orientation": ("prism orient", False, "Detection.prism_orientation standing|lying from the footprint fill + top-band strip (+ height when it differs); lying -> shape triangle, ridge_yaw, prism_xy"),
    "grasp.lying_prism": ("prism orient", False, "lying prism (needs perception.prism_orientation): jaw closes across the ridge (hand heading = ridge +- 90 deg), grasp at the triangle-section centroid height, configured close posture"),
    "grasp.lying_prism_arm_by_reach": ("prism orient", False, "lying prism (needs grasp.lying_prism): arm + heading chosen by reachability (default arm first); no reachable ridge heading on either arm -> abstain, never the standing-prism path"),
    "grasp.yaw_align": ("Oliver #5", False, "triangle grasp: finger direction from apex yaw (needs shape+pose)"),
    "grasp.yaw_phi_deg": ("Oliver #5", STOCK, "yaw_align: finger heading = apex + phi (stock 105)"),
    "grasp.yaw_tri_shift": ("Oliver #5", STOCK, "yaw_align: grasp point to the pinch midpoint (stock true)"),
    "grasp.tri_vertex_align": ("finals cage", False,
                                 "standing triangle: align the index-middle gap with a vertex yaw"),
    "policy.time_triage": ("Oliver #3", False, "skip plans that do not fit the time left, try the next object"),
    "policy.score_order": ("r22", False, "pick by expected value per second (p_success, p_wrong, t_cost); never EV <= 0; time-budget + endgame stop"),
    "policy.max_total_attempts": ("Oliver #3", None, "global per-episode cap on started picks"),
    "policy.max_attempts": ("stock param", 2, "tries per object before it is skipped (stock 2)"),
    "policy.retry_pass": ("Oliver #3", False, "once per episode: max_attempts-skipped objects get retry_extra_attempts"),
    "policy.time_margin_s": ("Oliver #3", 0.0, "plans must end this long before the time limit"),
    "policy.episode_summary_log": ("Oliver #3", False, "one log line per episode end"),
    "policy.retry_vary_grasp": ("retry-vary", False, "retry after the object remains in the next frame with another "
                                                     "grasp (posture/heading), never an identical one"),
    "policy.max_attempts_vary": ("retry-vary", None, "with retry_vary_grasp: tries per object up to this, only with time left"),
    "policy.retry_jitter_m": ("finals grasp", 0.0, "retry grasp-point jitter magnitude; 0 disables the deterministic +/-xy pattern"),
    "policy.first_object_speed_scale": ("finals safe", None,
                                          "null | (0,1]: arm speed x this for every plan made before the first pick starts"),
    "policy.verify_placement": ("Timi #6", False, "verify each release in the bins, correct wrong-bin, drop slots"),
    "policy.verify_landing": ("r22", False, "observe each release (bin depth occupancy / colour): bounce -> re-queue with jitter + refund, wrong bin / lost -> log; in-bin exclusion; attempt regrant while time is left"),
    "policy.place_gate": ("EV gate", False, "require colour and target-bin confidence before planning a placement"),
    "policy.place_gate_colour_min": ("EV gate", 0.6, "minimum HSV/blob colour confidence (used only by place_gate)"),
    "policy.place_gate_bin_min": ("EV gate", 0.5, "minimum residual-derived bin confidence (used only by place_gate)"),
    "policy.object_time_budget_s": ("EV gate", None, "cumulative planned motion budget per tracked object; null = off"),
    "policy.grasp_diag_log": ("Felix grasp", False, "passive [grasp_diag] line per pick: stall vs closed, held"),
    "grasp.held_detector": ("real held", "stall", "stall (legacy max error) | finger_sum (bias-corrected Dex3 fingers; empty -> checked vertical escape)"),
    "grasp.release_wait_s": ("lag R44", None, "null | s: hold at least this long after the open command before the retreat (Dex3 lags 300-380 ms)"),
    "grasp.release_confirm": ("lag R49", False, "after open + wait the fingers must read an empty hand: re-open once, else home and stop picking"),
    "grasp.held_calibration.enabled": ("M3", False, "finger_sum: after home/settle, close empty hands and accept the command-measured baseline only after N low-velocity samples"),
    "held.read_through_carry": ("finals grasp", False, "sample finger_sum after lift and through carry; abort an empty post-lift read"),
    "grasp.soft_hand": ("softgrip", False, "the hand's finger servos are soft (real Dex3 kp 1.5 / sim real_soft); also implied by config profile none/real"),
    "grasp.soft_overshoot": ("softgrip", [0.0, 0.10, 0.10, 0.50, 1.00, 0.50, 1.00], "grasp.soft_squeeze: per-joint over-closure beyond the posture, LEFT DDS order, magnitudes along the closing direction, clipped to the URDF limits"),
    "grasp.soft_squeeze": ("softgrip", False, "soft hand: close to posture + overshoot, hold >= lag+settle, slow first 2 cm of lift (lift_check phase)"),
    "held.soft_detector": ("softgrip", False, "soft hand: held = finger stall further open than the empty-close baseline (one empty close per hand at episode start) by > margin; re-read after 2 cm lift (needs grasp.held_detector=finger_sum)"),
    "policy.verify_yellow_unknown": ("Felix #6 rev", False, "#6: yellow in-bin detection = unknown, never mismatch"),
    "policy.max_corrections_per_episode": ("Felix #6 rev", None, "#6: cap corrections; counter picks go first"),
    "grasp.drop_slots": ("Timi 23", False, "standalone drop slots: next object in a bin -> next slot"),
    "grasp.drop_slot_grid": ("bin capacity", False,
                              "two reach-biased rows of object-separated slots (six slots with the stock count)"),
    "grasp.drop_slot_overflow": ("Felix #6 rev", "counter", "slots used up: counter (leave it) | stack (slot 0)"),
    "grasp.drop_height_extra_m": ("finals safe", 0.0,
                                    "extra bin-release height (capped at the carry height); IK failure uses the stock target"),
    "grasp.drop_lateral_margin_m": ("finals safe", 0.0,
                                      "move the bin release toward the bin centre (y only, footprint-checked); IK failure uses the stock target"),
    "motion.carry_scale": ("Felix grasp", 1.0, "<= 1: slower lift + carry only"),
    "motion.arm_vel_cap": ("Timi vel", None, "rad/s cap on motion.arm_max_vel (arm_max_acc scaled by the same ratio); null = profile"),
    "motion.output_caps": ("M4", None, "null = legacy output; dict = final per-joint arm/hand velocity and acceleration caps, with stateful limiting on every emitted route"),
    "motion.close_hold_s": ("Felix grasp", STOCK, "hold after closing the hand (sim profile 0.25 s)"),
    "motion.lift_on_stall.enabled": ("round 9", False, "lift as soon as the fingers stall (>= close_hold_s, <= timeout)"),
    "motion.lift_on_stall.timeout_s": ("round 9", STOCK, "lift_on_stall: longest close hold (stock 1.2 s)"),
    "grasp.point_in_palm": ("pinch fix", STOCK, "object centre in the LEFT palm frame (right = [1,-1,1] mirror)"),
    "grasp.cage_mode": ("finals cage", False,
                         "three-contact cage for metric cylinders/standing triangles; cubes keep their configured grasp"),
    "grasp.close_posture_by_shape": ("hybrid", False, "per pick: demo posture only for a confident metric triangle, else the configured one"),
    "grasp.posture_by_shape_min_conf": ("hybrid", 0.5, "close_posture_by_shape: min shape confidence (0..1) for the demo posture"),
    "grasp.posture_apex_gate": ("apex gate", False, "close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)"),
    "grasp.generic_fallback": ("generic grasp", False, "unknown/low-confidence shape: geometry-only top-down grasp, colour-only bin resolution"),
    "grasp.generic": ("novel R22", False, "URDF-opening-gated top-down/handle grasps for targeted novel detections"),
    "grasp.close_posture": ("pinch fix", "closed", "grasp posture: closed | closed_tight"),
    "grasp.border_posture": ("retry-vary", False, "label None from abstain_border -> demo posture on the first try"),
    "grasp.cylinder_no_yaw_vary": ("cyl yaw", False, "retry on a detected cylinder varies the posture only, never the finger yaw (rotationally symmetric)"),
    "grasp.staggered_release": ("finals grasp", False, "release thumb and first finger before DDS slots 5/6"),
    "grasp.staggered_release_delay_s": ("finals grasp", 0.30, "delay before DDS slots 5/6 begin opening when staggered_release is on"),
    "bin.geometry_runtime": ("bin geo", False, "every bin consumer (perception masks/hulls/near_bin/in_bin/gate, planner drop/slots/collision) follows bin_geometry(): nominal + the voted shift"),
    "bin.vote.enabled": ("bin vote", False, "per-bin position from 3 votes (depth rim, tape, prior) over the first frames"),
    "bin.vote.signed_rim_fit": ("C3", False, "fit physical rim displacement with its signed +x/+y direction"),
    "bin.vote.async_extract": ("H5", False, "extract bin votes in the low-priority look worker; act only polls/holds"),
    "bin.vote.frame_transform_snapshot": ("H4", False, "one immutable camera transform per frame shared by rim and tape"),
    "hand.closed_tight": ("pinch fix", STOCK, "the closed_tight posture (LEFT DDS order, right mirrored)"),
    "hand.mirror_index_middle": ("R42a", False, "real right Dex3 DDS slots 3,4 = MIDDLE, 5,6 = INDEX (not INTERFACE order): every emitted hand target gets index == middle per hand, so the order is irrelevant"),
    "debug.log_joint_offset": ("finals diag", False,
                                 "log cmd-minus-measured during steady holds in the first second (no control change)"),
    "policy.confidence_order": ("D1/D2", False, "order by 2*p/est_time (p_colour*p_bin*p_grasp), never the hardest first"),
    "policy.object_time_cap": ("D2", False, "per-object cap (remaining-15)/left 15..35 s, T-20 s no new object, T-8 s release/retract"),
    "policy.novel_obstacles": ("novel R22", False, "untargeted novel detections are clearance obstacles for every planned hand trajectory"),
    "policy.speed_tiers": ("D3", False, "1st object 0.7 + 1.5 cm clearance, clean 0.85/0.9, failure 0.6 + centre bin only"),
    "policy.confidence_tiers": ("D4", False, "hold 3-5 s, calibration C -> CONFIDENT / DEGRADED / PRIOR-ONLY"),
    "decision.isolated_first": ("isolated-first", False, "candidates within decision.isolated_min_m of another detection go after the isolated ones"),
    "instruction.novel_objects": ("novel R22", False, "closed-vocabulary colour/shape-noun matching onto visible novel candidates; unresolved nouns abstain or use the bounded brain"),
}


def mirror_left_to_right(left7):
    """Left DDS order (thumb0,1,2, middle0,1, index0,1) -> right DDS order (thumb0,1,2, index0,1, middle0,1),
    signs flipped (joint_spec.yaml: the hands are mirrored, the same motion has opposite signs)."""
    l = np.asarray(left7, float)
    return -np.array([l[0], l[1], l[2], l[5], l[6], l[3], l[4]])


OPEN_POSTURES = ("open", "preshape", "release")


def equalize_index_middle(v14, closed):
    """hand.mirror_index_middle (R42a): both hands have the index pair (joints 3,4) and the middle pair (5,6) in
    slots 3,4 / 5,6 of their 7-block, in an order that differs between sim docs and the real robot. Give both pairs
    the same value, joint by joint, so the order cannot matter: the larger |q| (more closed) for grasp postures
    (never under-grip), the smaller |q| (more open) for open/release and for emission-time repairs (never crush /
    never keep fingers curled over the object)."""
    v = np.array(v14, float)
    pick = np.maximum if closed else np.minimum
    for o in (0, 7):
        a, b = v[o + 3:o + 5].copy(), v[o + 5:o + 7].copy()
        m = np.where(pick(np.abs(a), np.abs(b)) == np.abs(a), a, b)
        v[o + 3:o + 5] = v[o + 5:o + 7] = m
    return v


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
    _apply_output_caps(cfg)
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


def _apply_output_caps(cfg):
    """Clamp planner limits to motion.output_caps; the policy also enforces these at its final output boundary."""
    mo = cfg.get("motion") or {}
    caps = mo.get("output_caps")
    if caps is None:
        return
    mo["arm_max_vel"] = min(float(mo["arm_max_vel"]), float(caps["arm_vel"]))
    mo["arm_max_acc"] = min(float(mo["arm_max_acc"]), float(caps["arm_acc"]))
    mo["hand_max_vel"] = min(float(mo["hand_max_vel"]), float(caps["hand_vel"]))
    mo["hand_max_acc"] = float(caps["hand_acc"])


def check_flag_values(cfg):
    """Hard limits on flag values (fail at load = at docker build): carry_scale in (0, 1], known overflow modes."""
    for k, (lo, hi) in LIMITS.items():
        v = float(_get(cfg, k))
        if not lo < v <= hi:
            raise ValueError(f"{k} = {v} outside ({lo}, {hi}]: never faster than the stock profile")
    for k, ok in CHOICES.items():
        if _get(cfg, k) not in ok:
            raise ValueError(f"{k} = {_get(cfg, k)!r}, want one of {ok}")
    for k in ("brain.timeout_s", "brain.request_timeout_s"):
        v = _get(cfg, k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{k} = {v!r}: want a finite number > 0")
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
    for key in ("safety.nan_guard", "policy.generation_workers", "grasp.held_calibration.enabled",
                "grasp.drop_slot_grid"):
        if not isinstance(_get(cfg, key), bool):
            raise ValueError(f"{key}: want true or false")
    caps = _get(cfg, "motion.output_caps")
    if caps is not None:
        keys = {"arm_vel", "arm_acc", "hand_vel", "hand_acc"}
        if not isinstance(caps, dict) or set(caps) != keys:
            raise ValueError(f"motion.output_caps = {caps!r}: want null or exactly {sorted(keys)}")
        for key in sorted(keys):
            value = caps[key]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"motion.output_caps.{key} = {value!r}: want a finite positive number")
        hard = {"arm_vel": 0.85, "arm_acc": 4.0, "hand_vel": 1.5}
        for key, maximum in hard.items():
            if float(caps[key]) > maximum:
                raise ValueError(f"motion.output_caps.{key} = {caps[key]} exceeds finals cap {maximum}")
    p = cfg.get("perception") or {}
    for key in ("sim_center_principal_point", "split_by_solidity", "tape_value_cut", "gate_below_bins",
                "bins_from_image", "reject_log", "still_vote"):
        if not isinstance(p.get(key), bool):
            raise ValueError(f"perception.{key}: want true or false")
    aim = p.get("aim_z_frac", 0.25)
    if isinstance(aim, bool) or not isinstance(aim, (int, float)) or not np.isfinite(aim) or not 0.0 <= aim <= 1.0:
        raise ValueError(f"perception.aim_z_frac = {aim!r}: want a finite fraction in [0, 1]")
    sv = p.get("still_vote_params") or {}
    n, k = sv.get("n_frames", 3), sv.get("min_hits", 2)
    if (not isinstance(n, int) or isinstance(n, bool) or not 3 <= n <= 5 or
            not isinstance(k, int) or isinstance(k, bool) or not 1 <= k <= n):
        raise ValueError("perception.still_vote_params needs integer 3 <= n_frames <= 5 and 1 <= min_hits <= n_frames")
    match = sv.get("match_m", 0.02)
    if isinstance(match, bool) or not isinstance(match, (int, float)) or not np.isfinite(match) or match <= 0:
        raise ValueError("perception.still_vote_params.match_m must be finite and positive")
    sp = p.get("split_by_solidity_params") or {}
    area = sp.get("area_px", [520, 15500])
    if not (isinstance(area, (list, tuple)) and len(area) == 2 and
            all(isinstance(x, int) and not isinstance(x, bool) and x > 0 for x in area) and area[0] <= area[1]):
        raise ValueError("perception.split_by_solidity_params.area_px needs two increasing positive integers")
    if not 0.0 <= float(sp.get("solidity", 0.82)) <= 1.0 or float(sp.get("max_aspect", 2.0)) < 1.0:
        raise ValueError("invalid split_by_solidity solidity/aspect")
    tp = p.get("tape_value_cut_params") or {}
    if any(not 0.0 <= float(tp.get(key, default)) <= 255.0 for key, default in (("min_s", 140), ("min_v", 140))):
        raise ValueError("tape_value_cut S/V thresholds must be in [0, 255]")
    if not 0.0 <= float(tp.get("min_core_frac", 0.2)) <= 1.0 or float(tp.get("min_aspect", 3.0)) < 1.0:
        raise ValueError("invalid tape_value_cut core/aspect")
    for k in ("brain.timeout_s", "brain.request_timeout_s"):
        v = _get(cfg, k)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v <= 0.0:
            raise ValueError(f"{k} = {v!r}: want a finite float > 0")
    for k in ("policy.chunk_ticks", "policy.hold_ticks"):
        v = _get(cfg, k)
        if v is not None and (not isinstance(v, int) or isinstance(v, bool) or v < 1):
            raise ValueError(f"{k} = {v!r}: want null or an int >= 1")
    if not isinstance(_get(cfg, "policy.done_confirm"), bool):
        raise ValueError("policy.done_confirm: want true or false")
    looks = _get(cfg, "policy.done_confirm_looks")
    if not isinstance(looks, int) or isinstance(looks, bool) or looks < 2:
        raise ValueError(f"policy.done_confirm_looks = {looks!r}: want an int >= 2")
    span = _get(cfg, "policy.done_confirm_span_s")
    if isinstance(span, bool) or not isinstance(span, (int, float)) or not np.isfinite(span) or span < 0.0:
        raise ValueError(f"policy.done_confirm_span_s = {span!r}: want finite seconds >= 0")
    deadline = _get(cfg, "perception.look_timeout_s")
    if deadline is not None and (isinstance(deadline, bool) or not isinstance(deadline, (int, float))
                                 or not np.isfinite(deadline) or deadline <= 0.0):
        raise ValueError(f"perception.look_timeout_s = {deadline!r}: want null or finite seconds > 0")
    max_inflight = _get(cfg, "perception.detect_max_inflight")
    if not isinstance(max_inflight, int) or isinstance(max_inflight, bool) or max_inflight < 1:
        raise ValueError(f"perception.detect_max_inflight = {max_inflight!r}: want an int >= 1")
    if not isinstance(_get(cfg, "planner.per_target_guard"), bool):
        raise ValueError("planner.per_target_guard: want true or false")
    cooldown = _get(cfg, "planner.per_target_cooldown_s")
    if isinstance(cooldown, bool) or not isinstance(cooldown, (int, float)) \
            or not np.isfinite(cooldown) or cooldown <= 0.0:
        raise ValueError(f"planner.per_target_cooldown_s = {cooldown!r}: want finite seconds > 0")
    if not isinstance(_get(cfg, "calib.wide_counter_window"), bool):
        raise ValueError("calib.wide_counter_window: want true or false")
    bp = p.get("bins_from_image_params") or {}
    if not isinstance(bp.get("hint", False), bool) or not isinstance(bp.get("with_depth", True), bool):
        raise ValueError("perception.bins_from_image_params hint / with_depth: want true or false")
    for key in ("hint_max_m", "hint_max_resid_px", "crosscheck_px", "rim_z_m"):
        if key in bp and (isinstance(bp[key], bool) or not np.isfinite(float(bp[key]))):
            raise ValueError(f"perception.bins_from_image_params.{key}: want a finite number")
    for k in ("policy.place_gate_colour_min", "policy.place_gate_bin_min"):
        v = float(_get(cfg, k))
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"{k} = {v}: want a confidence in [0, 1]")
    budget = _get(cfg, "policy.object_time_budget_s")
    if budget is not None:
        budget = float(budget)
        if not np.isfinite(budget) or budget <= 0.0:
            raise ValueError(f"policy.object_time_budget_s = {budget}: want null or a finite positive number of seconds")
    zp = _get(cfg, "scene.counter_z_prior")
    if zp is not None and not (isinstance(zp, (int, float)) and not isinstance(zp, bool) and 0.0 <= zp <= 0.3):
        raise ValueError(f"scene.counter_z_prior = {zp!r}: want null or a number in [0, 0.3] (m, pelvis frame)")
    ice = _get(cfg, "perception.inject_cam_error")
    if ice is not None and not (isinstance(ice, (list, tuple)) and len(ice) == 5
                                and all(isinstance(x, (int, float)) for x in ice)):
        raise ValueError(f"perception.inject_cam_error = {ice!r}: want null or [dpitch_deg, dyaw_deg, dx, dy, dz]")
    first_scale = _get(cfg, "policy.first_object_speed_scale")
    if first_scale is not None and not 0.0 < float(first_scale) <= 1.0:
        raise ValueError("policy.first_object_speed_scale must be null or in (0, 1]")
    for k in ("grasp.drop_height_extra_m", "grasp.drop_lateral_margin_m", "planner.bin_wall_extra_m"):
        v = float(_get(cfg, k))
        if not np.isfinite(v) or v < 0.0:
            raise ValueError(f"{k} must be a finite non-negative distance")
    distances = {"policy.retry_jitter_m": 0.02, "perception.assoc_radius_m": 0.25}
    for k, hi in distances.items():
        v = float(_get(cfg, k))
        if not np.isfinite(v) or not 0.0 <= v <= hi:
            raise ValueError(f"{k} must be finite and in [0, {hi}]")
    delay = float(_get(cfg, "grasp.staggered_release_delay_s"))
    if not np.isfinite(delay) or not 0.0 <= delay <= 1.0:
        raise ValueError("grasp.staggered_release_delay_s must be finite and in [0, 1]")
    for k in ("grasp.staggered_release", "held.read_through_carry", "grasp.soft_hand", "grasp.soft_squeeze",
              "held.soft_detector"):
        if not isinstance(_get(cfg, k), bool):
            raise ValueError(f"{k}: want true or false")
    _check_soft(cfg)
    pol = cfg["policy"]
    if first_scale is not None and pol.get("safety_shell") and float(pol.get("safety_shell_vel_reserve", 0.0)) > 0.0:
        # the shell wraps planner.plan with closures bound to the stock planner: the first-pick scale would not reach it
        raise ValueError("policy.first_object_speed_scale cannot be combined with policy.safety_shell_vel_reserve > 0")
    _check_color_gate(cfg)
    hg = cfg.get("perception", {}).get("hue_guard") or {}
    if not 0.0 <= float(hg.get("red_min_s", 110.0)) <= 255.0:
        raise ValueError(f"perception.hue_guard.red_min_s = {hg.get('red_min_s')}: want 0-255")
    pc = cfg.get("perception") or {}
    if not isinstance(pc.get("topface_min_px", 1024), int) or isinstance(pc.get("topface_min_px", 1024), bool) \
            or int(pc.get("topface_min_px", 1024)) < 1:
        raise ValueError("perception.topface_min_px: want an int >= 1")
    sa = pc.get("sat_floor_adaptive_params") or {}
    for key in ("abs_min", "bg_percentile", "margin"):
        v = sa.get(key)
        if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v):
            raise ValueError(f"perception.sat_floor_adaptive_params.{key}: want a finite number")
    if not 0.0 <= float(sa.get("abs_min", 45)) <= 255.0 or not 0.0 <= float(sa.get("bg_percentile", 90)) <= 100.0 \
            or float(sa.get("margin", 10)) < 0.0:
        raise ValueError("adaptive saturation needs abs_min 0..255, bg_percentile 0..100, margin >= 0")
    ag = pc.get("area_floor_geometric_params") or {}
    if not 0.0 < float(ag.get("min_frac", 0.35)) <= 1.0 or float(ag.get("max_factor", 2.0)) < 1.0:
        raise ValueError("geometric area needs min_frac in (0,1] and max_factor >= 1")
    ph = pc.get("partial_ok_near_hand_params") or {}
    for key in ("min_area_px", "boundary_px", "max_relooks"):
        v = ph.get(key)
        if not isinstance(v, int) or isinstance(v, bool) or v < 1:
            raise ValueError(f"perception.partial_ok_near_hand_params.{key}: want an int >= 1")
    if not isinstance(pc.get("relook_fresh_frames", 2), int) or isinstance(pc.get("relook_fresh_frames", 2), bool) \
            or int(pc.get("relook_fresh_frames", 2)) < 1:
        raise ValueError("perception.relook_fresh_frames: want an int >= 1")
    _check_depth_blob_recover(cfg)
    g = cfg.get("grasp") or {}
    hc = g.get("held_calibration") or {}
    nsteady = hc.get("steady_samples", 4)
    if not isinstance(nsteady, int) or isinstance(nsteady, bool) or nsteady <= 0:
        raise ValueError("grasp.held_calibration.steady_samples must be a positive int")
    vmax, held, timeout = (float(hc.get("max_velocity_rad_s", 0.08)),
                           float(hc.get("hold_after_settle_s", 0.4)), float(hc.get("timeout_s", 5.0)))
    if not (np.isfinite(vmax) and vmax > 0.0 and np.isfinite(held) and held >= 0.4
            and np.isfinite(timeout) and timeout > held):
        raise ValueError("grasp.held_calibration needs velocity > 0, hold >= 0.4 s, timeout > hold")
    if not 0.0 < float(g.get("lying_yaw_tol_deg", 10.0)) <= 45.0:
        raise ValueError("grasp.lying_yaw_tol_deg must be in (0, 45]")
    pp = (cfg.get("perception") or {}).get("prism_orientation_params") or {}
    if not 0.5 < float(pp.get("fill_min", 0.75)) < 1.0:
        raise ValueError("perception.prism_orientation_params.fill_min must be in (0.5, 1)")
    if pp.get("length_m") is not None and not 0.01 <= float(pp["length_m"]) <= 0.2:
        raise ValueError("perception.prism_orientation_params.length_m must be null or 0.01-0.2 m")
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
    thumb_extra = g.get("cage_thumb_extra_rad") or {}
    if set(thumb_extra) != {"cube", "cylinder", "triangle"}:
        raise ValueError("grasp.cage_thumb_extra_rad must contain cube/cylinder/triangle")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0.0
           for v in thumb_extra.values()):
        raise ValueError("grasp.cage_thumb_extra_rad values must be finite non-negative radians")
    if not 0.0 <= float(g.get("cage_shape_min_conf", 0.5)) <= 1.0:
        raise ValueError("grasp.cage_shape_min_conf must be in [0, 1]")
    if not 0.0 <= float(g.get("tri_vertex_phi_deg", 90.0)) <= 180.0:
        raise ValueError("grasp.tri_vertex_phi_deg must be in [0, 180]")
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
    for k in ("area_frac", "min_px_no_z", "edge_m", "near_margin_m", "fk_dilate_px"):
        num(k)
    dilate = g.get("fk_dilate_px")
    if isinstance(dilate, bool) or not isinstance(dilate, int) or dilate < 0:
        raise ValueError(f"perception.color_gate.fk_dilate_px = {dilate!r}: want an int >= 0")
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
    if not isinstance(g.get("fk_dilate_px"), int) or isinstance(g.get("fk_dilate_px"), bool):
        raise ValueError("perception.color_gate.fk_dilate_px needs an integer")
    for key, value in (g.get("sat_floor_params") or {}).items():
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0.0 <= float(value) <= 255.0:
            raise ValueError(f"perception.color_gate.sat_floor_params.{key} must be in [0, 255]")


EXPECTED_FLAGS = OVERLAY_DIR / "EXPECTED_FLAGS.json"  # written by tools/promote.sh; baked into the image with overlays/


def flags_dump(cfg):
    """{flag: value} for every FLAGS key, JSON-normalised (tuples -> lists)."""
    import json
    return {k: json.loads(json.dumps(_get(cfg, k))) for k in FLAGS}


def check_expected_flags(cfg, path=None):
    """Diff lines (empty = ok) between cfg's flags and EXPECTED_FLAGS.json. No file = no check (returns [])."""
    import json
    path = Path(path or globals()["EXPECTED_FLAGS"])
    if not path.exists():
        return []
    exp = json.loads(path.read_text())["flags"]
    got = flags_dump(cfg)
    return [f"{k}: expected={exp.get(k, '<missing>')!r} got={got.get(k, '<missing>')!r}"
            for k in sorted(set(exp) | set(got)) if exp.get(k, "<missing>") != got.get(k, "<missing>")]


def _check_depth_blob_recover(cfg):
    """Depth-recovery tuning is validated at load time, never in the look worker."""
    p = cfg.get("perception") or {}
    if not p.get("depth_blob_recover", False):
        return
    d = p.get("depth_blob_recover_params") or {}

    def number(key, lo, hi):
        value = d.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not np.isfinite(value) or not lo <= value <= hi:
            raise ValueError(f"perception.depth_blob_recover_params.{key} = {value!r}: want a finite number in [{lo}, {hi}]")

    for key, lo, hi in (("min_height_m", 0.012, 0.15), ("max_height_m", 0.012, 0.15),
                        ("sat_floor", 0, 255), ("min_colour_conf", 0, 1), ("min_colour_margin", 0, 1),
                        ("match_iou", 0, 1), ("match_centroid_px", 0, 640),
                        ("plane_z_window", 0.01, 0.5), ("plane_inlier_m", 0.001, 0.05),
                        ("shape_flatness_m", 0.001, 0.05), ("shape_prism_aspect", 1, 5)):
        number(key, lo, hi)
    if float(d["min_height_m"]) > float(d["max_height_m"]):
        raise ValueError("depth blob recovery needs min_height_m <= max_height_m")
    for key, lo in (("stride", 2), ("min_blob_cells", 1), ("max_blob_cells", 1),
                    ("min_plane_points", 3), ("min_colour_px", 1)):
        value = d.get(key)
        if isinstance(value, bool) or not isinstance(value, int) or value < lo:
            raise ValueError(f"perception.depth_blob_recover_params.{key} = {value!r}: want an int >= {lo}")
    if int(d["min_blob_cells"]) > int(d["max_blob_cells"]):
        raise ValueError("depth blob recovery needs min_blob_cells <= max_blob_cells")
    if not isinstance(d.get("shape_hint"), bool):
        raise ValueError("perception.depth_blob_recover_params.shape_hint: want true or false")


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


SOFT_PARAMS = {  # key -> (lo, hi) inclusive; all finite
    "grasp.soft_overshoot": (0.0, 2.0),
    "grasp.soft_limit_margin_rad": (0.02, 0.2),
    "grasp.soft_lag_s": (0.0, 2.0),
    "grasp.soft_settle_s": (0.0, 2.0),
    "grasp.soft_lift_check_m": (0.005, 0.06),
    "grasp.soft_lift_check_scale": (0.05, 1.0),
    "held.soft_margin_rad": (0.0, 1.5),
    "held.soft_lift_margin_rad": (0.0, 1.5),
    "held.soft_read_after_s": (0.0, 2.0),
    "held.soft_prior_ratio": (0.0, 1.0),
}


def _check_soft(cfg):
    for k, (lo, hi) in SOFT_PARAMS.items():
        vals = np.atleast_1d(np.asarray(_get(cfg, k), float))
        if not (np.all(np.isfinite(vals)) and np.all(vals >= lo) and np.all(vals <= hi)):
            raise ValueError(f"{k} = {_get(cfg, k)!r}: want finite values in [{lo}, {hi}]")
    if _get(cfg, "held.soft_detector") and _get(cfg, "grasp.held_detector") != "finger_sum":
        raise ValueError("held.soft_detector needs grasp.held_detector = finger_sum")


def soft_hand_active(cfg):
    """grasp.soft_squeeze ON and the hand is soft: hand.soft, or the config profile is none / real."""
    return bool(cfg["grasp"].get("soft_squeeze", False)
                and (cfg["grasp"].get("soft_hand", False) or str(cfg.get("profile", "")) in ("none", "real")))


def soft_detector_active(cfg):
    """held.soft_detector ON and the hand is soft (same rule as soft_hand_active, without needing soft_squeeze)."""
    return bool(cfg["held"].get("soft_detector", False)
                and (cfg["grasp"].get("soft_hand", False) or str(cfg.get("profile", "")) in ("none", "real")))


def soft_overclose(cfg, spec, base14, rest14):
    """The 14-vector `base14` (a closed posture) over-closed per joint by grasp.soft_overshoot, along the closing
    direction sign(base - rest) (rest = the preshape the hand closes from), never beyond the URDF limits shrunk by
    grasp.soft_limit_margin_rad (>= the safety filter's 0.02 hand margin, so the filter never clips it). Joints that do
    not move between rest and base (thumb0) get no overshoot. Right overshoot = the left vector in right DDS order."""
    ov_l = np.asarray(cfg["grasp"]["soft_overshoot"], float)
    ov = np.r_[ov_l, np.abs(mirror_left_to_right(ov_l))]
    base, rest = np.asarray(base14, float), np.asarray(rest14, float)
    m = float(cfg["grasp"]["soft_limit_margin_rad"])
    lo, hi = spec.hand_lower + m, spec.hand_upper - m
    d = np.sign(base - rest)
    room = np.where(d > 0, hi - base, np.where(d < 0, base - lo, 0.0))
    return base + d * np.minimum(ov, np.maximum(room, 0.0))


def hand_postures(cfg, spec: JointSpec | None = None):
    """{name: (14,) hand vector} for open / preshape / closed (+ extra postures), validated against the URDF limits."""
    spec = spec or JointSpec()
    h = cfg["hand"]
    missing = [n for n in POSTURES if n not in h]
    if missing:
        raise ValueError(f"hand postures missing: {missing}")
    right = h.get("right") or {}
    out = {}
    mirror = bool(h.get("mirror_index_middle", False))
    for name in [k for k in h if k not in ("right", "mirror_index_middle")]:  # open, preshape, closed (required) + extras
        left = np.asarray(h[name], float)
        r = np.asarray(right[name], float) if name in right else mirror_left_to_right(left)
        v = np.r_[left, r]
        if mirror:
            v = equalize_index_middle(v, closed=name not in OPEN_POSTURES)
        bad = [spec.hand_names[i] for i in np.flatnonzero((v < spec.hand_lower - 1e-9) | (v > spec.hand_upper + 1e-9))]
        if bad:
            raise ValueError(f"hand posture {name!r} outside joint_spec limits: {bad}")
        out[name] = v
    if cfg.get("grasp", {}).get("cage_mode", False):
        # These are commands, not measured contact poses. Start with the FK-derived finger posture, then put thumb_1
        # at the requested flexion past our existing closed posture. The soft hand stops on contact; clip both hands
        # to the same URDF-backed limits as the safety filter.
        g = cfg["grasp"]
        base = np.asarray(g["cage_finger_posture"], float)
        closed = np.asarray(h["closed"], float)
        for shape, extra in g["cage_thumb_extra_rad"].items():
            if float(extra) <= 0.0:  # cube=0 by default: no alternate cube posture or planning path
                continue
            left = base.copy()
            left[1] = closed[1] + float(extra)
            left = np.clip(left, spec.hand_lower[:7], spec.hand_upper[:7])
            right = np.clip(mirror_left_to_right(left), spec.hand_lower[7:], spec.hand_upper[7:])
            out[f"cage_{shape}"] = np.r_[left, right]
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
