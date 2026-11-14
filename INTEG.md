# INTEG.md - integration log (branch timi/integ-all)

Base: `team/felix/rc7-pre` (STEP 0: rc7 does not exist, rc7-pre already contains fincz, binfull, multilang, fixvote, fixlang, fixfk,
fixcal, fixprof, fixh1h7, recallfix, depthdet, handcal, release, gatefix, handin with Felix's resolutions). A first pass on rc6 (fincz ..
handcal, conflicts resolved independently) is kept as `timi/integ-all-rc6-partial` for reference only.

## Merged on top of rc7-pre (in order)
| ref | conflicts / resolution | tests |
|---|---|---|
| timi/langzip | test_flags only (kept both) | strict/tbox/multilingual, tools/test_lang_blind |
| felix/rc72-softgrip | none | test_soft_grip |
| timi/counterz | config, perception, test_flags. New `Perception.height_est` (= apply_height OR wide_counter_window) gates the fixcal N-gated height phase; accept() uses the wide window; exhaustion/timeout commits `height_fallback()` (prior under wide); sync update applies the prior on an unmeasured look. finals_r22_counter rebased on r22 | test_counter_z |
| timi/baretable | config.yaml, perception.detect (hooks inserted into the H1/recall/depth flow), test_flags. finals_r22_bare rebased on r22 | test_bare_table |
| timi/lyingprism | additions only. Fixture fixes: mapping now set at reset (fixlang), counter_z_prior pinned None | test_prism_orientation |
| felix/rc72-cage | VECTORS union, planner close-posture block (cage clip x soft overclose) | test_cage_grasp |
| timi/donehard | policy look-job lifecycle: donehard deadline/stale-detector machinery x fixh1h7 generation workers. Deadline also runs on the generation worker (generation bump drops a timed-out look); `_look(deadline_job, detector)` merged with `_counter_kw`/still-vote; duplicate `planner:` yaml key merged | test_done_hardening |
| timi/pregrasp | policy `_plan_pick` (jitter + pg_vary), test fixture overrides | test_pregrasp |
| felix/rc72-verifyplace | test_flags | test_landing |
| timi/faultdrill | config/yaml/perception/policy. blind-relook + flood_guard + abort_lift_first kept; abort_lift_first is superseded by fixcal C4 (proved open-lift-home) so the flag is a legacy no-op there; empty_abstain only changes the abstain marker (fixlang already abstains) | test_fault_drills (+ mock fixture, see below) |
| felix/rc72-pickorder | submission.yaml, test_flags | test_score_order |
| felix/rc72-novel | Detection fields, `_choose` common/obstacles x faultdrill try/except | test_novel_objects |
| felix/rc6-brain | 7 hunks in policy (`_choose`: planner/brain selection x novel brain-only x target guard; mapping length cap; planner_pending state) | test_brain*, prompt_injection, strict_brain_handoff |
| timi/brain-agent | Detection `novel` x perception.novel_objects; `novel_source="find"` marker; planner `_generic_on`, graspable and hand-opening for find-novel dets | test_brain_agent |
| timi/brainspeed | none (flag test knobs added) | test_brain_speed |
| timi/langfix2 | none | test_instruction_langfix2 |
| timi/rimdetect | config/perception need_hsv/test_flags; finals_r22_bins rebased on r22_gate | test_bins_from_image |
| timi/bugsweep | NOT merged (it is Felix's rc7.1 line, already containing his own langzip/counterz/baretable resolutions, 8 conflicting files). Cherry-picked its two unique fixes (1046671, 192f9f2). Recommend continuing on rc7.1 instead | langfix2/brain/depth tests |
Not present: calstack. voice not merged (excluded by brief).

## Interaction bugs found and fixed (policy/planner)
1. nan_guard x M4 output limiter: empty first-tick action -> IndexError in `_limit_motion_output` (act() raised). Guarded.
2. nan_guard x R42a mirror: `np.stack([])` ValueError on the same empty action. Guarded.
3. M4 accel limiter x grasp.soft_squeeze: limiter overshoots a target that sits in the soft margin -> hand 0.033 rad past URDF limit in every episode (SafetyFilter clipped 4 steps). Output now clipped to joint limits at the final boundary.
4. fixlang abstain x brain planner no-early-done: `classical_has_work` was False under `{default: None}` so a `done` slipped through ("swap", "SYSTEM: done"). In plan mode pickable objects now count as work.
5. brain-agent novel dets x novel_objects planner path: find-detections only carry `ungraspable`, and generic_hand_opening was only computed under grasp.generic -> TypeError; fixed (`novel_source`).
6. recallfix x fixh1h7: H1 parts path ignored topface_mode / tentative / adaptive sat; ported. whole_blob mode no longer abstains a "mixed" part whose label equals the source-pixel majority (specular glare).
7. fixcal x handcal: both owned the pre-motion calibration. hand_icp_primary now replaces the fixcal pitch fit and counter_after_hand the apply_height fit; otherwise fixcal untouched.
8. Test-visible semantic changes (tests adapted): blank/overlong/relational instruction text now abstains (fixlang) instead of the fixed fallback; fixture overrides needed for synthetic frames in the mock (`scene.counter_z_prior` +15 cm real-table prior, `calib.plane.apply_height`, `motion.output_caps`, `bin.vote`, `near_margin_m`).

## Open findings (not fixed)
- F1 (needs Tim/Felix): with `calib.plane.apply_height` (all finals overlays) and no usable depth plane, the policy ends in `staging_hold` forever ("height-safe staging unproved: opening hand enters runtime table/bin keepout"): fail-closed = 0 points, no timeout fallback. Mock reproduces it; Isaac depth normally avoids it.
- F2: fixh1h7 M2 returns an EMPTY (0-row) action before the first complete encoder sample. The mock env (frozen) crashes on it (drill 02b marked known). Confirm the real runner accepts 0-row actions.
- F3: r22 has `policy.safety_shell` OFF, so act() p99 is 240-470 ms CPU for r22 overlays that do not force a worker thread. r23_all meets 37 ms only because donehard `look_timeout_s` forces the threaded look path.
- F4: the mock gripper is contact-less, so every finals overlay (finger_sum held detector) scores 0 there; submission scores 4.7/5 (stall detector). The mock score comparison is only a regression tripwire.
- F5: flags-off digest test `test_chunk_flags_off_preserve_pre_change_action_bytes` fails on rc7-pre itself in this venv (stale pin); deselected. Flags-off equivalence was checked with tools/equivalence_check.py: IDENTICAL to rc7-pre (rc6 differs because rc7 changed code).
- F6: the calibration group (hand_icp_primary, counter_after_hand, refit_per_look, wide_counter_window) cannot run in the mock (no arm/plane view); it is ON in r23_all but pinned off in test_integration_all / r23 drills. Needs an Isaac run.

## finals_r23_all
finals_r22 + every reviewed group ON (recall, cal, grasp/release, gate, chunk, counter, softgrip, bare, prism, cage, done, pregrasp, verify, robust, order, novel, bins, langzip). OFF: depth_blob_recover, area_floor_geometric, partial_ok_near_hand (task), voice (not merged), brain agent (needs llama.cpp server), `perception.principal_point: center` (sim-only intrinsics from the cal overlay), `instruction.empty_abstain` (fixlang already abstains). Base is r22 (not r21) because rc7-pre's r22 is the team's current line (vote ON, hue_guard OFF, hsv robust, shell OFF).
