# Architecture: the classical sorter and its release discipline

Junction 2026, "Aaltonomous Humanoid". This describes the integration tree as it is now, with `finals_r22` as the
candidate for the finals. The baseline is the earlier `docs/ARCHITECTURE.md` (rc6). Where this text and the code
disagree, the code wins. Each claim names the file or function it was checked against.

## 1. Overview

A Unitree G1 has to sort objects into three bins (left, centre, right), either by colour or by a typed instruction.
The score is +2 for a correct language task, +1 for a correct sort without one, and -1 for a wrong drop. A
wrong drop therefore costs more than a missed one, so the whole design is built to hold or abstain rather than guess.
No learned policy sits on the control path. Perception is HSV masks plus depth, planning is inverse kinematics with
time-scaled trajectories, and the instruction is parsed by a frozen regex resolver. A local vision-language model
(the "brain") is optional. It may only fill gaps the resolver leaves open, and it is off in the submission image.
Two processes split the work. The runner owns the robot or simulator, the 30 Hz loop and the `SafetyFilter`. The
policy container is the image we submit. It serves joint targets over a websocket and never talks to the robot.

## 2. Pipeline

```
 head RGB-D, arm_q, hand_q, instruction
                 |
                 v
 +--------------------------------------------------------------+
 | PERCEPTION   Perception.detect (perception.py)               |
 |  HSV masks (hsv_set stock|robust|robust_v2)                  |
 |  depth -> pelvis-frame xyz, top-face shape (shape_id.py)     |
 |  split_blobs, hue_guard (red|yellow boundary -> abstain)     |
 |  colour_gate: area, workspace, bin footprint, FK arm mask    |
 |  counter-plane height (apply_height), pitch, bin vote        |
 +------------------------------+-------------------------------+
                                v
 +--------------------------------------------------------------+
 | BIN VOTE     bin_vote.vote_bins: rim, tape and prior votes   |
 +------------------------------+-------------------------------+
                                v
 +--------------------------------------------------------------+
 | INSTRUCTION  _instruction_mapping (policy.py)                |
 |  normalize.normalize -> instruction.resolve (frozen)         |
 |  strict_mapping abstains; insert_guard abstains everything   |
 |  fixed_mapping only when no instruction (sim or real map)    |
 +------------------------------+-------------------------------+
                                v
 +--------------------------------------------------------------+
 | DECISION     _choose / _choose_retry / _ordered_cands        |
 |  lookup_bin / lookup_detection_bin: None = no pick           |
 |  brain (brain/) only at after_look / after_fail / after_place|
 |  never changes a known bin; classical decides on any miss    |
 +------------------------------+-------------------------------+
                                v
 +--------------------------------------------------------------+
 | PLANNER      Planner.plan / plan_to_counter (planner.py)     |
 |  grasp variants, carry, drop slots, staggered release,       |
 |  IK via ArmKinematics (kinematics.py, Pinocchio, G1 URDF)    |
 +------------------------------+-------------------------------+
                                v
 +--------------------------------------------------------------+
 | OUTPUT       policy._nan_guard -> policy._act_shell          |
 |  MotionOutputLimiter (motion.output_caps) at the boundary    |
 +------------------------------+-------------------------------+
                                v
 +--------------------------------------------------------------+
 | SAFETY       SafetyFilter (interface/safety.py): per-tick    |
 |  velocity bound, workspace box, hold 0.2 s, stop 1.0 s       |
 +------------------------------+-------------------------------+
                                v
                         robot or simulator
```

Stage notes, in call order:

- **Instruction.** `policy._instruction_mapping` runs once per episode. The text is first capped by
  `policy._cap_text` (`instruction.max_chars`), then normalised (`normalize.normalize`, flag `instruction.normalize`),
  then resolved (`instruction.resolve`). `classical/instruction.py` is byte-identical to `policy/instruction.py` on
  the language lane, which is the frozen resolver. With `instruction.strict`, `normalize.strict_mapping` abstains on
  any unparsed or ambiguous key and never falls back. `policy.fixed_mapping` is used only when there is no instruction.
- **Perception.** `Perception.detect` builds the detections. Colour gating is the post-filter `_color_gate`
  (`perception.color_gate.*`). The hue guard is `_hue_guard` with `hue_guard.judge`. The FK arm mask uses the posed arms
  and hands. Shape comes from `classify_shape`.
- **Bin vote.** `bin_vote.vote_bins` combines the rim, tape and prior estimates (`combine`, `resolve`). Every bin consumer
  then reads `bin_geometry()` when `bin.geometry_runtime` is on.
- **Decision.** `policy._choose` walks `_ordered_cands`. The colour lookup is `planner.lookup_bin` (planner.py:32), or
  `lookup_detection_bin` (planner.py:42) with `policy.bin_lookup: resolve_bin`. That calls `scene.resolve_bin`
  (scene/scene.py:43), which tries (colour, shape), then colour, then shape, then `default`. A detection with no shape
  skips the shape rung, so it gets colour, then default. A generic grasp with no width gets no pick. `None` means no
  pick. `decision.py` holds the tiered ordering, which is off.
- **Planner.** `Planner.plan` chooses the grasp and carry, and `Planner.plan_to_counter` plans the drop. Release can be
  staggered (`staggered_release_ramp`, flag `grasp.staggered_release`). The ramp function is `ramp`, and `time_scale`
  stretches the trajectory to the velocity and acceleration limits.
- **Output.** `MotionOutputLimiter` (built in `policy.py` from `motion.output_caps`) limits every route, including home
  and abort. `policy._act_shell` (flag `policy.safety_shell`) holds the last target on an exception or a deadline miss.
- **Safety.** `SafetyFilter` in `interface/safety.py` is the last word and always on. It bounds each joint's change per
  tick, checks the workspace box, holds the last command after 0.2 s without an action, and latches a safe stop after 1.0 s.

## 3. Calibration

Calibration runs before motion, inside the policy state machine. It never runs a fit inside `act()`.

- **Preflight.** When `perception.online_pitch_cal` is set, the policy enters state `calibrate`
  (`policy.py`, around line 318). `_preflight_calibration` holds the measured joints while the pitch worker and the
  height job run. Each has a timeout. A timeout falls back to nominal values and logs the reason.
- **Counter height (`calib.plane.apply_height`).** The height comes from a fit of the counter plane in head depth.
  `Perception.accept_counter_height_sample` takes `calib.plane.looks` samples (default 3). The window is consistent
  when its spread is at most `max_spread_m` (0.015) and its median absolute deviation is at most `max_mad_m` (0.006).
  The result is clamped to the nominal height plus or minus `max_dz_m` (0.20) and latched once per episode. If no
  consistent window appears within 3N tries, the nominal height is kept. `scene.counter_z_prior` (0.15) is used only
  when no depth fit exists. The plane fit is `PlaneFit` and `PrimaryCalibrationFitter.fit` in `calibration_fit.py`.
- **Hand ICP and counter after hand.** `calib.hand_icp_primary` lets `PrimaryCalibrationFitter.fit` pick the full
  camera extrinsic from a hand and forearm ICP against the arm surface, falling back to plane-only or nominal. When it
  is on, the pitch-only fit is skipped. `calib.counter_after_hand` takes the height only after that pose is chosen,
  through `CounterHeightTracker` (`perception.py`, `update_counter_height`), and skips the pre-motion height phase in
  `policy.py`. `calib.refit_per_look` turns on the per-look refit and cross-check. The offline calibrator is `HandCalibrator`
  in `hand_calib.py`.
- **Pitch (`perception.online_pitch_cal`).** `table` fits the camera pitch to the counter plane. `true` fits it to the
  orange bin rims (`_pitch_mode` returns `bin_rim`). `Perception.calibrate_pitch` does the fit. A failure latches the
  nominal extrinsic (`make_pitch_failure`). `config.pitch_cal_conflicts` and `hand_cal_conflicts` list overlays that
  enable two calibrations that fight each other; `release_gate.py` reports them as failures.
- **Bins.** `bin.vote.enabled` places each bin from three votes over the first frames. Without it, the nominal bin
  positions are used.

Note that finals_r22 turns `calib.plane.apply_height` on. The submission overlay leaves it off, so the baked image
uses the nominal counter height until the next promotion.

## 4. Release discipline

Every behaviour change is a flag. The registry is `FLAGS` in `config.py`. Each entry has a group, a default and a
one-line description. An overlay key that is not in `FLAGS` is rejected at load time. Defaults are the off state.

Overlays, in `aaltonomous/baselines/classical/overlays/`:

| file | role |
|---|---|
| `submission.yaml` | the baked image overlay. Written only by `tools/promote.sh`. |
| `finals_r22.yaml` | the current candidate for the finals. |
| `finals_r22_recall.yaml` | r22 plus recall fixes: `area_floor_geometric`, `partial_ok_near_hand`, `relook_before_done`, `sat_floor_adaptive`, `topface_mode: whole_blob`. |
| `finals_r22_cal.yaml` | r22 plus calibration: `calib.hand_icp_primary`, `calib.counter_after_hand`, `calib.refit_per_look`, and `principal_point: center`. |
| `finals_r22_grasp.yaml` | r22 plus release reliability: `grasp.staggered_release` (delay 0.30 s), `held.read_through_carry`, `assoc_radius_m` 0.10, `retry_jitter_m` 0.004. |
| `finals_r22_gate.yaml` | r22 plus gate and RGB recall: `fk_silhouette`, `gate_below_bins`, `split_by_solidity`, `still_vote`, `tape_value_cut`, `reject_log`, and others. |
| `finals_r22_chunk.yaml` | built on r21, not r22: larger reply chunks (`chunk_ticks` 15), robust_v2 HSV with the hue guard on, `safety_shell` on, bin vote off. |

Rules that hold for all of them:

- Each variant group is a set of flags that are OFF in `finals_r22`. None of them enters the candidate without an A/B
  eval that justifies it. The four r22 groups say "NOT in r22" in their headers. The chunk group is built on r21.
- `tools/check_release_flags.py` holds `EXPECTED_FINALS_FLAGS`, the reviewed set of 46 flags that are ON in the finals
  build. It also pins `EXPECTED_OUTPUT_CAPS` (arm velocity 0.85, arm acceleration 4.0, hand velocity 1.5, hand
  acceleration 8.0) and requires the effective profile to be `none`. Any extra flag ON, or any missing one, fails.
- `tools/release_gate.py <rc>` compares the submission overlay with the candidate, flag by flag, and checks for
  conflicts. `policy_server.verify_expected_flags` refuses to serve when the loaded flags differ from
  `overlays/EXPECTED_FLAGS.json`, which `promote.sh` writes. A mismatch stops the container at startup.
- `tools/equivalence_check.py` runs seeded mock episodes non-threaded and digests every action, event and perception
  output. It is deterministic, so two trees can be diffed. Flags-off equivalence means the baseline is unchanged.
- `tools/promote.sh <overlay>` is the release path. It has eight steps: check the git tree is clean; write
  `submission.yaml`; run the release gate; run the finals allow-list check; write the flag dump; run the equivalence
  check with `AALTO_OVERLAY=submission`; write `RELEASE.md`; finish. It commits nothing and prints the commit command.
  `--dry-run` runs everything on a copy. A failed real run restores the files it wrote.

What the submission overlay really contains. It is the promoted set, not an all-off baseline. Eight boolean flags are
ON there (`instruction.resolve`, `policy.insert_guard`, `perception.classify_shape`, `perception.split_blobs`,
`grasp.yaw_tri_shift`, `policy.grasp_diag_log`, `grasp.close_posture_by_shape`, `grasp.posture_apex_gate`), along with
the closed-tight grasp posture and `policy.bin_lookup: resolve_bin`. `perception.color_gate.live_bins`,
`safety.nan_guard`, `instruction.strict`, `instruction.max_chars`, `motion.output_caps` and `calib.plane.apply_height`
are OFF in the submission but ON in r22, so the next promotion moves them. The brain and the hue guard are OFF in both.

## 5. The brain (optional local VLM)

What it is: a closed-set tool chooser. A local llama.cpp server returns one index into a list the code builds, plus a
confidence. Nothing else can be generated, because the output is constrained by a GBNF grammar (`decision_gbnf`) and
by a JSON schema (`decision_json_schema`). The model cannot name a tool, a bin or an object that is not on the list.

Where it runs: `policy._brain_select`, at the look decision points `after_look`, `after_fail` and `after_place`. The
`brain/` package (`core.py`, `registry.py`) holds the contract. `allowed_calls` builds the list from the world state:
`pick_place(obj, bin)`, `skip`, `retry` with a grasp variant, `verify`, `look` and `done`. `find` appears only with
`brain.find`, which is off. Requests are prefetched for the next state when possible (`_brain_prefetch`). A prefetched
answer is used only when its state key matches (`state_key`).

Trigger points. `_choose` (policy.py:1778) calls `_brain_select` (policy.py:1793). `_brain_points` (policy.py:2501) builds:
`after_fail` (policy.py:2513) alone, when the picked object is re-detected in place; `after_place` (policy.py:2515) after
a pick, with classical `verify` or `look`; and `after_look` (policy.py:2525) always last.

Guards, in the order they apply:

1. `validate_decision` checks the JSON shape, the index range, confidence in 0..1, the tool whitelist, argument types,
   that each object id exists in the world, that bins are left, center or right, and the step limit (`max_steps`, 32).
   For `pick_place` it also checks reachability through the planner (`reach_fn`).
2. A known bin is never changed. `BrainLayer.take` (brain/core.py:408) rejects a `pick_place` to a bin other than
   the object's `resolver_bin` when that key is set (core.py:428-435, `bin_reject`). The world carries `resolver_bin`
   from brain/core.py:47.
3. Every brain decision needs confidence at least `threshold` (0.8). A decision that differs from a classical choice
   also needs `override_conf` (0.95) (`ev_decide`). For an object the resolver gave no bin there is no classical
   choice to disagree with, so the threshold alone applies. That is the only place the brain fills a bin.
4. `done` is offered only when the state allows it. `_brain_points` sets `allow_done = first is None` for `after_look`
   (policy.py:2522) and `False` for `after_fail` and `after_place` (policy.py:2513, 2515). `registry.allowed_calls`
   adds `done` only when `allow_done` is true (brain/registry.py:73). The brain cannot end the episode while a mapped
   object is still on the counter.
5. Timing. `brain.timeout_s` is configurable, default 2.0 s (config.yaml:464); finals brain overlay 2.0 s (inherited, since `brain_llm.yaml` sets none). It is the outer clock of `AsyncRequest`
   (brain/core.py:341, :401; late = `timeout`, core.py:316-320). The HTTP call has its own 1.5 s deadline
   (`request_timeout_s`, config.yaml:451; `urlopen` at core.py:235). That one fires first, so a slower answer ends as
   `error`. `poll()` never blocks, so nothing waits in `act()`. The 0.5 s in `brain_llm.yaml` line 2 is stale.

Fallback. Every failure returns `None`, and the classical planner decides. That covers no answer, an invalid answer, a
low-confidence answer, a timeout, an exception in `_brain_select` or `take`, and the `stub` backend, which is the
classical choice. `brain.enabled` defaults to false (config.yaml:448). `submission.yaml` sets it false (line 28).
`finals_r22` does not set it. The overlay `brain_llm.yaml` turns it on for experiments only.

Origin. Only `classical/brain/` is in wt-rc7; no `policy/brain_lang.py` prototype is present.

## 6. Failure handling

The rule is that each layer answers "no" rather than a guess. A hold or an abstention scores 0. A wrong drop scores -1.

- **Instruction.** Unparsed text abstains every key under `instruction.strict`. Insert tasks abstain on every object
  (`policy.insert_guard`), so the lid is never released over the box.
- **Bin lookup.** `lookup_bin` returning `None` means no pick. The miss is logged once per object per episode, and the
  object stays on the counter ("never a -1 guess" in `policy.py`).
- **Perception.** The hue guard abstains on red and yellow hues too close to call (`perception.hue_guard.enabled`).
  The colour gate drops blobs outside the workspace or area range, inside a bin footprint, or covered by an arm.
  With many candidates it abstains rather than truncating (`max_cands`).
- **Calibration.** Pitch, height and hand ICP all fall back to nominal, with timeouts that never block motion.
- **Motion.** A non-finite joint value holds the last emitted action (`safety.nan_guard`, on in r22). An exception or
  deadline miss holds the last target (`policy.safety_shell`, on only in the chunk variant; OFF in r22 and in
  submission). The `SafetyFilter` holds after 0.2 s and stops after 1.0 s regardless of the policy.
- **Retries.** At most `max_attempts` (2) tries per object, `max_attempts_vary` (3) with varied grasps, and
  `max_total_attempts` (18) per episode. With `policy.retry_vary_grasp`, `_plan_pick` (policy.py ~2296) takes the first
  feasible variant whose (xy, yaw, posture) is not `_same_grasp` to any earlier try on that object (xy within 0.01 m,
  yaw within 20 deg, same posture). A retry must differ from every earlier try, not only the last one.
- **Release.** `grasp.release_confirm` (on in r22) re-opens once if the hand is not empty, then goes home and stops.
  `grasp.drop_slot_overflow: counter` leaves an object on the counter when the bin slots are full.
- **Brain.** Cannot override a known bin, cannot end the episode while work is left, and falls back to the classical
  choice on any failure.

## 7. Skill memory

Flag `skills.memory` (config.py:48), default OFF (submission.yaml:18). When on, it stores grasp-variant outcomes in
SQLite (`skill_memory.py`); `skills.memory_path` (config.py:49) unset means in-memory. It orders only the six retry-vary
arms in `ARMS`, never adds a motion, never changes a bin.

## 8. Evidence

Unit tests are `def test_` counts (wt-rc7).

| Area | Result |
|---|---|
| Brain (`test_brain*`) and skill memory (`test_skill*`) | 58 tests (42 + 16) |
| Calibration, pitch, hand ICP, depth, FK cross-check | 96 tests |
| Instruction, normalise, fixed mapping | 61 tests |
| Colour gate, shape, hue guard, bins, cylinder, place gate | 167 tests |
| Flags, release gate, motion caps, safety, fuzz, insert guard | 98 tests |
| All of `aaltonomous/baselines/classical` (63 files) | 802 tests |
| Test files outside that package | 111 tests |
| Release dry-run | PASS, `EXPECTED_FLAGS` sha256 `62cb7d8c…` |
| Local image smoke | 1.07 GB, cold start 5.2 s, 117 Hz |
| Box eval rows | pending |

## 9. Limits we know about

- The `brain_llm` overlay header says 0.5 s, which is stale. The live values are `brain.timeout_s` = 2.0 s
  (config.yaml:464) and the HTTP deadline = 1.5 s (config.yaml:451). The HTTP deadline is the one that bites first.
- Nothing in this document has been run on the venue hardware. The simulator and mock episodes are the evidence.
- The A/B results behind each variant group live in the eval logs, not in this repository.

## Candidate flags after rc7 (all OFF in finals_r22 and submission)

Every flag below defaults to OFF. None is on in `finals_r22` or `submission.yaml`. Each is turned on only by the overlay named in its row. Guard sites are relative to `aaltonomous/baselines/classical/` and were read from the branch tip of each worktree. `grasp.soft_overshoot` is a tuning vector, not a switch. It has a default value and does nothing unless `grasp.soft_squeeze` is on.

| flag | overlay | what it does | guard site (file:line) |
|---|---|---|---|
| `grasp.soft_hand` | `finals_r22_softgrip` | Models the hand's finger servos as soft, as the real hand is. | `config.py:614` |
| `grasp.soft_overshoot` | `finals_r22_softgrip` (default value, not set there) | Sets how far each finger closes past its grip posture when soft squeeze is on. | `config.py:628` |
| `grasp.soft_squeeze` | `finals_r22_softgrip` | Closes past the grip posture, holds longer, and lifts slowly for the first 2 cm. | `planner.py:939`; `policy.py:2877` |
| `held.soft_detector` | `finals_r22_softgrip` | Calls an object held when the fingers stop further open than in an empty close. The empty close is measured once at episode start. | `config.py:619` |
| `perception.novel_objects` | `finals_r22_novel` | Adds candidate objects from depth blobs above the counter that colour matching missed. | `perception.py:1436` |
| `grasp.generic` | `finals_r22_novel` | Allows a top-down or handle grasp on a targeted novel object when its width fits the hand. | `planner.py:783` |
| `instruction.novel_objects` | `finals_r22_novel` | Matches colour and object words in the instruction to visible novel objects. Unclear words abstain. | `policy.py:1499` |
| `policy.novel_obstacles` | `finals_r22_novel` | Treats untargeted novel objects as obstacles that every hand path must clear. | `planner.py:1144` |
| `grasp.cage_mode` | `finals_r22_cage` | Closes three contacts around metric cylinders and standing triangles. Cubes keep their normal grasp. | `planner.py:920` |
| `grasp.tri_vertex_align` | `finals_r22_cage` | Turns the yaw of a standing triangle so the grasp meets it at a vertex. | `planner.py:731` |
| `policy.score_order` | `finals_r22_order` | Picks the next object by expected value per second. Drops candidates with no positive expected value or no time to finish. | `policy.py:1774` |
| `instruction.structured` | `finals_r22_langzip` | Expands list and conditional phrases into full rules before the strict filter. Needs `instruction.resolve` and `instruction.strict`. | `policy.py:255` |
| `calib.wide_counter_window` | `finals_r22_counter` | Widens the counter-height search to -8 to +22 cm around nominal and uses the measured height everywhere. | `perception.py:1000` |
| `planner.bin_wall_extra_m` | `finals_r22_counter` | Adds extra height to the bin walls for release, carry and the rim keepout. Default 0.0. The overlay sets 0.01. | `planner.py:361` |
| `perception.glare_reject.enabled` | `finals_r22_bare` | Removes small bright highlights on the table from the colour masks. | `perception.py:1410` |
| `perception.reflection_reject.enabled` | `finals_r22_bare` | Drops a colour blob that mirrors a stronger blob above it. | `perception.py:1441` |
| `perception.local_bg.enabled` | `finals_r22_bare` | Drops a colour blob whose colour matches the ring of pixels around it, which is the table. | `perception.py:1413` |

**Known interactions**

- Cage mode and soft grip (`grasp.soft_squeeze`) both on: the soft squeeze overclose is applied on top of the posture the planner picks, and that includes the cage posture. The hand squeezes twice. (`planner.py:920`, `planner.py:939`)
- `score_order` drops any candidate whose expected value is zero or less. Expected value is `ps * (2 - 3 * p_wrong)`, so the cut falls at a p_wrong of 2/3 or more, not only above 2/3. (`policy.py:1837`)
- The arm-mask gate in `perception.py` (line 1319) also runs when only `perception.depth_blob_recover` is on. r22 is not affected, because `perception.color_gate.fk_mask` is on there.
