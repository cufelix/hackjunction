# Adversarial review: `felix/rc5b` finals policy

Reviewed `git diff team/felix/rc1 HEAD -- aaltonomous tools` (`7a4168e..f1e9115`), with runtime tracing concentrated on the flags enabled by `finals_r19b.yaml`, `finals_r19c.yaml`, and `finals_r20.yaml`.

## Executive verdict

Do not run the current image/manifests on the real G1. There are five stop-ship failures: the image bakes neither of the finals overlays, the server defaults to the simulation motion profile, r20's rim vote applies the estimated bin displacement with the wrong sign, r20 can retract a low failed grasp through the table/bins, and the robot moves to an unchecked fixed home before the table-height calibration can run. The remaining high-severity findings include two independent wrong-colour/wrong-bin paths, mixed calibration frames, and synchronous work that can block `act()` well past the control-loop budget.

Counts: **5 Critical, 7 High, 4 Medium, 1 Low**.

## Critical findings

### C1 — Critical — The submitted image does not enable any finals overlay

**Affected overlays:** `finals_r19b`, `finals_r19c`, `finals_r20` (all are bypassed).

**Evidence:** `tools/submission.Dockerfile:4-6,35-36` says the baked file is the only switch when organizers start the container without environment, but it copies `overlays/submission.yaml`, not a finals overlay. That overlay keeps the finals switches off.

**Impact:** The deployed policy lacks finals safety shell, strict language, hue guard, pitch calibration, live bins, bin vote, and runtime height application. Tests that load finals YAMLs do not describe the image.

**Concrete fix:** Require a build argument such as `FINAL_OVERLAY=finals_r20.yaml`, copy exactly that file to `overlay.yaml`, and fail the build unless the resolved config records the expected overlay and exact release flag allow-list. Archive the printed effective flags.

### C2 — Critical — The policy server uses the simulation profile on the real robot

**Affected overlays:** `finals_r19b`, `finals_r19c`, `finals_r20`.

**Evidence:** `aaltonomous/baselines/classical/config.yaml:6-10` defaults to `profile: sim` and says real execution must select `none`. `aaltonomous/baselines/classical/policy_server.py:33-40` constructs `ClassicalPolicy` without choosing a profile. The container/manifests do not set `AALTO_PROFILE`. Sim overrides at `config.yaml:489-503` replace real home and raise arm acceleration from `1.5` to `8.0 rad/s²`, arm velocity from `0.5` to `0.97 rad/s`, and hand velocity from `1.2` to `2.5 rad/s`. The finals `0.90` cap still leaves about `7.42 rad/s²` acceleration.

**Impact:** First motion uses a sim-only home and substantially more aggressive dynamics than the real defaults, creating collision and hardware risk.

**Concrete fix:** Make the real server pass `profile="none"` explicitly and bake `AALTO_PROFILE=none` as defense in depth. Fail release builds if effective profile is `sim`. Test the exact container entry point against real home and limits.

### C3 — Critical — r20's rim vote mirrors the physical bin displacement

**Affected overlays:** `finals_r20` only (`bin.vote.enabled: true`).

**Evidence:** `aaltonomous/baselines/classical/bin_vote.py:124-140` scores `Q + off` against the nominal box, then returns `nominal_center + off`. If the physical rim is at `nominal + δ`, the minimizing offset is `-δ`, so the reported center is `nominal - δ`. `Planner.bin_xy`, `drop_xy`, and `drop_slots` consume it at `planner.py:150-178`.

**Impact:** The correction moves away from the real bin and doubles target error, potentially placing a release outside the inner footprint or colliding with the rim.

**Concrete fix:** Use one convention throughout: score `Q - shift` and return `nominal + shift`, or keep the current score and return `nominal - off`. Add synthetic depth tests for all `±x/±y` shifts and assert final drop slots.

### C4 — Critical — Empty-grasp abort retracts directly through unchecked joint space

**Affected overlays:** `finals_r20` only (`held_detector.mode: finger_sum`).

**Evidence:** `aaltonomous/baselines/classical/policy.py:2169-2171` may invoke `_abort_empty_grasp` at the low grasp/lift transition. `_abort_empty_grasp` at `policy.py:2111-2125` replaces the queue with `planner.go_to(...home...)`. `Planner.go_to` at `planner.py:908-916` merely time-scales joint space; it has no vertical escape, bin/table clearance check, or safety dry-run.

**Impact:** The hand can sweep through a higher table or bin wall while retracting a failed grasp.

**Concrete fix:** Precompute a checked abort: open/hold as appropriate, lift vertically above runtime bin rims, then follow a collision-checked route home through the same bin-clearance and dry-run filters. If unavailable, hold rather than synthesize a direct `go_to`.

### C5 — Critical — The robot moves before measuring the real table height

**Affected overlays:** all three; r20 is additionally misleading because it enables `calib.plane.apply_height`.

**Evidence:** The first state transition commands `planner.go_to` to configured home at `aaltonomous/baselines/classical/policy.py:667-691`. Height estimation occurs only later inside `Perception.detect` (`perception.py:1036`), after move/settle and, in r20, bin-vote collection. The path is the unchecked `planner.py:908-916` path. `config.yaml:367-373` documents only about 6 cm nominal rim clearance. r19b/r19c never apply height. Runtime scene height also does not update fixed safety keepouts in `aaltonomous/interface/safety.py:60-65`.

**Impact:** A higher real table/bin can intersect the nominal home or route before calibration runs. Later geometry correction cannot undo the collision.

**Concrete fix:** Initially hold measured joints; calibrate height before motion; atomically update scene and safety keepouts; then collision-check a route to a height-safe staging pose. If calibration fails, use a conservatively prevalidated path or refuse motion.

## High findings

### H1 — High — Touching red/yellow objects are split only after receiving one shared colour

**Affected overlays:** `finals_r19b`, `finals_r19c`, `finals_r20` (`robust_v2`, `hue_guard`, `split_blobs`).

**Evidence:** `aaltonomous/baselines/classical/perception.py:1075-1091` unions red/yellow masks, creates one component, and labels it from one top-face HSV statistic. Only afterward does `detect` split it at `perception.py:1050-1067`; all parts inherit the shared label. Existing hue tests use separated patches.

**Impact:** Touching orange/yellow objects can both receive the dominant component colour. One then goes confidently to the wrong bin (`-1`).

**Concrete fix:** Split the union first and classify/guard every part independently, or require per-part source-pixel dominance. Abstain on mixed parts. Add touching cross-colour tests in both spatial orderings and unequal areas.

### H2 — High — Strict language mode still guesses fixed mapping on empty input or exceptions

**Affected overlays:** `finals_r19b`, `finals_r19c`, `finals_r20` (`instruction.strict: true`).

**Evidence:** `aaltonomous/baselines/classical/policy.py:151-153` returns `None` on empty text; `policy.py:162-164` does the same on resolver/normalizer exception. `reset` turns either into `FIXED_MAPPING` at `policy.py:190-193`. `test_instruction_strict.py:113+` codifies the empty-text fixed fallback.

**Impact:** Missing or malformed language silently routes objects using fixed colours, producing wrong-bin penalties instead of the promised abstention.

**Concrete fix:** With strict on, every missing/error path must return a truthy abstain mapping such as `{"default": "abstain"}`. Keep fixed fallback only for non-strict `sort_fixed`. Test empty, malformed, and exception-injected inputs.

### H3 — High — Pitch calibration can commit in the middle of the first look

**Affected overlays:** `finals_r19b`, `finals_r20`; not r19c.

**Evidence:** `_start_pitch_worker` launches a daemon at `aaltonomous/baselines/classical/policy.py:341-361`, started at `policy.py:624-626`. Neither bin vote (`policy.py:697-705`) nor threaded look (`policy.py:721-730`) waits. `Perception.commit_pitch` swaps the transform at `perception.py:336-340`, while projection helpers snapshot per call (`perception.py:287-311`), not per complete frame.

**Impact:** A detect/vote can mix nominal and corrected frames among keep mask, object deprojection, bin hull, FK mask, and height estimation. r20 may complete its one-shot bin vote before correction exists.

**Concrete fix:** Add a hold-only calibration phase before vote/first plan. Also snapshot one immutable transform/version per frame and pass it through all consumers; discard work whose version changes.

### H4 — High — Tape votes bypass the corrected camera transform

**Affected overlays:** `finals_r20` only.

**Evidence:** `aaltonomous/baselines/classical/bin_vote.py:201,204-205` calls raw `perception.cal.pixel_to_pelvis` for measured depth and rays, bypassing `Perception._pixel_to_pelvis` and runtime `pitch_T`. Rim voting uses corrected projection.

**Impact:** Rim and tape evidence occupy different frames after pitch calibration, causing false disagreement or biased false agreement and an incorrect drop target.

**Concrete fix:** Project tape data through the same frame-snapshotted method as rims. Test nonzero injected pitch with agreeing synthetic tape/rim observations.

### H5 — High — r20 performs the five-frame bin vote synchronously in `act()`

**Affected overlays:** `finals_r20` only.

**Evidence:** `aaltonomous/baselines/classical/policy.py:697-705` directly invokes `vote_bins`. It handles every frame synchronously at `bin_vote.py:240-252`; rim voting performs three nested 2-D grid searches over up to 1,500 points per bin (`bin_vote.py:122-136`) plus connected-components/deprojection. The shell at `policy.py:571-598` checks elapsed time only after `_act` returns and cannot interrupt it.

**Impact:** A request can block far beyond 30 Hz and the 200 ms shell budget. A late hold does not restore missed deadlines.

**Concrete fix:** Put voting in the background look executor with copied frames and an episode token. `act()` should poll immutable results and hold while pending. Bound work and measure deployment-hardware p99 latency.

### H6 — High — “Median of 3 looks” applies the first plane immediately

**Affected overlays:** `finals_r20` only.

**Evidence:** `aaltonomous/baselines/classical/perception.py:816-836` appends a plane estimate and immediately assigns the median of however many samples exist, including one. Config declares `looks: 3` at `config.yaml:13-19`.

**Impact:** One transient but parallel plane can move grasp, clearance, and bin geometry by the allowed clamp before the first pick. Later samples cannot protect that motion.

**Concrete fix:** Do not commit until all N valid, mutually consistent samples exist; use spread/MAD rejection and commit once per episode. Test that samples 1 and 2 cannot alter runtime height.

### H7 — High — Background look can cross reset; the attempted guard blocks reset up to 30 seconds

**Affected overlays:** `finals_r19b`, `finals_r19c`, `finals_r20`.

**Evidence:** Reset replaces job state without an episode token. `_look_job` at `aaltonomous/baselines/classical/policy.py:757-762` runs `_look`, which mutates detections, scene/calibration, planner IK/vary, and policy fields. Acknowledging non-thread-safe state, server reset busy-waits up to 30 seconds at `policy_server.py:43-51`; after timeout it resets while the old worker may still publish.

**Impact:** Reset can violate watchdog/control availability, or stale detections/plans/calibration can corrupt the next episode concurrently.

**Concrete fix:** Use one executor and generation token. Workers compute local immutable results and publish only on a matching token; never mutate shared planner/scene directly. Reset invalidates immediately instead of sleeping.

## Medium findings

### M1 — Medium — A nonempty shift dictionary disables the live-bin hedge even with no estimate

**Affected overlays:** `finals_r20` only.

**Evidence:** `_color_gate_boxes` uses `bool(shift)` as “estimate exists” at `aaltonomous/baselines/classical/perception.py:1614-1626`. Bin vote returns a dictionary covering all bins even for prior-only zero shifts, and its exception fallback reports prior confidence at `bin_vote.py:240-255`.

**Impact:** With no usable evidence, a physically closer bin wall can fall outside the exclusion footprint and be detected/picked as an object.

**Concrete fix:** Trust per-bin geometry only at agreed high `planner.bin_conf`. For prior/disagree, exclude the conservative union of nominal footprint and full search/near margin.

### M2 — Medium — NaN guard can command an unobserved joint toward home

**Affected overlays:** `finals_r19b`, `finals_r19c`, `finals_r20`.

**Evidence:** `_fallback_pose` at `aaltonomous/baselines/classical/policy.py:477-489` replaces nonfinite measurements with configured home. `_nan_guard` at `policy.py:512-522` uses it before any trusted emitted command. Repeated bad startup frames therefore produce a finite home target; velocity limiting advances the unknown joint toward it.

**Impact:** Encoder loss at startup causes blind motion, not a hold; with C2, the target is sim home.

**Concrete fix:** Latch no-motion until one complete finite encoder sample. Before a trusted command exists, make the final filter retain its last known target rather than inventing home values.

### M3 — Medium — Finger-sum bias is learned during initial motion

**Affected overlays:** `finals_r20` only.

**Evidence:** `aaltonomous/baselines/classical/held.py:36-54` collects commanded-minus-measured hand error during the first second without requiring steady motion. That interval includes the initial open/home transition.

**Impact:** Tracking lag contaminates the baseline, causing false-empty aborts or false-held grasps depending on startup timing.

**Concrete fix:** Calibrate after home/settle only when commanded and measured hand velocities stay below a threshold for N samples. Reject high spread; disable early abort if calibration is invalid.

### M4 — Medium — Acceleration caps are not applied to every motion

**Affected overlays:** `finals_r19b`, `finals_r19c`, `finals_r20`.

**Evidence:** Normal arm planning uses acceleration-aware `time_scale` (`aaltonomous/baselines/classical/planner.py:88-115,809-856`), but `SafetyFilter.filter` at `aaltonomous/interface/safety.py:219-236` has no acceleration state/clamp. Hands use velocity-only `ramp` at `planner.py:118-124,825-834`. Queue replacement can create command-velocity discontinuities.

**Impact:** Hand motion and emergency/retry transitions can exceed intended acceleration even when every sample passes velocity limits; C2 magnifies this.

**Concrete fix:** Add per-joint arm/hand acceleration limiting to the final safety layer using prior commanded velocity and real elapsed time. Use acceleration/jerk-limited hand trajectories and dry-run through the identical stateful filter.

## Low finding

### L1 — Low — Finals velocity-cap tests exercise sim, where the flag means something different

**Affected overlays:** `finals_r19b`, `finals_r19c`, `finals_r20`.

**Evidence:** `_apply_arm_vel_cap` at `aaltonomous/baselines/classical/config.py:204-215` only lowers the existing value. In real profile, `0.90` is inert against `0.5`; in sim it lowers `0.97` and scales `8.0` acceleration. Finals flag tests load default sim configuration.

**Impact:** Passing tests conceal the deployment-profile error and do not validate real effective values.

**Concrete fix:** Parameterize tests over explicit `sim` and `none`, and snapshot the effective config from the exact server/container entry point.

## Configuration and coverage audit

- Every key in the three finals overlays exists in `FLAGS`; no silently ignored overlay key was found. `load_overlay` rejects unknown keys at `aaltonomous/baselines/classical/config.py:163-173`. The real configuration failure is selection, not a misspelling.
- `overlap_look` and the skill-memory writer are absent from this checkout's current ancestry/code path; related commits in repository history are not ancestors of `HEAD`, so there is no enabled rc5b implementation to audit.
- Largest test gaps: exact built-image config, exact real profile, signed synthetic bin displacement, touching cross-colour blobs, calibration commit during detect, corrected-pitch tape/rim agreement, and safe abort clearance.
- Static verification completed with a clean worktree and `git diff --check`. Full unit execution was unavailable in the checkout interpreter because `numpy` is not installed; pure strict-mapping tests ran, while NumPy-dependent suites failed at import rather than exercising code.

## Release recommendation

Treat C1-C5 as stop-ship. After fixes, build the exact final image, assert its overlay and real profile, hold through calibration, replay all targets through collision and acceleration checks, and run signed-bin/touching-colour/reset-overlap tests before enabling the G1.
