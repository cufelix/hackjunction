# Adversarial review: `timi/pregrasp` (`3a875bc`) vs `team/felix/rc6`

Scope: `git diff 4540b9ba -- aaltonomous tools` (one commit). Flag `grasp.pregrasp_relocalise` defaults off. `finals_r22_pregrasp` is `finals_r21` plus that flag. With the flag off, `_plan_pick` / `_start` / `act` do not enter the new path (`_pg_on` is false, `_pg` stays `None`). No flag-off behaviour change and no new URDF-limit command on the flag-off path. Corrected plans still go through `Planner.dry_run_safety`; the abort `go_to` interpolates between the current command and home/open, which stays inside the joint box if those endpoints do.

Findings below are flag-on only.

## 1. High — reset races the pregrasp worker on the shared planner / Pinocchio data

`aaltonomous/baselines/classical/policy.py:2274` starts a daemon thread (`_pg_job` → `_pg_run` → `_pg_plan`). `_pg_plan` (`policy.py:2346-2355`) sets `pl.vary` on the live `Planner` and calls `_plan_side_hints`, which runs IK on `ArmKinematics.data` (`kinematics.py:80-86`, one `pin.Data`, not thread-safe). `policy.py:250-251` clears `_pg` and does not bump a generation or join that thread. `policy_server.py:43-51` waits only for `pol._job` (the look thread) and says the planner is not thread-safe, then calls `reset()`.

Failure: the 120 s episode ends while the arm is held at the pre-grasp pose and the worker is inside IK (that is the normal pause, or the `PG_MAX_JOB_S` timeout path which keeps executing while the worker is still running). The runner calls `reset()` for the next episode. `_job` is `None` during the pause, so the wait returns immediately. The next episode's look thread calls `planner.plan` on the same `kin.data` while the old worker is in `_fk` / `_lsq`. The old worker's `finally` then does `pl.vary = None` (`policy.py:2355`) and can clear a `vary` the new episode just set. Torn FK produces a bad but finite trajectory (wrong bin drop) or NaNs. NaN targets are not covered by `safety.nan_guard` (that checks measured `arm_q` / `hand_q` only) and are outside the URDF.

Minimal fix: give the worker the same treatment as `_job`. Store the `Thread` on the policy, increment a generation in `reset()` before clearing `_pg`, and have `_pg_job` return before `_pg_plan` if the generation changed. In `ClassicalServerPolicy.reset`, wait for that thread the same way as `_job` (join on timeout too, and ignore a late `job["result"]`).

## 2. High — pause and the replacement motion are outside the 120 s check

`_has_time` (`policy.py:2150-2151`) runs once, before `_pg_begin`, against the original plan: `t + plan.duration <= time_limit`. `finals_r22_pregrasp` sets `policy.time_margin_s: 0`, so a plan may be accepted to finish at exactly 120. Nothing after that rechecks.

The pause is not in `plan.duration`. `wait_s` is 0.15 and `motion.hold_ticks` is 6, so the first hold already advances episode time by 0.2 s (`policy.py:2262-2263`, `config.yaml:391`). If the worker is slow, holds continue until `PG_MAX_JOB_S` (3.0 s, `policy.py:2231` and `2280-2283`) and the tail is still executed. End time of a keep/tail pick is `t0 + original_duration + pause`, which is at least 120.2 s and up to about 123 s for a plan that was accepted at the limit.

A corrected plan is worse: `_pg_finish` (`policy.py:2382-2384`) starts `res["plan"]` with no `_has_time` call. That plan is a new full trajectory from the pre-grasp pose (its own `to_pregrasp` plus descend, the 1.2 s close hold, carry, home). The original approach has already been spent. An abort (`policy.py:2410-2414`) enqueues `go_to` home with no duration check at all. Example: at t=116 a 4 s plan is accepted. The approach uses 2.5 s, the pause 0.3 s, and the retreat home is ~2 s. The episode is still commanding at t>120, mid-retreat or mid-carry. A cutoff during carry over a bin drops the object into the wrong bin (−1) or onto the floor.

`test_added_episode_time_per_pick_is_below_0_8_s` (`test_pregrasp.py:318-332`) allows that extra 0.8 s and never places it against `time_limit`. `Run` does not stop at 120.

Minimal fix: before `_start` of the tail or the corrected plan, require `t + duration <= time_limit`; otherwise do not move. On abort, same check for the retreat; if it does not fit, hold and set `state = "done"` instead of `go_to`. Count the pause in the budget that `_has_time` used at look time (subtract `wait_s + PG_MAX_JOB_S`, or the measured pause).

## 3. High — corrected grasp keeps the first look's bin, and skips the gates that prevent a swap

`relocalise` (`perception.py:1116-1136`) keeps the nearest blob of the prior's raw HSV class inside `gate_m` (0.075 m). It does not run `hue_guard`, `color_gate`, or `sat_floor`. All three are on in `finals_r22_pregrasp`. The docstring says it is the same pipeline as `detect()`; `detect()` applies those filters at `perception.py:1081-1084`. `_pg_plan` (`policy.py:2352-2353`) then replans with `plan.target_bin` from the first look and never calls `lookup_bin` / `resolve_bin`. The overlay sets `policy.bin_lookup: resolve_bin` and `instruction.resolve: true`, so shape (and an abstain) changes the bin.

Failure: yellow cube planned to the left bin, yellow cylinder 5 cm away (inside the gate, outside `min_shift_m` 8 mm, inside `max_shift_m` 7 cm). The hand mask covers the cube; the cylinder is the only fragment. Shift is ~5 cm, kind `corrected`, cylinder is carried to the cube's bin. Score −1. Same outcome if the true object is hidden and a yellow sheen with S in [35, 45) survives the raw yellow mask (`hsv_robust_v2` yellow S floor is 35; `sat_floor` would reject S < 45) on a red neighbour: the red object is placed in the yellow bin. `color_gate`'s bin-edge / in-bin rejection is also skipped, so a same-colour rim blob within 7.5 cm of the prediction can be picked out of a bin and dropped in the first look's bin.

Minimal fix: run the same post-filters as `detect()` on the relocalise candidate, and abort (do not correct) unless the kept detection matches the prior's shape and `lookup_detection_bin` returns `plan.target_bin`. If the nearest in-gate blob is a different object, treat it as `too_far`.

## 4. Medium — abort loop is not bounded by the attempt cap or the failed list

`_pg_abort` (`policy.py:2398-2409`) does not increment `n_started` or `attempts`. `max_total_attempts` (18 in the overlay, checked at `policy.py:734-738`) therefore does not count these cycles. The comment says the failed list bounds the loop. It does not:

- The abort is stored at the first-look xy. `n` uses `_attempt_radius()` (`policy.py:1041-1044`), which is `retry_assoc_radius` 0.06 m because `retry_vary_grasp` is on.
- The next look drops a candidate only inside `same_object_radius` 0.03 m (`policy.py:1084`, `config.yaml:424`).

Failure: each approach sweeps the object by 4 cm (between 3 cm and 6 cm) and the hand then hides it, so relocalise returns `gone`. Abort 1 records p0. The next detection is p1, 4 cm away, not in `failed`. Abort 2 records p1; p0 is within 6 cm, so `n >= 2` and `failed` gets p1 only. The next detection is p2, 4 cm past p1, outside every 3 cm ball. That repeats until a later `_has_time` rejects a full plan. The last retreat is the overrun in finding 2. `max_total_attempts` exists for "knocked objects reappear as new" (`config.yaml:433-434`); this path never touches it.

`test_repeated_aborts_leave_the_object_on_the_counter` (`test_pregrasp.py:223-229`) passes because `Run` shows the same `first` frame on every look (`test_pregrasp.py:98`). The detected xy never moves, so two aborts hit the same point.

Minimal fix: count an abort in `n_started` (or a dedicated counter capped by `max_total_attempts`). On abort, append the failed marker at the prior xy and, when the re-detection exists, at that xy too. Use one radius for the count and the filter: `max(same_object_radius, retry_assoc_radius)`.

## 5. Medium — `act()` runs the full re-detect and IK on the control thread when not threaded

`policy.py:2274-2277`: if `self.threaded` is false, `_pg_step` calls `_pg_job` inline. That is `relocalise` plus `_plan_side_hints` (IK with restarts). That is well over 37 ms, inside `act()`, and an exception there is caught only after the stall. `policy.safety_shell` defaults false; the shell is what forces `threaded = True` (`policy.py:114-118`). `finals_r22_pregrasp` happens to set the shell, so the overlay does not hit this. Turning the flag on alone (or any overlay that sets the flag and not the shell) stalls the 30 Hz loop for a full plan. `run.py` passes `threaded=False` for the mock backend; only the shell override saves the r22 overlay.

Minimal fix: always start `_pg_job` on a thread and return `_hold()` on that tick, including when `self.threaded` is false. Do not call `_pg_run` from `act()`.

## 6. Medium — tests that pass without checking the property

- `test_act_never_blocks_threaded_and_the_pause_ends` (`test_pregrasp.py:350-354`) does not call `act()` while the look or pregrasp job is in flight (`sleep` + `continue`). The latency sample is only ticks where the worker is already idle, so a GIL stall or the inline path in finding 5 cannot fail it. The 20 ms / 50 ms asserts never see the tick that can block.
- `test_retreat_goes_home_with_the_open_hand_then_looks` (`test_pregrasp.py:221`) treats any event containing `"retreat"` as proof of a retreat. The abort note itself is `"retreat + re-look"` (`policy.py:2389`), so the count is 1 even if `go_to` was never queued. The `arm_cmd` check above it is real; this assertion is not.
- `test_episode_actions_are_deterministic_and_unchanged_by_the_new_code_path` (`test_pregrasp.py:280-306`) runs the new code twice with the flag off. It does not compare against `team/felix/rc6`. It cannot catch a flag-off behaviour change. (The `relocalise` patch and the r21-vs-r22 config equality test are real checks; this one is not an equivalence test.)
- The abort-bound and deadline tests are the stationary / no-deadline cases in findings 2 and 4.

Minimal fix: call `act()` on a deadline while the worker holds the GIL (and once with `safety_shell` false); assert a home `go_to` from the queue, not the substring `"retreat"`; diff action streams against the parent commit with the flag off; add a case whose detection walks by 4 cm per abort and a case whose accepted plan ends at t=120.
