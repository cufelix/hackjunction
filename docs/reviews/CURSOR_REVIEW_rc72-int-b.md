# Adversarial review: `team/felix/rc7-pre...team/felix/rc72-int-b` (`aaltonomous`, `tools`)

Read-only. Line numbers are `team/felix/rc72-int-b`. Scoring assumed: +2 correct bin, -1 wrong bin, 0 left on the counter; episode 120 s; control loop must stay at or above 27 Hz (act under 37 ms) with zero safety clips.

Flags-off behavior was checked (`done_confirm`, `look_timeout_s`, `per_target_guard`, `score_order`, `cage_mode`, `tri_vertex_align`). Each new branch is gated, and stock `config.yaml` keeps them off. No flag-off behavior change found. Cage joint targets are clipped to `JointSpec` hand limits (`config.py` `hand_postures`, `planner.py` command seam); the values in `config.yaml` sit inside the Dex3 URDF, and `act()` equalizes index/middle on emit. No URDF-limit violation found.

`finals_r22_done` turns on `done_confirm`, `look_timeout_s=5`, `per_target_guard`, and the pre-existing `generation_workers`. `finals_r22_order` turns on `score_order` with `generation_workers` and `done_confirm` left off.

---

## 1. High — per-target cooldown never reaches the live policy when `generation_workers` is on

**Where:** `aaltonomous/baselines/classical/policy.py:96` (`_LOOK_COMMIT_FIELDS`), `:1360` (`_look_snapshot`), `:2131` (`_cool_target`), `:393` (comment that workers read `_latest_t`).

**What happens:** `finals_r22_done` plans on a `copy.copy` of the policy. `_cool_target` rebinds `self._target_cooldowns` on that copy (`self._target_cooldowns = [...]` then `append`). The list is not in `_LOOK_COMMIT_FIELDS`, so `_commit_look_result` drops it. The live policy keeps `[]`. The guard also does not append to `failed` (`:2098`). The next look therefore plans the same infeasible target again.

`_latest_t` does not save this. The snapshot copies the float by value (`:393` says workers use the live clock). A look whose IK exceeds `per_target_cooldown_s` (2 s) stamps `until` in the past even if the list were committed.

**Failure:** one workspace-edge object is first in candidate order and IK fails in ~3 s. Every later look pays that IK again before a reachable object. Five looks burn ~15 s of the 120 s episode. A later +2 is started after `time_limit - score_safety` and is refused, or the episode hits the limit with the object still on the counter (0). `test_infeasible_target_is_cooled_not_episode_fatal` calls `_choose` on the live policy and never sets `generation_workers`, so it stays green.

**Fix:** deep-copy `_target_cooldowns` into the snapshot and add it to `_LOOK_COMMIT_FIELDS`. Stamp `until` from the live episode clock inside `_commit_look_result` (`until = self._latest_t + cooldown`), not from the worker’s frozen `t`.

---

## 2. High — endgame keeps one candidate, then an IK miss ends the episode

**Where:** `policy.py:1990-1992` (`_score_order`), `:2118` (`return ("none", ...)`), `:3068-3070` (`kind == "none"` → `done`).

**What happens:** `finals_r22_order` has `score_order` on and `done_confirm` off. When usable time is under `score_endgame_s` (25 s), `_score_order` replaces the candidate list with only the minimum-`t_cost` certain object (`pw <= 0.1`). `_choose` plans that one object. If the plan is infeasible it is appended to `failed` and, with no further candidates, the look returns `("none", None, len(dets))`. `_after_look` sets `state = "done"` while other detections are still on the counter.

**Failure:** `t=96`, `time_limit=120`, safety margin 3 s, so `left=21 < 25`. Two certain cubes, estimates 12 s and 14 s, both inside 21 s. The 12 s plan fails IK (wrist limit). The 14 s cube is never planned. Episode ends at t=96. That cube scores 0 instead of +2, with time left for the pick.

**Fix:** in the endgame, keep every certain object, sorted by `t_cost`, and plan the next one when the quickest is infeasible or over the real duration check. Return `none` only after those plans have been attempted.

`test_score_order.py:64` `test_endgame_only_quickest_certain` only asserts `len(c) <= 1`. An empty list and the wrong singleton both pass.

---

## 3. High — `done_confirm` retries `uncorrectable` and never plans counter objects

**Where:** `policy.py:1566-1571` (`_look` returns `uncorrectable` before `_choose_retry`), `:3040-3044` (`_after_look` sends that result back to `settle`).

**What happens:** with `policy.verify_placement` on, `max_corrections_per_episode` null (the stock default), and `planner.per_target_guard` off, a wrong-bin detection that has no safe corrective returns `uncorrectable` before any counter candidate is planned. `done_confirm` treats that as “keep looking”: `state = "settle"` and `_hold()` until `t >= time_limit`. The same short-circuit runs on every retry.

`finals_r22_done` itself has `verify_placement: false` and `per_target_guard: true`, so this overlay does not take the branch. The branch is live for `done_confirm` plus verify, which is the combination that both sees wrong-bin objects and refuses to finish.

**Failure:** one cube already in the wrong bin (score -1 if it stays) and two cubes on the counter (+2 each). The corrective IK is infeasible. The robot settles and re-looks until t=120. The wrong-bin cube stays (-1). The counter cubes are never grasped (0). Net versus planning the counter first: -5.

**Fix:** when `done_confirm` is on and the corrective cannot be planned, do not return `uncorrectable`. Set the same `guarded_corrections` flag the per-target guard uses and fall through into `_choose_retry`. Stop the episode only at `time_limit`.

---

## 4. Medium — `look_timeout_s` without `generation_workers` races reset on the live policy

**Where:** `policy.py:189` (deadline forces `threaded=True`), `:1416-1420` (no snapshot: `_look` runs on `self`), `:1525-1530` (cancel is noticed only after `detect` returns), `:377-390` (`reset` still swaps `planner.scene` via `_replace_stale_perception`).

**What happens:** the deadline flag turns on a background look even when `generation_workers` is off. Cancel is stored on the job, and `_detector_finished` reads it only once `detect` has returned and set `phase` to `"plan"`. Planning after that point ignores `cancelled`. `reset` clears `failed`, installs the new mapping, then replaces `self.perception` and `self.planner.scene` while that thread is still inside `_choose` on the same object, using the same `self.kin` Pinocchio `Data`.

**Failure:** episode 1’s look is in IK when the runner calls `reset` for episode 2. The thread then `failed.append`s episode 1’s xy onto episode 2’s fresh list, and both threads step the same `Pinocchio` `Data`. Episode 2 skips a real object that sits on that xy (0 instead of +2). A concurrent FK update can throw inside `act()`.

`finals_r22_done` also sets `generation_workers`, which keeps planning on a snapshot and drops a stale generation at commit. The deadline flag alone does not.

**Fix:** give the deadline path the same snapshot worker as `generation_workers`, or stop planning when `cancelled` is set (check the flag between targets) and do not touch `self.planner` / `self.kin` / `self.failed` after `reset` has started.

---

## 5. Medium — estimated `t_cost` deletes candidates, then `none` ends the episode

**Where:** `policy.py:1964-1988` (drop when `tc > left`), `:2118`, `:3068-3070`. Real duration is checked later at `:2083`, only for candidates that survived the estimate.

**What happens:** `finals_r22_order` has `done_confirm` off. `_so_estimate` uses `score_overhead_s` (8 s) plus path / `score_speed_mps` (0.12 m/s) and never calls IK. Anything over `left` is removed. An empty list returns `none`, and `_after_look` sets `done`.

**Failure:** one cube, reach 0.35 m, bin 0.40 m away. `path = 2*0.35+0.40 = 1.10`, `tc = 8 + 1.10/0.12 = 17.2 s`. At `t=100` with `time_limit=120` and verify off, `left = 17`. The cube is removed. The episode ends at t=100 with the object on the counter (0). The IK duration check, which would accept a plan under 17 s, never runs.

**Fix:** keep over-estimate candidates in order and apply only the real `p.duration` test (`:2083`). Return `notime` / `done` only when a computed plan does not fit, or when every remaining plan has been tried.

---

## Tests that pass without the behavior they name

- `aaltonomous/baselines/classical/test_score_order.py:64` — `assertLessEqual(len(c), 1)` passes for `[]` and for the wrong object. It does not check that the survivor is the quickest certain cube, or that an IK failure still tries the next one (finding 2).
- `aaltonomous/baselines/classical/test_cage_grasp.py:188` `test_blob_min_area_rect_fallback_produces_finite_robot_yaw` — any finite yaw in `[0, 120°)` passes. A 90° short-axis error still goes green.
- `aaltonomous/baselines/classical/tools_fault_drill.py:86` and `:216` — for `depth_zero`, `camera_frozen`, `bin_missing`, `joint_nan`, and `instruction_bad`, `FaultSupervisor.act` holds and never calls `policy.act` on the bad observation. `test_tools_fault_drill.py` `test_all_faults_hold_and_meet_contract` then reports the policy passed the fault.
- `aaltonomous/baselines/classical/test_tools_fault_drill_ext.py:31` — `test_ws_drop_reconnect` does not assert `p99 < 37`. A slow `act()` still passes.
- `tools/test_guard_eval_args.py:20` — the two tests that would require `evaluate` / `aalto.sh` to run `guard_eval_args.py` are skipped. Nothing else in the tree calls `check()`. `python -m aaltonomous.sim.evaluate --container ...` without `--policy` still uses the default oracle. The unit tests of `check()` pass anyway.

---

## Checked, no bug filed

- New flags off: threaded look, done-confirm, guard, score order, cage posture, and vertex yaw stay on the previous branches.
- Cage commands: `cage_finger_posture` plus thumb extras `0.152` / `0.177` rad, clipped to `hand_lower`/`hand_upper`. Index and middle match, so emit-time `equalize_index_middle` does not move them. Inside the URDF this is a silent margin clamp, not a counted safety clip.
- Deadline hold when `detect_max_inflight` wedged jobs are outstanding returns `_hold()` only until `t >= time_limit` (`robust_look` check is before that hold). It does not stick past the episode.
