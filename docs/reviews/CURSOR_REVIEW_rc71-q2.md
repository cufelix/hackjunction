# Adversarial review: `team/felix/rc71-q2` vs `team/felix/rc7-pre`

Read-only review of `git diff team/felix/rc7-pre...team/felix/rc71-q2 -- aaltonomous tools`. New behaviour is behind `brain.planner` (off; on in `overlays/ab_planner.yaml`) and `perception.prism_orientation` / `grasp.lying_prism` (off; on in `overlays/finals_r22_lyingprism.yaml`). `finals_r22_brain` turns the brain on and leaves `brain.planner` at the default false.

With both new flags off, a normal-length instruction does not take the new branches. Lying-prism joint targets still go through the existing IK and safety filter. The bugs below are with the new flags on, including the finals stack (`policy.generation_workers`) together with the planner.

---

## 1. High — a rejected `done`, or any adopted plan with nothing left to run, ends the episode

`aaltonomous/baselines/classical/brain/planner.py:409` (`adopted = bool(steps)`), `aaltonomous/baselines/classical/brain/planner.py:449-450` (`can_done` is true whenever nothing is classically binned), `aaltonomous/baselines/classical/policy.py:2570` (`("stop", …)` once the plan is adopted and `next_call()` is `None`), `aaltonomous/baselines/classical/policy.py:1838` (`"stop"` and `"done"` both become `("none", …)`).

`adopted` is set from a non-empty step list, including a list whose every step is invalid. The no-early-done guard only marks that one step rejected and then `continue`s. The next `next_call()` is `None`, and the adopted plan returns `("stop", …)`. `_choose` turns that into `("none", …)`, and `_after_look` sets `done=True`.

Sequence, `brain.planner` on, instruction “first put the red ones left, then the blue ones right”. The resolver binds red→left and blue→right, so classical sorting would score +2 per object.

1. The model returns `[{"tool":"done","args":{}}]`, or a red pick plus `done`. `classical_has_work` is true, so `done` fails `explicit_plan_satisfied` and is stored as an invalid step. The list is non-empty, so `adopted` is true.
2. `next_call()` skips the invalid `done` and returns `None`.
3. `_planner_select` returns `("stop", …)`. The episode ends. Blue, and red if it was never picked, stay on the counter (0).

Same exit when the strict mapping binds nothing (`can_done` returns true immediately) and the model emits `done`. Same exit when every step is junk (`obj_id` not in the scene, extra keys, a bare array recovered from the instruction text): the injection tests only assert that no step has `outcome=="ok"`. `test_c02_system_done_is_rejected_while_work_remains`, `test_c04_…`, `test_c11_done_while_objects_remain_is_rejected`, and `test_c13_…` stay green while the episode is already over.

**Fix:** treat a plan as adopted only when it contains a valid step. On a rejected `done` or an exhausted plan, return `None` from `_planner_select` so `_choose` keeps the classical candidates. Delete the `if not classical_has_work: return True` shortcut; allow `done` only when `explicit_plan_satisfied` is true. Do not append those skipped ids to `self.failed` on the way out, or the classical pass will skip them too.

---

## 2. High — `if` / `unless` / `otherwise` proves the whole instruction after one move, and `closest` / `farthest` never check the bin

`aaltonomous/baselines/classical/brain/planner.py:241-245` (`if` / `unless` / `otherwise`), `aaltonomous/baselines/classical/brain/planner.py:223-240` (`closest` / `nearest` / `farthest`), used from `planner.py:337` and `planner.py:449-454`. The trigger regex is `_COMPLEX` (`planner.py:20-25`), which includes `if`, `unless`, `otherwise`, `closest to`, and `nearest to`.

`if` branch: any object whose colour or shape word appears anywhere in the sentence is “mentioned”. The proof returns true when `moves[0]` is one of those ids. It does not look at the other clause.

Sequence: “if you see a red cube put it in the left bin, otherwise put the blue cube in the right bin”, with both colours classically mapped so `classical_has_work` is true. The model emits `pick_place` of the red cube into left, then `done`. `described()` sees “red” in the full sentence, `moves[0]` is that object, and `validate_plan` accepts `done`. Blue stays on the counter (0). `test_complex_trigger` only checks that the sentence enables planner mode.

`closest` / `farthest`: the proof compares `moves[0]`’s id with the nearest or farthest object and returns. The bin in the sentence is not read. `pick_place` of an object whose `resolver_bin` is `None` is not rejected (`planner.py:294-296`, `policy.py:2602-2608`). Instruction “put the object closest to the center bin into the center bin”, resolver abstains on that object. The model picks it into the right bin and emits `done`. The proof returns true. `Scene.classify_cube` scores that as `wrong` (−1) when the task mapping’s bin for that colour is not right. `tools/planner_eval.py` `correct()` for tag `"condition"` has the same hole: any plan that also picks the star into the right bin is counted correct, whatever else it bins.

**Fix:** delete the `if` / `unless` / `otherwise` branch, or require every mapped object the sentence names to be moved to the bin that sentence names. For `closest` / `farthest`, require `moves[0]`’s bin to be the bin named in the sentence, and do not accept `done` while any other mapped object is still on the counter. Add a two-colour `if` / `otherwise` case and a closest case whose `resolver_bin` is `None` and whose bin is wrong.

---

## 3. High — `planner_wait` plans on the live Pinocchio data, and `reset()` does not retire that thread

`aaltonomous/baselines/classical/policy.py:1232-1246` (`planner_wait` starts `_choose_job` on `self`), `aaltonomous/baselines/classical/policy.py:2520-2524` (`_choose_job` calls `self._choose` with no generation), `aaltonomous/baselines/classical/policy.py:2545-2555` (`reach` calls `self.planner.plan` on `self.arm_cmd` / `self.kin`), `aaltonomous/baselines/classical/policy.py:299-304` and `:352-355` (`reset` drops `_job` and `_look_thread` and does not join).

`brain.planner` forces `threaded = True`. The look that sees a still-pending plan returns `("planner_pending", …)` and `_after_look` enters `planner_wait`. The follow-up `_choose` is a daemon thread on the live policy, not on the generation snapshot `_look_snapshot` builds for look workers (`policy.py:1296`). `validate_plan` then runs a full `planner.plan` per `pick_place` on `self.kin`.

`ab_planner` does this with `generation_workers` off. `finals_r22_brain` and `finals_r22_lyingprism` set `policy.generation_workers: true`, which snapshots look workers and still does not cover this thread. The runner stops calling `act()` at `time_limit` and the next trial calls `reset()` (`aaltonomous/deploy/runner.py` ends the episode on the clock; `reset` does not join).

Sequence:

1. Episode 1 is in `planner_wait`. The choose thread is inside `planner.plan` (Pinocchio `data` on `self.kin`), or inside `plan_brain.take` / `next_call`.
2. The env hits 120 s. `reset()` sets `arm_cmd = None`, calls `plan_brain.reset()` (new `AsyncRequest`, `steps = None`), installs episode 2’s mapping, and sets `_job = None`. The thread keeps the old job dict and keeps calling methods on `self`.
3. Episode 2’s `go_to` home uses the same `ArmKinematics.data`. `framesForwardKinematics` is not thread-safe. That is a control-thread exception, or a home trajectory built from episode 1’s mid-solve state.
4. The thread then appends to `self.failed` and can `mark` / `request` on episode 2’s `PlanLayer`. Episode 2 skips those xy positions for the rest of the 120 s (0).

`test_timeout_returns_no_plan_so_classical_can_continue` only checks `PlanLayer.take` on a `Scripted` backend. It never enters `planner_wait` or `reset`.

**Fix:** give `_choose_job` the same snapshot as `_look_job` (its own `Planner` and `ArmKinematics.data`). Capture the look generation when the thread starts; if `reset` has bumped it, write nothing to `self`. Bump that generation at the top of `reset`, before `plan_brain.reset()` and before clearing `arm_cmd`.

---

## 4. High — with `generation_workers` and `brain.planner` both on, the settled look confirms the wrong step

`aaltonomous/baselines/classical/policy.py:100-114` (`_LOOK_COMMIT_FIELDS` has no `_planner_execution`), `aaltonomous/baselines/classical/policy.py:1296` (shallow copy: `plan_brain` is shared, `_planner_execution` is not published), `aaltonomous/baselines/classical/policy.py:1950` (execution record written on whichever object runs `_choose`), `aaltonomous/baselines/classical/policy.py:2675-2694` (`_planner_observe` clears the record and `mark`s the shared plan).

`finals_r22_*` sets `policy.generation_workers`. Look runs on a snapshot worker. `planner_wait` then runs `_choose` on the real policy and stores `_planner_execution` there. The next look copies that dict into the worker. `plan_brain` was not copied, so `mark` advances the real plan. `self._planner_execution = None` rebinds only the worker. The commit list does not include `_planner_execution`, so the real policy keeps the first move’s record forever.

Sequence: plan is pick red→left, pick blue→right, `done`. Red’s move was started from `planner_wait`.

1. Next look’s worker observes red’s record, marks step 0 ok, and starts blue. The real policy still holds red’s record, and it executes the blue plan.
2. The look after that copies red’s record again. Red is gone, so `ok = not at_source` is true. `mark("ok")` runs on the current index, which is the blue step, whether or not blue is still on the counter.
3. The same look’s `_choose` sees `done`. `can_done` counts the false blue ok. The episode ends. Blue scores 0. One replan never runs, because `_planner_failed` is only reached from `_planner_observe` when the stale record says the move failed.

The other timing (the plan is already ready inside the snapshot `_look`, as with `brain.debug_wait`) never publishes `_planner_execution` at all. Observe no-ops. `next_call()` returns the same step every look. `_choose` then does `cands = [d0] + …` (`policy.py:1841`), which puts the object back in front even after `_attempt_excluded` dropped it. The same grasp repeats until `max_total_attempts` (18) or `_has_time` fails. That burns the 120 s on one object.

`test_strict_brain_handoff` does not set `generation_workers`. `test_act_never_blocks_with_a_lying_prism_in_view` never turns the planner on; `any(m for _, m in pol.events)` is true for the home move alone.

**Fix:** run `_planner_observe` on the policy that owns the execution record, after the look result is committed, and include `_planner_execution` in the commit (clear it on the real policy, not on the worker). Do not `mark` a step from a record whose `call` is not the step at `plan_brain.index`. Do not reinsert a forced target that `_attempt_excluded` already dropped.

---

## 5. Medium — a colour the resolver did not bind can be placed in any bin

`aaltonomous/baselines/classical/brain/planner.py:294-296` (authority only when `resolver_bin` is not `None`), `aaltonomous/baselines/classical/policy.py:2602-2608` (execution `_lookup` misses the same way and returns the model’s bin).

`sort_language` still scores every colour: `make_instruction` returns `dict(zip(colors, bins))`, and `Scene.classify_cube` returns `wrong` when the object is at rest in a different bin (−1). Leaving it on the counter is 0.

Sequence: instruction “Put the red objects in the left bin.” Strict binds red and abstains on blue and yellow (`_strict_leaves_colour`). The model emits `pick_place` of a blue id into `"right"`. `resolver_bin` is `None`, so validation and `_planner_select` both accept it. Blue in the right bin is −1 when the task bin for blue is not right. The same hole is a homoglyph colour (`rеd`): it is absent from the mapping, and a `pick_place` into any of the three bins is legal.

`test_planner_is_asked_about_abstained_objects_and_red_keeps_its_bin` requires blue→right and yellow→center for that red-only sentence. `test_c07_homoglyph_colour_cannot_bind_a_bin_in_the_planner` only asserts `"red" not in mapping`. `wrong_bin()` never emits a step for an object with `resolver_bin is None`, so a hostile bin for the unbound colour is untested.

**Fix:** for an object with `resolver_bin is None`, accept `pick_place` only when the instruction names that object’s colour and the bin the sentence gives it; otherwise the only legal tools are `skip` and `wait`. Keep the existing authority check for objects that do have a resolver bin.

---

## Tests

| Test | What it actually locks |
| --- | --- |
| `test_c02_…`, `test_c11_…`, `test_c04_…`, `test_c13_…` | The hostile step is invalid or has no `outcome=="ok"`. They do not fail when finding 1 ends the episode with every object still on the counter. |
| `test_c07_homoglyph_…` | The homoglyph colour is absent from `mapping`. The scripted plan never tries to bin that colour. |
| `test_timeout_returns_no_plan_so_classical_can_continue` | `PlanLayer.take` returns `None` after a short timeout. `planner_wait`, classical fallback, and `reset` are not in the test. |
| `test_strict_brain_handoff.py` | Red’s bin is enforced, and the scripted blue/yellow bins are executed. `generation_workers` is off, so findings 3 and 4 do not run. |
| `test_act_never_blocks_with_a_lying_prism_in_view` | Wall time of `act()` under the threaded finals overlay. The event check passes on the home log line. It does not require a lying-prism plan, and it does not run `brain.planner`. |
| `test_flags_on_change_nothing_without_prisms`, `test_standing_prism_plan_is_unchanged_by_the_flags` | Cube streams and a standing-prism plan match the flags-off planner. These do lock the prism flags. |

---

## Checked, not a finding

**Flags off.** `brain.planner` false leaves `plan_brain` as `None`, so `_choose` stays on the classical / brain-decision path. `grasp.lying_prism` false makes `_lying_on` return false before any heading or height change. `perception.prism_orientation` false does not call `_apply_prism_orientation`. The new 2000-character bail-out in `_instruction_mapping` sits after `instruction.max_chars`; finals sets that cap to 2000, so a capped instruction never hits it. A resolve-on, max-chars-off instruction longer than 2000 now abstains instead of running the quadratic resolver. That is the 1 s reset watchdog, and it does not change a normal sentence.

**`act()` over 37 ms, and exceptions into `act()`.** Planner validation runs on the look or `planner_wait` thread (`threaded` is forced when `brain.enabled` and `brain.planner` are both true). `act()` returns `_hold()` while that thread runs. `_apply_prism_orientation` catches and returns. `_choose_job` and `_look_job` catch and store `("error", …)`, which `_after_look` turns into `done` rather than a raise. I did not find a path that raises out of `act()` or that runs the new IK on the control thread. The 27 Hz test above does not measure the worker.

**URDF.** `_plan_side_lying` uses the existing `_ik` / safety-filter path and restores `kin.hint` and `_lying` in `finally`. `test_joint_targets_inside_urdf_limits` checks the lying plans that succeed. No new hand posture is introduced; the configured `close_posture` is reused. `place_at` still goes through `plan_place_at` and the 2 cm / 6 cm gate in `_planner_place_at`.

**Retry cap in `PlanLayer`.** `replan` returns false once `replans >= 1` and does not submit another request. That cap is what makes finding 1’s stop permanent, and what finding 4 skips by never calling `_planner_failed`.
