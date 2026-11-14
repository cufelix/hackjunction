# Fault drills

`test_fault_drills.py` runs the real `ClassicalPolicy` through short mock-world episodes (`mock_world.make_mock_env`,
virtual time, `AaltoEnv` + `SafetyFilter` unchanged) with one fault injected, against three overlays: `submission`
(as shipped), `finals_r21` and `finals_r22_robust` (r21 + the fixes below). Run:

    PYTHONPATH=. python -m unittest aaltonomous.baselines.classical.test_fault_drills -v      # ~4 min

Every drill asserts: no exception from `act()`/`reset()`; every row finite, 14 wide, inside the URDF joint limits;
per-tick speed <= `motion.arm_max_vel` / `hand_max_vel` (per segment, a runner reset restarts at the measured pose);
no hand point inside the counter or a bin wall (planner geometry) and wrist/palm inside the SafetyFilter workspace +
keep-out; the SafetyFilter never clipped or stopped. Plus per drill: unknown instruction -> no pick; no objects -> hold
and look again; latency p99 < 1/27 s (wall; a wall miss only counts when the calling thread's CPU time misses too).

Known failures are `expectedFailure` marks (`known(overlay, reason=...)`) so they stay visible and a fix shows up
as an unexpected success. Result key: PASS / KNOWN (recorded failure) / --.

| # | Drill | submission | finals_r21 | finals_r22_robust | Fix |
|---|---|---|---|---|---|
| 0 | clean control: safe + p99 latency + picks | KNOWN latency (p99 CPU 160-200 ms) | PASS | PASS | `policy.safety_shell` (already in r19b+) |
| 0b | configured `arm_max_acc` honoured | KNOWN (all) | KNOWN | KNOWN | none, see "Not fixed" |
| 1a/b/c | head depth None / zeros / intermittent | PASS | PASS | PASS | |
| 2 | NaN/inf in arm_q or hand_q on random ticks | PASS | PASS | PASS | `safety.nan_guard` |
| 2b | whole hand_q NaN on the very first tick | KNOWN | KNOWN | KNOWN | none, see "Not fixed" |
| 3 / 3b | stale RGB(+depth) while joints move | PASS | PASS | PASS | attempts stay <= `max_total_attempts` |
| 4a/b | empty / whitespace instruction abstains | KNOWN picks via fixed mapping | KNOWN | KNOWN | `instruction.empty_abstain` (NOT in r22, see below; test 4g proves it) |
| 4c/d | 5000-char / injection text abstains | KNOWN picks via FIXED fallback | PASS | PASS | `instruction.strict` (already in r19+) |
| 4e/f | instruction None; 5000-char real sentence | PASS / KNOWN latency | PASS | PASS | `instruction.max_chars`, shell |
| 5a | brain on, backend connection refused | KNOWN latency (shell off) | PASS | PASS | |
| 5b | brain on, backend sleeps 3 s | KNOWN latency (shell off) | PASS | PASS | brain request is async, classical still picks |
| 6a/b/c | reset mid-motion / twice quickly / before any act | PASS | PASS | PASS | |
| 7 | no objects visible: hold, look again, never done early | KNOWN `done` within seconds | KNOWN | PASS | `policy.no_objects_relook` |
| 8a/b/c | objects beyond reach / at the near edge / floor+behind | PASS | PASS | PASS | planner declines unreachable plans |
| 9a/b | all-black / all-white image: no pick | PASS | PASS | PASS | |
| 9c | saturated-orange image (tape): no pick | KNOWN picks "red" objects | KNOWN | PASS | `perception.flood_guard: 0.6` |
| 10a/b | act() at 60 Hz / 10 Hz with jitter | KNOWN latency | PASS | PASS | shell |
| 11 | empty grasp abort retreat clears counter + bins | PASS (stall detector never aborts) | KNOWN 15-16 mm into the counter, crosses a bin wall | PASS | `grasp.abort_lift_first` |

Latency measured on the dev laptop (p99 per `act()`): shell ON (r21/r22) wall < 1 ms to a few ms, CPU < 1 ms;
shell OFF (submission) 150-210 ms CPU on every look/plan, up to 1.2 s wall in the control episode. Submission
therefore fails the 27 Hz criterion whenever it plans; the shell is the existing fix and it is ON in every finals overlay.

## Findings and fixes

New flags (all default OFF, registered in `config.py` FLAGS, `config.yaml`, `overlays/submission.yaml` as OFF,
`test_flags.py`; flags-off behaviour is unchanged, the new branches are guarded):

* `grasp.abort_lift_first` (REAL BUG, r21): the empty-grasp abort (`held_detector: finger_sum`) retreats on a single
  joint-space line from the grasp pose to home with no clearance check. For some grasps (e.g. the right hand reaching
  to y=+0.09) the fingertips go 15-16 mm below the counter surface and across a bin wall. On the real robot an empty
  grasp is common. Fix: open the hand in place, replay the plan's own clearance-checked lift, then go home from the
  top. Falls back to the stock path if the arm is not on the plan's close/lift path.
* `policy.no_objects_relook`: with no detection at all and nothing picked yet, `_after_look` ended the episode
  (`done=True`). Now it settles and looks again until the time limit (never after a pick: the normal finish is unchanged).
* `perception.flood_guard` (0.6): one colour class covering more than 60 % of the frame (nominal frames: red/orange
  bins 35-37 %) is no colour information; no detections. Without it an orange-filled frame labels the real objects red.
* `instruction.empty_abstain`: empty / whitespace text abstains instead of the `policy.fixed_mapping` fallback.
  Deliberately NOT in `finals_r22_robust`: the fixed mapping is the organisers' real-robot sort_fixed (yellow left,
  blue center, red right) and abstaining forfeits those points if that task sends no text. Tim decides; flip it in the
  overlay if the finals always send text.

`overlays/finals_r22_robust.yaml` = finals_r21 + `grasp.abort_lift_first`, `policy.no_objects_relook`,
`perception.flood_guard: 0.6`.

## Not fixed (recorded)

* **Acceleration cap (0b)**: `planner.ramp()` / `go_to` start at the speed cap in one tick (24-46 rad/s^2 vs
  `arm_max_acc` 7-8); only `time_scale()` honours it and the SafetyFilter has no accel limit. Pre-existing, also in
  the clean episode. A change would alter every trajectory (not flag-small).
* **First-tick NaN (2b)**: whole `hand_q` NaN before any action: no pose to hold, nan_guard falls back to the open
  posture (1.5 rad hand jump, the filter clips 7 ticks). Safe but not 0 clips. NaN on any later tick is held correctly.
* **Mock limits**: the mock world has no contacts (grasps always read empty, hand tracks the command exactly), no
  arm self-occlusion and virtual time. The rate drills vary wall-clock pacing only; env time is virtual.
* One transient `arm speed 0.978 > cap 0.9` was seen once in the intermittent-depth drill on r22 before the abort fix
  and did not reproduce in 3 reruns or the final full runs (timing-dependent; watch for it).
