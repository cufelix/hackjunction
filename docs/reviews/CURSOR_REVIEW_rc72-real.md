# Adversarial review: `team/felix/rc7-pre...team/felix/rc72-real`

Scope: `git diff team/felix/rc7-pre...team/felix/rc72-real -- aaltonomous tools` (commit `33aebe6`). `tools/` has no changes. The diff adds three default-off flags: `perception.color_gate.workspace_fail_open`, `perception.hue_guard.yellow_fix`, `perception.rim_z_from_depth`.

Scoring assumed below: +2 correct bin, −1 wrong bin, 0 left on the counter. Episodes are 120 s. Head camera from `calibration.yaml`: origin `(0.054, 0.018, 0.474)` m in the pelvis frame. Workspace keep is pelvis `x <= 0.39` and `|y| <= 0.40`. Bin near edge is pelvis `x = 0.40`. Nominal counter z is `0.01`; `scene.counter_z_prior` on both new overlays is `0.15`.

---

## 1. High — fail-open accepts a plane blob and the planner still grasps the uncorrected point

**Where:** `aaltonomous/baselines/classical/perception.py:2307`, `:2365-2376`, `:2178`. Overlay: `overlays/finals_r22_failopen.yaml:8` (`scene.counter_z_prior: 0.15`) and `:78` (`workspace_fail_open: true`). Gate order is `_color_gate_workspace(det) or _color_gate_bin(det)`.

**What is wrong:** The reprojected `(x, y)` are locals. `det.top` is not written. `_color_gate_bin` and the planner both consume `det.top`. `dz` is `workspace_assumed_counter_z - nominal_counter_z` (`0.15 - 0.01 = 0.14`), but `_from_plane` already intersects `scene.counter_z`, which returns the prior `0.15` whenever `counter_z_runtime` is still `None`. Any exception in the reprojection, including a zero denominator, `return None` and skips the workspace test.

**Failure:** `finals_r22_failopen` runs `calib.plane.apply_height`. `Planner.commit_counter_height(None)` (`planner.py:249-251`) leaves `counter_z_runtime` as `None` on a rejected fit or the 3 s preflight timeout (`policy.py:562-567`). For the rest of the episode the plane is already at the prior and fail-open shifts the test point by another 14 cm toward the camera.

A no-depth blob whose true aim point sits in the 2 cm lip in front of a bin (mat `y` just under `0.26`, pelvis `x` about `0.395-0.40`) fails the workspace test on its real `xy`. The shifted test point lands near pelvis `x ≈ 0.25` and passes. The bin rule does not reject it: `color_gate:bin_edge` is depth-only (`perception.py` `_color_gate_bin`). The grasp stays on the lip, about 0.42 m from the pelvis, inside reach. Bin plastic is orange and falls in the robust red hue, so the fixed map sends that pick to the right bin. The hand is driven into the near wall; a cube knocked into a neighbor scores −1.

The same hole exists with the prior unset, which is the configuration the unit test builds. A back-row object at mat `(0.56, 0.19)` (inside `layout.region_x/y`) is kept by the full gate: the nominal-plane hit is pelvis `(0.476, 0.154)`, 1 mm outside every bin box so not `in_bin`, while the corrected test point `(0.34, 0.11)` is inside the workspace. Reach of the stored point is 0.50 m. The arm goes to the gap between the center and left bins, 15 cm behind the cube.

**Vacuous test:** `test_gate_failopen.py:53` `test_on_unmeasured_keeps_real_object` only calls `_color_gate_workspace`. Its fixture, mat `(0.45, 0.155)` on a 0.15 counter, stores pelvis `(0.424, -0.009)`, mat `(0.441, 0.284)`, which is inside the center bin (`y` `0.26-0.47`). The production `or _color_gate_bin` returns `color_gate:in_bin`. The test never loads `finals_r22_failopen`, so it never combines the prior with `counter_z_runtime is None`.

**Minimal fix:** Reproject once, write that `xy` and a consistent `z` onto `det.top`, then run workspace and bin on the stored point. Set `dz` from the plane `_from_plane` used (`scene.counter_z`), so a live prior makes `dz = 0`. On exception, return `color_gate:workspace`. Assert the full gate and `det.top` for the lip point above and for mat `(0.56, 0.19)`, both with and without `counter_z_prior`.

---

## 2. High — measured rim is used for the first picks, then later drops fall back to the nominal wall

**Where:** `perception.py:1078-1094` and `:1359`. `policy.py:104-105`, `:1290`, `:1295`, `:1310`. Overlay: `overlays/finals_r22_rimz.yaml:75` and `:77` (`generation_workers: true` together with `rim_z_from_depth: true`).

**What is wrong:** With `policy.generation_workers`, `_look_snapshot` shallow-copies the scene and the perception `__dict__`. `update_rim_z` assigns `worker.scene.rim_z_runtime`. `_commit_look_result` copies `counter_z_runtime` only. `_LOOK_PERCEPTION_FIELDS` has `_ah_est` / `_ah_tries` and not `_rz_est` / `_rz_tries`.

`_rz_est` is a dict, so the worker appends into the live policy's dict. `_rz_tries += 1` rebinds the worker's own int; the live counter stays 0. The stop condition is "every bin that has appeared already has `looks` samples", and a bin is published as soon as it has one sample (`if v`, not `len(v) >= n_looks`). `update_rim_z` runs at the start of `detect`, before the FK mask is built, and `_rim_z_looks` does not receive `robot_q`.

**Failure:** On `finals_r22_rimz`, look 1 appends one wall sample and plans the first pick on the worker scene at that single-frame rim (up to `max_delta_m = 0.04`). A hand or cube in the wall ring is part of the p90. Object bottom at release is `wall + drop_above_rim` (`0.06 + 0.04`). A rim 4 cm low puts the bottom on the nominal rim; the cube clips the wall and can bounce into the neighbor (−1).

After three samples, `_rz_est` is full on the live object, so later workers hit the early return at `perception.py:1078` and never assign `rim_z_runtime` on that look's scene copy. The main scene is still `None`. Picks 2-5 drop at `wall_height` 0.06. A real rim 3 cm higher leaves about 1 cm of the 4 cm `drop_above_rim` margin; a rim at the +4 cm clamp leaves the bottom on the rim. The same early return freezes any bin the first looks did not see (arm occluding one box at home): once the visible bins have three samples, the occluded bin is never measured.

`reset_calibration` (`policy.py:382`) rebinds `_rz_est` and clears `rim_z_runtime` only on the live objects. A stale look already holds the old dict and writes the old scene copy, and the generation check drops its plan. The hole is the opposite one: the estimate that should survive on the live scene is the value that is never committed. H7 does not cover this write.

**Vacuous tests:**

- `test_rim_z.py:78` plans on the same `Scene` `detect` just wrote. It never builds a `ClassicalPolicy`, never `_look_snapshot`, never `_commit_look_result`. A second look after `n_looks` is not planned.
- `test_rim_z.py:113` `test_clamped` comments that +6 cm is outside the window and must stay nominal, then asserts `rim_z <= nominal + 0.04`. Nominal and a wrong clamp to +4 cm both pass.
- `test_rim_z.py:99-102` skips the clearance assert whenever `bin_clearance` is infinite, which is the home pose the test uses (the comment on line 98 says so). `hits_bin_wall` then compares two falses.

**Minimal fix:** Put the applied rim dict on `_LookResult` and assign `self.scene.rim_z_runtime` in `_commit_look_result`. Deep-copy `_rz_est` and `_rz_tries` in `_LOOK_PERCEPTION_FIELDS`. Publish a bin only after `n_looks` samples. Keep scanning until every bin has `n_looks` or the try cap hits. Run the rim measure on depth with the FK arm removed, or skip a bin whose ring contains the hand. Tests: after a committed look, a fresh snapshot's drop `z` is still the measured rim; `test_clamped` asserts equality with nominal; clearance uses a hand pose that is over the left bin and asserts the delta with no `isfinite` guard.

---

## 3. High — `yellow_fix` does not run on its overlay, and with `hue_guard` plus `hsv_set: robust` it places the hue the guard abstains

**Where:** `hue_guard.py:46-52`. `perception.py:1486` (`if self.hue_guard`) and `:1714-1715`. Overlay: `overlays/finals_r22_failopen.yaml:57` (`perception.hue_guard.enabled: false`), `:19` (`perception.hsv_set: robust`), `:77` (`yellow_fix: true`). Fixed map on that overlay: yellow → left, red → right.

**What is wrong:** `judge(..., yellow_fix=)` is only reached from `_hue_guard`, which `detect` skips when `hue_guard.enabled` is false. The overlay turns the child flag on and the parent off. Relabel runs before `low_s_red` and uses `classify_hue`, which does not look at saturation. `hsv_robust` yellow is H 18-52 (`config.yaml` `hsv_robust`). `hsv_robust_v2` yellow starts at 24, which is what the tests pass in.

**Failure:** On `finals_r22_failopen` as committed, a yellow cube whose component label is red and whose own top face is H 40, S 80 stays red. The fixed map puts it in the right bin. Score −1. The flag that was added to relabel it never runs. `r22` already does this; the new overlay does not change it, so the A/B cannot show the fix.

Turning `hue_guard.enabled` on so the flag can run, and leaving `hsv_set: robust`, does the other wrong thing. H 20, S 80 is inside robust yellow, so `yellow_fix` relabels it and `low_s_red` no longer sees a red label. `judge` returns `("yellow", None)`. The hue-guard comment (`hue_guard.py:39-40`) is that a yellow top that has drifted to H 20 at S 73 is labelled red and must not pass. Under robust_v2, H 20 is still red (red goes to 22) and stays abstained. Under the overlay's robust set it is placed in the left bin. An orange or bin sheen at that hue scores −1. Real orange in the same config comment is H 9-15; H 18-22 is the band this rule exists to refuse.

**Vacuous test:** `test_gate_failopen.py:77-90` calls `judge` directly with `hsv_robust_v2` from `load_config(overlay="none")`. It never constructs `Perception` with `hue_guard.enabled` false, and it never classifies H 20 against `hsv_robust`. `test_on_real_low_s_orange_still_abstains` uses H 12, which is red in both sets.

**Minimal fix:** If `yellow_fix` is set, run that relabel even when the rest of hue_guard is off, or refuse to load the overlay unless `hue_guard.enabled` is true. Relabel only when the hue is inside the v2 yellow range (H ≥ 24), and leave H 18-23 on the existing `low_s_red` / band abstain. Set the overlay's `hsv_set` to the set the relabel was measured on. Add one `detect` test on `finals_r22_failopen` for a red-labelled H 40 candidate, and one `judge` test at H 20, S 80 on `hsv_robust` that still returns `(None, "low_s_red")`.

---

## Checked, no bug filed

- **Flags off.** `rim_z()` / `rim_wall()` return `wall_height` while `rim_z_runtime` is `None`. The workspace block is behind `workspace_fail_open`. `yellow_fix` defaults false, and `judge` is unchanged for that default. `test_rim_z.py:131` compares waypoints with the flag off.
- **`act()` budget and exceptions.** `update_rim_z` runs inside `detect` on the look worker. `act()` polls that job. Rim failures are caught at `perception.py:1084`. Hue-guard failures were already fail-closed. No new path raises out of `act()` or holds the control thread for the rim pass.
- **URDF.** This diff does not emit a joint target. A bad Cartesian point fails IK and the plan is discarded; `motion.output_caps` is unchanged.
- **Hold-only states, infinite retries, 120 s budget.** No new state whose only exit is `_hold()`, and no new attempt loop. Extra false plane blobs cost one failed plan each, then `failed`, bounded by `max_total_attempts`.
