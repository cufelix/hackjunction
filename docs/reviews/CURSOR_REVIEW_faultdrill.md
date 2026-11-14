# Adversarial review: `timi/faultdrill`

Base is `team/felix/rc6` (`4540b9b`). This branch is one commit, `b87da42`. Flags-off paths were checked and do not change behaviour: `instruction.empty_abstain`, `grasp.abort_lift_first`, and `policy.no_objects_relook` return before any new work, and `perception.flood_guard` is `0` so the new branch in `detect()` does not run.

`grasp.abort_lift_first` was dry-run on spawn-region grasps (both arms, including the test_11 layouts). The open-in-place + plan lift + joint-space home path stayed above the counter, out of bin walls, inside the SafetyFilter workspace, inside the velocity cap, and shorter than the plan it replaces (about 3–4 s vs 10–13 s). No URDF-limit, `act()` block, reset-generation, or >120 s overrun showed up on that path.

## 1. High — `perception.flood_guard` uses the wrong pixels

**Where:** `aaltonomous/baselines/classical/perception.py:1063`

**What is wrong:** The trip is `mask.mean() / 255` over the raw HSV mask of the whole frame, taken before `& keep`. That count includes the orange bins (already ~37% of a normal mock frame, all inside the red range of `hsv_robust_v2`) and pixels with depth outside `depth_range_mm`, which `detect()` never turns into objects. `finals_r22_robust` sets the threshold to `0.6` and also turns on `policy.no_objects_relook`.

Two failures on a seed-0 mock frame (head RGB 480×640, r22 config, flood compared at 0.6):

1. **Wrong bin, guard silent.** Paint rows 240:400 orange `(255, 106, 0)` — the band that covers the objects. Red fraction of the frame is **0.568**, so the guard does not fire. `detect()` then returns the yellow object at pelvis `(0.27, 0.02)` and the blue object at `(0.31, 0.09)` as **red**. r22's mapping is yellow→left, blue→center, red→right, so both are carried to the right bin: **−1 each** instead of a correct placement. Same calls with the guard forced off show this is the guard missing the flood, not a later filter.
2. **Correct colours, guard wipes the episode.** Blend the same frame 9% toward that orange. Every new red pixel is invalid depth (`new red valid = 0`). Object labels stay yellow/blue/red, identical to the unblended frame. Frame red fraction jumps **0.372 → 0.637**, the guard returns no detections, and `policy.no_objects_relook` (`policy.py:2106`) holds and looks again until `t` is within 1 s of the time limit. A scene r21 would sort scores **0** and sits until the deadline. Fraction of red among valid-depth ∩ keep pixels stays **0.457**, the same as the nominal frame.

A fully orange frame is still 1.0 on every denominator, so the tape case the drill actually tests can keep tripping.

**Why the suite stays green:** `test_09c_image_saturated_orange_tape` (`test_fault_drills.py:545`) replaces every pixel with orange and only asserts that nothing is picked. Fraction 1.0 cannot see a 57% band that still publishes red labels, or a 64% frame whose labels are still right. `test_00` uses an unblended frame (red ≈ 0.37), so it also passes.

**Minimal fix:** Trip on the fraction of one colour inside the pixels `detect()` can actually group: valid depth and the keep mask (bins already removed), not `m.mean()` of the raw frame. On these three images that fraction is nominal **0.457**, object-band **0.699** (would trip, blocking the wrong-bin), 9% background blend **0.457** (would not trip), full orange **1.0** (still trips). Keep the 0.6 threshold. Add two asserts next to test_09c: the row band must not yield a yellow or blue object labelled red, and a blend that only reddens invalid-depth pixels must still return those colours.
