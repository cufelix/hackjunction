Reviewed the requested commit `fbbdbd5` against its parent/integration base. The branch moved to `96b07ab` while I was reviewing; findings below are for `fbbdbd5`, as requested. No files were modified.

## Findings

### Critical

1. **Top-3 truncation can permanently starve valid objects and make the policy declare the episode finished.**  
   `perception.py:1298-1304` always returns the same three largest blobs whenever more than six survive. The downstream failed/attempt filtering happens later in `policy.py:831-894`. Therefore, if those three are unplannable, exhausted, or false-positive large blobs, the hidden candidates never reach the policy; it can return `none` and finish at `policy.py:1772-1775` while real objects remain.

   Raw `n_px` also favors nearby, square, unsplit, or arm/bin blobs rather than candidate quality. The sort is technically deterministic—stable sort over deterministic component order—but deterministic starvation is still starvation.

   **Fix:** Do not destructively truncate in perception. Pass all bounded candidates downstream and let the policy filter failed/attempted objects before selecting top-k. If a hard perception bound is required, retain at least one candidate per color/connected region and rotate or exclude previously failed tracks. Append truncated candidates to `rejected` with `color_gate:top_k`.

### High

2. **A real object 0–2 cm outside a bin is rejected whenever its depth is missing or insufficient.**  
   Plane fallback is intentionally produced for depth holes/missing streams at `perception.py:967-979`, but `perception.py:1340-1347` categorically rejects a near-edge plane detection because `_color_gate_rim_ok()` requires `method == "depth"`. The new test at `test_color_gate.py:119-123` explicitly codifies this false negative.

   With a NaN or absent depth frame, `_keep_mask()` retains unknown pixels and detection falls back to the plane path; the gate then rejects a perfectly real object whose center lies in this 2 cm band. This is especially likely for objects directly in front of a bin, where the bin already causes depth occlusion and clipped blobs.

   **Fix:** Treat missing depth as “cannot disambiguate,” not “bin.” Keep an accepted plane detection unless independent image evidence identifies the rim, or reduce the outside edge band to zero for plane detections. A safer rule is to reject only inside a trusted live footprint and merely clear shape/confidence near its edge.

3. **The nominal footprint is inconsistent with the available live/perceived-bin geometry, and fixing only `_color_gate_bin()` is insufficient.**  
   The new gate reads nominal mat-frame centers directly at `perception.py:1330-1339`. The frame conversion itself is correct—pelvis `(x,y)` is swapped into mat `(y,x)` by `Scene.pelvis_to_mat()`—but the geometry source is stale.

   Earlier stages also consume nominal bins:

   - `perception.py:1115-1118`: depth mask
   - `perception.py:1122-1137`: no-depth projected hull
   - `perception.py:1139-1147`: near-bin shape guard
   - `perception.py:1278-1282`: final stock bin rejection
   - `planner.py:132-180`: nominal `bin_xy`, `drop_xy`, and slots

   Consequently, a live footprint added only at the post-filter cannot recover an object already removed by `_keep_mask()` or `_reject()`.

   **Fix:** Introduce one runtime bin-geometry accessor and use it in all five perception paths plus planner targets. Do not mutate `scene.cfg`, because scoring and nominal layout consumers share it.

4. **The “rim OK” predicate also admits the exact false positive it is meant to suppress.**  
   `perception.py:1344-1347` accepts any depth detection 3–7 cm above the counter. The configured bin rim is 6 cm and a real object top is 4.5 cm, so height cannot distinguish them. If calibration error leaves a strip of the physical rim just outside the nominal footprint, that strip has `method="depth"`, positive depth support, and height ≈6 cm—therefore it passes.

   **Fix:** First align the footprint to live geometry. For residual edge blobs, use evidence that distinguishes an object from a rim: compactness/aspect ratio, metric top-cloud extent, object-sized support, or distance of the entire object footprint from the wall. Do not call a shared 3–7 cm band a rim discriminator.

### Medium

5. **The area floor conflicts with `split_blobs` and can reject parts the splitter explicitly accepted.**  
   Split components may be accepted down to `0.3 * expected_area` at `perception.py:1040-1059`. The post-filter then independently demands `0.4 * (fx*0.045/Z)^2` at `perception.py:1313-1316`. These two expected-area definitions use different coefficients and can disagree, so a valid touching-object split can immediately be discarded.

   **Fix:** Compute expected area once and carry it with the split result, or share one helper and one minimum-part fraction. Tests should exercise `detect()` with touching objects rather than only synthetic `Detection` instances.

6. **The area formula hard-codes scene size and assumes square pixels.**  
   `perception.py:1313-1315` hard-codes `0.045` despite `self.size` being initialized from configurable perception/scene geometry at `perception.py:174`. It also uses `fx²`; projected area should normally scale with `fx * fy`. This silently becomes wrong for another object set, scene, crop, or non-square intrinsics.

   **Fix:** Use `self.size` and `fx * fy`, with an explicit shape/fill coefficient or the existing `expected_area_px()` machinery. Validate the threshold against cylinders and triangular prisms, not just oversized synthetic square blobs.

7. **The advertised “3D footprint reject” is effectively unreachable in normal `detect()`.**  
   Objects inside a nominal footprint are removed before grouping by `perception.py:1115-1118`, or rejected with a larger `bin_exclusion_margin` at `perception.py:1278-1282`. `_color_gate_bin()` only receives already accepted detections, so its `gap == 0` branch at `perception.py:1337-1338` normally cannot fire. The unit test calls `_color_gate()` directly and therefore misses this integration behavior.

   **Fix:** Consolidate the bin decision into one geometry layer, or apply the new rule at the earlier mask/reject seam. Add an end-to-end `detect()` test proving that the new gate changes the intended case.

8. **Malformed color-gate tuning can raise inside `detect()`.**  
   `config.py:194-219` validates several other flag parameters but none of the color-gate block. Runtime code assumes all keys exist and are well-typed:

   - missing `max_cands`/`top_k`: `KeyError` at `perception.py:1298-1299`
   - malformed `edge_h`: unpacking error at `perception.py:1346`
   - nonnumeric values: conversion errors at `perception.py:1298`, `1313-1315`, or `1339`
   - NaN/negative/zero parameters produce silent nonsensical behavior

   **Fix:** Validate at config load: finite `area_frac >= 0`, positive integer thresholds, `0 <= top_k <= max_cands`, finite nonnegative `edge_m`, and two finite ordered `edge_h` values.

### Low

9. **Top-k removals disappear from debug output.**  
   `perception.py:1303` slices them away without appending them to `rejected`, contradicting `_color_gate()`’s documentation at lines 1287-1289. The test at `test_color_gate.py:139-147` actually requires `rejected == []`.

   **Fix:** Append `(det, "color_gate:top_k")` for every truncated detection. This is important for diagnosing whether real objects were hidden.

10. **Clearing a plane triangle leaves stale shape metadata and logs the pre-gate label.**  
    `perception.py:1351-1352` clears only `shape` and `shape_conf`. `shape_path` and `shape_diag["label"]` remain from the blob classifier, and shape diagnostics were already logged at `perception.py:987-988`. The returned detection can therefore say `shape=None` while diagnostics still report `triangle`.

    **Fix:** Clear or rewrite `shape_path`, shape diagnostics, and any triangle-only fields together, and perform diagnostic logging after the post-filter.

## Live-bin wiring

There are two distinct concepts that must not be conflated:

- `calibrate_bins()` at `perception.py:701-864` solves a global correction satisfying approximately `observed_points + estimate = nominal_bins`. It stores that correction as `self.xy_offset` at lines 862-864, and detections receive it at lines 1226 and 1258.
- Therefore, under its intended interpretation—camera translation error—comparing corrected detections against nominal bin centers is already algebraically consistent. Adding that offset to nominal centers again would double-correct.
- If interpreted as actual bin displacement instead, the observed live center is `nominal_center_pelvis - cal.xy_offset`, because of the sign used in the search at `perception.py:798-817`. But then applying the same offset to every counter-object detection is wrong; camera error and physical bin movement need separate state.

True per-bin/drop-target estimates exist on other branches:

- `team/felix/bin-vote:planner.py:131-162` exposes `Planner.bin_shift` in pelvis `(dx,dy)` and already applies it to `bin_xy`, `drop_xy`, and `drop_slots`.
- `team/felix/bin-vote:policy.py:234-243` latches the voted shifts.
- `team/timi/bin-drop:perception.py:97-130` exposes `Perception.bin_layout`, including perceived center/yaw.
- `team/timi/bin-drop:planner.py:138-184` consumes that layout and validates perceived drop targets.

The clean wiring is to feed one of those accepted runtime layouts into a shared geometry provider. Convert a pelvis shift to mat with `[dy, dx]`; no sign flip is needed for `Planner.bin_shift`. Then use the same provider in `_keep_mask`, `_bin_hulls_mask`, `_near_bin`, `_reject`, `_color_gate_bin`, `Planner.bin_xy`, `drop_xy`, and `drop_slots`.

## Other conclusions

- I found no pelvis↔mat axis swap or optical-depth unit error in the new fbbdbd5 gate: `Z` is in metres and the mat conversion is correct.
- Missing/NaN sensor depth generally falls back to the plane path; the dangerous behavior is specifically the near-bin rejection described above.
- Shape-only instructions will deliberately become unresolved after a plane triangle is cleared (`test_color_gate.py:167-171`). That is safe abstention but means such an object cannot be sorted without a default/color mapping.
- I could not execute the tests in this checkout because the available Python environment lacks NumPy. Static review and Git inspection completed; the worktree remained unchanged.
