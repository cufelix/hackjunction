# Prism speckle floor (perception.prism_speckle_floor, default OFF)

Branch oliver/prism-speckle (felix/rc72-real 33aebe6 + 1 commit). Overlay ab_prism_speckle = finals_r22_gate + flag.
Offline, drop_2 ep<40, colour-aware, visible-only. Raw: /home/oliver/work/ps2/{base,final,...}.

## Rule
In `_detect_blob`, red/orange blobs get `det.prism_like` = compact (hull solidity >= 0.85), not a sliver (minAreaRect aspect <= 2.0,
short side >= 22 px) and area / minEnclosingTriangle(hull) >= 0.75. In `_color_gate_why` a prism_like blob uses
min(color_gate.area_frac, 0.08) instead of 0.4. Everything else (bin, workspace, FK mask) unchanged.
Params: perception.prism_speckle_floor_params {area_frac, min_fill, min_solidity, max_aspect, min_short_px, min_px}.

## Result (default params)
| arm | overall | orange-prism | orange-cube | purple | yellow | FP | FP tape/wall/hand/other/phantom | FP red |
|---|---|---|---|---|---|---|---|---|
| finals_r22_gate | 377/538 | 56/79 | 36/62 | 113/113 | 113/188, 59/96 | 36 | 1/0/16/0/19 | 4 |
| ab_prism_speckle | 401/538 | 64/79 | 52/62 | 113/113 | 113/188, 59/96 | 40 | 1/0/16/0/23 | 8 |
Acceptance met: prism +8, FP 40 <= 41, no class down. The gain is shared with orange-cube (+16): small cubes also pass the triangle test.

## Sweep (all shape-sliver gates on unless noted; fill = min_fill, a = area_frac)
| fill / a | overall | prism | cube | FP |
|---|---|---|---|---|
| .75 / .10, no aspect/short gate (sol .85) | 389 | 56 | 48 | 38 |
| .55 / .10, sol .75, no aspect gate | 397 | 62 | 50 | 182 (tape flood: 89) |
| .65 / .10 same | 392 | 57 | 50 | 110 |
| .55-.60 / .10, asp 2.8, short 18, sol .80 | 395 | 60 | 50 | 48-52 |
| .60 / .10, asp 2.0, short 22, sol .85 | 395 | 60 | 50 | 45 |
| .65 / .10 (strict gates) | 391 | 57 | 49 | 42 |
| .65 / .08 | 403 | 65 | 53 | 45 |
| .60 / .08 | 403 | 65 | 53 | 54 |
| .70 / .08 | 402 | 64 | 53 | 42 |
| .72 / .08 | 402 | 64 | 53 | 42 |
| **.75 / .08 (default)** | **401** | **64** | **52** | **40** |
| .70 / .06 | 402 | 64 | 53 | 47 |
| .75 / .05 | 402 | 64 | 53 | 44 |
Lesson: tape flaps pass a loose triangle fill; the aspect (<= 2.0) and short side (>= 22 px) gates are what remove them.
Prism blobs measure fill 0.57-0.74, aspect 1.0-3.3. Remaining 15 prism misses: no blob (21 GT rows), gate_below_bins, split_by_solidity:area_low.

## Checks
- test_flags, test_release_gate, test_color_gate (3 new), test_recall: 81 passed. Full classical suite: 816 passed; failures = known
  test_finger_lag x5 + test_calibration_fit warm-fit CPU; msgpack installed so those 7 pass. test_recall pinned the Detection field list: updated.
- tools/equivalence_check.py flags off: IDENTICAL vs felix/rc72-real.
- Recommended: area_frac 0.08, min_fill 0.75 (the defaults). Enable only on top of finals_r22_gate; not validated on drop_2 ep 40-47 (held out) or live.
