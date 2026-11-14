# Why the real camera misses purple cubes (and orange prisms), and a flag-gated fix

Tree: `oliver/purple-recall` (base origin/felix/rc72-real 33aebe6), overlay finals_r22 (baseline) vs ab_purple_recall.
Frames: drop_2 head RGB, ep 0-39 only (40-47 never opened), depth=None, robot_q from the parquet. Colour-aware
visible-only recall, greedy 25 px, aliases orange<->red purple<->blue (tools/recall_ab.py on oliver/recall-ab).

Baseline note: the brief's numbers (0.355, purple 20/113, FP 116+7) reproduce on /home/oliver/work/ab-r21 with
finals_r21 (checked: 191/538 = 0.355, 20/113). On the rc72 tree finals_r22 reads 144/538 = 0.268, purple 18/113,
prism 6/79, FP 14. Different tree/overlay (scene.counter_z_prior 0.15, hue_guard off, bin.vote on), same failure.

## 1. Attribution (finals_r22, unmatched visible GT)

| rule that drops it | purple-cube (95 missed of 113) | orange-prism (73 missed of 79) |
|---|---|---|
| `_keep_mask`: nominal bin hull removes the pixels before blob grouping ("no blob") | 55 | 50 |
| `color_gate:speckle` (blob left after the hull cut is below area_frac x expected) | 38 | 23 |
| `color_gate:in_bin` (nominal 3D footprint) | 2 | 0 |
| `color_gate:workspace`, `sat_floor`, `hue_guard`, band / v-floor / min_area | 0 | 0 |

No-blob stage, per GT (HSV mask in a 36x36 window, then keep mask, FK arm mask, morph open): all 55 purple and all 50
prism no-blob misses die at the keep mask (HSV pixels present, 0 kept). The speckle rows are the same cause: the cube
straddles the hull edge and only a ~22 % remnant (median 911 px of an expected 4185, ratio 0.20-0.23) is left.
Evidence: with the hull replaced by the observed bin mass, purple speckle drops 38 -> 1.

Cause: with depth=None and no bin vote, the bins are at their nominal mat position. Their projected hull
(counter to rim) covers image rows 124-314 at col 400, but the real bins sit in rows 0-125. Everything on the counter
in front of the centre bin (rows 124-314, cols ~220-470) is masked. Purple (-> centre bin) and the prism sit exactly there.
Overlay picture: `scratchpad/e00a.png` (red = masked).

HSV of the missed purple tops (brightest 35 % of the cube pixels): median H 110, S 152, V 228; p10-p90 H 109-112,
S 144-177, V 196-247. Band blue = H 100-145, S>=70, V>=50; sat_floor blue 110 (S p10 is 144). Colour is not the problem.
Missed prism tops: median H 13, S 224, V 205 (red band H 0-16, S>=70, V>=70). Also fine.

The earlier suspects do not apply on rc72/r22: workspace rejects 0 purple (and workspace_fail_open does not help:
finals_r22_failopen 137/538 = 0.255, purple 18/113, FP 79 here). Turning color_gate off (brief: purple ~0.51, FP 1680) was not re-run on rc72.

## 2. Fix: `perception.bin_hull_observed` (default OFF)

When live bins are untrusted (no bin vote confidence, the state of every first look and of this offline replay), the
observed bin/tape mass (the existing `_observed_bin_bottom` detector) bounds the object mask instead of the nominal hull;
`margin_px` rows below it stay masked. `gate_follow` (default true) makes the colour gate's in_bin / workspace rules
stand down too. Without `gate_follow` the gate re-rejects the rescued cubes (purple 21/113, see row nf3).
Trusted bins / bin.vote still use the exact 3D footprint. No bin mass in view -> nominal hull (unchanged).
Note: gate_below_bins (oliver gatefix, in finals_r22_gate, not in r22) already does the same job. The new flag is its
mask-and-gate core without the rejected-list/blocked-region bookkeeping; at margin 3 it reproduces gate_below_bins
alone exactly (363/538 = 0.675).

## 3. A/B (colour-aware, visible-only, rc72 tree, overlay finals_r22 + flag)

| arm | overall | purple-cube | orange-prism | orange-cube | yellow-cube | yellow-cyl | FP total (blue/yellow/red) | dFP |
|---|---|---|---|---|---|---|---|---|
| baseline finals_r22 | 144/538 = 0.268 | 18/113 = 0.159 | 6/79 = 0.076 | 26/62 | 58/188 | 36/96 | 14 (5/9/0) | - |
| margin_px=3  | 363 = 0.675 | 112/113 = 0.991 | 50/79 = 0.633 | 33/62 | 111/188 | 57/96 | 34 (15/18/1) | +20 |
| margin_px=10 | 366 = 0.680 | 112/113 = 0.991 | 50/79 = 0.633 | 33/62 | 114/188 | 57/96 | 30 (15/15/0) | +16 |
| **margin_px=20** | 365 = 0.678 | 108/113 = 0.956 | 50/79 = 0.633 | 33/62 | 117/188 | 57/96 | 26 (14/12/0) | +12 |
| margin_px=30 | 357 = 0.664 | 104/113 = 0.920 | 45/79 = 0.570 | 33/62 | 119/188 | 56/96 | 22 (12/10/0) | +8 |
| margin_px=40 | 316 = 0.587 | 73/113 = 0.646 | 45/79 = 0.570 | 33/62 | 119/188 | 46/96 | 21 (11/10/0) | +7 |
| margin_px=50 | 283 = 0.526 | 41/113 = 0.363 | 45/79 = 0.570 | 33/62 | 118/188 | 46/96 | 20 | +6 |
| ablation 3, gate_follow=false | 186 = 0.346 | 21/113 = 0.186 | 38/79 = 0.481 | 33/62 | 55/188 | 39/96 | 31 | +17 |
| ref: finals_r22_gate (all gatefix flags) | 377 = 0.701 | 113/113 | 56/79 | 36/62 | 113/188 | 59/96 | 36 | +22 |

Acceptance (colour-aware recall up, FP <= +5): recall criterion met by every row; the FP criterion is NOT met by any
row that keeps purple (+8 at 30 gives purple 0.92, +12 at 20 gives 0.956). The extra FPs are 9 yellow (large cylinder-sized
blobs near x 190 in e17/e18 and e34-e39, with no matching visible GT; cause not investigated) and
~9 blue (image edges x 150-600, 2000-7000 px). Recall gained per FP is ~18-25 true matches. Recommendation: margin_px=20.

Leftover after margin 20: purple 5 missed (all speckle); orange-prism 29 missed (19 speckle with n_px/expected 0.12-0.30,
10 no blob). The prism remainder is the area floor (a triangle top is 1/sqrt3 of a cube top plus occlusion); the existing
`area_floor_geometric` flag is the lever, not touched here.

Caveat: this replays first-look state (no bin vote). On the robot, once bin.vote is confident the 3D footprint is shifted
and trusted, and the flag stays inactive, so the live gain is limited to untrusted looks.
