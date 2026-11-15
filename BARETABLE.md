# Bare glossy table audit (perception / calibration / bin voting)

Finding: the policy never reads the mat TEXTURE. There is no grid / line detection, no mat-colour mask and no global
background colour statistic or white balance. "mat" in the code is only a coordinate frame (scene.yaml mat_origin /
mat_size), which is geometry from calibration and does not depend on a print. The real risks of a bare glossy table are
specular glare, mirror images and depth dropouts. Line numbers are in aaltonomous/baselines/classical/perception.py.

| # | Place | Assumption / risk | Fix |
|---|-------|-------------------|-----|
| 1 | `_masks_hsv` (~l.1000), `hsv_robust_v2` yellow `[24,35,90]-[60,255,255]` (config.yaml) | Yellow accepts S >= 35, V >= 90. A highlight (V 255, S 3) blended with the brown table is pale yellow (H 25, S 55) and passes. On the grey mat this never occurred. | `perception.glare_reject`: small V>=245 & S<=40 components on the counter plane, plus a low-S halo, removed from all colour masks (`_glare_filter`, `bare_table.glare_kill_mask`). |
| 2 | `detect` per-blob, `_detect_blob` (~l.1230) | A mirror image under an object is a same-colour blob with lower S. It has no valid depth (glossy returns holes) so `_from_plane` places it on the counter as an extra object. `_keep_mask` keeps depth holes. | `perception.reflection_reject`: B below stronger A, centred, similar width, lower S, flipped-IoU >= 0.45, and depth (if any) at or below the plane (`_reflection_skip`, `bare_table.reflection_indices`). Two real stacked-in-image objects with valid object-height depth are kept. |
| 3 | `colour_confidence` (l.29-93), `sat_floor` (`_color_gate_sat_low`, l.~1730: yellow 45, blue 110) | Floors are absolute S values tuned on the mat/lab lighting; a glossy table lowers S on everything lit by glare. Yellow floor 45 passes glare halo (S~55). | Covered by glare_reject (removes the pixels before blobs form). sat_floor itself left as is (already ON in finals_r21). |
| 4 | `table_pitch_calibration` / `_counter_plane` (l.775, l.~600) | Plane fit uses depth only, not colour: mat-independent. Glossy specular spots give depth holes or noise; RANSAC with 8 mm inlier + 60 % inlier ratio rejects them. Risk: heavy dropout -> "insufficient valid depth" -> nominal pitch (safe fallback). | None needed. |
| 5 | `calibrate_bins`, pitch cal orange mask (l.~452, RGB ratios `orange_*`) | Orange-tape colour test on RGB ratios; a brown-orange glossy table (R>G>B, g/r 0.2-0.82) could pass as "orange". ROI is limited to the nominal rim rows and the dominant-edge histogram rejects isolated orange. Flag-gated (online_pitch_cal) and the finals use `table` mode (depth), not rims. | None for the finals config. If rim mode is ever used on a brown table, tighten `orange_*` or use `local_bg`. |
| 6 | `bin_vote.tape_votes` (bin_vote.py l.~174), `tape_hsv [[3,150,70],[8,255,180]]` | High S >= 150 requirement already excludes a brown/grey table and glare (S low); additionally requires rim height via depth. bin.vote is OFF in finals_r21/r22. | None. |
| 7 | Workspace bounds: `_reject` off_mat (l.1657, `counter_margin`), `_keep_mask` (l.1390), `_color_gate_workspace` (l.1744), `_counter_plane` mask window (l.~782) | Bounds come from scene.yaml geometry (mat_size / counter_size in the pelvis frame), not from a detected mat edge. Independent of whether the mat is printed. | None. |
| 8 | Mat-dependent global background statistics | None exist (checked: no median-of-background, no white balance, no mat colour). To be safe against future use, `perception.local_bg` provides the local alternative: a blob whose median H/S matches the ring of non-object pixels around it (the local table colour) is dropped (`bare_table.low_contrast_with_ring`). | `perception.local_bg` (ON in r22). Window 10 px, dH<=5, dS<=25. |
| 9 | `mock_world.py` `MAT_RGB` + checker grid (tests only) | Tests only ever saw the grid mat. | `test_bare_table.py` renders a bare brown-grey gradient table with highlights and reflections. |
| 10 | `hue_guard`, `colour_kmeans` | k-means (`_fit_colour_kmeans`) fits on object blob pixels only (after keep mask); glare/reflection blobs could pollute it if kept. Off in finals. | Glare/reflections are removed before blobs form when the flags are on. |

## Flags (default OFF, flags-off byte-identical, never raise: every helper fails open)

- `perception.glare_reject.enabled` (v_min 245, s_max 40, max_px 900, dilate_px 6, halo_s_max 65, plane_h_max 0.5)
- `perception.reflection_reject.enabled` (max_gap_px 40, dx_frac 0.6, width_ratio [0.5,1.8], s_ratio 0.9, min_iou 0.45, h_max 0.5)
- `perception.local_bg.enabled` (ring_px 10, dh 5, ds 25)

Overlay: `overlays/finals_r22_bare.yaml` = finals_r21 + the three flags. Code: `bare_table.py`, hooks in `Perception.detect`.
Tests: `test_bare_table.py`.

## Limits

Synthetic frames only; real glare/reflection thresholds are untuned against the organiser's recordings. Reflection test
needs the mirror blob to be dimmer (S ratio) than its source; a mirror with equal saturation is not caught. local_bg
alone can cost up to ~6 ms on a frame with 6 blobs (full-frame nonzero per blob; not optimised); all three flags together
measured about +1 ms on the synthetic frame (24.3 vs 25.7 ms), and test_added_cost_per_frame_under_3ms asserts < 3 ms.
