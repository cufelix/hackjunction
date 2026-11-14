# No-blob misses on drop_2 head frames (ep 0-39, depth=None, finals_r21 @4540b9b)

Baseline reproduced: visible recall 195/538 = 0.362, FP 123. Missed visible = 343.
No-blob by my definition (no accepted/rejected/colour-gate-rejected det within 25 px of GT centroid): **205** of the 343
(not 315/506; the user's count used a different denominator/definition, not reconciled). Scripts: analyze.py, cls.py (this dir).

## Headline
HSV band, S/V floors, 3x3 opening and the 120 px floor explain **0 of 205**. Every no-blob GT object has a
colour mask covering ~55-70 % of its bbox (median raw_frac 0.56-0.71) and medians inside the band. The loss is elsewhere:

1. **Nominal bin-hull pixel mask (_keep_mask -> _bin_hulls_mask, depth=None)** removes the object entirely: 111/205.
   The projected hull of the 3 nominal bins covers image rows 124-314; the real bins end near row ~216 (tape 182-216).
   Objects on the mat in front of/at the bin edge (rows 190-285) are inside the hull -> 0 mask pixels, no blob.
2. **GT partial label vs whole blob (centroid offset 26-59 px)**: 94/205. A blob >= 120 px exists over the GT bbox but
   its centroid is > 25 px from the label centroid. GT notes: "shadowed lower face; top+side split", "merged with
   yellow cube in bootstrap; manual box", "cap+side split". These are label-geometry / merge cases, not mask losses.

## Cause table (no-blob, visible, per class)
| class | n | hull mask removes object | blob exists, centroid >25 px off (partial label / merge) | hue/S/V out | opening / area floor | FK arm mask |
|---|---|---|---|---|---|---|
| orange-cube | 13 | 2 | 11 | 0 | 0 | 0 |
| orange-prism | 49 | 22 | 27 | 0 | 0 | 0 |
| purple-cube | 55 | 42 | 13 | 0 | 0 | 0 |
| yellow-cube | 50 | 11 | 39 | 0 | 0 | 0 |
| yellow-cylinder | 38 | 18 | 20 | 0 | 0 | 0 |

GT median HSV (H,S,V) of the misses vs band (orange: H<=22|>=170 S>=70 V>=70; purple: H100-145 S>=70 V>=50; yellow: H24-60 S>=35 V>=90):
- orange-cube hull 9.5,212,135 | offset 11,177,208
- orange-prism hull 12,220,188 | offset 11,213,186
- purple-cube hull 112,140,207 | offset 111,151,192  (H 109-115 sits well inside 100-145)
- yellow-cube hull 39,77,230 | offset 24,144,130 (shadow face; H 23 at p10 is below the 24 yellow edge, S/V fine)
- yellow-cylinder hull 34,92,209 | offset 36,95,229
All inside the bands; nothing a band edge would recover.

Blob stats for the offset cases: area median orange-prism 340 px, purple 775, yellow-cube 5953 (merged), cylinder 2783.

## Measured arms (real-frame harness, same tree/overlay)
| arm (--set) | recall | orange-prism | purple | FP total (hand) |
|---|---|---|---|---|
| baseline finals_r21 | 195/538 = 0.362 | 27/79 | 20/113 | 123 (116) |
| perception.bin_pixel_margin=-0.015 | 201 = 0.374 | 31 | 19 | 158 (132; phantom 26) |
| perception.bin_pixel_margin=-0.03 | 226 = 0.420 | 54 | 20 | 152 (145) |
| perception.bin_pixel_margin=-0.045 | 215 = 0.400 | 39 | 20 | 152 (149) |
| perception.bin_pixel_margin=-0.06 | 201 = 0.374 | 28 | 19 | 146 |
| perception.bin_pixel_margin=-0.10 | 198 = 0.368 | 28 | 19 | 140 |
| bin_exclusion_margin=-0.06 alone | 195 = 0.362 | 27 | 20 | 123 (no effect) |
| pixel -0.03 + exclusion -0.03 | 226 = 0.420 | 54 | 20 | 152 |
| pixel -0.06 + exclusion -0.06 / -0.10 | 196 = 0.364 | 28 | 19 | 146 |

Rule: recall up AND FP <= +10. **No arm passes.** Best recall arm (-0.03, +31 recall, +29 FP, all "hand" class: the
newly unmasked area lets coloured pieces of the hands pass; the hand-FP class is already 116 in baseline).
Non-monotonic: past -0.03 the next gate (in_bin:* reject, which also uses the nominal footprint) takes the blob back;
purple stays at 20/113 because its blob is rejected as in_bin:center (purple cubes sit at the nominal bin front edge).

## Recommendation
- Do NOT retune HSV / morphology / min_area for this loss: no evidence it helps (0/205).
- The real fix is geometry, not a config key we can set: the nominal bin footprint is ~100 px too close to the camera in
  these frames. Needs a correct bin position (bin.geometry_runtime with bin.vote, currently OFF) or a pitch calibration.
  CAVEAT: the harness has depth=None, so perception.online_pitch_cal=table cannot run; on the robot the camera
  projection is corrected and the hull may line up far better. The 111 hull losses may be largely a harness artifact. Verify on
  a depth-bearing run before acting.
- If a stop-gap is wanted: perception.bin_pixel_margin=-0.03 gives +31 matches (0.362 -> 0.420) but fails the FP gate
  (+29 hand FPs); pair it with a hand FP filter first. Not recommended as shipped.
- Not config keys: the hull height (scene wall_height) and bin outer_size live in scene config; the 25 px matching radius
  is harness. The offset class (94) is label/merge geometry; the existing split_blobs handles part of it.
