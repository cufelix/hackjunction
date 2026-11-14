# DR fit plan: Drop 1 statistics -> `dr/train.yaml` knobs

Purpose: when real frames arrive, turn `stats.json` into a new `dr/*.yaml` in minutes. No data and no GPU were needed to write this.
Sources read (public starter repo `zupermiez/aaltonomous-humanoid`, main, 2026-11-13): `aaltonomous/data/drop1/stats.py`, `capture.py`, `test_drop1.py`, `aaltonomous/sim/dr/train.yaml`, `aaltonomous/sim/aalto_task/dr.py` (`sample`, `corrupt`). `hooks.py` was only grepped, not read.
Integrity: this plan uses only the train profile and our own captures. It never touches the held-out config.

## 0. Read first

1. **The MockSource numbers are not measurements.** `capture.py:137` defaults (rgb 2.0, depth 3.0 mm, holes 2 %, object holes 20 %) are fixture values for offline tests. `test_drop1.py:22` uses 3.0 / 0.02 / 0.25 only to check that `stats.py` recovers what was injected. They say nothing about the real camera. Do not describe them as "the organisers' noise model", and do not tune against them. Real numbers exist only after a real drop.
2. **`stats.json` has `calibration_extrinsics_default: true` hard-coded** (`stats.py` `analyse`). Until a real `calibration.yaml` is in the session, `calib_offset_px` measures the default extrinsics, not the robot.
3. **Same code on both sides.** `capture.py --backend sim` writes the same session format. Run `stats.py` on a sim session and on the real session and compare the two `stats.json` files. Fit the difference, not the absolute value.
4. **One venue, a few minutes of data.** Treat a real value as evidence for where the range must reach, not as the centre. Keep train ranges wider than the real value (section 4).

## 1. How the sim applies the camera knobs (`dr.py:128`)

- RGB: `rgb * gain + N(0, rgb_noise_std)`, clipped to uint8. One gain per episode, noise iid per pixel and frame.
- Depth: `d += N(0,1) * (depth_noise_mm + depth_noise_rel * d_mm)`, then iid holes with probability `depth_hole_frac`. Holes are set to 0.
- Note the depth std model is **linear in range**. `stats.py` fits **`k * z^2`**. They cannot match everywhere, so fit over the working range only (section 2.2).
- There is **no mask input to `corrupt()`**, so no per-object hole rate exists. See the new-knob table.

## 2. Table: statistic -> knob -> how to compute -> if real is worse

### 2.1 RGB (per camera, per `lights` condition; use `head_rgb` first)

| `stats.json` statistic | Knob | Compute | If real exceeds train |
|---|---|---|---|
| `noise_std_median` | `camera.rgb_noise_std` | real median minus sim median at the same lights, added to the current 3.0. Both read from `stats.py` output | set the knob to the real **p95 across lights**, not the median |
| `noise_std_p95` | `camera.rgb_noise_std` (tail check) | if p95 / median > 2 the noise is spatially uneven (JPEG blocks, dark corners). A single sigma cannot model it | add nothing; note it. Cheaper to test robustness at 2x sigma |
| `noise_std_per_channel` | none (one scalar knob) | if blue std > 1.5x green std, the scalar underestimates blue | extension: per-channel sigma. Low priority |
| `luma_mean` per lights | `lighting.key_intensity_scale`, `dome_intensity_scale`, `textures.mat_scale` | ratio real / sim luma at the **same** lights. Range needed = [min ratio, max ratio] over lights, then intersect with the current [0.6, 1.4] / [0.5, 1.5] | widen the side that is missed, with 10 % margin |
| `luma_p05`, `luma_p95` | same | spread (p95 - p05) real vs sim: more spread in real means harder shadows or glare. Raise `key_pos_jitter` | raise `key_pos_jitter` (0.4, 0.4, 0.3) in steps of 25 % |
| `channel_mean`, `wb_r_over_g`, `wb_b_over_g` | `lighting.key_rgb_jitter` (0.08), `dome_rgb_jitter` (0.05), `textures.mat_rgb_jitter` | `max(|real_wb / sim_wb - 1|)` over lights and over the session. Use the ratio against sim, because the scene is not grey so absolute gray-world is not 1 | set jitter to that max, plus 25 % |
| `clipped_high_frac`, `clipped_low_frac` | upper/lower end of `key_intensity_scale` | real clipped fraction larger than sim at scale 1.4 / 0.6: the range is too narrow | raise 1.4 or lower 0.6 until sim clips as much as real |
| `distinct_frames` (of `n_frames`) | none | below `n_frames` means stale or duplicated frames. This is a latency fact, not a DR fact | pass to the latency notes, not here |

### 2.2 Depth, empty scene (head camera)

| statistic | Knob | Compute | If real exceeds train |
|---|---|---|---|
| `std_vs_depth` bins (`z_mm`, `std_mm_median`) | `camera.depth_noise_mm` = `a`, `camera.depth_noise_rel` = `b` | least squares `std_mm = a + b * z_mm` over bins with `z_mm` in the **working range** (head to counter, about 350-600 mm; check `range_mm_median`). If `a < 0`, set `a = 0` and refit `b` | train currently gives 1.0 + 0.002 z = 2.0 mm at 500 mm. If real at 500 mm is above 2.0, take the fitted a, b and add 25 % margin |
| `model_std_mm_eq_k_z2.k_mm_per_m2` | same | convert for cross-checking only: std at z metres = `k * z^2`. At 0.5 m: `k * 0.25` mm. Compare to the linear fit at 500 mm. They should agree within 10 % in the working range | if they disagree a lot, trust the bins, not the quadratic |
| `std_mm_median`, `std_mm_p95` | `depth_noise_mm` | p95 / median > 3 means a few pixels carry the noise (edges, shiny parts). Do not raise the global sigma for it | keep global sigma at the median fit; cover the tail with the object hole knob below |
| `hole_frac_sometimes` | `camera.depth_hole_frac` (0.01) | **restrict to the counter region first.** `stats.py` averages the whole image, so out-of-range background dominates. Use the projected counter ROI (the same projection `object_masks` uses) | set to the real value, plus 25 % |
| `hole_frac_always` | none today | permanent holes (specular, out of range) are not iid, so `depth_hole_frac` cannot model them. If non-zero **inside the counter ROI**, it is a new knob: fixed hole blobs per episode | add `depth_hole_blobs` only if the counter ROI shows it |
| `range_mm_median` | none, sanity check | compare to sim. A mismatch above about 1 cm means the real head pose or the extrinsics differ from the sim | feeds `pose_jitter_m` check below |

### 2.3 Objects (head camera; only trust rows where `mask_source == "colour"`)

| statistic | Knob | Compute | If real exceeds train |
|---|---|---|---|
| `hole_frac` per object, split by `shape` | **NEW `camera.depth_hole_frac_obj`** (does not exist in `train.yaml`) | iid part = `hole_frac - hole_frac_always`. Take the **p90 across objects and lights**, per shape. Prism is the case to watch | train has no equivalent (0 effectively beyond the global 1 %). Add the knob (section 3) and set to the real p90 |
| `hole_frac_always` per object | none | pixels never valid inside the mask: edge pixels or registration error. If large, the mask is wrong, not the camera | check the mask overlay PNGs (`masks_<step>.png`) before using any object number |
| `depth_std_mm_median` per object | `depth_noise_mm` / `depth_noise_rel` | ratio to the empty-counter std at the same z. Ratio near 1 means no change. Above 1.5 means objects are noisier (matte PLA, slant) | add optional `depth_noise_obj_scale` (new), else raise the global fit to the object p90 |
| `top_height_m_measured` - `expected_height_m` | **NEW `camera.depth_bias_mm`** | difference in mm over objects. Median bias is a systematic depth offset, the p90 height is noise-biased high by roughly 1 sigma | train has no bias. If the median bias exceeds 3 mm, add a bias range of +/- that value |
| `calib_offset_px`, `calib_offset_px_median` | `camera.pose_jitter_m` (0.01), `pose_jitter_deg` (1.0) | metric error = `calib_offset_px * z / fx` (`fx` from the calibration). Compare with 0.01 m. **Valid only if `calibration_extrinsics_default` is false** | if the metric error exceeds 1 cm, widen. Otherwise leave it |
| `mask_sources` | none, a **gate** | if most objects are `projection`, the object stats are box stats and must not be fitted | re-capture with better light or fix the HSV windows before fitting |

### 2.4 Knobs `stats.py` cannot fit today

| Knob | Why no stat | What would give one |
|---|---|---|
| `objects.hue_jitter` (0.02), `sat_scale`, `val_scale` | `stats.py` uses HSV only to build the mask. It does not report the object colour | add to `object_stats`: median H, S, V inside the mask. Then `hue_jitter = max |H - H_nominal| / 360` (turns), `sat_scale` and `val_scale` ranges from S/S_nom and V/V_nom across lights. Nominal colours: `scene.yaml` `objects.colors` |
| `camera.rgb_gain` [0.9, 1.1] | `stats.py` averages over frames, so auto-exposure drift is invisible | add per-frame mean luma per step and report its relative std and min/max. That is the gain range |
| `textures.counter_rgb_jitter`, `bin_rgb_jitter` | no ROI statistic | median colour in the bin and counter ROIs, per lights |
| `mass_scale`, `friction_scale` | not visible in images | measure on the robot only. Organisers: pads unmeasured, expect less grip than sim 1.2 / 1.0. Keep the train range [0.75, 1.1]; consider lowering the lower bound |
| `xy_jitter`, `yaw_jitter_deg` | staff placement | not a camera quantity |

## 3. Code change needed for the new object knobs

`corrupt(rgb, depth, cam, rng)` (`dr.py:128`) has no object mask. Minimal change, kept out of the held-out path:

1. `sample()`: read `depth_hole_frac_obj` and `depth_bias_mm` from the `camera:` section into `out["camera"]`, default 0 (so old configs behave the same).
2. `corrupt(..., obj_mask=None)`: hole probability `np.where(obj_mask, hole_obj, hole)`; add `depth_bias_mm` to valid pixels.
3. `hooks.py:203`: pass the object mask. Source unknown, `hooks.py` was not read. If the sim has an instance-segmentation channel use it; otherwise project the layout the way `stats.py:object_masks` does.

Owner: GPU lane (needs a sim run to verify). Nothing here is applied yet.

## 4. Procedure when a drop arrives (about 20 minutes, CPU)

1. `python -m aaltonomous.data.drop1.stats data/drop1/<session>` writes `stats/stats.json`. Check `mask_sources` and `calibration_extrinsics_default` first (the gates).
2. Capture the same steps in sim: `capture.py --backend sim --yes --out runs/drop1`, then the same `stats.py`. Needs the GPU box.
3. Fill the table above, one row at a time, real vs sim. Write the result as `dr/fitted.yaml`, a **new file**. Do not edit `train.yaml`.
4. Rule for every range: the real value (p95 across lights, 25 % margin) is the **upper end** of the new range, the nominal stays inside it. Do not narrow any train range to the real value. One venue is one sample.
5. Keep two profiles: `train` (wide) and `fitted` (real-centred). Report the sim score on both. The pair is the sim-to-real story for the 25 % generality score.

## 5. Open items

- Is the real head camera running auto-exposure and auto-white-balance with JPEG q85? It was reported as a lead, unverified. `wb_*` per session and the `rgb_gain` addition above would show it.
- Real data format when a drop comes: LeRobot v3.0, `lerobot==0.7.0`, `observation.images.head_depth` as 480x640 HEVC 12-bit lossless depth, linear 0-4.095 m at 1 mm per code, 0 = invalid. `stats.py` reads `session/<step>/<cam>/*.png`, so a LeRobot drop needs a converter to PNG frames first.
- Grasp consequence (from Felix's handbook 3.2.2): if object holes reach 20 %, the top-face centroid should be a trimmed median with a minimum point count. That is a policy change, independent of this table.
