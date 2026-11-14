# Adversarial review: `timi/rimdetect`

Scope: `02f57de` vs `team/timi/gatefix` (`7c3bee5`), the parent this branch is one commit ahead of. Diff is `aaltonomous/baselines/classical` only (`perception.py`, `rim_detect.py`, config, `finals_r22_bins`, tests). Flag-off detection on `finals_r22_gate` matches a config with the new keys removed (the byte-identity test passes). No joint commands in this diff, so nothing new vs the URDF. Rim fitting on a 640×480 frame is a few milliseconds, under the 37 ms `act()` budget. No new retry loop and no new time-budget path.

Reproduced with the `finals_r22_bins` perception flags (color gate, live bins, hue guard, solidity split, shape) on frames rendered from the head calibration. Bins shifted +14 cm, a yellow cube already in the left bin, a yellow cube on the table in front of the centre bin, and a synthesized depth image whose valid fraction passes `min_valid_frac`.

## 1. High — one found rim turns off every in-bin check, including columns with no rim

`rim_detect.py:29`, `rim_detect.py:151`, `rim_detect.py:152`, `perception.py:1089`, `perception.py:1655`, `perception.py:1674`, `perception.py:1923`, `perception.py:2036`, `perception.py:2055`

`min_bins: 2` is never read. Any non-empty `RimResult` is a success. `row_limit` is `-1e6` everywhere except the columns of the segments that survived, and `segs[:max_bins]` drops any run past the third. `detect()` then sets `_image_bin_gate_active` for the whole frame. That one flag skips all of the following, not only the covered columns:

- projected bin hulls and the 3D footprint test in `_keep_mask`
- `in_bin:<name>` in `_reject`
- `color_gate:in_bin` and the workspace near-edge / `ws_x` upper bound (`ws_x` is `[0.12, 0.45]`)

`finals_r22_bins` also sets `gate_below_bins`, but that full-width fallback does not run once the image gate is active. A dim, occluded, or out-of-`row_frac` side bin is therefore a hole: every pixel in its columns is a candidate, and a 3D point sitting inside that bin is still accepted.

Failure: with only the centre rim found, the same frame returns the table yellow (good) and the yellow cube already in the left bin at pelvis xy about `(0.62, 0.25)`, labelled `cylinder`, 2615 px. Valid depth does not stop it (`with_depth` defaults true, so a good depth image still takes the image-gate branch). The flag-off path on that frame returns no detections. The overlay uses `policy.bin_lookup: resolve_bin`, so the target is `(colour, shape)` then colour. A yellow cube that was already in the yellow bin and is now seen as a yellow cylinder is placed in the cylinder’s bin. That is −1 instead of +2. Colour-only mapping re-picks the sorted cube out of its own bin and tries to drop it back; a missed drop lands in the neighbour or off the counter (−1 / 0).

Minimal fix: if `len(bins) < min_bins`, or the names are not a unique cover of the nominal bins, return no rims and let `gate_below_bins` / nominal exclusion run. For a result you do keep, fill uncovered columns with the median measured rim row (the same fallback `_observed_bin_bottom` uses), and keep the 3D `in_bin` / `color_gate:in_bin` test wherever `row_limit` is still the sentinel.

## 2. Medium — rim hint shifts only the centre bin, and it survives a miss and `reset_calibration`

`perception.py:1265`, `perception.py:1287`, `perception.py:1642`, `perception.py:305`, `test_bins_from_image.py:157`

`hint` defaults false and `finals_r22_bins` does not set it. With `hint: true` the path is live, and the test that claims it tracks a 10 cm shift does not catch the failure.

On the head camera the left and right rims are `clipped` (`u0 == 0` or `u1 == 639`; nominal near edges project to about u 181..−40 and 506..727). `_rim_hint` skips clipped rims and any residual above 6 px. A uniform +10 cm shift therefore stores a hint for **centre only** (`dx ≈ 0.094`). `bin_geometry` adds that as pelvis `(dx, 0)` → mat `(0, dx)`, so the centre box moves to mat y ≈ 0.46 and left/right stay at 0.365. The test only loops the bins that made it into the dict, then checks `set(centres) == bin names`. Both pass while two bins did not move.

`_detect_rims` returns `None` before clearing `bin_rim_hint` when the next frame finds nothing, and `reset_calibration` never clears it either. After a good frame, a blank frame still reports the old centre shift. Episode 2 therefore starts with episode 1’s centre box until a later frame both finds rims and recomputes the hint. While the image gate is off (trusted poses, or `with_depth: false` with a valid depth image), that stale centre box is the 3D exclusion volume: a cube in the real centre bin is outside it and becomes a pick, and a table cube that falls in the stale box is labelled `in_bin` and left behind.

Minimal fix: clear `bin_rim_hint` at the start of `detect()` and in `reset_calibration`. Apply a hint only when every nominal bin has one, or apply the median `dx` to all three; otherwise apply nothing. Assert the three `centre_mat` y values, and assert a failed frame and a reset both leave the hint empty.

## 3. Medium — the depth-path test never feeds depth

`test_bins_from_image.py:144`

`test_depth_path_uses_the_image_masks_too` calls `detect` with `np.zeros((480, 640), uint16)`. Valid depth is 150–2000 mm and `min_valid_frac` is 0.2, so `_depth_ok` is false and `_keep_mask` takes the no-depth return. The assertion (the front cube is blue) is the same branch as `depth=None`. It stays green if a real depth image stops applying the rim mask, and it never places an object inside a bin. That is the case in finding 1: with a depth image of median ~635 mm and `_depth_ok` true, the in-bin yellow is still returned.

Minimal fix: build a depth image with a valid fraction above `min_valid_frac`, include a cube inside a shifted bin and one in front of it, and assert only the front cube survives. Add the one-rim case (side bins absent) and assert the in-bin cube is absent too.
