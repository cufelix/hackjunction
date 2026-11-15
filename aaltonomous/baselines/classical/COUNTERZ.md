# Counter height (counter_z): consumers and the wide window

Flags (default OFF, flags-off byte-identical): `calib.wide_counter_window`, `planner.bin_wall_extra_m` (finals 0.01).
Overlay: `overlays/finals_r22_counter.yaml` = finals_r21 + both. Params: `calib.plane.wide_lo_m` 0.08, `wide_hi_m` 0.22,
`counter_z_prior_m` (absolute pelvis z, null = none).

## Estimation
- `Perception._counter_plane` (plane fit, shared by calibrate_bins and the bin.vote rim vote): window was +-6 cm (`plane_z_window`)
  around nominal. Wide: when no runtime height exists, `_wide_seed` takes the lowest well-populated layer of a 1 cm z histogram
  inside nominal -8..+22 cm over the mat, then the usual +-6 cm / 3 iteration robust fit (inlier + MAD-style trimming) runs around it.
- `Perception._plane_height`: wide = seed pass, reject estimate outside the window, then the two tight passes; tilt gate
  (`max_tilt_deg`) and `min_inliers` unchanged. Failure -> nominal (or the prior, see below).
- `update_counter_height` (called each detect): runs with wide even without `apply_height`; median of first N looks; clamp is the
  window (-8..+22 cm) instead of +-max_dz_m. Unmeasured (depth None / rejected) -> `counter_z_prior` (scene attribute, else
  `calib.plane.counter_z_prior_m`), else nominal runtime=None and the colour-gate workspace rule fails OPEN.
- `Scene.mat_z_follows_counter` (set by the flag): `mat_to_pelvis` / `pelvis_to_mat` z shifts with the runtime counter, so every
  mat-frame z (bin rim corners 0..wall_height, bin hulls, rim votes) moves with it. Off: mat z is relative to the nominal origin.

## Consumers of the runtime height (all read `scene.counter_z`, never `origin[2]` / the scene constant)
- perception: `_keep_mask` height gate, `_from_plane` ray/plane z, `_color_gate` height + `_color_gate_rim_ok`, `_reject` height,
  dz_top, area-by-distance fallback (`counter_z + height`), `_calibrate_bin_rims` rim z, `_calibrate_table` plane_height,
  `calibrate_bins` height delta, bin hulls (`_bin_hulls_mask`, `_bin_footprint_mask`), `_near_bin`.
- bin_vote: rim depth window (`min_wall_height_m .. wall_height + 0.015` above the fitted plane), tape `rim_z`.
- planner: `clearance` (hand above counter), `bin_clearance` / `hits_bin_wall` rim keepout, grasp z, carry z, drop z, relocation z.
- policy: scored-bin estimate (`pelvis_to_mat`, bin footprint is xy only).
- sim/oracle.py, mock_world renderer, scene.suggested_workspace, run.py keepout read `counter_z` too.
- NOT covered (interface/ is off limits): `safety.py` `counter_keepout_max` z default 0.03 is a fixed pelvis-frame number; run.py
  sets it from the NOMINAL scene counter at startup. A +15 cm counter sits inside that keepout box: check at the venue.

## planner.bin_wall_extra_m
Added to the wall height in `Planner._wall`: release height `z_drop`, carry height `z_carry` (when wall+H+0.03 dominates),
`bin_clearance` (descend/transit rim check) and `hits_bin_wall` (rim keepout). 0.0 = stock.

## Known behaviour with the flag OFF at +15 cm (documented in test_counter_z)
Plane/bin evidence window +-6 cm and the height gates use nominal: the counter is not found, objects land at the wrong z/xy and
are rejected (e.g. `color_gate:workspace`). counter_z stays nominal.
