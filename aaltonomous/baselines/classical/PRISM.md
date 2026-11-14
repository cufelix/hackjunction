# Lying prisms (perception.prism_orientation + grasp.lying_prism)

Organiser fact: prism orientation is random. A prism stands on a triangular face (top view = triangle) or lies on a
rectangular side (top view = rectangle, ridge along the long axis, lower than the standing height). The stock code
(classify_shape, yaw_tri_shift, closed_demo posture) assumes standing. Both flags are default OFF; overlay
`overlays/finals_r22_prism.yaml` = finals_r21 + both ON. Flags off = byte-identical (tested).

## Perception (`prism_orientation.py`, called from `Perception._from_depth`)
Inputs: all blob points and the top-band points (pelvis frame). Three cues, each normalised to a margin in [-1, 1]:
- fill = convex-hull area / min-area-rect area of the footprint. Triangle 1/2, rectangle 1; threshold `fill_min` 0.75 = midpoint.
- strip ratio = width of the top band across its own long axis / width of the footprint across it. Standing: 1 (top band
  is the whole face). Lying: 2*depth/tan 60 deg / side, with depth = (1 - top_percentile/100)*altitude + top_band
  (depth of the band below the ridge, sloped faces at 60 deg from horizontal). Threshold = midpoint of that and 1.
- height cue: only when |length - altitude| >= 2 top bands; standing top = length, lying top = altitude. In our scene
  altitude = 45 mm = length (objects.size, height_ratio 1) so this cue is silent; set `length_m` to the measured part
  length and it votes automatically.
Lying needs all active cues to agree, standing needs the two geometry cues to say standing, else None = current behaviour.
A footprint aspect ratio is reported but not used (lying footprint is side 52 x length 45 mm, nearly square).
Output: `Detection.prism_orientation`, `ridge_yaw` (mod pi, from the top-band major axis), `prism_xy` (centre of the ridge
strip = footprint centroid by symmetry, with xy_offset / y_affine applied), `prism_conf`. A lying prism gets shape
"triangle", path "lying" (so the standing-prism demo posture is not picked).

## Grasp (`Planner._plan_side_lying`)
- Closing axis: from URDF FK (`tips_in_palm`, closed/closed_tight) thumb tip and index/middle tips differ in palm x only
  (thumb 0.072 vs fingers 0.101 m, same palm y), so the jaw closes along palm x. Across the ridge = palm-x heading
  ridge +- 90 deg (`lying_headings`), tried nearest the arm's natural direction first (yaw_ref_deg / yaw_max_dev_deg, existing).
  Planned with the existing strong heading IK hint (yaw_hint_weight); accepted only if the achieved heading is within
  `lying_yaw_tol_deg` (10).
- Grasp xy = `prism_xy`. Grasp z = counter + altitude/3 (centroid of the triangular section; `object_dims("triangle")`
  apothem = size/3 = 15 mm). The jaw there spans the section width side*(1 - z/altitude) = 34.6 mm vs closed_tight gap
  ~29 mm -> ~5 mm squeeze (tested). No new hand posture: the configured `grasp.close_posture` (closed_tight) with the existing
  point_in_palm; the demo posture is skipped for lying prisms.
- lying_yaw_tol_deg: 22.5 mm half length * sin 10 deg = 3.9 mm < the squeeze. Not applied to relocation / corrective picks
  (stock path). No heading reachable -> falls back to the stock plan (offline: ridge yaws 0-60 deg need a heading beyond
  the natural range for the right arm and fall back).
- Retry-vary yaw variants still override (stock behaviour on a second try).

## Check on the first real run
1. Log a few lying and standing prisms with `perception.shape_diag`; print `prism_orientation`, `prism_conf`, fill, strip ratio
   (`prism_orientation()` returns them). Real depth is noisier: if standing prisms read lying (or the reverse), tune `fill_min`.
2. Measure the real prism length and altitude; set `perception.prism_orientation_params.length_m` and check object size.
3. Ridge yaw sign: put a lying prism with its ridge along pelvis x and y; hand heading must be across it.
4. Grasp height 15 mm: fingertip clearance (12 mm) may raise z (existing loop); check the soft fingers (kp 1.5, lag 300-380 ms)
   hold with the ~5 mm squeeze, raise `grasp.close_hold_s` effect, watch `[grasp_diag]` held yes/no.
5. A lying prism is near-cube-like in the footprint: confirm it is not routed as a cube (shape "triangle", path "lying").
