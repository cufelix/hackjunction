# Three-contact cage: derivation and first real run

Use `overlays/finals_r22_cage.yaml`. Both `grasp.cage_mode` and `grasp.tri_vertex_align` are off in the base config;
`finals_r22` remains unchanged.

Expected on the soft hand:

- Cubes use the same posture, point-in-palm, and heading path as r22.
- A trusted metric cylinder or standing triangular prism approaches with the object centred between index and middle.
  Both fingers close symmetrically (`finger_0` moves farther closed), while the thumb command continues past `closed`.
  Contact should stop the soft joints before the command.
- A standing prism rotates so the index-middle gap follows one vertex line. A pose estimate is preferred; otherwise
  the colour blob's minimum-area rectangle supplies the line.
- The existing 1.2 s close hold covers the measured 300--380 ms finger lag before lift.

## Values derived from our hand

All coordinates below are in the left `hand_palm_link` frame used by `ArmKinematics`; the right side mirrors palm y.
FK uses `third_party/xr_teleoperate/assets/g1/g1_body29_hand14.urdf` and the postures in
`aaltonomous/baselines/classical/config.yaml`.

1. At `hand.closed`, the index and middle fingertip pads are
   `[0.100927, -0.088917, +0.028500]` m and `[0.100927, -0.088917, -0.028500]` m. Their midpoint is therefore
   `[0.100927, -0.088917, 0]` m. The midpoint of the supported 40--60 mm object-width interval is 50 mm, so its
   centre is placed 25 mm behind the pads along palm x: `[0.075927, -0.088917, 0]` m. Rounded to 1 mm,
   `cage_point_in_palm = [0.076, -0.089, 0.000]`. Repeating the calculation at every width from 40 to 60 mm puts
   the ideal FK centre no more than 5.1 mm from this command, inside the required 10 mm bound.
2. `cage_finger_posture` starts with the existing r22 `closed_tight` posture
   `[0, 0, 1.2, -1.0, -0.6, -1.0, -0.6]`. Holding each distal finger joint fixed and flexing both `finger_0`
   joints solves `x_pad(closed_tight) - x_pad(q) = 0.010 m` at 0.109165 rad. Rounded to 0.001 rad, this moves
   both pads 9.985 mm inward in FK. The resulting full left-hand DDS vector is
   `[0, 0, 1.2, -1.109, -0.6, -1.109, -0.6]`; every component is inside the URDF limits.
3. Thumb overlap is measured from our existing `closed` posture, with thumb_0 and thumb_2 unchanged and only
   thumb_1 flexing. We target the middle of the requested band, 10 mm, along each object's contact normal. At the
   rear centre of a cylinder that normal is palm x, so solving FK for 10 mm palm-x pad travel gives 0.152161 rad.
   With
   the triangle's vertex line along the finger gap, the contacted equilateral-prism side has palm-x normal component
   `cos(30 deg)`; 10 mm normal penetration therefore needs `10/cos(30 deg) = 11.547` mm of palm-x travel, which FK
   reaches at 0.177369 rad. Rounded to 0.001 rad, `cage_thumb_extra_rad = {cube: 0, cylinder: 0.152, triangle:
   0.177}`. The generated
   thumb_1 targets are 0.052 and 0.077 rad respectively from a `closed` value of -0.100 rad, inside its URDF range
   `[-0.724, 1.047]`. Commands are clipped to the URDF limits after construction as a final guard.

First-run checks:

1. Start with one cylinder in the workspace centre and an operator on the stop. Confirm right DDS slots 3/4 move the
   middle finger and 5/6 move the index finger, with equal closure and no palm/self collision.
2. Confirm the object sits on palm z near zero: one finger on each side, thumb behind, no sideways ejection. Repeat
   once with about 10 mm lateral placement error in each direction.
3. Repeat with one standing prism. Confirm the gap points along a vertex and the derived thumb command compresses the foam
   without folding, extruding, or pinning it under a fingertip.
4. Watch the first lift for delayed settling or slip. Stop and reduce overlap if the object squirts out, a joint stays
   hard against its limit, or the foam remains deeply crushed after release.
5. Verify release before allowing a full episode. A deep cage can retain soft foam; stop if either finger carries the
   object out of the bin after the release command.
6. Run one cube last and compare its command/trajectory log with r22; any cube difference is a regression.
