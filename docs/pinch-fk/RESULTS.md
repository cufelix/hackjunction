# pinch FK results (CPU, pinocchio + URDF 817fb00, starter 36daf5b, sim profile)

LEFT hand, palm frame, mm. Signed distance: negative = penetration. i/m/t = index/middle/thumb tip.
Right hand = left * [1,-1,1] (verified exact mirror). Scripts in this folder.

## Current config (point_in_palm [99,-84,15], hand.closed)
- grasp height 22.5 at pelvis (0.30,0.10)/(0.25,0.15)/(0.35,0.05), no raise, preshape clearance 18.9, tilt 0.
- closed: thumb-index 46.0, index-middle 57.0.
- thumb NEVER touches: +11.7 cube0, +12.3 cube45, +14.9 cyl, +8..+22 prism. index penetrates -9.0 / -11.6 / -8.9 / -1.5..-6.6.
- sweep preshape->closed: index first contact at 0.70 (cube0), 0.75 (cube45, cyl), 0.65-0.95 (prism); thumb and middle never.
- centring alone (midpoint [82.85,-88.7,14.25]) with hand.closed: cube0 -4.4/-4.4, cube45 +0.4/+0.4, cyl +0.5/+0.5:
  no squeeze. Grid search: no point_in_palm gives >=3 mm on both thumb and index for cube and cylinder with hand.closed.
- lead per mm is ~0.018 rad/mm on thumb_1 (not 0.0039).

## pinch_v1 (deliverable)
```yaml
hand.closed_tight: [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]   # jaw 41.3 mm, thumb joints only
grasp.point_in_palm: [0.0862, -0.0864, 0.0143]                # right [0.0862, 0.0864, 0.0143]
```
- thumb only reaches 41 mm (thumb_1 -0.1 -> 0.0, thumb_2 1.2), in limits t0 +-1.047, t1 [-0.724,1.047], t2 [0,1.745]; thumb0 = 0.
- anti-wedging holds: palm z 14.3.
- penetration i/m/t: cube0 -7.8/+20.3/-7.8, cube45 -2.0/+18/-2.0, cyl -2.0/+22.7/-2.0; prism 0 +4.7/-/-0.3, 30 -0.7/-/+4.9, 60 -0.3/-/+4.7, 90 +4.9/-/-0.7.
- max lead (rad): cube0 0.143, cube45 0.052, cyl 0.050, prism 0/30/60/90 = 0.005/0.099/0/**0.200** (thumb sweep clips the apex early).
- backup 39.5 mm [0,0.05,1.2,-1,-0.6,-1,-0.6], point [0.0878,-0.0852,0.0143]: worst lead 0.228 (cube0 0.170) -> fails 0.15.
- widening: only jaw >= 45.6 mm passes 0.15 for all shapes, and then cube45/cyl get no squeeze. 41 mm is the narrowest passing all except prism 90.
- IK (0.30,+-0.10) ok both arms, err 0.003 mm, tilt 0; grasp height 22.5, preshape clearance 21.3, lowest finger point while closing 20.0.
- descent: no preshape tip lands on the top face; footprint clearance thumb >=15.4, index >=40.7, middle >=51.6.

## pinch_demo (deliverable; our preshape; joint order matches config.yaml + URDF)
```yaml
hand.closed_demo: [0.00, 0.92, 0.25, -0.50, -1.41, -0.49, -1.21]
grasp.point_in_palm: [0.1094, -0.0694, 0.0142]
```
- thumb-index 29.1 mm. 'auto' = [0.1067,-0.0689,0.0] is correct but averages index+middle (z=0 = wedging case); jaw midpoint z 14.2.
- penetration i/m/t: cube0 -8.2/+20.2/-8.2, cube45 -11.0/+14/-11.0, cyl -8.1/+21/-8.1, prism -0.8..-6.1; middle clear.
- lead from our preshape (rad): cube0 0.525, cube45 0.42, cyl 0.41, prism 0.36/0.39/0.22/0.60 -> far above 0.15.
- planner raises grasp point to 30.2 (preshape clearance 12.0, top limit 37). IK ok, err 0.003 mm. Descent: no tip on top; min clearance index 20.6, thumb 35.7.

## Handbook note (not a deliverable): demo preshape [-0.67, 0.92, 0.15, -0.28, -0.02, -0.19, -0.05]
- right via config.py mirror (thumb0 flipped): [0.67,-0.92,-0.15,0.19,0.05,0.28,0.02] -> right tips 98.8 mm from the true mirror.
- exact mirror keeps thumb0 = -0.67 (error 0.00 mm). The config.py mirror bug is real for any non-zero thumb0.

## Assumptions
fingertip = point at kinematics.TIP (52 mm along distal link), rigid point contact, no pad geometry; linear joint
interpolation preshape->closed with the object fixed; lead = remaining joint travel at first contact; object centred on
the grasp point; prism centred on its centroid; sim profile, fingers forward.

## Prism yaw sweep (prism_yaw_sweep.py / .out, 2 deg steps over 0-120)
Convention: pelvis frame at a real left-arm IK pose (tilt 0); rel = apex heading - finger heading (palm x), CCW from
above; planner phi = -rel mod 120 (left heading = apex + phi + 120k; right: s = -1).
- pinch_v1 (41.3 mm): NO window with thumb AND index >= 1 mm. Thumb inside only for rel 120..34 (to -4.1), index
  inside only for rel 60..94 (to -4.2) while the thumb is +4.8..+5.5 outside. Best compromise rel 106: both still
  outside (+2.7 / +2.8). No yaw alignment makes pinch_v1 two-sided on the prism.
- pinch_demo (29.1 mm): two 30-deg windows, middle clear (>= +13):
  rel 52..82, centre 67: i/t -2.6/-2.4, lead 0.247 (index_1), max 0.332 -> phi 53 (recommended, lower lead)
  rel 114..24 (wraps), centre 9: i/t -2.6/-2.5, lead 0.412 (thumb_1), max 0.532 -> phi 111
  Outside the windows both tips still penetrate >= 0.6 mm at every yaw (weak, not one-sided).
- Sign check: oracle old optimum phi 105 = rel 15, inside the phi-111 window. Right arm (s = -1, config mirror,
  tip error 0.00 mm) reproduces the left distances at both centres.
- Assumed: point tips at kinematics.TIP, rigid contact, linear interpolation from our preshape, object fixed,
  centroid on the grasp point, no yaw_tri_shift, one IK pose per arm (0.30, +-0.10).

## Prism grasp-point offset along the clamp axis (prism_offset.py / .out)
Coverage = share of yaws with index AND thumb >= 1 mm (in brackets >= 0 mm). Width-midpoint offset +-7.5 mm, width 45-52 mm.
- pinch_v1: no shift 0%; best constant 2%; yaw-dependent along-axis shift 3% (max +1.3 mm). Along-axis shift cannot
  fix pinch_v1. A 2D shift (+-15 mm sideways) reaches >= 1 mm everywhere, middle/lead NOT checked.
- pinch_demo: no shift 53% (100%); best constant d = -0.5 mm 68% (100%); yaw-dependent d*(rel) 100% (100%), still 100%
  with +-5 and +-10 deg yaw error. d* = -0.61 * midoff (corr 0.999), not -1 * midoff (that drops to 54% at +-10 deg).
  Middle >= +13 mm; worst lead 0.65 rad (exact d*), 0.48 (constant).
- pinch_demo table, rel = (apex_yaw - finger heading) mod 120, linear interpolation, wraps:
  rel 0 10 20 30 40 50 60 70 80 90 100 110 -> d mm -1.2 +0.2 +1.8 +3.2 +4.5 +2.8 +1.2 -0.2 -1.8 -3.5 -4.5 -2.8
- Apply in pelvis xy at grasp time: target = centroid - d * u, u = horizontal unit vector index tip -> thumb tip
  (for demo 98 deg CCW from finger heading, i.e. nearly sideways). Needs apex_yaw from triangle_pose (mod 120);
  perception._yaw cannot be used. Left arm only; right expected rel -> -rel, NOT checked.
- Assumed: left arm top-down, small shift as pure translation (no IK re-solve), rigid tips, 0.25 mm grid.
