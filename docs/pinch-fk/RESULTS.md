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
