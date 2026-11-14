| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | held yes/no/n.a. | gate fails | clips | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 20 | **52/82** | 52/0/30/0 | 21/25 | 18/28 | 13/29 | 15/18 | 6/7 | 6.75 | 0/0/0 | 0 | 0 | 29.99 | 0.416 |
| ab_pinch_v1_yaff_a7978dd | 20 | **67/82** | 67/0/15/0 | 22/25 | 25/28 | 20/29 | 16/18 | 6/7 | 5.30 | 77/29/0 | 0 | 0 | 29.97 | 0.318 |
| ab_slots_6a65371 | 20 | **61/82** | 61/0/21/0 | 23/25 | 24/28 | 14/29 | 16/18 | 7/7 | 6.85 | 61/76/0 | 0 | 0 | 29.97 | 0.319 |
| ab_pinch_hybrid_14a2b67 | 20 | **65/82** | 65/1/16/0 | 17/25 | 24/28 | 24/29 | 11/18 | 6/7 | 5.60 | 76/36/0 | 0 | 0 | 29.95 | 0.339 |
| ab_pinch_hybrid_yaff_strict_4f039af | 20 | **69/82** | 69/0/13/0 | 21/25 | 26/28 | 22/29 | 14/18 | 7/7 | 5.50 | 80/30/0 | 0 | 0 | 29.97 | 0.331 |
| ab_pinch_v1_yaff_slots_4f039af | 20 | **62/82** | 62/0/20/0 | 18/25 | 25/28 | 19/29 | 13/18 | 5/7 | 5.75 | 70/45/0 | 0 | 0 | 29.97 | 0.327 |

| arm | left cube | left cylinder | left triangle | right cube | right cylinder | right triangle | never picked |
|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 14/16 | 7/10 | 8/14 | 7/9 | 11/18 | 5/15 | 0 |
| ab_pinch_v1_yaff_a7978dd | 14/16 | 10/10 | 11/13 | 8/8 | 15/18 | 9/14 | 3 |
| ab_slots_6a65371 | 14/16 | 10/11 | 7/14 | 9/9 | 14/17 | 7/15 | 0 |
| ab_pinch_hybrid_14a2b67 | 10/15 | 10/11 | 12/13 | 7/8 | 14/17 | 12/16 | 2 |
| ab_pinch_hybrid_yaff_strict_4f039af | 13/16 | 10/11 | 9/14 | 8/9 | 16/17 | 13/15 | 0 |
| ab_pinch_v1_yaff_slots_4f039af | 13/17 | 10/10 | 7/13 | 5/7 | 15/18 | 12/14 | 3 |

Spawn-y side (pelvis y: left > +5 cm, right < -5 cm) x picking arm, correct/n per shape:

| arm run | spawn side | picked by | cube | cylinder | triangle | all |
|---|---|---|---|---|---|---|
| baseline box A (flags off) | left | left | 5/5 | 5/5 | 5/10 | 15/20 |
| baseline box A (flags off) | left | right | 2/4 | 3/4 | 3/5 | 8/13 |
| baseline box A (flags off) | middle | left | 1/1 | 2/2 | 2/3 | 5/6 |
| baseline box A (flags off) | middle | right | 2/2 | 4/7 | 1/4 | 7/13 |
| baseline box A (flags off) | right | left | 8/10 | 0/3 | 1/1 | 9/14 |
| baseline box A (flags off) | right | right | 3/3 | 4/7 | 1/6 | 8/16 |
| ab_pinch_v1_yaff_a7978dd | left | left | 5/5 | 5/5 | 8/9 | 18/19 |
| ab_pinch_v1_yaff_a7978dd | left | right | 3/3 | 3/4 | 3/5 | 9/12 |
| ab_pinch_v1_yaff_a7978dd | left | none | 0/1 | - | 0/1 | 0/2 |
| ab_pinch_v1_yaff_a7978dd | middle | left | 1/1 | 2/2 | 2/3 | 5/6 |
| ab_pinch_v1_yaff_a7978dd | middle | right | 2/2 | 5/7 | 3/4 | 10/13 |
| ab_pinch_v1_yaff_a7978dd | right | left | 8/10 | 3/3 | 1/1 | 12/14 |
| ab_pinch_v1_yaff_a7978dd | right | right | 3/3 | 7/7 | 3/5 | 13/15 |
| ab_pinch_v1_yaff_a7978dd | right | none | - | - | 0/1 | 0/1 |
| ab_slots_6a65371 | left | left | 5/5 | 5/5 | 5/10 | 15/20 |
| ab_slots_6a65371 | left | right | 4/4 | 4/4 | 4/5 | 12/13 |
| ab_slots_6a65371 | middle | left | 1/1 | 2/3 | 2/3 | 5/7 |
| ab_slots_6a65371 | middle | right | 2/2 | 5/6 | 1/4 | 8/12 |
| ab_slots_6a65371 | right | left | 8/10 | 3/3 | 0/1 | 11/14 |
| ab_slots_6a65371 | right | right | 3/3 | 5/7 | 2/6 | 10/16 |
| ab_pinch_hybrid_14a2b67 | left | left | 4/5 | 4/5 | 9/10 | 17/20 |
| ab_pinch_hybrid_14a2b67 | left | right | 3/4 | 2/4 | 4/5 | 9/13 |
| ab_pinch_hybrid_14a2b67 | middle | left | 1/1 | 3/3 | 2/2 | 6/6 |
| ab_pinch_hybrid_14a2b67 | middle | right | 2/2 | 6/6 | 4/5 | 12/13 |
| ab_pinch_hybrid_14a2b67 | right | left | 5/9 | 3/3 | 1/1 | 9/13 |
| ab_pinch_hybrid_14a2b67 | right | right | 2/2 | 6/7 | 4/6 | 12/15 |
| ab_pinch_hybrid_14a2b67 | right | none | 0/2 | - | - | 0/2 |
| ab_pinch_hybrid_yaff_strict_4f039af | left | left | 4/5 | 4/5 | 7/10 | 15/20 |
| ab_pinch_hybrid_yaff_strict_4f039af | left | right | 3/4 | 4/4 | 4/5 | 11/13 |
| ab_pinch_hybrid_yaff_strict_4f039af | middle | left | 1/1 | 3/3 | 2/3 | 6/7 |
| ab_pinch_hybrid_yaff_strict_4f039af | middle | right | 2/2 | 5/6 | 4/4 | 11/12 |
| ab_pinch_hybrid_yaff_strict_4f039af | right | left | 8/10 | 3/3 | 0/1 | 11/14 |
| ab_pinch_hybrid_yaff_strict_4f039af | right | right | 3/3 | 7/7 | 5/6 | 15/16 |
| ab_pinch_v1_yaff_slots_4f039af | left | left | 4/5 | 5/5 | 5/9 | 14/19 |
| ab_pinch_v1_yaff_slots_4f039af | left | right | 3/3 | 3/4 | 4/5 | 10/12 |
| ab_pinch_v1_yaff_slots_4f039af | left | none | 0/1 | - | 0/1 | 0/2 |
| ab_pinch_v1_yaff_slots_4f039af | middle | left | 1/1 | 2/2 | 2/3 | 5/6 |
| ab_pinch_v1_yaff_slots_4f039af | middle | right | 1/2 | 5/7 | 4/4 | 10/13 |
| ab_pinch_v1_yaff_slots_4f039af | right | left | 8/11 | 3/3 | 0/1 | 11/15 |
| ab_pinch_v1_yaff_slots_4f039af | right | right | 1/2 | 7/7 | 4/5 | 12/14 |
| ab_pinch_v1_yaff_slots_4f039af | right | none | - | - | 0/1 | 0/1 |

ab_pinch_v1_yaff_a7978dd flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_pinch_v1_yaff.yaml
  ON   perception.y_affine                  [-0.0326, 0.0039]  Timi 23 §13   y -= a*y + b (perceived - true y model, pelvis frame)
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
6 of 28 flags on

ab_slots_6a65371 flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_slots.yaml
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   grasp.drop_slots                     true    Timi 23       standalone drop slots: next object in a bin -> next slot
2 of 29 flags on

ab_pinch_hybrid_14a2b67 flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_pinch_hybrid.yaml
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
7 of 31 flags on

ab_pinch_hybrid_yaff_strict_4f039af flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_pinch_hybrid_yaff_strict.yaml
  ON   perception.y_affine                  [-0.0326, 0.0039]  Timi 23 §13   y -= a*y + b (perceived - true y model, pelvis frame)
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
8 of 33 flags on

ab_pinch_v1_yaff_slots_4f039af flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_pinch_v1_yaff_slots.yaml
  ON   perception.y_affine                  [-0.0326, 0.0039]  Timi 23 §13   y -= a*y + b (perceived - true y model, pelvis frame)
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   grasp.drop_slots                     true    Timi 23       standalone drop slots: next object in a bin -> next slot
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
7 of 33 flags on
