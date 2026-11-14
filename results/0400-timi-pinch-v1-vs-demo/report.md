| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | held yes/no/n.a. | gate fails | clips | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 20 | **52/82** | 52/0/30/0 | 21/25 | 18/28 | 13/29 | 15/18 | 6/7 | 6.75 | 0/0/0 | 0 | 0 | 29.99 | 0.416 |
| ab_pinch_v1_a48ef4f | 20 | **60/82** | 60/0/22/0 | 20/25 | 23/28 | 17/29 | 14/18 | 6/7 | 5.55 | 71/40/0 | 0 | 0 | 29.97 | 0.312 |
| ab_pinch_demo_9c3fde0 | 20 | **51/82** | 51/0/31/0 | 7/25 | 21/28 | 23/29 | 3/18 | 4/7 | 6.20 | 74/50/0 | 0 | 0 | 29.97 | 0.323 |

| arm | left cube | left cylinder | left triangle | right cube | right cylinder | right triangle | never picked |
|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 14/16 | 7/10 | 8/14 | 7/9 | 11/18 | 5/15 | 0 |
| ab_pinch_v1_a48ef4f | 13/15 | 8/10 | 9/14 | 7/9 | 15/18 | 8/13 | 3 |
| ab_pinch_demo_9c3fde0 | 5/14 | 10/11 | 10/13 | 2/9 | 11/17 | 13/15 | 3 |

Spawn-y side (pelvis y: left > +5 cm, right < -5 cm) x picking arm, correct/n per shape:

| arm run | spawn side | picked by | cube | cylinder | triangle | all |
|---|---|---|---|---|---|---|
| baseline box A (flags off) | left | left | 5/5 | 5/5 | 5/10 | 15/20 |
| baseline box A (flags off) | left | right | 2/4 | 3/4 | 3/5 | 8/13 |
| baseline box A (flags off) | middle | left | 1/1 | 2/2 | 2/3 | 5/6 |
| baseline box A (flags off) | middle | right | 2/2 | 4/7 | 1/4 | 7/13 |
| baseline box A (flags off) | right | left | 8/10 | 0/3 | 1/1 | 9/14 |
| baseline box A (flags off) | right | right | 3/3 | 4/7 | 1/6 | 8/16 |
| ab_pinch_v1_a48ef4f | left | left | 5/5 | 4/5 | 7/9 | 16/19 |
| ab_pinch_v1_a48ef4f | left | right | 3/4 | 3/4 | 4/5 | 10/13 |
| ab_pinch_v1_a48ef4f | left | none | - | - | 0/1 | 0/1 |
| ab_pinch_v1_a48ef4f | middle | left | 1/1 | 2/2 | 1/4 | 4/7 |
| ab_pinch_v1_a48ef4f | middle | right | 1/2 | 6/7 | 2/3 | 9/12 |
| ab_pinch_v1_a48ef4f | right | left | 7/9 | 2/3 | 1/1 | 10/13 |
| ab_pinch_v1_a48ef4f | right | right | 3/3 | 6/7 | 2/5 | 11/15 |
| ab_pinch_v1_a48ef4f | right | none | 0/1 | - | 0/1 | 0/2 |
| ab_pinch_demo_9c3fde0 | left | left | 4/5 | 4/5 | 9/10 | 17/20 |
| ab_pinch_demo_9c3fde0 | left | right | 0/4 | 1/4 | 4/4 | 5/12 |
| ab_pinch_demo_9c3fde0 | left | none | - | - | 0/1 | 0/1 |
| ab_pinch_demo_9c3fde0 | middle | left | 0/1 | 3/3 | 1/2 | 4/6 |
| ab_pinch_demo_9c3fde0 | middle | right | 0/2 | 3/6 | 3/5 | 6/13 |
| ab_pinch_demo_9c3fde0 | right | left | 1/8 | 3/3 | 0/1 | 4/12 |
| ab_pinch_demo_9c3fde0 | right | right | 2/3 | 7/7 | 6/6 | 15/16 |
| ab_pinch_demo_9c3fde0 | right | none | 0/2 | - | - | 0/2 |

ab_pinch_v1_a48ef4f flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_pinch_v1.yaml
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
5 of 25 flags on

ab_pinch_demo_9c3fde0 flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_pinch_demo.yaml
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.1094, -0.0694, 0.0142]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.92, 0.25, -0.5, -1.41, -0.49, -1.21]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
5 of 27 flags on
