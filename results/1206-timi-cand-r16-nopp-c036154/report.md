| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | held yes/no/n.a. | gate fails | clips | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 20 | **52/82** | 52/0/30/0 | 21/25 | 18/28 | 13/29 | 15/18 | 6/7 | 6.75 | 0/0/0 | 0 | 0 | 29.99 | 0.416 |
| cand_r16_nopp_c036154 | 27 | **97/111** | 97/0/14/0 | 28/32 | 42/43 | 27/36 | 11/12 | 17/20 | 5.44 | 103/44/0 | 0 | 0 | 29.96 | 0.306 |

| arm | left cube | left cylinder | left triangle | right cube | right cylinder | right triangle | never picked |
|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 14/16 | 7/10 | 8/14 | 7/9 | 11/18 | 5/15 | 0 |
| cand_r16_nopp_c036154 | 13/16 | 24/25 | 12/18 | 15/16 | 18/18 | 15/18 | 0 |

Spawn-y side (pelvis y: left > +5 cm, right < -5 cm) x picking arm, correct/n per shape:

| arm run | spawn side | picked by | cube | cylinder | triangle | all |
|---|---|---|---|---|---|---|
| baseline box A (flags off) | left | left | 5/5 | 5/5 | 5/10 | 15/20 |
| baseline box A (flags off) | left | right | 2/4 | 3/4 | 3/5 | 8/13 |
| baseline box A (flags off) | middle | left | 1/1 | 2/2 | 2/3 | 5/6 |
| baseline box A (flags off) | middle | right | 2/2 | 4/7 | 1/4 | 7/13 |
| baseline box A (flags off) | right | left | 8/10 | 0/3 | 1/1 | 9/14 |
| baseline box A (flags off) | right | right | 3/3 | 4/7 | 1/6 | 8/16 |
| cand_r16_nopp_c036154 | left | left | 8/8 | 17/17 | 6/6 | 31/31 |
| cand_r16_nopp_c036154 | left | right | 3/3 | 4/4 | 5/7 | 12/14 |
| cand_r16_nopp_c036154 | middle | left | 4/5 | 3/3 | 5/6 | 12/14 |
| cand_r16_nopp_c036154 | middle | right | 5/5 | 2/2 | 5/5 | 12/12 |
| cand_r16_nopp_c036154 | right | left | 1/3 | 4/5 | 1/6 | 6/14 |
| cand_r16_nopp_c036154 | right | right | 7/8 | 12/12 | 5/6 | 24/26 |

cand_r16_nopp_c036154 flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/cand_r16_nopp.yaml
  ON   instruction.resolve                  true    Felix #1      instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)
  ON   policy.insert_guard                  true    insert        rung 3 (insert_shapes task id or the hole/lid sentence): pick nothing (0 pts, never -1 on the lid)
  ON   policy.bin_lookup                    "resolve_bin"  round 8       colour (stock) | resolve_bin: (colour,shape)->colour->shape->default
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   perception.hsv_set                   "robust"  real colour   stock | robust: one HSV set for sim + the real head cam (hsv_robust)
  ON   perception.split_blobs               true    real colour   split touching same-colour objects at distance-transform peaks
  ON   policy.max_total_attempts            18      Oliver #3     global per-episode cap on started picks
  ON   policy.retry_vary_grasp              true    retry-vary    retry after the object remains in the next frame with another grasp (posture/heading), never an identical one
  ON   policy.max_attempts_vary             3       retry-vary    with retry_vary_grasp: tries per object up to this, only with time left
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   grasp.drop_slots                     true    Timi 23       standalone drop slots: next object in a bin -> next slot
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.posture_apex_gate              true    apex gate     close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
17 of 41 flags on
