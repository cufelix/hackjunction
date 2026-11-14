| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | held yes/no/n.a. | gate fails | clips | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 20 | **52/82** | 52/0/30/0 | 21/25 | 18/28 | 13/29 | 15/18 | 6/7 | 6.75 | 0/0/0 | 0 | 0 | 29.99 | 0.416 |
| submission_2380e83 | 20 | **62/82** | 62/0/20/0 | 16/25 | 25/28 | 21/29 | 10/18 | 6/7 | 5.80 | 73/43/0 | 0 | 0 | 29.96 | 0.349 |
| ab_depthgate_786ae65 | 20 | **55/82** | 55/1/26/0 | 17/25 | 22/28 | 16/29 | 12/18 | 5/7 | 6.25 | 74/51/0 | 0 | 0 | 29.97 | 0.346 |
| cand_r14_786ae65 | 20 | **64/82** | 64/0/18/0 | 20/25 | 23/28 | 21/29 | 13/18 | 7/7 | 5.85 | 75/42/0 | 0 | 0 | 29.97 | 0.316 |
| ab_retry_vary_6efbd05 | 20 | **68/82** | 68/1/13/0 | 19/25 | 25/28 | 24/29 | 12/18 | 7/7 | 5.60 | 92/20/0 | 0 | 0 | 29.97 | 0.322 |
| ab_retry_vary_slots_6efbd05 | 20 | **75/82** | 75/0/7/0 | 22/25 | 27/28 | 26/29 | 16/18 | 6/7 | 5.75 | 92/23/0 | 0 | 0 | 29.96 | 0.335 |
| ab_pp_center_7cd01bf | 20 | **72/82** | 72/0/10/0 | 23/25 | 24/28 | 25/29 | 16/18 | 7/7 | 5.20 | 76/28/0 | 0 | 0 | 29.97 | 0.334 |
| ab_prism_offset_1b65ffb | 20 | **65/82** | 65/0/17/0 | 20/25 | 24/28 | 21/29 | 14/18 | 6/7 | 5.80 | 76/40/0 | 0 | 0 | 29.96 | 0.310 |
| cand_r14_1b65ffb | 20 | **71/82** | 71/0/11/0 | 22/25 | 26/28 | 23/29 | 15/18 | 7/7 | 5.65 | 81/32/0 | 0 | 0 | 29.97 | 0.309 |

| arm | left cube | left cylinder | left triangle | right cube | right cylinder | right triangle | never picked |
|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 14/16 | 7/10 | 8/14 | 7/9 | 11/18 | 5/15 | 0 |
| submission_2380e83 | 10/14 | 10/11 | 9/14 | 6/9 | 15/17 | 12/15 | 2 |
| ab_depthgate_786ae65 | 10/15 | 9/11 | 5/13 | 7/9 | 13/17 | 11/16 | 1 |
| cand_r14_786ae65 | 12/15 | 10/11 | 9/14 | 8/8 | 13/17 | 12/15 | 2 |
| ab_retry_vary_6efbd05 | 12/15 | 8/10 | 13/15 | 7/9 | 17/18 | 11/14 | 1 |
| ab_retry_vary_slots_6efbd05 | 13/14 | 11/11 | 13/14 | 9/9 | 16/17 | 13/15 | 2 |
| ab_pp_center_7cd01bf | 14/16 | 10/10 | 11/14 | 9/9 | 14/18 | 14/15 | 0 |
| ab_prism_offset_1b65ffb | 13/15 | 10/11 | 7/13 | 7/8 | 14/17 | 14/16 | 2 |
| cand_r14_1b65ffb | 14/15 | 11/11 | 11/15 | 8/9 | 15/16 | 12/14 | 2 |

Spawn-y side (pelvis y: left > +5 cm, right < -5 cm) x picking arm, correct/n per shape:

| arm run | spawn side | picked by | cube | cylinder | triangle | all |
|---|---|---|---|---|---|---|
| baseline box A (flags off) | left | left | 5/5 | 5/5 | 5/10 | 15/20 |
| baseline box A (flags off) | left | right | 2/4 | 3/4 | 3/5 | 8/13 |
| baseline box A (flags off) | middle | left | 1/1 | 2/2 | 2/3 | 5/6 |
| baseline box A (flags off) | middle | right | 2/2 | 4/7 | 1/4 | 7/13 |
| baseline box A (flags off) | right | left | 8/10 | 0/3 | 1/1 | 9/14 |
| baseline box A (flags off) | right | right | 3/3 | 4/7 | 1/6 | 8/16 |
| submission_2380e83 | left | left | 5/5 | 5/5 | 6/10 | 16/20 |
| submission_2380e83 | left | right | 3/4 | 4/4 | 4/5 | 11/13 |
| submission_2380e83 | middle | left | 1/1 | 3/3 | 2/3 | 6/7 |
| submission_2380e83 | middle | right | 0/2 | 5/6 | 4/4 | 9/12 |
| submission_2380e83 | right | left | 4/8 | 2/3 | 1/1 | 7/12 |
| submission_2380e83 | right | right | 3/3 | 6/7 | 4/6 | 13/16 |
| submission_2380e83 | right | none | 0/2 | - | - | 0/2 |
| ab_depthgate_786ae65 | left | left | 5/5 | 5/5 | 3/10 | 13/20 |
| ab_depthgate_786ae65 | left | right | 2/4 | 3/4 | 3/5 | 8/13 |
| ab_depthgate_786ae65 | middle | left | - | 3/3 | 1/2 | 4/5 |
| ab_depthgate_786ae65 | middle | right | 3/3 | 4/6 | 4/5 | 11/14 |
| ab_depthgate_786ae65 | right | left | 5/10 | 1/3 | 1/1 | 7/14 |
| ab_depthgate_786ae65 | right | right | 2/2 | 6/7 | 4/6 | 12/15 |
| ab_depthgate_786ae65 | right | none | 0/1 | - | - | 0/1 |
| cand_r14_786ae65 | left | left | 5/5 | 5/5 | 7/10 | 17/20 |
| cand_r14_786ae65 | left | right | 4/4 | 2/4 | 3/5 | 9/13 |
| cand_r14_786ae65 | middle | left | 1/1 | 3/3 | 2/3 | 6/7 |
| cand_r14_786ae65 | middle | right | 2/2 | 5/6 | 4/4 | 11/12 |
| cand_r14_786ae65 | right | left | 6/9 | 2/3 | 0/1 | 8/13 |
| cand_r14_786ae65 | right | right | 2/2 | 6/7 | 5/6 | 13/15 |
| cand_r14_786ae65 | right | none | 0/2 | - | - | 0/2 |
| ab_retry_vary_6efbd05 | left | left | 5/5 | 4/5 | 9/10 | 18/20 |
| ab_retry_vary_6efbd05 | left | right | 3/4 | 4/4 | 4/5 | 11/13 |
| ab_retry_vary_6efbd05 | middle | left | 1/1 | 2/2 | 3/4 | 6/7 |
| ab_retry_vary_6efbd05 | middle | right | 1/2 | 7/7 | 3/3 | 11/12 |
| ab_retry_vary_6efbd05 | right | left | 6/9 | 2/3 | 1/1 | 9/13 |
| ab_retry_vary_6efbd05 | right | right | 3/3 | 6/7 | 4/6 | 13/16 |
| ab_retry_vary_6efbd05 | right | none | 0/1 | - | - | 0/1 |
| ab_retry_vary_slots_6efbd05 | left | left | 5/5 | 5/5 | 9/10 | 19/20 |
| ab_retry_vary_slots_6efbd05 | left | right | 4/4 | 4/4 | 5/5 | 13/13 |
| ab_retry_vary_slots_6efbd05 | middle | left | 1/1 | 3/3 | 3/3 | 7/7 |
| ab_retry_vary_slots_6efbd05 | middle | right | 2/2 | 5/6 | 4/4 | 11/12 |
| ab_retry_vary_slots_6efbd05 | right | left | 7/8 | 3/3 | 1/1 | 11/12 |
| ab_retry_vary_slots_6efbd05 | right | right | 3/3 | 7/7 | 4/6 | 14/16 |
| ab_retry_vary_slots_6efbd05 | right | none | 0/2 | - | - | 0/2 |
| ab_pp_center_7cd01bf | left | left | 5/5 | 5/5 | 7/10 | 17/20 |
| ab_pp_center_7cd01bf | left | right | 4/4 | 2/4 | 5/5 | 11/13 |
| ab_pp_center_7cd01bf | middle | left | 1/1 | 2/2 | 3/3 | 6/6 |
| ab_pp_center_7cd01bf | middle | right | 2/2 | 5/7 | 4/4 | 11/13 |
| ab_pp_center_7cd01bf | right | left | 8/10 | 3/3 | 1/1 | 12/14 |
| ab_pp_center_7cd01bf | right | right | 3/3 | 7/7 | 5/6 | 15/16 |
| ab_prism_offset_1b65ffb | left | left | 5/5 | 4/5 | 5/10 | 14/20 |
| ab_prism_offset_1b65ffb | left | right | 3/4 | 3/4 | 5/5 | 11/13 |
| ab_prism_offset_1b65ffb | middle | left | 1/1 | 3/3 | 2/2 | 6/6 |
| ab_prism_offset_1b65ffb | middle | right | 2/2 | 4/6 | 4/5 | 10/13 |
| ab_prism_offset_1b65ffb | right | left | 7/9 | 3/3 | 0/1 | 10/13 |
| ab_prism_offset_1b65ffb | right | right | 2/2 | 7/7 | 5/6 | 14/15 |
| ab_prism_offset_1b65ffb | right | none | 0/2 | - | - | 0/2 |
| cand_r14_1b65ffb | left | left | 5/5 | 5/5 | 7/10 | 17/20 |
| cand_r14_1b65ffb | left | right | 4/4 | 3/3 | 5/5 | 12/12 |
| cand_r14_1b65ffb | left | none | - | 0/1 | - | 0/1 |
| cand_r14_1b65ffb | middle | left | 1/1 | 3/3 | 3/4 | 7/8 |
| cand_r14_1b65ffb | middle | right | 1/2 | 5/6 | 3/3 | 9/11 |
| cand_r14_1b65ffb | right | left | 8/9 | 3/3 | 1/1 | 12/13 |
| cand_r14_1b65ffb | right | right | 3/3 | 7/7 | 4/6 | 14/16 |
| cand_r14_1b65ffb | right | none | 0/1 | - | - | 0/1 |

submission_2380e83 flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/submission.yaml
  ON   instruction.resolve                  true    Felix #1      instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)
  ON   policy.insert_guard                  true    insert        rung 3 (insert_shapes task id or the hole/lid sentence): pick nothing (0 pts, never -1 on the lid)
  ON   policy.bin_lookup                    "resolve_bin"  round 8       colour (stock) | resolve_bin: (colour,shape)->colour->shape->default
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   perception.hsv_set                   "robust"  real colour   stock | robust: one HSV set for sim + the real head cam (hsv_robust)
  ON   perception.split_blobs               true    real colour   split touching same-colour objects at distance-transform peaks
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.posture_apex_gate              true    apex gate     close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
13 of 37 flags on

ab_depthgate_786ae65 flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_depthgate.yaml
  ON   instruction.resolve                  true    Felix #1      instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)
  ON   policy.bin_lookup                    "resolve_bin"  round 8       colour (stock) | resolve_bin: (colour,shape)->colour->shape->default
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   perception.hsv_set                   "robust"  real colour   stock | robust: one HSV set for sim + the real head cam (hsv_robust)
  ON   perception.split_blobs               true    real colour   split touching same-colour objects at distance-transform peaks
  ON   perception.require_depth_for_new     true    Felix review  no pick target from a blob without valid depth (no plane fallback)
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.posture_apex_gate              true    apex gate     close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
13 of 37 flags on

cand_r14_786ae65 flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/cand_r14.yaml
  ON   instruction.resolve                  true    Felix #1      instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)
  ON   policy.bin_lookup                    "resolve_bin"  round 8       colour (stock) | resolve_bin: (colour,shape)->colour->shape->default
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   perception.hsv_set                   "robust"  real colour   stock | robust: one HSV set for sim + the real head cam (hsv_robust)
  ON   perception.split_blobs               true    real colour   split touching same-colour objects at distance-transform peaks
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.posture_apex_gate              true    apex gate     close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
12 of 37 flags on

ab_retry_vary_6efbd05 flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_retry_vary.yaml
  ON   instruction.resolve                  true    Felix #1      instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)
  ON   policy.insert_guard                  true    insert        rung 3 (insert_shapes task id or the hole/lid sentence): pick nothing (0 pts, never -1 on the lid)
  ON   policy.bin_lookup                    "resolve_bin"  round 8       colour (stock) | resolve_bin: (colour,shape)->colour->shape->default
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   perception.hsv_set                   "robust"  real colour   stock | robust: one HSV set for sim + the real head cam (hsv_robust)
  ON   perception.split_blobs               true    real colour   split touching same-colour objects at distance-transform peaks
  ON   policy.retry_vary_grasp              true    retry-vary    retry after held=no with another grasp (posture/heading), never an identical one; held=no -> abort the carry
  ON   policy.max_attempts_vary             3       retry-vary    with retry_vary_grasp: tries per object up to this, only with time left
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.posture_apex_gate              true    apex gate     close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
15 of 40 flags on

ab_retry_vary_slots_6efbd05 flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_retry_vary_slots.yaml
  ON   instruction.resolve                  true    Felix #1      instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)
  ON   policy.insert_guard                  true    insert        rung 3 (insert_shapes task id or the hole/lid sentence): pick nothing (0 pts, never -1 on the lid)
  ON   policy.bin_lookup                    "resolve_bin"  round 8       colour (stock) | resolve_bin: (colour,shape)->colour->shape->default
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   perception.hsv_set                   "robust"  real colour   stock | robust: one HSV set for sim + the real head cam (hsv_robust)
  ON   perception.split_blobs               true    real colour   split touching same-colour objects at distance-transform peaks
  ON   policy.retry_vary_grasp              true    retry-vary    retry after held=no with another grasp (posture/heading), never an identical one; held=no -> abort the carry
  ON   policy.max_attempts_vary             3       retry-vary    with retry_vary_grasp: tries per object up to this, only with time left
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   grasp.drop_slots                     true    Timi 23       standalone drop slots: next object in a bin -> next slot
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.posture_apex_gate              true    apex gate     close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
16 of 40 flags on

ab_pp_center_7cd01bf flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_pp_center.yaml
  ON   instruction.resolve                  true    Felix #1      instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)
  ON   policy.insert_guard                  true    insert        rung 3 (insert_shapes task id or the hole/lid sentence): pick nothing (0 pts, never -1 on the lid)
  ON   policy.bin_lookup                    "resolve_bin"  round 8       colour (stock) | resolve_bin: (colour,shape)->colour->shape->default
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   perception.hsv_set                   "robust"  real colour   stock | robust: one HSV set for sim + the real head cam (hsv_robust)
  ON   perception.principal_point           "center"  Oliver lane 3 file | center: Isaac renders cx,cy at the image centre
  ON   perception.split_blobs               true    real colour   split touching same-colour objects at distance-transform peaks
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.posture_apex_gate              true    apex gate     close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
14 of 38 flags on

ab_prism_offset_1b65ffb flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/ab_prism_offset.yaml
  ON   instruction.resolve                  true    Felix #1      instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)
  ON   policy.bin_lookup                    "resolve_bin"  round 8       colour (stock) | resolve_bin: (colour,shape)->colour->shape->default
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   perception.triangle_pose             true    Oliver #5     triangles: apex yaw + robust xy centre
  ON   perception.hsv_set                   "robust"  real colour   stock | robust: one HSV set for sim + the real head cam (hsv_robust)
  ON   perception.split_blobs               true    real colour   split touching same-colour objects at distance-transform peaks
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.posture_apex_gate              true    apex gate     close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)
  ON   grasp.prism_offset                   "table"  Oliver e344273 pinch_demo prism picks: grasp shift along the clamp axis by apex yaw (needs triangle_pose)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
14 of 38 flags on

cand_r14_1b65ffb flags (tools/flags.py, first episode):
profile=sim  overlay=/tmp/work/aaltonomous/baselines/classical/overlays/cand_r14.yaml
  ON   instruction.resolve                  true    Felix #1      instruction text -> colour/shape->bin mapping (else FIXED_MAPPING)
  ON   policy.bin_lookup                    "resolve_bin"  round 8       colour (stock) | resolve_bin: (colour,shape)->colour->shape->default
  ON   perception.classify_shape            true    Oliver #5/#7  Detection.shape from the top face (shape_id)
  ON   perception.hsv_set                   "robust"  real colour   stock | robust: one HSV set for sim + the real head cam (hsv_robust)
  ON   perception.split_blobs               true    real colour   split touching same-colour objects at distance-transform peaks
  ON   policy.grasp_diag_log                true    Felix grasp   passive [grasp_diag] line per pick: stall vs closed, held
  ON   motion.close_hold_s                  1.2     Felix grasp   hold after closing the hand (sim profile 0.25 s)
  ON   grasp.point_in_palm                  [0.0862, -0.0864, 0.0143]  pinch fix     object centre in the LEFT palm frame (right = [1,-1,1] mirror)
  ON   grasp.close_posture_by_shape         true    hybrid        per pick: demo posture only for a confident metric triangle, else the configured one
  ON   grasp.posture_apex_gate              true    apex gate     close_posture_by_shape: demo also needs apex_conf + max vertex angle (cut-cube guard)
  ON   grasp.close_posture                  "closed_tight"  pinch fix     grasp posture: closed | closed_tight
  ON   hand.closed_tight                    [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]  pinch fix     the closed_tight posture (LEFT DDS order, right mirrored)
12 of 38 flags on
