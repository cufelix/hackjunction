# Soft-finger grip (grasp.soft_squeeze, held.soft_detector)

Problem. The real Dex3 finger servos run kp 1.5 / kd 0.2, about 67x softer than the sim default (kp ~100), with a
finger lag of 300-380 ms. In Isaac with those gains (real_soft profile) rc1 got 0/8: the stall-based held detector
read the fingers stopping short of the closed target (they lag and are soft) as an object (held=yes on 17/18
closes), so the policy lifted an empty hand. At close track_err_max was 0.06-0.43 rad soft vs 0.00-0.11 stiff.

All three pieces are default OFF and flags-off behaviour is byte-identical (checked by test_soft_grip).

| Flag | What |
|---|---|
| `grasp.soft_hand` | the finger servos are soft. Soft is also implied by config profile `none` / `real`. |
| `grasp.soft_overshoot` | per-joint over-closure (rad, LEFT DDS order, magnitudes along the closing direction). |
| `grasp.soft_squeeze` | on a soft hand: over-closed target, hold >= lag+settle+read-after, slow `lift_check`. |
| `held.soft_detector` | on a soft hand: held = stall further open than the empty-close baseline (needs `grasp.held_detector: finger_sum`). |

`overlays/finals_r22_softgrip.yaml` = `finals_r22_grasp` + `grasp.soft_hand`, `grasp.soft_squeeze`, `held.soft_detector`.
Non-flag params (config.yaml): `grasp.soft_limit_margin_rad` 0.03, `soft_lag_s` 0.38, `soft_settle_s` 0.4,
`soft_lift_check_m` 0.02, `soft_lift_check_scale` 0.35, `held.soft_margin_rad` 0.12, `soft_lift_margin_rad` 0.08,
`soft_read_after_s` 0.4, `soft_prior_ratio` 0.15.

## 1. Over-closed squeeze: the derivation

Grip force on a position-controlled joint is tau = kp x (commanded - actual) (kd term ~0 at stall), clipped by the
joint effort limit (Dex3 finger joints 1.4 N m, joint_spec.yaml).

- Stiff sim, default closure: commanded squeeze ~9 mm on a 45 mm object over a ~80 mm finger lever = ~0.11 rad,
  tau = 100 x 0.11 = 11 N m, far above the effort limit, so the sim grip is **saturated at 1.4 N m**.
- Soft real hand: reaching 1.4 N m needs commanded - actual = 1.4 / 1.5 = **0.93 rad** per joint, vs 0.11 rad now.
  The actual joint stalls short of the closed posture by a typical 0.06-0.43 rad (measured track error, take 0.2),
  so the overshoot beyond `closed` that is still needed is 0.93 - 0.2 = ~0.73 rad.
- URDF headroom beyond `closed` (left; right is mirrored), keeping `soft_limit_margin_rad` 0.03 (the safety filter's
  own margin is 0.02, so it never clips the command):

| joint | closed | limit | headroom (limit - 0.03) | wanted | default overshoot | cmd | torque at stall short 0.2 |
|---|---|---|---|---|---|---|---|
| middle0 / index0 | -1.0 | -1.571 | 0.541 | 0.73 | 0.50 | -1.50 | 1.5 x (0.50+0.2) = 1.05 N m (75 % of saturation) |
| middle1 / index1 | -0.6 | -1.745 | 1.115 | 0.73 | 1.00 | -1.60 | 1.5 x (1.00+0.2) = 1.8 -> 1.4 N m (saturated, = stiff sim) |
| thumb1 | -0.1 | -0.724 .. +1.047 | 1.15 (closing +) | small | 0.10 | 0.00 | thumb only pinches; kept small |
| thumb2 | 1.2 | 0 .. 1.745 | 1.17 (closing -) | small | 0.10 | 1.10 | same |
| thumb0 | 0.0 | +-1.047 | does not move | none | 0 | 0 | |

`soft_overclose(cfg, spec, base, rest)`: overshoot is applied along sign(base - rest) per joint (rest = preshape),
clipped at the shrunk limits, so it can never exceed them whatever overshoot is requested. A joint that does not
move (thumb0) gets none. Right overshoot = the left vector in right DDS order (slots 3,4 index / 5,6 middle).
Index and middle share one value so the order does not matter in the default.

Timing. With the ramp at `motion.hand_max_vel` the over-closed ramp is longer than before. The close hold is
`max(close_hold_s, soft_lag_s + soft_settle_s + soft_read_after_s)` = max(1.2, 0.38+0.4+0.4) = 1.2 s (1.18 s floor).
Only after the hold does the arm lift, first `soft_lift_check_m` (2 cm) at `soft_lift_check_scale` 0.35 of the
carry speed as a separate `lift_check` phase (a time-based phase in the queued plan; act() just pops it), then the
normal `lift`.

## 2. Soft held detector

Existing `finger_sum` reads `cmd - q` of the index/middle joint pairs minus the bias reference from episode
second one. On soft fingers the empty close alone leaves a lag/stiction error above the 0.045 rad threshold, so
the legacy test calls every close held.

Soft mode reuses that finger_sum score and compares it to what an EMPTY close of the same command leaves:

1. At episode start, at home with both hands open (state `settle`), the policy queues one empty close of both hands
   with the grasp command (same over-closed target, ramp and hold) and reopens (`_soft_baseline_start`, ~3 s,
   queued ticks, act() never blocks). In the last ~0.4 s of the hold it reads `ratio = finger_sum / travel`
   per finger (travel = closing-positive pair sum of command - preshape); `[soft_baseline]` logs it.
2. Held at the close read (>= lag + settle + 0.4 s after the ramp ends, i.e. the hold end):
   `excess = finger_sum - ratio x travel > soft_margin_rad (0.12)`. Per-posture travel scales the expectation, so
   `closed_demo` etc. work without their own baseline.
3. Lift test: at the end of `lift_check` (2 cm up) the excess must still be > `soft_lift_margin_rad` (0.08): an
   object keeps pushing back, a slipped one lets the fingers close and the error collapses. The carry reads use the
   same margin. An empty or slipped read aborts: no carry, open, home, and the existing retry path (vary_hist
   held=False, retry jitter) takes the next look.
4. No baseline (hand_q missing at start) falls back to `soft_prior_ratio`; no bias reference -> unsure, never empty
   (as before). Non-finite or wrong-shape data is also never empty.

Simulated numbers (test rig, first-order servo with stiction, both hands; the object stops the fingers at 60 %
of the closing travel): soft baseline ratio 0.029, empty excess 0.00 (raw legacy finger_sum 0.066 > 0.045,
i.e. the legacy detector says held); object excess 1.75. Stiff: ratio 0.00, empty 0.00, object 1.82. A servo still
converging at the read (rate limited, raw 0.08+ rad) is also read empty because the baseline measures the same lag.

## Tests

`test_soft_grip.py`: empty close reads empty and object close reads held (stiff and soft, left and right), legacy
misread documented, slip during lift_check -> empty -> abort -> retry, overshoot inside the URDF limits (all
postures, both hands, even for a 2 rad request), hold >= lag+settle, lift only after the hold, flags-off plan
byte-identical, act() latency unchanged. `test_flags.py`: the overlay is r22_grasp + the 3 flags.

Not validated on the real robot or in a soft-gain Isaac run yet: the margins (0.12 / 0.08) and the overshoot are
sized from the numbers above and should be tuned on the real_soft A/B.
