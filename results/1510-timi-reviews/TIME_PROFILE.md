# G1 policy time profile

Scope: 49 complete episodes, 254 pick attempts from both requested runs. `held=yes` at lift is used as the placed-object proxy: 186 objects; 68 `held=no`; 46 attempts were explicit varied retries.

## Timing

| Phase | n | Median (s) | P90 (s) |
|---|---:|---:|---:|
| look/detect | 254 | 0.20 | 0.20 |
| plan | 254 | 0.03 | 0.77 |
| approach | 254 | 3.20 | 3.77 |
| close + hold | 254 | 1.47 | 1.80 |
| lift + carry | 254 | 1.67 | 4.47 |
| release | 254 | 0.80 | 1.00 |
| retract | 254 | 2.97 | 3.53 |
| idle/wait | 254 | 0.30 | 0.30 |
| retry (whole cycle; non-additive) | 46 | 12.17 | 14.52 |
| **cycle total** | 254 | **11.47** | **13.55** |

Active elapsed time / `held=yes` object is **16.62 s pooled** (per-episode median 14.75 s, p90 28.93 s). A successful attempt alone is 11.17 s median / 12.98 s p90; the gap is failed attempts, retries, and terminal planning.

Measurement note: execute start, total duration, close-end and lift-end ticks are logged directly. Logs omit the other `Plan.phases` boundaries. The script reconstructs hand ramps from the sim config, assigns the known 0.30 s settle to idle, and estimates (a) look vs plan from the 0.20 s hold polling quantum and (b) retract by home-path symmetry, assigning the residual to lift+carry. Those two splits are directional, not instrumentation-grade. Add a passive log of `plan.phases` for exact future profiles.

## Five largest safe savings

| Rank | Change | Estimated saving | Location | Risk / guardrail |
|---:|---|---:|---|---|
| 1 | **Abort an empty grasp after the lift check**: on high-confidence `held=no`, skip carry/open and retract home for the retry. | **5.45 s per failed attempt** median (p90 7.54); **1.54 s amortized per attempt** in these logs. | `policy.py:1881-1912` (`_grasp_diag`, currently passive); queued tail is `planner.py:693-698`. | A false negative abandons a held object. Require repeated low-error samples after lift (all active fingers at commanded close), optionally vision/force confirmation; keep hand closed until safely back over the source/counter. |
| 2 | **Adaptive close hold**: use `motion.close_hold_s: 0.4`, `motion.lift_on_stall.enabled: true`, `timeout_s: 1.2`. This retains the long fallback but lifts once fingers settle. | **~0.7-0.8 s/attempt**; current overlay forces a 1.2 s hold. | `config.yaml:272-277`; cut logic `policy.py:1919-1944`; phase build `planner.py:687-693`. | Validate stall thresholds on G1. Early lift can reduce grip seating; never shorten below the tested minimum, and fall back to 1.2 s if hand telemetry is absent/noisy. |
| 3 | **Prefetch the next settled frame + plan during the camera-clear end of retract/home**, planning from the known final home state. | **0.23 s/attempt median**, **0.97 s p90** for sequential look+plan; up to **0.53 s typical** if it also removes the following fixed settle (do not add to rank 4). | Current serialization: `policy.py:555-557`, then `506-534`. | Only snapshot after release and after both hands clear the object ROI; discard the plan if the scene/frame hash changes. Real-robot occlusion or object settling makes earlier capture unsafe. |
| 4 | **Make the home settle conditional**, e.g. end after 0.10 s once measured arm velocity and image sharpness are stable; retain 0.30 s timeout. | **~0.20 s/attempt** when stable. | `motion.settle_s` (`config.yaml:411` sim); wait at `policy.py:506-510`. | Fixed reduction alone risks motion blur/arm lag. Gate on measured state, not commanded queue completion; no saving on unstable frames. |
| 5 | **Adaptive release hold**: finish after 0.20 s when finger opening has stalled/reached target, retain 0.40 s timeout. | **~0.20 s/attempt** (the release phase is 0.80 s median). | `motion.open_hold_s` (`config.yaml:278`, sim override `:410`); `planner.py:695`. | A sticky object can remain hooked. Confirm hand aperture (and preferably object separation); do not replace the dedicated `release` posture with `open`—the code notes that objects previously stayed wedged. |

Do **not** raise sim arm speed for this: `motion.arm_max_vel` is already 0.97 rad/s versus the 1.0 safety-filter limit (`config.yaml:406`). `grasp.carry_height` is 0.16 m, while the planner's geometric floor is about 0.135 m (`wall 0.06 + object 0.045 + clearance 0.03`, `planner.py:612-623`); lowering to 0.14 m might save only ~0.1-0.2 s and leaves ~5 mm margin, so it is not a top safe real-robot change without collision trials.

Reproduce with:

```bash
python3 timeprof/profile_policy.py
```
