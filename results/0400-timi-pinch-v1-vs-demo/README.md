# Pinch posture arms: pinch_v1 vs pinch_demo

## Setup

Engine: classical policy (`aaltonomous/baselines/classical`) from a `git archive` snapshot of the stated commit, overlay selected with `AALTO_OVERLAY`, Isaac Sim on Modal (L40S), one episode per container (`modal/ab_arm.sh`), `sort_fixed`, settle 3 s, nominal DR, time limit 120 s, gate 0 clips / >= 27 Hz. Baseline row = box A STEP 0 classical, flags off, seeds 1000-1019 (52/82). Noise between repeated runs is about +-3 objects on 20 episodes and about +-5 on 30.


## Results

| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | held yes/no/n.a. | gate fails | clips | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 20 | **52/82** | 52/0/30/0 | 21/25 | 18/28 | 13/29 | 15/18 | 6/7 | 6.75 | 0/0/0 | 0 | 0 | 29.99 | 0.416 |
| ab_pinch_v1_a48ef4f | 20 | **60/82** | 60/0/22/0 | 20/25 | 23/28 | 17/29 | 14/18 | 6/7 | 5.55 | 71/40/0 | 0 | 0 | 29.97 | 0.312 |
| ab_pinch_demo_9c3fde0 | 20 | **51/82** | 51/0/31/0 | 7/25 | 21/28 | 23/29 | 3/18 | 4/7 | 6.20 | 74/50/0 | 0 | 0 | 29.97 | 0.323 |

## Conclusion

- **pinch_v1: 60/82 (+8 vs baseline)**, 0 wrong. Gain on cylinders (23/28 vs 18/28) and triangles (17 vs 13), cubes hold (20/25). Fewer picks per episode (5.55 vs 6.75), so fewer repeat attempts. The predicted weakness on the right hand did not show up.
- **pinch_demo: 51/82 (about baseline)** but with a completely different profile: cubes collapse to 7/25 (flat 3/18), triangles are the best of all arms (23/29). The demo posture (thumb-index 29 mm) does not close on a 5 cm cube but does catch thin triangle edges. Idea: demo posture for triangles only, pinch_v1 for the rest (became the hybrid arm).
- pinch_v1 vs baseline (+8) is above noise, pinch_demo vs baseline (-1) is not.
- Note: seeds 1010-1019 of both arms were re-run on the same SHA after a Modal warm-container bug (old policy server kept port 8000). Scores of the original copies were valid; the re-run is what is counted (pinch_v1 re-run 28 vs 26 on those 10 seeds, i.e. run-to-run noise ~2).


## Files

- `summary.json`: machine-readable per-arm numbers (outcomes, per shape, picks, held yes/no, gate).
- `report.md`: full `modal/ab_report.py` output (per hand, spawn side, flags).
- Raw episode data and logs are not copied.
