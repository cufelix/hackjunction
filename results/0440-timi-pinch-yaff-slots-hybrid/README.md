# Pinch + y_affine, drop slots, hybrid posture (arms 3-7)

## Setup

Engine: classical policy (`aaltonomous/baselines/classical`) from a `git archive` snapshot of the stated commit, overlay selected with `AALTO_OVERLAY`, Isaac Sim on Modal (L40S), one episode per container (`modal/ab_arm.sh`), `sort_fixed`, settle 3 s, nominal DR, time limit 120 s, gate 0 clips / >= 27 Hz. Baseline row = box A STEP 0 classical, flags off, seeds 1000-1019 (52/82). Noise between repeated runs is about +-3 objects on 20 episodes and about +-5 on 30.


## Results

| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | held yes/no/n.a. | gate fails | clips | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 20 | **52/82** | 52/0/30/0 | 21/25 | 18/28 | 13/29 | 15/18 | 6/7 | 6.75 | 0/0/0 | 0 | 0 | 29.99 | 0.416 |
| ab_pinch_v1_yaff_a7978dd | 20 | **67/82** | 67/0/15/0 | 22/25 | 25/28 | 20/29 | 16/18 | 6/7 | 5.30 | 77/29/0 | 0 | 0 | 29.97 | 0.318 |
| ab_slots_6a65371 | 20 | **61/82** | 61/0/21/0 | 23/25 | 24/28 | 14/29 | 16/18 | 7/7 | 6.85 | 61/76/0 | 0 | 0 | 29.97 | 0.319 |
| ab_pinch_hybrid_14a2b67 | 20 | **65/82** | 65/1/16/0 | 17/25 | 24/28 | 24/29 | 11/18 | 6/7 | 5.60 | 76/36/0 | 0 | 0 | 29.95 | 0.339 |
| ab_pinch_hybrid_yaff_strict_4f039af | 20 | **69/82** | 69/0/13/0 | 21/25 | 26/28 | 22/29 | 14/18 | 7/7 | 5.50 | 80/30/0 | 0 | 0 | 29.97 | 0.331 |
| ab_pinch_v1_yaff_slots_4f039af | 20 | **62/82** | 62/0/20/0 | 18/25 | 25/28 | 19/29 | 13/18 | 5/7 | 5.75 | 70/45/0 | 0 | 0 | 29.97 | 0.327 |

## Conclusion

- **pinch_v1_yaff: 67/82 (+15 vs baseline, +7 vs pinch_v1)**, 0 wrong. y_affine mostly helps objects spawned on the right (21/30 -> 25/30), as the y-error model predicts. At the edge of noise but consistent across shapes and sides.
- **slots alone: 61/82 (+9)**: gain probably from placing into slots (fewer collisions in the bin), not from the grasp. Triangles stay at 14/29. Over half of closes are empty with the stock posture.
- **hybrid (demo posture only for a confident metric triangle): 65/82**, best triangles (24/29) but cubes dropped to 17/25: 6 of 34 cube picks were classified triangle with conf 1.0 (all at the far-left spawn edge, y ~ +0.165 m, top face only 15-28 % visible; likely occluded by the left forearm in the home pose). Apex-angle check (max vertex angle ~89.7 deg, apex_conf 0.00) would reject them correctly. No confidence threshold separates them.
- **hybrid_yaff_strict: 69/82 (+17 vs baseline)**, best of this series, 0 wrong. Versus hybrid without yaff +4 and versus pinch_v1_yaff +2, both within noise.
- **pinch_v1_yaff_slots: 62/82**, 5 worse than pinch_v1_yaff: slots do not add on top of yaff. Recommendation was not to take slots into the submission at that time.


## Files

- `summary.json`: machine-readable per-arm numbers (outcomes, per shape, picks, held yes/no, gate).
- `report.md`: full `modal/ab_report.py` output (per hand, spawn side, flags).
- Raw episode data and logs are not copied.
