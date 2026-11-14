# Submission baseline, depthgate, pp_center, prism_offset, retry_vary (seeds 1000-1019)

## Setup

Engine: classical policy (`aaltonomous/baselines/classical`) from a `git archive` snapshot of the stated commit, overlay selected with `AALTO_OVERLAY`, Isaac Sim on Modal (L40S), one episode per container (`modal/ab_arm.sh`), `sort_fixed`, settle 3 s, nominal DR, time limit 120 s, gate 0 clips / >= 27 Hz. Baseline row = box A STEP 0 classical, flags off, seeds 1000-1019 (52/82). Noise between repeated runs is about +-3 objects on 20 episodes and about +-5 on 30.


## Results

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

## Conclusion

- Same overlay cand_r14 on two commits (786ae65 / 1b65ffb, only flags default-off differ) gave 64 vs 71: noise is above +-4 objects, differences below ~5 mean nothing.
- **retry_vary_slots +13 vs submission** (empty closes 43 -> 23, counter 20 -> 7), retry_vary +6.
- **pp_center +10 vs submission** (empty closes 43 -> 28), above noise at that time.
- depthgate -9 vs cand_r14 (worse, 1 wrong). prism_offset -6 vs cand_r14 on the same commit (no gain, triangles 21 vs 23).
- Seed 1000 of retry_vary_slots hit an infra error (scene service timeout) and was re-run on the same SHA.
- `submission_2380e83` also holds seeds 1020-1039 from another session; the report counts only 1000-1019.


## Files

- `summary.json`: machine-readable per-arm numbers (outcomes, per shape, picks, held yes/no, gate).
- `report.md`: full `modal/ab_report.py` output (per hand, spawn side, flags).
- Raw episode data and logs are not copied.
