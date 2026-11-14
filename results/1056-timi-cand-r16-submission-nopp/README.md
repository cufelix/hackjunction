# Candidate r16 (retry_vary_slots + pp_center) vs submission, fresh seeds 1040-1069

## Setup

Engine: classical policy (`aaltonomous/baselines/classical`) from a `git archive` snapshot of the stated commit, overlay selected with `AALTO_OVERLAY`, Isaac Sim on Modal (L40S), one episode per container (`modal/ab_arm.sh`), `sort_fixed`, settle 3 s, nominal DR, time limit 120 s, gate 0 clips / >= 27 Hz. Baseline row = box A STEP 0 classical, flags off, seeds 1000-1019 (52/82). Noise between repeated runs is about +-3 objects on 20 episodes and about +-5 on 30.


## Results

| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | held yes/no/n.a. | gate fails | clips | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 20 | **52/82** | 52/0/30/0 | 21/25 | 18/28 | 13/29 | 15/18 | 6/7 | 6.75 | 0/0/0 | 0 | 0 | 29.99 | 0.416 |
| submission_9a28c41 | 30 | **105/124** | 105/0/19/0 | 32/36 | 42/46 | 31/42 | 12/12 | 20/24 | 5.30 | 112/47/0 | 0 | 0 | 29.97 | 0.321 |
| cand_r16_9a28c41 | 30 | **111/124** | 111/0/13/0 | 32/36 | 44/46 | 35/42 | 10/12 | 22/24 | 4.70 | 127/14/0 | 0 | 0 | 29.96 | 0.321 |
| cand_r16_nopp_9a28c41 | 30 | **111/124** | 111/0/13/0 | 34/36 | 43/46 | 34/42 | 11/12 | 23/24 | 5.50 | 139/26/0 | 0 | 0 | 29.96 | 0.334 |

## Conclusion

Branch `timi/integ-r16`, commit **9a28c41** = 75e95b2 + merge of retry-vary (6efbd05) + principal_point (7cd01bf) + overlays `cand_r16` / `cand_r16_nopp`. 215 unit tests, flags-off equivalence IDENTICAL against 75e95b2. 30 episodes per arm (124 objects). The baseline row (52/82, seeds 1000-1019) is not the same seed set as these arms; compare the arms to `submission`.

- Both candidates +6 vs submission (105 -> 111), at the edge of noise (+-5). retry_vary_slots had given +13 on seeds 1000-1019.
- pp_center on top of nopp added nothing (111 vs 111): its earlier +10 is not confirmed as an independent gain once retry is on.
- **Recommendation: nopp variant** (same result, no sim-only offset). pp_center fixes the sim image centre (320,240) vs calibration (~325,251); on the real robot the calibration is considered correct, so pp_center must be OFF there (`perception.principal_point: file`).


## Files

- `summary.json`: machine-readable per-arm numbers (outcomes, per shape, picks, held yes/no, gate).
- `report.md`: full `modal/ab_report.py` output (per hand, spawn side, flags).
- Raw episode data and logs are not copied.
