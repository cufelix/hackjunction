# cand_r16_nopp at c036154 (retry safety hardening)

## Setup

Engine: classical policy (`aaltonomous/baselines/classical`) from a `git archive` snapshot of the stated commit, overlay selected with `AALTO_OVERLAY`, Isaac Sim on Modal (L40S), one episode per container (`modal/ab_arm.sh`), `sort_fixed`, settle 3 s, nominal DR, time limit 120 s, gate 0 clips / >= 27 Hz. Baseline row = box A STEP 0 classical, flags off, seeds 1000-1019 (52/82). Noise between repeated runs is about +-3 objects on 20 episodes and about +-5 on 30.


## Results

| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | held yes/no/n.a. | gate fails | clips | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 20 | **52/82** | 52/0/30/0 | 21/25 | 18/28 | 13/29 | 15/18 | 6/7 | 6.75 | 0/0/0 | 0 | 0 | 29.99 | 0.416 |
| cand_r16_nopp_c036154 | 27 | **97/111** | 97/0/14/0 | 28/32 | 42/43 | 27/36 | 11/12 | 17/20 | 5.44 | 103/44/0 | 0 | 0 | 29.96 | 0.306 |

## Conclusion

Commit **c036154**: "Harden retry safety: next-frame miss check, attempt cap 18, corrective picks via _plan_pick; finals_real overlay" on top of 9a28c41. Seeds 1040-1064, 1068, 1069 (27 episodes; seeds 1065-1067 are missing from the local run dir). Different seed set than the 20-seed baseline row, so compare with `cand_r16_nopp_9a28c41` (111/124 on seeds 1040-1069, folder 1056) rather than with the baseline row.

Result: **97/111 correct, 0 wrong, 14 left on the counter**, cubes 28/32, cylinders 42/43, triangles 27/36; gate pass, 0 clips, min 29.96 Hz. Correct rate 87.4 % vs 89.5 % for 9a28c41 nopp on its 30 seeds; difference within noise, so the hardening costs nothing measurable while adding the safety caps.


## Files

- `summary.json`: machine-readable per-arm numbers (outcomes, per shape, picks, held yes/no, gate).
- `report.md`: full `modal/ab_report.py` output (per hand, spawn side, flags).
- Raw episode data and logs are not copied.
