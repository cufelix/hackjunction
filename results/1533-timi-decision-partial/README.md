# ab_decision (c20b2b8) vs cand_r16_nopp (c036154): PARTIAL, arm clearly worse

Status: INCOMPLETE. 21 of 30 planned seeds (1040-1060) finished. Seeds 1061-1069 are missing because the Modal run stopped early (scene-service timeouts in the container logs).

## Setup

- Arm: engine commit c20b2b8, overlay `ab_decision`.
- Baseline: `cand_r16_nopp` at c036154 (as the job specified), 27 seeds (1040-1064, 1068, 1069).
- Task sort_fixed, settle 3 s, nominal DR, time limit 120 s, gate 0 clips / >= 27 Hz.

## Results

| arm | seeds | score | correct/wrong/counter | cube | cyl | tri | picks/ep | never picked | Hz min |
|---|---|---|---|---|---|---|---|---|---|
| cand_r16_nopp_c036154 | 27 | **97/111** (87.4 %) | 97/0/14 | 28/32 | 42/43 | 27/36 | 5.44 | 0 | 29.96 |
| ab_decision_c20b2b8 | 21 | **57/85** (67.1 %) | 57/1/27 | 17/24 | 32/36 | 8/25 | 3.86 | 12 | 29.97 |

## Conclusion

The decision overlay is much worse on this seed set: 12 objects never picked, triangles 8/25, and 1 wrong pick, against 0 wrong for the baseline. Do not promote it; the 9-seed gap is large enough that the missing seeds are unlikely to change the verdict.
