# ab_stagger vs cand_r16_nopp (d04fbea): PARTIAL

Status: INCOMPLETE. 24 of 30 planned seeds (1040-1063) finished. Seeds 1064-1069 are missing because the Modal run stopped early (scene-service timeouts in the container logs). The result is not final.

## Setup

- Engine commit: d04fbea (same commit for both arms).
- Arm: overlay `ab_stagger` (cand_r16_nopp + `grasp.staggered_release`).
- Baseline: overlay `cand_r16_nopp` at d04fbea, 30 seeds.
- Task sort_fixed, settle 3 s, nominal DR, time limit 120 s, gate 0 clips / >= 27 Hz.
- Runner: `modal/aalto_eval.py` on Modal L40S, one episode per container.

## Results

| arm | seeds | score | correct/wrong/counter | cube | cyl | tri | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|
| cand_r16_nopp_d04fbea | 30 | **113/124** (91.1 %) | 113/0/11 | 36/36 | 46/46 | 31/42 | 29.96 | 0.323 |
| ab_stagger_d04fbea | 24 | **84/99** (84.8 %) | 84/0/15 | 23/27 | 37/41 | 24/31 | 29.96 | 0.341 |

The two arms do not cover the same seeds (baseline 30, stagger 24), so these are not a paired comparison.

## Conclusion

Stagger is 6 points lower on 24 seeds, with no wrong picks and the gate passing on both arms. Not decided yet: the seed set is incomplete and the difference is within the noise of about +-5 objects on 30 episodes.
