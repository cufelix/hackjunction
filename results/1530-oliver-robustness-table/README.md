# Robustness table snapshot (oliver, 1530)

Snapshot of existing results only, no new runs. Built with `tools/robustness_table.py` (branch `oliver/robustness-table`). Score cell = mean ± std per episode, correct/objects, wrong count, n episodes.

Conditions:
- **nominal**: Modal stock sim seeds 1040-1069, plus adv-sim nominal runs.
- **real_mild**: real colours, bin offset dx 7±3 cm, camera pitch -7.5±1.5°, mild depth noise.
- **real_hard**: as real_mild, plus lighting/exposure/white-balance variation and more noise.
- **score**: +2 per correct object, -1 per wrong, safety stop = -6 per episode.

Sha suffix = commit prefix. Local adv copies of `dir`-named G runs were renamed to `9a28c41` (ckout `ck_9a28c41e`) so they merge with the Modal `9a28c41` rows.

## (a) Everything found, default rows

| overlay | nominal | real_mild | real_hard |
|---|---|---|---|
| K1_baseline | +3.25 ± 0.66 (26/38 obj, 0 wrong, n=8) | – | – |
| K2_baseline | +1.25 ± 1.09 (10/38 obj, 0 wrong, n=8) | – | – |
| ab_decision | +2.53 ± 1.26 (39/59 obj, 1 wrong, n=15) | – | – |
| ab_drop2 | +1.95 ± 1.02 (40/82 obj, 0 wrong, n=20) | – | – |
| ab_slots@6a65371 | +3.05 ± 1.02 (61/82 obj, 0 wrong, n=20) | – | – |
| ab_slots@a48ef4f | +3.00 ± 0.00 (3/3 obj, 0 wrong, n=1) | – | – |
| ab_stagger | +3.39 ± 1.06 (61/73 obj, 0 wrong, n=18) | – | – |
| ab_verify | +2.17 ± 0.90 (65/124 obj, 0 wrong, n=30) | – | – |
| cand_r16 | +3.70 ± 1.04 (111/124 obj, 0 wrong, n=30) | – | – |
| cand_r16_nopp@9a28c41 | +3.69 ± 0.94 (133/147 obj, 0 wrong, n=36) | +0.30 ± 0.64 (3/40 obj, 0 wrong, n=10) | +0.10 ± 0.30 (2/40 obj, 1 wrong, n=10) |
| cand_r16_nopp@9fd8ac0 | +3.70 ± 0.84 (74/82 obj, 0 wrong, n=20) | – | – |
| cand_r16_nopp@c036154 | +3.59 ± 0.83 (97/111 obj, 0 wrong, n=27) | – | – |
| cand_r16_nopp@d04fbea | +3.76 ± 1.07 (94/104 obj, 0 wrong, n=25) | – | – |
| cand_r16_nopp@eb31647 | +3.56 ± 0.98 (90/104 obj, 0 wrong, n=25) | – | – |
| cand_r17@e5bbd90 | – | – | +0.10 ± 0.30 (2/40 obj, 1 wrong, n=10) |
| cand_r17@fbbdbd5 | +3.33 ± 1.70 (10/13 obj, 0 wrong, n=3) | – | – |
| rc1 | +3.77 ± 0.89 (98/109 obj, 0 wrong, n=26) | – | – |
| submission | +3.50 ± 0.85 (105/124 obj, 0 wrong, n=30) | – | – |

Cell: mean score per episode ± population std · correct/objects · wrong · n episodes · safety stops (scored -6).

### Paired vs base `K1_baseline` (same seeds, candidate − base)

| candidate | condition | n | Δ mean | Δ (mean − std) |
|---|---|---|---|---|
| K2_baseline | nominal | 8 | -2.00 | -2.43 |
| ab_decision | nominal | 15 vs 8 | seeds differ, not paired | |
| ab_drop2 | nominal | 20 vs 8 | seeds differ, not paired | |
| ab_slots@6a65371 | nominal | 20 vs 8 | seeds differ, not paired | |
| ab_slots@a48ef4f | nominal | 1 vs 8 | seeds differ, not paired | |
| ab_stagger | nominal | 18 vs 8 | seeds differ, not paired | |
| ab_verify | nominal | 30 vs 8 | seeds differ, not paired | |
| cand_r16 | nominal | 30 vs 8 | seeds differ, not paired | |
| cand_r16_nopp@9a28c41 | nominal | 36 vs 8 | seeds differ, not paired | |
| cand_r16_nopp@9fd8ac0 | nominal | 20 vs 8 | seeds differ, not paired | |
| cand_r16_nopp@c036154 | nominal | 27 vs 8 | seeds differ, not paired | |
| cand_r16_nopp@d04fbea | nominal | 25 vs 8 | seeds differ, not paired | |
| cand_r16_nopp@eb31647 | nominal | 25 vs 8 | seeds differ, not paired | |
| cand_r17@fbbdbd5 | nominal | 3 vs 8 | seeds differ, not paired | |
| rc1 | nominal | 26 vs 8 | seeds differ, not paired | |
| submission | nominal | 30 vs 8 | seeds differ, not paired | |

## (b) Focused: cand_r16_nopp, cand_r17, rc1; nominal / real_mild / real_hard

| overlay | nominal | real_mild | real_hard |
|---|---|---|---|
| cand_r16_nopp@9a28c41 | +3.69 ± 0.94 (133/147 obj, 0 wrong, n=36) | +0.30 ± 0.64 (3/40 obj, 0 wrong, n=10) | +0.10 ± 0.30 (2/40 obj, 1 wrong, n=10) |
| cand_r16_nopp@9fd8ac0 | +3.70 ± 0.84 (74/82 obj, 0 wrong, n=20) | – | – |
| cand_r16_nopp@c036154 | +3.59 ± 0.83 (97/111 obj, 0 wrong, n=27) | – | – |
| cand_r16_nopp@d04fbea | +3.76 ± 1.07 (94/104 obj, 0 wrong, n=25) | – | – |
| cand_r16_nopp@eb31647 | +3.56 ± 0.98 (90/104 obj, 0 wrong, n=25) | – | – |
| cand_r17@e5bbd90 | – | – | +0.10 ± 0.30 (2/40 obj, 1 wrong, n=10) |
| cand_r17@fbbdbd5 | +3.33 ± 1.70 (10/13 obj, 0 wrong, n=3) | – | – |
| rc1 | +3.77 ± 0.89 (98/109 obj, 0 wrong, n=26) | – | – |

Cell: mean score per episode ± population std · correct/objects · wrong · n episodes · safety stops (scored -6).

### Paired vs base `cand_r16_nopp@9a28c41` (same seeds, candidate − base)

| candidate | condition | n | Δ mean | Δ (mean − std) |
|---|---|---|---|---|
| cand_r16_nopp@9fd8ac0 | nominal | 20 vs 36 | seeds differ, not paired | |
| cand_r16_nopp@c036154 | nominal | 27 vs 36 | seeds differ, not paired | |
| cand_r16_nopp@d04fbea | nominal | 25 vs 36 | seeds differ, not paired | |
| cand_r16_nopp@eb31647 | nominal | 25 vs 36 | seeds differ, not paired | |
| cand_r17@e5bbd90 | real_hard | 10 | +0.00 | +0.00 |
| cand_r17@fbbdbd5 | nominal | 3 vs 36 | seeds differ, not paired | |
| rc1 | nominal | 26 vs 36 | seeds differ, not paired | |

## Commands

```
# 1. copy episodes.json of FINISHED runs only (box F and box G; hosts omitted here)
ssh -o ConnectTimeout=8 <box> 'cat /data/work/oliver/adv/runs/<run>/episodes.json' </dev/null > adv_eps/<run>/episodes.json
# 2. local rename: __dir__ -> __9a28c41__ for the two G cand_r16_nopp runs (ckout ck_9a28c41e)
# 3. (a)
python tools/robustness_table.py --adv adv_eps --modal modal_extract --out table_a.md
# 4. (b)
python tools/robustness_table.py --adv adv_eps --modal modal_extract --rows cand_r16_nopp,cand_r17,rc1 --out table_b.md
```

`--rows ...,ab_plane_table,ab_plane_rim` was tried and makes the script exit 1 ("no runs for ab_plane_table"), so both were left out.

## Unfinished, excluded

- Box F: `real_mild__cand_r17__e5bbd90__s1000n10` (running).
- Box G: `real_hard__ab_plane_table__dir__s1000n10` (ck_7688de2), `real_hard__ab_plane_rim__dir__s1000n10` (ck_7688de2).

Result: no ab_plane rows exist in either table.
