# Adversarial sim policy logs (learned_p)

`adv_policy_logs.tgz` holds the policy logs and per-episode results of finished adversarial-sim runs (sort_fixed task). Only runs with `episodes.json` (finished) are included. 35 episodes in 4 groups.

## Layout

```
<group>/<group>_s<seed>/policy.log
<group>/<group>_s<seed>/adv_summary.json
```

`group` = `<profile>__<overlay>_<sha>`. Episode dir = `<group>_s<seed>`, same pattern as the Modal extract in `1510-timi-policy-logs`.

| group | episodes | seeds |
|---|---|---|
| `nominal__cand_r16_nopp_9a28c41e` | 5 | 1000-1004 |
| `real_hard__cand_r16_nopp_9a28c41e` | 10 | 1000-1009 (two runs: 1000-1004 + 1005-1009) |
| `real_hard__cand_r17_e5bbd90` | 10 | 1000-1009 |
| `real_mild__cand_r16_nopp_9a28c41e` | 10 | 1000-1009 |

## policy.log

Split from the run's `policy_server.log` at each `reset task=... seed=...` line. An episode's log runs from its reset line up to the next reset. Startup config lines (before the first reset) are prepended to the first episode of each run.

## adv_summary.json

| field | meaning |
|---|---|
| `group`, `episode_id` | group key and `<group>_s<seed>` |
| `profile`, `overlay`, `sha` | sim profile, policy overlay, code sha |
| `seed`, `episode_index_in_run` | scene seed and index within its run |
| `task`, `variant`, `instruction` | task id, DR variant, instruction text |
| `n_objects`, `correct`, `wrong`, `counter`, `off` | object outcomes |
| `score`, `raw_score`, `max_score`, `all_correct` | scoring |
| `safety_stop`, `clips` | safety-clip counts |
| `end_reason` | e.g. `done` |
| `wall_s`, `sim_s`, `ticks` | episode wall time, sim time, control ticks |
| `rtf`, `command_hz`, `policy_hz_rt` | sim real-time factor and rates |
| `mapping`, `layout` | colour to bin mapping and the randomised object layout |
| `source_run` | run dir the episode came from |

## Profiles

- `nominal`: stock-like sim, no domain randomisation.
- `real_mild`: real colours, bin positions shifted by 7 +/- 3 cm, camera pitch -7.5 +/- 1.5 degrees, mild depth noise.
- `real_hard`: as `real_mild`, plus lighting/exposure changes and more noise.

## Notes

- The two `real_hard__cand_r16_nopp` runs (seeds 1000-1004 and 1005-1009) share one group.
- Dropped one duplicate: the nominal seed-1000 episode from a second run matched the kept episode in layout, outcomes and mapping, with different timing. The kept copy is from the n5 run.
- Simulator `.npz` recordings and eval `.jsonl` raw files are not included.
- Not included (still in progress, no `episodes.json`): `real_mild__cand_r17` and `real_hard__ab_plane_rim` / `real_hard__ab_plane_table`.
