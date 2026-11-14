# Baseline sanity: sim clock vs wall clock on Modal (K1/K2)

## Setup

Engine: classical policy (`aaltonomous/baselines/classical`) from a `git archive` snapshot of the stated commit, overlay selected with `AALTO_OVERLAY`, Isaac Sim on Modal (L40S), one episode per container (`modal/ab_arm.sh`), `sort_fixed`, settle 3 s, nominal DR, time limit 120 s, gate 0 clips / >= 27 Hz. Baseline row = box A STEP 0 classical, flags off, seeds 1000-1019 (52/82). Noise between repeated runs is about +-3 objects on 20 episodes and about +-5 on 30.

Setup note: this folder is a different protocol from the other A/B folders (baseline commit, seeds 0-7, no overlay, DR off); the baseline row in the table is the box A STEP 0 row for reference only.

## Results

| arm | ep | score | correct/wrong/counter/off | cube | cyl | tri | cube flat | cube diag | picks/ep | held yes/no/n.a. | gate fails | clips | Hz min | RTF |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| baseline box A (flags off) | 20 | **52/82** | 52/0/30/0 | 21/25 | 18/28 | 13/29 | 15/18 | 6/7 | 6.75 | 0/0/0 | 0 | 0 | 29.99 | 0.416 |
| K1_baseline_36daf5b | 8 | **26/38** | 26/0/12/0 | 7/10 | 13/15 | 6/13 | 5/7 | 2/3 | 7.88 | 0/0/0 | 0 | 0 | 29.98 | 0.348 |

## Conclusion

K1 (default sim clock, unchanged baseline 36daf5b, seeds 0-7, DR off): **26/38**, gate PASS, RTF 0.34-0.36. The organizer reports 29-31/38 (RTF ~0.65), so K1 is 3-5 objects lower, at the edge of the quoted +-10 % and of 8-episode noise.

K2 (`AALTO_SIM_CLOCK=0`, wall-clock pacing): **10/38**, gate PASS (per-episode 3/5, 0/4, 1/5, 3/5, 1/5, 0/5, 1/4, 1/5). At RTF ~0.35 a wall-paced 30 Hz command stream drives physics about 3x too fast, so the run falls apart. Sim clock is mandatory on Modal (and in general at low RTF). `ab_report.py` cannot parse the K2 dir (no policy log), so its numbers come from the lab notes, not from `summary.json`.


## Files

- `summary.json`: machine-readable per-arm numbers (outcomes, per shape, picks, held yes/no, gate).
- `report.md`: full `modal/ab_report.py` output (per hand, spawn side, flags).
- Raw episode data and logs are not copied.
