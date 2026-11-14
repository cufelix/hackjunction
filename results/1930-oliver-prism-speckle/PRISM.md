# Prism speckle A/B (offline, drop_2 ep<40, colour-aware, visible-only, tree purple-recall 77bf905)

All arms overlay finals_r22_gate unless noted. Overrides are `--set` keys under `perception.`. No code changed.
FP budget: arm 2 FP (36) + 5 = 41.

| arm | overrides | overall | purple | orange-prism | FP | tape / bin_wall / hand / other / phantom | FP by colour b/y/r | FP <= 41 |
|---|---|---|---|---|---|---|---|---|
| 1 | finals_r22 (baseline) | 144/538 .268 | 18/113 | 6/79 | 14 | 0/0/5/0/9 | 5/9/0 | yes |
| 2 | gate (gate_below_bins) | 377 .701 | 113/113 | 56/79 | 36 | 1/0/16/0/19 | 18/14/4 | yes |
| 3a | area_floor_geometric, min_frac 0.35 (default) | 385 .716 | 113 | 52/79 | 257 | 101/23/57/4/72 | 69/30/158 | NO |
| 3b | area_floor_geometric, min_frac 0.20 | 408 .758 | 113 | 56/79 | 364 | 114/24/127/6/93 | 147/34/183 | NO |
| 3c | area_floor_geometric, min_frac 0.50 | 379 .704 | 113 | 51/79 | 136 | 61/10/28/1/36 | 33/19/84 | NO |
| 4a | color_gate.area_frac 0.25 (default 0.4) | 394 .732 | 113 | 58/79 | 114 | 10/4/53/4/43 | 59/20/35 | NO |
| 4b | color_gate.area_frac 0.12 | 413 .768 | 113 | 59/79 | 408 | 112/24/169/7/96 | 188/34/186 | NO |
| 4c | color_gate.area_frac 0.35 | 385 .716 | 113 | 57/79 | 48 | 1/1/23/0/23 | 25/14/9 | NO (+12) |
| 4d | color_gate.area_frac 0.30 | 391 .727 | 113 | 58/79 | 71 | 3/2/32/1/33 | 37/17/17 | NO |

## Reading
- No arm meets the FP budget AND beats arm 2 on prism. Best prism gain inside any tolerable FP is +1 (4c: 57 vs 56, FP +12).
- Speckle relaxation (4a/4b) buys +2/+3 prism for +78/+372 FP: the extra blobs are mostly hand, phantom and tape, plus red/blue FP.
- area_floor_geometric (3a-3c) does not help prism (52/56/51) and floods FP (tape, hand) at every tried value; the overall gain is
  other classes (orange-cube, yellow), not the prism.
- Prism misses at gate are partly "no blob" (10), a mask problem no area threshold can fix.
- Recommendation: keep finals_r22_gate unchanged. Prism recall needs a code change (shape/colour-aware speckle floor for prism
  only), which this task forbids; stopping here.
Raw: /home/oliver/work/ps/{a1..a4d}, score.md/score.json.
