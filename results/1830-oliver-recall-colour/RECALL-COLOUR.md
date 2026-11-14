# Colour-aware real-frame recall (drop_2 ep 0-39, visible-only, 538 GT)

Tool: `tools/recall_ab.py score --colour-aware` (oliver/recall-ab). Same E1 visibility filter and greedy 25 px match as `e1_fair_recall.py`; colour-aware only lets a detection match a GT of the same colour.
Detection colours found in all detections_e1.jsonl: `blue` 14888, `red` 4162, `yellow` 4266 (sim names only, no real names). Aliases: orange<->red, purple<->blue.
Sanity: with the colour constraint off, the reimplementation reproduces the e1_fair_recall headline, per-class counts and FP census exactly on all 33 arm dirs (a_r21_rc6 195/538 = 0.362, repro_submission 321/538 = 0.597).

`wrong_colour` = unmatched detection within 25 px of a GT object of another colour (also counted in FP). Held-out ep 40-47 never opened.

## Main A/B arms (ab/)

| arm | agnostic | colour-aware | drop | orange-cube | orange-prism | purple-cube | yellow-cube | yellow-cyl | wrong_colour | FP agn -> col |
|---|---|---|---|---|---|---|---|---|---|---|
| ab/a2_r21_recalltree | 195/538 = 0.362 | **191/538 = 0.355** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 123 -> 127 |
| ab/a_r21_rc6 | 195/538 = 0.362 | **191/538 = 0.355** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 123 -> 127 |
| ab/b_r22_recall | 139/538 = 0.258 | **136/538 = 0.253** | -0.005 | 25/62 = 0.403 | 26/79 = 0.329 | 9/113 = 0.080 | 32/188 = 0.170 | 44/96 = 0.458 | 6 | 3004 -> 3007 |
| ab/d1_sat_floor_adaptive | 194/538 = 0.361 | **190/538 = 0.353** | -0.008 | 38/62 = 0.613 | 27/79 = 0.342 | 19/113 = 0.168 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 263 -> 267 |
| ab/d2_area_floor_geometric | 137/538 = 0.255 | **135/538 = 0.251** | -0.004 | 25/62 = 0.403 | 26/79 = 0.329 | 8/113 = 0.071 | 32/188 = 0.170 | 44/96 = 0.458 | 2 | 302 -> 304 |
| ab/d3_topface_whole_blob | 195/538 = 0.362 | **191/538 = 0.355** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 123 -> 127 |
| ab/d4_partial_ok_near_hand | 196/538 = 0.364 | **191/538 = 0.355** | -0.009 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 59/188 = 0.314 | 47/96 = 0.490 | 7 | 2433 -> 2438 |
| ab/d5_relook_before_done | 195/538 = 0.362 | **191/538 = 0.355** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 123 -> 127 |
| ab/repro_submission | 321/538 = 0.597 | **319/538 = 0.593** | -0.004 | 51/62 = 0.823 | 30/79 = 0.380 | 57/113 = 0.504 | 129/188 = 0.686 | 52/96 = 0.542 | 10 | 1569 -> 1572 |
| ab/x1_no_colorgate | 282/538 = 0.524 | **270/538 = 0.502** | -0.022 | 39/62 = 0.629 | 29/79 = 0.367 | 58/113 = 0.513 | 97/188 = 0.516 | 47/96 = 0.490 | 17 | 1680 -> 1693 |
| ab/x2_no_hueguard | 218/538 = 0.405 | **218/538 = 0.405** | +0.000 | 51/62 = 0.823 | 28/79 = 0.354 | 20/113 = 0.177 | 69/188 = 0.367 | 50/96 = 0.521 | 0 | 128 -> 128 |
| ab/x3_hsv_robust | 187/538 = 0.348 | **183/538 = 0.340** | -0.008 | 37/62 = 0.597 | 24/79 = 0.304 | 20/113 = 0.177 | 58/188 = 0.309 | 44/96 = 0.458 | 4 | 123 -> 127 |
| ab/x4_hsv_robust_no_hueguard | 220/538 = 0.409 | **220/538 = 0.409** | +0.000 | 51/62 = 0.823 | 28/79 = 0.354 | 20/113 = 0.177 | 70/188 = 0.372 | 51/96 = 0.531 | 0 | 125 -> 125 |
| ab/x5_r22bisect_no_cg | 323/538 = 0.600 | **321/538 = 0.597** | -0.003 | 52/62 = 0.839 | 30/79 = 0.380 | 58/113 = 0.513 | 129/188 = 0.686 | 52/96 = 0.542 | 8 | 1733 -> 1736 |
| ab/x6_no_fkmask | 193/538 = 0.359 | **190/538 = 0.353** | -0.006 | 38/62 = 0.613 | 27/79 = 0.342 | 19/113 = 0.168 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 235 -> 239 |
| ab/x7_no_satfloor | 195/538 = 0.362 | **191/538 = 0.355** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 158 -> 162 |

## gate-failopen arms (gf/)

| arm | agnostic | colour-aware | drop | orange-cube | orange-prism | purple-cube | yellow-cube | yellow-cyl | wrong_colour | FP agn -> col |
|---|---|---|---|---|---|---|---|---|---|---|
| gf/base | 195/538 = 0.362 | **191/538 = 0.355** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 123 -> 127 |
| gf/both | 264/538 = 0.491 | **260/538 = 0.483** | -0.008 | 38/62 = 0.613 | 29/79 = 0.367 | 58/113 = 0.513 | 88/188 = 0.468 | 47/96 = 0.490 | 4 | 180 -> 184 |
| gf/both2 | 299/538 = 0.556 | **295/538 = 0.548** | -0.008 | 38/62 = 0.613 | 29/79 = 0.367 | 58/113 = 0.513 | 117/188 = 0.622 | 53/96 = 0.552 | 4 | 144 -> 148 |
| gf/ws | 264/538 = 0.491 | **260/538 = 0.483** | -0.008 | 38/62 = 0.613 | 29/79 = 0.367 | 58/113 = 0.513 | 88/188 = 0.468 | 47/96 = 0.490 | 4 | 180 -> 184 |
| gf/ws2 | 264/538 = 0.491 | **260/538 = 0.483** | -0.008 | 38/62 = 0.613 | 29/79 = 0.367 | 58/113 = 0.513 | 88/188 = 0.468 | 47/96 = 0.490 | 4 | 143 -> 147 |
| gf/yf | 211/538 = 0.392 | **207/538 = 0.385** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 69/188 = 0.367 | 53/96 = 0.552 | 4 | 123 -> 127 |
| gf/yf2 | 211/538 = 0.392 | **207/538 = 0.385** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 69/188 = 0.367 | 53/96 = 0.552 | 4 | 123 -> 127 |

## nb arms (nb/)

| arm | agnostic | colour-aware | drop | orange-cube | orange-prism | purple-cube | yellow-cube | yellow-cyl | wrong_colour | FP agn -> col |
|---|---|---|---|---|---|---|---|---|---|---|
| nb/base | 195/538 = 0.362 | **191/538 = 0.355** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 123 -> 127 |
| nb/e06 | 195/538 = 0.362 | **191/538 = 0.355** | -0.007 | 38/62 = 0.613 | 27/79 = 0.342 | 20/113 = 0.177 | 59/188 = 0.314 | 47/96 = 0.490 | 4 | 123 -> 127 |
| nb/m015 | 201/538 = 0.374 | **199/538 = 0.370** | -0.004 | 43/62 = 0.694 | 31/79 = 0.392 | 19/113 = 0.168 | 60/188 = 0.319 | 46/96 = 0.479 | 2 | 158 -> 160 |
| nb/m03 | 226/538 = 0.420 | **223/538 = 0.414** | -0.006 | 40/62 = 0.645 | 54/79 = 0.684 | 20/113 = 0.177 | 61/188 = 0.324 | 48/96 = 0.500 | 3 | 152 -> 155 |
| nb/m045 | 215/538 = 0.400 | **210/538 = 0.390** | -0.010 | 41/62 = 0.661 | 39/79 = 0.494 | 20/113 = 0.177 | 62/188 = 0.330 | 48/96 = 0.500 | 5 | 152 -> 157 |
| nb/m06 | 201/538 = 0.374 | **195/538 = 0.362** | -0.012 | 41/62 = 0.661 | 28/79 = 0.354 | 19/113 = 0.168 | 59/188 = 0.314 | 48/96 = 0.500 | 6 | 146 -> 152 |
| nb/m10 | 198/538 = 0.368 | **196/538 = 0.364** | -0.004 | 42/62 = 0.677 | 28/79 = 0.354 | 19/113 = 0.168 | 59/188 = 0.314 | 48/96 = 0.500 | 2 | 140 -> 142 |
| nb/x03_03 | 226/538 = 0.420 | **223/538 = 0.414** | -0.006 | 41/62 = 0.661 | 54/79 = 0.684 | 20/113 = 0.177 | 61/188 = 0.324 | 47/96 = 0.490 | 3 | 152 -> 155 |
| nb/x06_06 | 196/538 = 0.364 | **194/538 = 0.361** | -0.003 | 41/62 = 0.661 | 28/79 = 0.354 | 19/113 = 0.168 | 59/188 = 0.314 | 47/96 = 0.490 | 2 | 146 -> 148 |
| nb/x06_10 | 196/538 = 0.364 | **194/538 = 0.361** | -0.003 | 41/62 = 0.661 | 28/79 = 0.354 | 19/113 = 0.168 | 59/188 = 0.314 | 47/96 = 0.490 | 2 | 146 -> 148 |

## Takeaways for the real robot

- **Colour confusion is a minor loss, not the recall problem.** Colour-aware recall is 0.003 to 0.022 below agnostic on every arm (a_r21_rc6 0.362 -> 0.355, repro_submission 0.597 -> 0.593, x5 0.600 -> 0.597). Only 2 to 17 detections per arm sit on a GT of the wrong colour. The 0.597 headline survives the colour check.
- **Who gets confused with whom:** orange-cube read as yellow is the only recurring pair on the r21 family (4 detections, all arms); once the colour gate is relaxed or removed (x1_no_colorgate, repro_submission, x5) yellow-cube and yellow-cylinder read as red, and orange-cube / yellow-cube read as blue (orange-prism too on repro_submission) (x1: 17 wrong-colour, drop -0.022, the largest). The colour gate is what protects colour correctness, at the price of recall. x2/x4 (no hue guard) show zero confusions.
- **The real losses are misses and blue phantoms, not mislabels.** purple-cube is only 0.177 (20/113) on the r21 family (0.50 to 0.51 on repro_submission, x1, x5, both) and orange-prism 0.34; almost all of those misses are non-detections (only 2 to 17 wrong-colour detections per arm). Blue is also about 90% of all FPs (116/127 on a_r21_rc6, 1395/1572 on repro_submission) and explodes where recall is bought (1395 to 1557 blue FPs in repro_submission, x1, x5; 2951 in b_r22_recall vs 116 in a_r21_rc6), so a policy that picks "blue" may pick tape/wall/phantoms rather than a purple cube.
