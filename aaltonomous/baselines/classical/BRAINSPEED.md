# Brain VLM speed (find + plan), measured live

Box E, Qwen3-VL-8B Q8 on llama.cpp (:8080, RTX PRO 6000; the finals GPU is a 5090, absolute ms will differ, ratios should hold). Sequential calls, warm, temperature 0.
Find path = `/v1/chat/completions` with the PNG head frame; plan path = `/completion` + GBNF, text only (no image). Scratch dir `/data/work/brainspeed/`, bench scripts `bench_find.py` / `bench_plan.py` (kept in the scratchpad, not in the repo).

## Find (VLM grounding): 20 frames (`ep00..19_s_f0`, 640x480) x up to 10 instruction types = 149 scored queries per row

Types: colour+shape, paraphrase ("the round yellow piece"), colour-only, alias (red for orange, violet for purple), leftmost, rightmost, ordinal from left/right ("the second object from the left"), relational ("the object farthest from the robot", scored as lands on any object), unseen noun ("the pan": correct = empty). Hit = top-1 point <= 25 px from the GT object (`/data/brain/find/gt.txt`). Ordinal/left/right rows are skipped on frames where the ordering is ambiguous (<20 px apart). Row `640` is the full frame (the frames are 640x480), so it is the "full" row. n=149 means one query is about 0.7 points: differences under ~3 points are noise.

| max side px | n_predict | layout | acc % | p50 ms | p95 ms | cut-off outputs |
|---|---|---|---|---|---|---|
| 384 | 64 | image_first | 65.8 | 321 | 366 | 7 |
| 384 | 128 | image_first | 65.8 | 319 | 352 | 0 |
| 384 | 160 | image_first | 65.8 | 316 | 355 | 1 |
| 384 | 400 | image_first | 66.4 | 316 | 357 | 0 |
| 448 | 64 | image_first | 67.1 | 323 | 367 | 1 |
| 448 | 128 | image_first | 67.8 | 322 | 353 | 0 |
| 448 | 160 | image_first | 67.1 | 319 | 353 | 0 |
| 448 | 400 | image_first | 67.1 | 319 | 350 | 0 |
| 512 | 64 | image_first | 63.8 | 325 | 383 | 7 |
| 512 | 128 | image_first | 64.4 | 326 | 362 | 0 |
| 512 | 160 | image_first | 64.4 | 325 | 357 | 0 |
| 512 | 400 | image_first | 64.4 | 325 | 355 | 0 |
| 640 (full) | 64 | image_first | 63.8 | 332 | 415 | 5 |
| 640 (full) | 128 | image_first | 63.1 | 334 | 368 | 1 |
| 640 (full) | 160 | image_first | 63.8 | 335 | 366 | 1 |
| 640 (full) | 400 | image_first | 63.8 | 336 | 364 | 0 |
| 384 | 160 | static_first | 61.7 | 273 | 309 | 0 |
| 448 | 160 | static_first | 64.4 | 274 | 316 | 0 |
| 512 | 160 | static_first | 59.7 | 276 | 327 | 0 |
| 640 (full) | 160 | static_first | 61.1 | 286 | 361 | 0 |

n_predict 400 = the previous hard-coded `max_tokens` (so `640 / 400 / image_first` is the old behaviour). 'cut-off' = the raw output did not end in `]` (64 tokens clips multi-box answers; 128 does not). `static_first` = system prompt (static prefix, cacheable) first, then image, then the phrase; it is ~45 ms faster but loses 3 points at 448/512, so it stays off.

### Accuracy by type (hits / n)

| type | 448 / 128 (chosen) | 640 / 400 (old) |
|---|---|---|
| colour+shape | 20/20 | 20/20 |
| paraphrase | 20/20 | 20/20 |
| colour-only | 19/20 | 20/20 |
| alias | 7/20 | 7/20 |
| leftmost | 0/8 | 0/8 |
| ordinal-L | 1/8 | 1/8 |
| relational | 5/20 | 2/20 |
| unseen-noun | 20/20 | 19/20 |
| rightmost | 4/7 | 4/7 |
| ordinal-R | 5/6 | 2/6 |

Weak spots that do not depend on the knobs: ordinal/leftmost/rightmost (the 8B model grounds the wrong object), and the colour aliases. Both are for the planner/validator layer, not for resizing.

## Planner (text only, `plan_prompt` + `plan_gbnf`, agent=true): 4 scenes x 10 instructions = 40 calls per row

| n_predict | p50 ms | p95 ms | first step correct % (28 scored) | cut-off |
| 64 | 274 | 534 | 71 | 3 |
| 128 | 272 | 642 | 71 | 0 |
| 160 | 272 | 646 | 71 | 0 |
| 512 | 267 | 642 | 71 | 0 |

Plan latency does not depend on the cap: the GBNF closes the array and generation stops there, so n_predict only matters as a ceiling (64 cuts 3 of 40 plans). p95 ~640 ms is the cold prefill of the first call on each new scene (the static prefix is cached, the scene JSON is not); p50 is ~270 ms. The 35% fully-valid rate is the strict validator (no-early-done guard, unseen nouns, relational steps), identical for every cap. Caps stay `plan_n_predict` 512 (an initial plan of 5 picks needs ~180 tokens, more than 160) and `agent_n_predict` 160 (agent loop plans at most 4 steps).

## Pareto choice for `finals_r22_agent`

Rule: p95 < 600 ms and accuracy within 2 points of the best row.

- Every find row already has p95 < 420 ms on this GPU, so latency does not discriminate; accuracy does. Best row = 448 / 128 (67.8 %).
- Chosen: **`image_max_side_px: 448`, `find_n_predict: 128`, layout `image_first`**: 67.8 % acc, p50 322 ms, p95 353 ms. Versus the old full-size / 400: +4.0 points, -14 ms p50 (about -4 %), p95 -11 ms. n_predict 128 loses nothing against 400 and avoids the 64-token cut-offs.
- Image size changes latency by only ~15 ms between 384 and 640 (decode of the box JSON dominates, not the vision encoder), so the real gain here is the cap and the accuracy, not the resize. The resize still shrinks the PNG upload (matters on a slower CPU/loopback path) and the image-token count (KV memory).
- `static_first` (cacheable system prefix): ~-45 ms but -3.4 points at 448, outside the 2 point band, left off. The image comes before the phrase, so only the ~50-token prefix is cacheable anyway.
- Resizing keeps the find point mapping in original pixels: the model emits boxes normalised to 0-1000, `find_tool.find(..., orig_size=(w, h))` scales them by the ORIGINAL frame size (`test_brain_speed.py` checks the same point/bbox at 384/448/512 and full).

## New config (all defaults = old behaviour)

`brain.image_max_side_px` (0 = full; applies to the find image and the decide image), `brain.find_n_predict` (400), `brain.find_layout` (`image_first`). `plan_n_predict` / `agent_n_predict` already existed. `cache_prompt: true` is now also sent on find requests.
