# RC videos: cand_r16_nopp at c036154

Sim footage of the cand_r16_nopp policy (commit c036154, no overlay flags beyond the candidate), recorded in Isaac Sim on the sim box during the RC run. Each file is one episode, with a text overlay (instruction, task, seed, time, picks done, final result) drawn by `vbuild.py`.

## Files

| file | task | seed | episode | run result (trial) | length |
|---|---|---|---|---|---|
| `sort_fixed_s1040.mp4` | sort_fixed | 1040 | last episode of the run | 3/3 correct, end: done | 3.9 s (117 frames) |
| `sort_fixed_s1041.mp4` | sort_fixed | 1041 | last episode of the run | 5/5 correct, end: done | 3.3 s (100 frames) |
| `sort_language_s1.mp4` | sort_language | 1 | last episode of the run | 5/5 correct, end: done | 3.6 s (109 frames) |

Instructions: sort_fixed = red to left bin, yellow to center, blue to right. sort_language = red to right bin, yellow to left, blue to center.

## Encoding

- H.264 (yuv420p), crf 30, preset slow, from 30 fps JPG frames (`vbuild.py`).
- Width 800 px (the builder's `scale=800:-2`), not the 640 px in the folder convention. Kept as built; each file is about 0.2 MB.

## Setup

- Engine: cand_r16_nopp @ c036154 (as written on the overlay).
- Recorded by `/data/work/rcvideo/run_rc.sh` on the sim box; frames built with `vbuild.py <task> <seed>`.
- Only the last episode of each run is in the video (the builder picks the last episode of the seed).
- Summary numbers for these seeds are in the sibling folders (`1206-timi-cand-r16-nopp-c036154`).
