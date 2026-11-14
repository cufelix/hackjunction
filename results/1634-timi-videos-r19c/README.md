# finals_r19c: simulation videos (Isaac sim, whole episode, timelapse)

Four short clips of the current finals candidate (overlay `finals_r19c`) running in the stock Isaac sim. Each clip shows the whole episode, sped up to 20 s. The head camera is the main view, the two wrist cameras are on the right, and the text overlay shows the instruction, task, seed, sim time, picks done, and the episode result.

| file | task | seed | instruction | result | length |
|---|---|---|---|---|---|
| sort_fixed_s1040_timelapse.mp4 | sort_fixed | 1040 | Put the red objects in the left bin, the yellow objects in the center bin and the blue objects in the right bin. | 2/3 correct, end: done | 20 s (episode 60.5 s sim time, about 3x) |
| sort_fixed_s1041_timelapse.mp4 | sort_fixed | 1041 | Put the red objects in the left bin, the yellow objects in the center bin and the blue objects in the right bin. | 4/5 correct, end: done | 20 s (episode 69.2 s sim time, about 3.5x) |
| sort_language_s1_timelapse.mp4 | sort_language | 1 | Drop the red ones into the right bin, the yellow ones into the left bin, the blue ones into the center bin. | 5/5 correct, end: done | 20 s (episode 60.9 s sim time, about 3x) |
| sort_language_s3_timelapse.mp4 | sort_language | 3 | Blue goes to the left bin, yellow to the right bin, red to the center bin. | 3/5 correct, end: done | 20 s (episode 87.8 s sim time, about 4.4x) |

Encoding: H.264, yuv420p, 800 px wide, crf 28, 15 fps, 20 s each. Each clip is one episode, one run, no cuts.

Setup:
- Code: `felix/rc5` at `eb0f25d`, overlay `finals_r19c`, time limit 120 s, one trial per seed.
- Simulator: stock Isaac sim (no changes to the sim).
- Frame dump: the recordings used a debug-only frame dumper on the policy server (annotated head and wrist frames). It only records and does not change the actions.
- Overlay times are the policy clock mapped linearly onto the frame index, because the dump keeps every other tick.
- Results come from the eval harness (`eval.json`) of each episode. The overlay "picks done" counts planned picks in the policy log, so it can exceed the number of objects.
