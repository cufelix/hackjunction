# drop_2 analysis (zupermiez/g1_junction_drop_2) — OpenCode lane, CONTRACT v2

48 episodes / 14,548 frames, LeRobot v3.0, 3x RGB 640x480, **no depth**. Tasks: orange cube -> right (16)
/ left (13), purple cube -> center (14), orange triangular prism -> right (5). README: "a significant part
of the data is suboptimal". Scripts: `tools/drop2_extract.py` (192 frames: 20/50/80/97% per episode),
`tools/drop2_hue.py` (hue separation + placement heuristic), `tools/drop2_vgate.py` (V-gate + sim colour
targets). All CPU-only; `--root` points at the downloaded dataset (default /var/tmp/hf_drop2).

## 1. Scene geometry DIFFERS from scene.yaml / drop-1  (headline)

Bins are much closer/larger in frame: the row spans beyond both image edges, far rims at v~0-30
(drop-1: v~30-90). A pitch fit that assumes nominal bin xy (tools/pitch_fit.py, valid on drop-1)
**rejects 158/192 frames** (rms ~9 cm). Consequence (agreed with Timi): online pitch cal must NOT depend
on nominal bins — Codex's `perception.online_pitch_cal: table` mode takes pitch/roll from the counter
plane in depth, bins detected afterwards in the corrected frame. Five objects on the mat: orange cube,
orange prism, purple cube, yellow cube, **yellow cylinder** (new shape).

## 2. Colour: orange objects vs orange bins (assumption-free spatial split, v<0.38H bins / v>0.45H mat)

| class | H p50 | V p50 | S p50 | mean RGB |
|---|---|---|---|---|
| bin walls | 6.2 | 108 | 141 | [122, 69, 55] |
| tape | 5.6 | 112 | 182 | [112, 52, 39] |
| orange objects | 12.1 | 152 | 183 | [152, 90, 47] |
| purple cube | 104.1 | 119 | 148 | [60, 93, 131] |
| yellow objects | 32.8 | 170 | 102 | [156, 159, 99] |

- Hue separates at the MEDIAN (objects ~12 vs bins/tape ~6, consistent with drop-1's objects H9-15 /
  bins H2-7), but the prism sits at H~9-16 and per-frame distributions overlap: **hue alone is not a
  safe object-vs-bin gate**; geometry (mat region / detected bins) must carry it.
- **V-gate is a soft prior, not a hard gate**: object V_p5 across frames = 50 vs tape V_p95 = 179
  (shaded object faces are dark, lit tape is bright). Lit TOP faces reach V~250 vs tape ~119 — so a
  V>=~200 cut works on top faces only. Combine V with hue + geometry.
- **Purple is already covered by the configured `blue` HSV range: 82-97% of its pixels** (mean over
  tasks). The resolver's purple/violet -> blue alias design holds (Claude verified on origin/submission).
- Mean RGB above = starting targets for a sim colour overlay (organisers: sim colours are wrong, finals
  use purple/yellow/orange; "make the sim accurate to our setup"). Top-face values preferred over these
  whole-object means when building the overlay.

## 3. Placement outcomes (coarse, temporal diff final-vs-20% frame)

First attempt (nominal-hull centroid test) was invalid; v2 uses new-orange-in-bin / purple-in-bin with
per-episode bin boundaries from column gaps. Result: 44 "wrong_bin", 4 "not_placed", 0 "placed_correct"
— **not trustworthy**: gap detection finds only 1-2 bins in many episodes (tape merges them), and at 97%
the hand often still holds the object over a bin. Treat as an open item; the reliable outcome signal for
retry/memory stays the sim/verify_placement path ("object still on the table next frame", r16-fix).
Annotated finals: <root>/ov2/e00z.png ... e05z.png.

## 4. Lane / architecture context (CONTRACT v2, Timi 10:08)

BASE = `submission` (cand_r16_nopp). (A) sim = + instruction.normalize + r16-fix + staggered_release.
(B) finals = (A) + online_pitch_cal (table mode) + held_detector finger_sum + Oliver's PERCEPTION module
(bins/V-gate/fusion-track, after sim A/B) + brain on the 5090. Lanes: Felix language/brain, Claude
PERCEPTION module, **OpenCode = drop_2 analysis + memory**, Timi integration/A-B/builds.
`perception.reject_orange` must NEVER be ON in build B (it would reject the real orange objects).
