# Pan in real Isaac (box G, AALTO_PAN=1 sim), 2026-11-15 ~00:20Z — throwaway branch

Instruction: "put the pan in the middle bin". Sim pan: 20 cm disc + 12 cm handle, grey. Overlay finals_r23_universal.
Per episode: `*.mp4` (head camera recording from the runner), `*.npz` / `*.jsonl` / `*_traj.json` (runner logs),
`policy_server.log` (the policy's reasons). No grasp phase happened in any run (all plan-rejected / left).

- `A_gd_4e7741d_fpoff/` felix/generality-demo 4e7741d, fp_filter off: pan blob merges with the hovering hand
  (height 200 mm) on *_in; diag: "wider than the hand, no handle".
- `B_sha3cand_fpon/` 4e7741d + oliver/novel-fp-v2 94aa3a8, fp_filter true: pan clean, but the disc is cut by the
  bottom/right image edge -> box-like, "wider than the hand, no handle".
- `C_fullview_sha3cand/` same code, pan at x=0.37 (whole pan in view): far_left matched with-handle, rejected
  "handle too short/occluded" (the bin band cuts the disc's far half); far_right "white elongated", no handle.
- `frames/pan_<variant>.npz`: head RGB-D every 3rd tick (rgb uint8 [N,480,640,3], depth [N,H,W], arm_q, hand_q, t)
  for the first 8-15 s; `.json` = layout. Replay: `tools/novel_fp/replay_novel.py frames --fp-filter on`.
- `D_toypan_sha3_4af094c/` felix/brain-r23 4af094c, toy pan (r 6 cm, handle 8 cm beyond the rim): small_right/left plan + execute on the handle 3x each, hand closes on nothing (held=no); small_left lifted 3 cm; rdiag graspability 0.35 (border), ldiag no handle.
- `E_showcam/` third-person RTX camera (1280x720, 30 fps sim time, side-front) + head cam, toy pan small_left on SHA3 4af094c (3 empty closes on the handle). Grid = frames at 15/35/55/75 %.

## F_egg15_4c2b055 — 15 cm egg pan, felix/handle-grasp-fix 4c2b055 (2026-11-15 01:05)
Procedural pan (own geometry, `make_pan_real.py PAN_RIM_R=75`): rim 15 cm, base 11.7 cm, 4 cm walls, handle
15 x 2.2 x 0.8 cm at +10 deg, 150 g, pale-cream pancake prop inside. Instruction 'put the pan in the middle bin'.
Both arms pinch the HANDLE and lift (right 11.2 cm, left 12.5 cm) with the pancake in the pan; the pan then
slips out of the hand during the carry and lands tipped next to the centre bin (in_center_bin False on both).
`grid_egg_right.jpg` = 1 frame/s of the show-cam clip.

## frames_skillet18 — head RGB-D of the 18 cm skillet (real_right / real_left)
dump_frames.py format (rgb, depth, arm_q, hand_q, t) on 017cbb2. The 18 cm body does not fit the visible strip
(image bottom edge ~pelvis x 0.23, bin band x 0.37): blob truncated -> 'elongated' / 'wider than the hand'.
For checking felix/pan-edge locally.

Demo mass 80 g; at 150 g the one-hand pinch slips during carry (F_egg15_4c2b055 is the honest 150 g run).

## G_egg15_80g_4c2b055 — same as F at 80 g (2026-11-15 01:12)
egg_left: handle pinch, lift 11.6 cm, carried to the centre bin with the pancake still inside; the pan ends upright
resting on the bin rim (z 0.10), not inside. egg_right: lift 6.2 cm, then the pan slips. `grid_egg_left.jpg` 1 frame/s.

## H_meat_hold_327e947 — 'take the pan and hold it steady' (felix/pancake-demo 327e947, AALTO_FORCE_SKILL=hold)
80 g egg pan + 60 g meat slab (9x6x2 cm, RGB 0.24/0.20/0.18, no known HSV class) + 4 clutter objects.
meat_right: left-arm handle pinch, lift 10.7 cm, held 13.8 s, put back; meat in pan at end of hold + end: yes/yes.
meat_left: right arm, lift 12.7 cm, held 15.3 s; yes/yes. Meat inside the rim the whole time lifted on both.

## I_meat_hold_150g_327e947 — same hold scene, honest 150 g pan
meat_right: lift 11.5 cm, held 13.8 s, meat in pan at end of hold + end yes/yes (slid ~30 mm toward the wall, still inside).
meat_left: lift 10.7 cm, held 15.9 s, yes/yes. 2/2 at 150 g: the hold skill keeps the pinch where the carry slipped.
