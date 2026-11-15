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
