pan_tools_data  (rescued copy, read-only copy from aalto-sim-g and aalto-sim-v00128; nothing committed or pushed)

g_pan_dir/            copy of /data/work/oliver/pan on aalto-sim-g (without ck_* checkpoints and without runs/)
  tools/              make_pan_tree.sh (cp -a stock tree + pan_sim.patch + make_pan_stl.py), pan_sim.patch, pan_env.sh,
                      start_pan_sim.sh, run_pan.sh, run_pan.py (+ .pre_* backups), make_pan_real.py, make_pan_stl.py
  meat_patch.py, tools_patch.py, tools_patch_lib.py, scenes_patch.py
                      Oliver's runtime patches: meat scenes, tools scenes (screwdriver/wrench/hammer/pliers, tools_left/right),
                      scene variants (egg_left/right, meat_*, thirsty_left, tidy)
  run_*.out, dbg_*, frames_* (jpg), pancake_check.py, miss.py, frame_cam.py   logs and debug frames
pan_tree_vs_stock_aaltonomous.diff
                      diff -ruN stock /data/aaltonomous-humanoid/aaltonomous vs the live pan tree /data/aaltonomous-humanoid-pan
                      (state on G when copied; *.stl excluded here, see next)
pan_tree_assets/      object_*.stl from the pan tree: pan (pan, pan_20cm, pan_egg15, pan_real, pan_toy), pancake, meat, lime,
                      teacup, knife, screwdriver, wrench, hammer, pliers.  NO drill mesh exists anywhere; no USD/URDF were
                      created (objects are STLs loaded by scene_cfg).
runs_show_clips/      26 *_show.mp4 (third-person recordings, 56 MB) + pan_results.json from Oliver's runs on G.
                      The rest of runs/ (143 MB: jsonl/npz/mp4/logs) stays on G; also see git origin/oliver/pan-video
                      (108 MB: A_..K_ clips incl. D_toypan, E_showcam, F/G egg, H/I meat, J/K vlm flip).
cine_camera_patch/    my cinematic camera, for reuse:
  cine_vs_stock.diff  hooks.py + scene_cfg.py vs the stock tree (cine_vs_rc2form.diff = vs the earlier rec patch)
                      adds: CINE_* env (MODE orbit|orbit2|side|overhead|front, T, W/H, FOCAL, LIGHT, NEAR, DIST), camera path
                      set via USD xform each step (demo camera follows the active palm and blends to a low close-up when
                      hand z < 0.97), rec ts.tsv columns, resets.tsv, dark floor/black sky/rim lights (CINE_LIGHT=1)
  run.sh              run.sh <slot> <gpu> <cores> <name> <mode> <task> <seed> <trials> [image]; starts the sim from
                      /data/work/cine/tree, policy docker, rec_start, eval, rec_stop. Needs --camera_include ...,demo_camera
                      (already in the script) and NOVEL_CAM=1. Run from a copy of the sim tree with the diff applied.
  pull.sh, edit.py, endcard.png   fetch frame range from the box and encode; cut + xfade + rap audio + vertical + end card
Still on v00128: /data/work/cine/{tree,out,run.sh} (1080p frames of the sorting shots, 2-3 GB), not copied.

Rerun the pan/tools sim (on a box you own; one sim per box, default ports):
  1. bash g_pan_dir/tools/make_pan_tree.sh /data/aaltonomous-humanoid /data/aaltonomous-humanoid-pan
  2. apply meat/tools/scenes patches as run_pan.py imports them (they sit next to it in /data/work/oliver/pan)
  3. source tools/pan_env.sh; bash tools/start_pan_sim.sh; bash tools/run_pan.sh <checkout> <overlay> <variants> [port]
Honesty: screwdriver never ran in Isaac, no drill scene exists (per Felix).
