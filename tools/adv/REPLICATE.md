# Replicate the adversarial sim on another box

What you get: a COPY of the sim tree (`/data/aaltonomous-humanoid-adv`) with a patch (camera pitch/yaw/xyz, bin shift, render
colours + tape, exposure/WB/gamma, depth holes/noise, widened object spawn) driven by profiles. The original tree and its running
sims are never touched. The adv sim is its own instance (instance `c`: DDS domain 8, cams 57555-7, rgbd 57560, scene 57580,
clock 57581, teleimager req 62000) so it can run next to the default sim and instance b.

Prereqs on the box: the stock tree at `/data/aaltonomous-humanoid` (conda env `isaac`), docker with image
`hackjunction/policy:integ-770888c`, a policy checkout (any dir containing `aaltonomous/`, e.g. the r16 worktree).

1. Copy `tools/adv/` of this repo to the box, e.g. `/data/work/<you>/adv/tools/` (scp -r), `chmod +x *.sh`.
2. Build the tree (once, ~1 s; refuses if it exists; remove it only when no sim runs from it):
   `bash /data/work/<you>/adv/tools/make_adv_tree.sh`   (args: `[orig_tree] [adv_tree]`)
   It does `cp -a` of the stock tree, `patch -p1 < adv_sim.patch`, and copies `profiles/*.yaml` into `aaltonomous/sim/dr/`.
3. Start the adv sim (3-5 min, log `/tmp/adv_sim_c.log`). Pin cores so it does not fight other sims:
   `ADV_CORES=0-9 bash /data/work/<you>/adv/tools/start_adv_sim.sh`    (default cores 15-29; ~13 cores per sim)
   On a one-sim box stop the default sim first (`kill -INT <its python pid>`; never `pkill sim_main`, never stop sims that are not yours).
4. Frame check (sim idle): `source adv_env.sh; cd $ADV_TREE; bash aaltonomous/cloud/aalto.sh py <tools>/dump_profile_frame.py <out_dir> real_hard 1000`
5. Run: `bash <tools>/run_adv.sh <checkout_dir|branch|sha> <overlay> <profile> <seed0> <n> [policy_port=8002]`
   Needs `ADV_REPO` (git repo used to resolve a branch/sha, default `/data/work/oliver/cand`) or pass a checkout dir; results go to
   `$ADV_RUNS` (default `/data/work/oliver/adv/runs`). One eval at a time per sim.
6. Compare: `python3 adv_report.py compare <runA> <runB>` (paired by seed, safety stop = -6, mean-std). Why objects were lost:
   `python3 loss_breakdown.py real_hard=<run_dir>[,<run_dir>] ...` (run inside a checkout so `aaltonomous.scene` imports).

Profiles (switch at run time, no restart): `nominal` (stock, no perturbation), `real_mild`, `real_hard`, `out_of_range`.
`run_adv.sh` switches via `set_profile.sh <name>`. Files: `profiles/adv_<name>.yaml`; key `adv:` holds the knobs
(cam_pitch_deg, cam_yaw_deg, cam_xyz_cm, bin_dx_cm, bin_dx_jitter_cm, exposure_ev, wb_jitter, gamma, depth_noise_mm,
depth_hole_frac, depth_hole_blob_frac, depth_hole_near_boost, render_colors, bin_rgb, tape_rgb, ensure_object, object_region).
Everything is seeded per episode with the trial seed (`randomize`). Not implemented: latency jitter (needs interface changes).
Policy-side scene.yaml / calibration are NOT changed: the policy keeps believing the nominal geometry, that is the point.

If the stock tree differs from the one the patch was cut from, `patch` fails loudly; regenerate with
`diff -u` of `aaltonomous/sim/aalto_task/{dr,hooks,scene_cfg}.py`.
