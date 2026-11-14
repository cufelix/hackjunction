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

## adv-real2 profiles (branch `oliver/adv-real2`)

New knobs in the same patch (rebuild the tree with `make_adv_tree.sh` after pulling; the old profiles draw exactly the
same values per seed as before, because the new knobs use their own seed stream):
`table_dy_cm`, `counter_dz_mm`, `pp_px`, `dark_near_keep` (+ `dark_lum_max`, `dark_near_mm`), `hand_kp` / `hand_kd` /
`hand_gain_scale`, `prism_lying_frac`, `n_objects`, `shape_weights`, `colour_weights` (layout variation; also applied to the client layout of `reset`), `start_arm_jitter_rad` / `start_arm_q` / `start_hand_open`. The evaluator now calls `randomize`
BEFORE homing the arms, so a profile's start pose applies (it changes nothing for profiles without one).

| profile | what it adds (on top of nominal unless noted) | decides |
|---|---|---|
| `pp` | head image rendered at the real principal point 325.0/250.9 (Isaac renders 319.5/239.5) | how much this sim artefact moved every past sim number |
| `real_soft` | + pp, Dex3 PD gains kp 1.5 / kd 0.2 (real firmware); fit lag with `hand_step.py` | timi/cage, timi/release |
| `displaced` | + pp, bins +5..14 cm, counter +0..20 mm, head pitch -1.5..-9 deg, table y +-4 cm | timi/handcal, timi/gatefix, calibration |
| `dark_links` | + pp, depth kept on dark near-field pixels: 80 / 40 / 0 % (one per episode) | timi/handcal (hand ICP) |
| `start_pose` | + pp, arm start +-0.2 rad per joint around ARM_HOME, hands exactly open | anything calibrating before motion |
| `layout_var` | + pp, `prism_lying_frac` 0.5 (triangle prisms lie on a rectangular face, random yaw), `n_objects` [3,6], uniform `shape_weights` / `colour_weights` (all sim-side, seed stream 0xAD9, capped at the 2 copies per type) | organiser variation: lying prisms, random count/shapes/colours |
| `real2_all` | real_hard + all of the above (incl. `prism_lying_frac` 0.5) (worst case, report only) | readiness story |

**Held-out rule:** seeds with `seed % 4 == 3` are never used for tuning. With 10 seeds from 1000, that is 1003 and 1007.
`adv_report.py compare` prints them on their own line; a change ships only if the held-out delta is not negative.

### 10-seed A/B recipe (per profile, ~25-40 min per arm on one adv sim)
```bash
source tools/adv/adv_env.sh
# 0) once per tree: hand lag fit (sim idle). Target close.t90_median_s 0.30-0.38 s
cd $ADV_TREE && bash tools/adv/set_profile.sh real_soft && bash aaltonomous/cloud/aalto.sh py <tools>/hand_step.py /tmp/hand_step_soft.json
# 1) base vs candidate, same profile, same seeds 1000..1009
bash <tools>/run_adv.sh <base_branch|sha>  <overlay>   <profile> 1000 10
bash <tools>/run_adv.sh <cand_branch|sha>  <overlay_c> <profile> 1000 10
# 2) paired score (+ held-out line) and why objects were lost + localisation error (median / p95 / max, held-out split)
python3 <tools>/adv_report.py compare $ADV_RUNS/<profile>__<overlay>__<base_sha>__s1000n10 $ADV_RUNS/<profile>__<overlay_c>__<cand_sha>__s1000n10
python3 <tools>/loss_breakdown.py base=$ADV_RUNS/<profile>__<overlay>__<base_sha>__s1000n10 cand=$ADV_RUNS/<profile>__<overlay_c>__<cand_sha>__s1000n10
```
Suggested pairs: `real_soft`: rc7-pre finals_r22 vs timi/cage (+ timi/release); `displaced` and `dark_links`:
finals_r22 vs timi/handcal; `pp`: rc1 under `nominal` vs `pp` (same code, measures the artefact).
Not simulated: command latency jitter (needs interface changes), finger-pad friction (unmeasured).
