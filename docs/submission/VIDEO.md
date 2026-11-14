# Demo video plan

Length: 2:00 maximum. Target 1:50 so the cut has slack. Record voice-over last.

## Shot list

| Time | Shot | Voice-over | On-screen text |
|---|---|---|---|
| 0:00-0:10 | Sim sort run, head camera view, objects leaving the counter | We built a policy that sorts coloured objects into bins on the Unitree G1. | hackjunction / Sorting on the G1 |
| 0:10-0:30 | Text instruction on screen, then the debug overlay with colour, shape and bin labels per object | An instruction goes in as text. We turn it into a colour and shape to bin map. Then colour and depth give each object a 3D position. | 1 Resolver / 2 Perception |
| 0:30-0:50 | Arm planning and a pick in the sim, then the drop in the bin | The grasp is chosen by shape. Inverse kinematics plans the arm around the bins, and the motion is time-scaled to speed limits. | 3 Grasp / 4 Planning / 5 Motion |
| 0:50-1:10 | Terminal: evaluator summary table, then the gate result | In the official-style sim, the release candidate placed 33 of 38 objects on the fixed task and 30 of 36 on the language task, with no wrong bins. The safety gate showed no clipped steps. | 33/38 fixed, 30/36 language, 0 wrong bins |
| 1:10-1:35 | Adversarial sim: side by side nominal and `real_hard` frames, then the report | The stock simulator is too kind. Our adversarial copy shifts bins, moves the camera, changes the light and softens the hands. On its hardest profile, 42 percent of early losses came from camera pitch. That is why we built online calibration. | Adversarial sim / 42% pitch, 42% bin rims |
| 1:35-1:50 | Terminal: release gate PASS and the fault-drill output | Every change sits behind a flag. A release gate checks the shipped flags. If an instruction does not parse, the object stays on the counter. | Release gate: PASS / abstain, never guess |
| 1:50-2:00 | Closing card | One Docker image, websocket policy on port 8000, joint targets only. We have not run on the real robot yet, and we say so. | Docker image / hackjunction |

Edit the 1:50 line if a real-robot clip exists by recording time. Never show real-robot footage we do not have.

## What to capture

All commands run from a checkout of the repo. Paths are repo-relative. Where a tool lives only on a branch, the branch is named.

### 1. Sim eval run with video

Starts an episode run against the sim with a head-cam mp4 per episode.
`aaltonomous/deploy/runner.py` has the `--video` flag. It writes `runs/<team>/<run>/episode_NNN.mp4`. The command is from `aaltonomous/deploy/README.md`.

```bash
python -m aaltonomous.deploy.runner --backend sim --container hackjunction/policy:rc1-7a4168e --team hackjunction --episodes 5 --video --export
```

### 2. Evaluator summary table (the 33/38 and 30/36 shot)

Use the same evaluator as the readiness numbers. It prints a table and writes JSON per trial. Match the organiser seeds and image used for the cited run. Seeds are not in our sources, so [fill the organiser seed value].

```bash
python -m aaltonomous.sim.evaluate --container hackjunction/policy:rc1-7a4168e --team hackjunction --task sort_fixed --trials 20 --seed <seed> --gate-max-clips 0 --gate-min-hz 27 --out runs/eval
python -m aaltonomous.sim.evaluate --container hackjunction/policy:rc1-7a4168e --team hackjunction --task sort_language --trials 20 --seed <seed> --gate-max-clips 0 --gate-min-hz 27 --out runs/eval
```

Trial counts differ from the cited run (38 and 36 objects). Quote the numbers from the new run on screen if you re-record.

### 3. Overlay and debug images

- Annotated frames (blobs with colour and shape, planner phase, last action): `aaltonomous/baselines/classical/framedump.py`. Set `debug.frame_dump_dir` (and `debug.frame_dump_every`, default 2). It is on `origin/timi/framedump` and not in `origin/main` or `origin/felix/rc7-pre`. Debug only, never in a release overlay. Put the key in a local overlay file, not in `submission.yaml`.
- Raw sim frames for a perturbation profile (RGB and depth PNG): `tools/adv/dump_profile_frame.py`, branch `origin/oliver/adv-real2`, sim idle.

```bash
source tools/adv/adv_env.sh
cd $ADV_TREE && bash aaltonomous/cloud/aalto.sh py <tools>/dump_profile_frame.py <out_dir> real_hard 1000
```

Run it again with `nominal` for the side-by-side shot. The profile names are in `tools/adv/REPLICATE.md`.

### 4. Adversarial sim run and report

Branch `origin/oliver/adv-real2`, needs a box with the adv sim up (`tools/adv/start_adv_sim.sh`). Setup steps are in `tools/adv/REPLICATE.md`.

```bash
bash tools/adv/run_adv.sh <checkout_dir|branch|sha> <overlay> real_hard <seed0> <n>
python3 tools/adv/adv_report.py compare <runA> <runB>
python3 tools/adv/loss_breakdown.py real_hard=<run_dir>
```

`loss_breakdown.py` gives the per-cause loss table for the 42 percent shot. Run it inside a checkout so `aaltonomous.scene` imports.

### 5. Readiness gate output

On `origin/main` for the gate. `tools/promote.sh` and `tools/equivalence_check.py` are on `origin/main` as well.

```bash
python3 tools/release_gate.py rc1
python3 tools/flags.py
tools/promote.sh --dry-run rc1
```

`release_gate.py rc1` exits 0 on PASS and prints differing flags on FAIL. Check that it prints PASS on the tree you film. `promote.sh --dry-run` changes nothing in the repo.

### 6. Fault drills

Branch `origin/felix/fault-drill-rc1`. We have no verified command in the sources. [Ask Felix for the exact drill command before filming.]

## Production notes

- Capture at 1080p. Increase terminal font size so numbers read on a phone.
- Keep each terminal shot under 8 seconds. Zoom on the one line that matters.
- Subtitles on. Judges may watch without sound.
- Record voice-over at one take per row. Total spoken words should stay near 230.
