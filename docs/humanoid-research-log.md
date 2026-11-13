# Humanoid research log

Research only. Append new findings at the bottom, newest last. Companion to `humanoid-starter-notes.md`.

## 2026-11-13 ~21:00 | Official challenge page (pasted by Oliver) and first web pass

### New facts from the official page
- Boxes: 21 cm wide, 5 cm apart, 40 cm from the robot's torso. Counter in front, robot legs and waist held stiff.
- Mentors on site: Erkka Lehtonen and Viktor Solianoi. Plus about 30 Aalto Robotics Club volunteers.
- Top ~5 teams are chosen **by sim performance** and run live on the real G1 on Sunday. Best real-robot score wins: 2000 EUR + Aalto robotics lab visit.
- Page says "Status: Closed" with a countdown (T -13:16:45 when pasted). UNVERIFIED what it refers to. Ask a mentor whether submission or registration is closed.
- Sim image: `aalto-sim-v1.tar.zst` (download link shared at kickoff). Run with `docker load`, then `docker run --gpus all --network host --ipc host`. Inside: `aalto sim` (first start 5-10 min while shaders compile), `aalto check` (8 PASS / 0 FAIL expected), `aalto baseline`, `aalto eval --policy ws://localhost:8000`.
- No Docker route: `bash aaltonomous/cloud/setup_instance.sh`, 30-40 min, no sudo.
- Package a submission with `aaltonomous/docker/submission.Dockerfile`. Policy server template in `aaltonomous/deploy/`.
- GPU: RTX-class, 24 GB or more. On-demand, not spot. Keep work on a persistent volume. A100/H100 fine for training only.
- Real-world data arrives during the event on Hugging Face, links in Junction Discord.
- Page asks for: robustness to sim-real mismatch, general ideas beyond one setup, a lot from very little real data. This matches the originality (20%) and generality (25%) criteria.

### Web search findings (standard search, not yet verified against primary docs)
- GR00T N1.5 few-demo run on a G1 tabletop task: LoRA (rank 16, alpha 32, 10k steps) on 5 demos, batch size 1, 11.3 GB GPU. Full fine-tune ran out of memory on 16 GB. No success rate reported. Source: https://discuss.huggingface.co/t/fine-tuning-gr00t-for-chess-setup-on-unitree-g1/172323
- NVIDIA's own G1 workflow uses Dex3-1 hands, a head RealSense and a Jetson AGX Thor, targets GR00T 1.7, and has a dataset `nvidia/Arena-G1-Static-PickNPlace-Task` (apple to plate, not Dex3). Source: https://docs.nvidia.com/learning/physical-ai/gr00t-e2e-workflow/latest/index.html
- NVIDIA says N1/N1.5 were post-trained on 1K G1 teleop episodes. That is far more than our ~20. Source: https://research.nvidia.com/labs/gear/gr00t-n1_5
- For a new embodiment the forum suggests `embodiment_tag="new_embodiment"`.
- `unitree_sim_isaaclab` has Dex3 pick-and-place tasks (e.g. `Isaac-PickPlace-Cylinder-G129-Dex3-Joint`) and the same DDS topics as the real robot. No published Dex3 sim-to-real results found.
- Isaac Lab Mimic humanoid docs cover GR-1; G1 is in the AutoData DexMimicGen guide (one sample run: 5/10 generated demos successful). GR-1 reference: 65-80% generation success over 1000 demos, BC policy 75-86%. Sources: https://isaac-sim.github.io/IsaacLab/main/source/features/imitation-learning/humanoids_imitation.html and https://isaac-sim.github.io/AutoData/main/pages/workflows/humanoid_dexmimicgen/step_3_generate_dataset.html
- Benchmarks on cheap arms (SO-101): SmolVLA 32.5%, ACT 33.75%, pi0.5 higher, with only 20 eval episodes per task. Source: https://arxiv.org/pdf/2606.08881. The SmolVLA paper's own numbers (78.3% vs ACT 48.3% vs pi0 61.7%) come from a Medium summary, so verify. No inference Hz figures found.

### Read so far
- With about 20 demos per rung, pure imitation learning is risky. Evidence says VLAs need hundreds to 1K demos on G1. The synthetic route (Mimic or own scripted policy with domain randomisation) is where data volume would come from.
- Hybrids look most plausible: parse the instruction once (rule-based or VLM), then run perception + IK skills with learned grasping as a stretch.

### Open
- Research bot (grokbot "Research") was sent a deep-research request, not answered yet.
- Subagent is extracting INTERFACE, SUBMISSION, FAQ, data card, baseline README. Results not in yet.
- xAI API tools (`grok_search_*`, `grok_ask`) fail: no credentials. Needs `grokbot login` or `XAI_API_KEY`. Only the bot channel works.

## 2026-11-13 ~21:10 | Remaining starter docs (INTERFACE, SUBMISSION, FAQ, DATA_CARD, baseline README, QUICKSTART)

Extracted by a haiku subagent from the repo; numbers not independently re-checked. RULES_AND_SCORING and STARTER_PATHS were read directly earlier.

### Interface
- Obs: `head_rgb` (480,640,3) uint8; `head_depth` (480,640) uint16 mm, 0 = invalid; `left_wrist_rgb`, `right_wrist_rgb` (480,640,3); `arm_q` (14), `arm_dq` (14), `hand_q` (14) float32; `instruction` str; `t` float s.
- Action: `arm_q_target` (14,) or (k,14) absolute rad, `hand_q_target` same k, optional `done`. Missing key, wrong shape or NaN counts as a clip.
- Arm order: 0-6 left (shoulder pitch, roll, yaw, elbow, wrist roll, pitch, yaw), 7-13 right. Hand joint order differs between left and right. Left hand closes negative, right positive. Zero = open.
- Cameras: head RealSense D435i, fx=fy=615. Wrists fx=fy=400, **extrinsics null, intrinsics placeholder (uncalibrated)**. Head extrinsics from URDF, about 48 degrees down, unverified.
- Frame: pelvis, x forward, y left, z up, metres. Counter surface z = 0.01.
- Safety filter: wrist/palm workspace box x 0..0.80, y -0.70..0.70, z -0.25..0.60. Counter keep-out x 0.14..1.20, y -0.55..0.55, z < 0.03. Margin 0.05 rad arms, 0.02 rad hands. Watchdog 0.2 s hold, 1.0 s safe stop. **Self-collision and fingertip-to-table are NOT checked, staff will e-stop for them.**
- Scene: object band x 0.26..0.33, y -0.17..0.17. Bin centres x 0.505, y +0.26 / 0 / -0.26. Comfortable reach about 0.40 m, straight arm 0.55 m.

### Submission
- msgpack over websocket, protocol v1. Server sends `metadata` on connect. Client sends `reset` then repeated `act`. One request in flight. Slow `act` is held up to 30 s. An error reply starts the 1 s watchdog and ends the episode.
- Docker: x86_64, CUDA 12.8 or newer, weights baked in, listening within 120 s, fits 32 GB VRAM, image at most 40 GB, no volumes or args. Finals PC has an RTX 5090.
- Hand-in: push to Docker Hub or GHCR, pin the digest, submit through a web form. Deadline Sunday. **The last image before the deadline runs in the finals.**
- Local test ladder: synthetic replay, then `runner --backend mock`, then `sim.evaluate --trials 20` (gate: 0 clips, 27 Hz).
- Inside the sim container there is no docker CLI, run the policy image on the host and point at `ws://localhost:8000`.

### Data
- DATA_CARD.md is still an unfilled template: episode counts and dates unknown.
- LeRobot v3.0 (with depth) and GR00T copy (no depth). State 28 = arm 14 + hand 14. Action 28. Depth is 12-bit HEVC, clips at 4095 mm.
- Demos were recorded at arm cap 0.9 rad/s. No latency compensation. Frame drops and stale ticks are in `warnings`. Depth holes on the dark mat and near bin walls.
- Filter with `success`, `notes`, `warnings` in `meta/aalto_episodes.jsonl`.

### Baseline
- HSV blobs + head depth, top-down grasp, Pinocchio IK. Rung 1 only, ignores shape and instruction.
- Sim rung 1, seeds 0-7: **76-82% of objects sorted**, 0 clips, 30 Hz, gate PASS. Misses: mostly triangular prisms, plus the 120 s limit.
- Real-robot status conflicts: baseline doc says never run, FAQ says it did one pick.

### Quickstart
- Needs driver 580.65.06 or newer, Ubuntu 22.04/24.04, 8+ cores, 32 GB RAM, 80 GB disk, nvidia-container-toolkit. First `aalto sim` start about 9 min. `aalto baseline` about 75 s per trial.
- Ports: head 55555, wrists 55556/55557, head RGB-D 55560, scene service 55580. DDS domain 1.

### Traps and sharp edges
1. Sim fingertips are about 57 mm apart, a 45 mm object can wedge between index and middle. Pinch with the thumb.
2. Validation failures count as clips, and more than 30 clips ends the episode.
3. Held ticks do not count toward the 27 Hz gate, and your own latency must also allow 27 Hz.
4. Calibration changes mid-event and the baseline bakes it into the image: rebuild after each update.
5. Any policy leaning on wrist-camera geometry is uncalibrated today.

### Strategic read (my opinion, not a fact)
1. The baseline already sorts about 80% of objects on rung 1 in sim. Beating it is the real bar, so a from-scratch learned policy with about 20 demos starts far behind it.
2. The baseline ignores the instruction, so rung 2 (worth +2 per object, and the x1.5 final) is the biggest gap. A parser for the 8 templates plus paraphrases is cheap, a VLM is the robust version.
3. Triangles are the main grasp failure. Better grasps there are worth more than a new architecture.
4. Originality and generality need a story: robustness to sim-real mismatch (depth noise model, calibration check from Drop 1, DR) on top of the classical stack.

## 2026-11-13 ~21:20 | Web pass 2: latency, depth noise, grasping, challenge buzz

Standard web search, summaries not yet checked against the primary pages.

### Challenge-specific
- Nothing public found on the Aaltonomous Humanoid challenge, DigiTwin or "Väiski" outside the repo and the Junction page. Mentors and Discord are the only live source.

### Inference speed (matters for the 27 Hz gate; finals PC is an RTX 5090 32 GB)
- GR00T N1.6-3B on RTX 5090: NVIDIA path about 27 Hz per call (4 denoising steps, single view, torch.compile). Community FlashRT engine reports 13 ms (about 76 Hz). Pi0.5 on RTX 5090 with FlashRT: 17.6 ms (about 57 Hz). Source: https://forums.developer.nvidia.com/t/real-time-inference-on-thor-rtx-pi0-5-gr00t-n1-6-1-7-thor-23-hz-rtx-5090-50-80hz/368788
- Caveat: those are model calls per second, not action steps. Returning a chunk of k actions per call decouples command rate from model rate. The wall-clock rule (`policy_hz_rt` at least 27) still applies to ticks that carry a command, so chunking is required for anything slower than 27 calls/s. Configs differ (steps, views), so the spread between 27 and 76 Hz is not hardware alone.
- Our policy would see 3 camera views plus state, not one, so expect the slower end. UNVERIFIED for our exact input.

### Depth noise recipes (for matching Drop 1 real depth)
- NVIDIA Isaac Lab guide: depth camera randomization (uniform value noise plus deliberate data gaps): https://docs.nvidia.com/learning/physical-ai/getting-started-with-isaac-lab/latest/transferring-robot-learning-policies-from-simulation-to-reality/03-bridging-the-gap-simulation-enhancement/02-depth-camera-randomization.html
- Parkour in the Wild (arXiv 2505.11164): D435-class recipe with edge dropouts, Perlin-noise hole patches that stay consistent across frames, Gaussian blur. Hole value is max depth there, but our data uses 0 = invalid, so convert.
- Camera Depth Models (arXiv 2509.02530, ICLR 2026): learned D435 hole and noise model, reports transfer without extra randomisation. RaSim (arXiv 2404.03962) simulates the stereo pipeline physically.
- Not found: an Isaac Sim built-in D435 noise model.

### Grasping
- Nothing on Dex3-1 specifically. General three-finger work: single point-cloud grasp predictors (CMG-Net, arXiv 2303.13182) and optimisation-based precision grasps at about 0.4-0.5 s per grasp (arXiv 1904.07332, 1902.10841). The challenge repo's own tip (pinch with the thumb, not index-middle) is more direct.

### Read
- Latency is not the blocker for a VLA path on the 5090 as long as it returns chunks. Data volume (about 20 demos) is.
- Depth realism is a concrete, citable originality angle: fit a hole/noise model to Drop 1 depth and randomise the sim with it.

## 2026-11-13 ~21:35 | Pass 3: prior G1 grasping work, synthetic data, instruction parsing

### Prior G1 work
- Åbo Akademi thesis, Ahmad Alkhaldi, 2026, "Reinforcement Learning for Humanoid Grasping with the Unitree G1" (https://doria.fi/handle/10024/195766). RL in Isaac Lab, sim only for the grasp policy. 91.9% grasp success in sim, 72.3% when the object must also end upright. Progress-based reward shaping beat the alternatives. Reward hacking (unintended contact, object displacement, standing still) forced reward and training changes. Sim-to-sim (MuJoCo) showed dynamics, timing and control-interface mismatches. No hardware grasp result. Hand model and details NOT read: the PDF came back as unreadable bytes (saved locally). Treat as abstract-level only.
- `sl628/g1_hackathon` (a G1 hackathon repo): teleop with Quest 3 plus GR00T N1.6 fine-tune, no numbers, no Dex3 code, no licence. Not useful.

### Synthetic data
- DexMimicGen (arXiv 2410.24185, ICRA 2025): dexterous tasks used 5 source demos per task, expanded to about 21K synthetic demos in total across tasks. A secondary summary claims 40 validated synthetic rollouts from 4 real demos gave 90% real success on a can-sorting task vs 0% for real-only. UNVERIFIED, check the paper. https://arxiv.org/abs/2410.24185
- Isaac Lab docs: G1 path is in the AutoData DexMimicGen guide, one sample run produced 5/10 successful generated demos (from the earlier pass).

### Instruction parsing (rung 2)
- No controlled study found of small local LLM vs rules. One self-published model card: Qwen2.5-1.5B 28% exact match at about 2 s, 0.5B 18% at about 3.8 s on its own robot-command set, a custom fine-tune 76% at 0.5 s. Not independently checked, different task. https://huggingface.co/AMFORGE/foros-v5.3
- Our instruction space is tiny: each of 3 colours maps to one of 3 bins, so at most 27 mappings. That makes this a classification problem with a closed answer set, not open-ended parsing. A rule-based parser with fuzzy matching and a constrained fallback can be tested exhaustively against paraphrases. (My opinion.)
- Note from STARTER_PATHS "difficulty knobs": the harder finals settings may add **colour + shape instructions, negation, and held-out paraphrases**. The code today only does colour, 8 training templates. A parser that only handles the 8 templates is at risk in the x1.5 final.
- Latency is a non-issue if parsing happens once in `reset()`, not per tick.

### Read
- RL grasp policy in sim looks feasible (92% in a thesis) but the real-robot gap and reward hacking are the risks, and it is the highest-effort route.
- Best value-for-effort ordering so far: (1) instruction parser handling colour+shape+negation, (2) better triangle/cylinder grasp, (3) depth-noise randomisation, (4) synthetic demos for a learned residual. None built.

## 2026-11-13 ~21:45 | Scoring question from Felix, answered from the rules

- Scoring is **final state at episode end**, not on touch. IN = centre inside the inner walls, below the rim, at rest (< 2 cm/s). A misplaced object can be moved again.
- No timeout or "incomplete" penalty. Episode ends at 120 s, points so far kept. Safety abort keeps points too.
- Per object: fixed +1 / wrong bin -1 / on counter 0 / knocked off -1. Language: +2 / -1 / 0 / -1.
- Break-even for attempting an object: P(correct) above 1/2 on rung 1, above 1/3 on rung 2, if a failed attempt costs -1. A drop that stays on the counter costs 0, so the true risk is lower.
- Felix's read (from the Warren thread): bin choice is an `if`, the hard part is the three-finger grasp. Matches the baseline's measured misses (mostly triangular prisms). Abstention should key on grasp confidence and instruction ambiguity, not colour.

## 2026-11-13 ~21:55 | tasks.py read: CORRECTION to the "27 mappings" line above

Source: `aaltonomous/interface/tasks.py` in the starter repo (read directly).

- **The generator only makes permutations**: `rng.permutation(COLORS)` and `rng.permutation(BINS)`, zipped. So training instructions have **6 possible mappings (3!)**, each colour to a different bin. My earlier "at most 27" was wrong for the training generator. Finals may differ: the repo says held-out templates are deliberately not in this file.
- 8 training templates, all mention all three colours and all three bins. Two orderings exist: colour then bin ("red into left", "Drop the red ones into the left bin") and **bin then colour** ("The left bin is for red objects", "Use the left bin for red"). A parser has to pair each colour with its bin in both orders.
- Bins are always named by position: left, center, right. Never by colour or "the middle one". Spelling is "center" (US).
- Shapes in code: cube (extruded square), cylinder, triangle (equilateral triangle prism). Colours: red, yellow, blue.
- `sample_layout`: 3 to 6 objects by default (task text says 3-5), at most 2 per colour, shape drawn per object after positions. `layout_card` prints the layout for staff on the real robot.
- Implication: a parser that finds each colour word and its nearest bin word, tested against all 8 templates x 6 mappings, would be right on every training case. The open risk is the held-out finals phrasing, possibly with shape words or negation (see difficulty knobs in STARTER_PATHS).

## 2026-11-13 ~22:05 | Baseline README and grasp config read directly

Source: `aaltonomous/baselines/classical/README.md` and `config.yaml` in the starter repo.

- **Never run on the real robot** ("not run yet (M8)"). All tuning was in sim. HSV floor, grasp point and `open` posture are "only checked in the sim". Treat the 76-82% as a sim number.
- History: 0/24 through the runner before the 2026-11-11 fixes, because perception was 15-20 mm short (lit top faces washed out, so HSV saturation floor lowered 120 to 70) and grasps slipped.
- Sim ran at about 0.65x real time with 2-3 Isaac instances, making the 30 Hz motions look about 1.5x faster to the physics. Expect +-10% between runs.
- Grasp = "claw from above": palm faces the counter, fingers curl down in front of the object, thumb swings in from behind. Pinch is thumb vs ONE finger, because the Dex3 index and middle tips are 57 mm apart and a 45 mm object wedges between them.
- Grasp point in the palm frame is hand-tuned: `point_in_palm [0.099, -0.084, 0.015]`, grasp height at 0.5 x object size, three finger-direction hints tried in turn. It does not use object yaw or shape, which is a plausible cause of the 6 missed triangles. (My inference, not stated in the README.)
- Planner runs in a low-priority thread and returns 6-tick hold chunks while planning, which keeps the command rate at 30 Hz.
- Files: `perception.py` 9 KB, `planner.py` 24 KB, `kinematics.py` 12 KB, `policy.py` 12 KB. CPU only image, about 1.5 GB.

## 2026-11-13 ~22:15 | Baseline source read: planner.py, perception.py, policy.py

Source: baseline files in the starter repo, read directly (460-line planner, grep plus targeted reads).

- **Perception is explicitly shape-agnostic.** It finds coloured blobs, takes the top-face centre from depth, and computes an in-plane `yaw` with `cv2.minAreaRect`, but the field is commented "info only". Planner code has no use of yaw or shape (grep for yaw/shape/prism returns nothing outside the time-scaling "triangle profile"). This confirms the likely cause of the 6 triangle misses: the grasp ignores object orientation and shape.
- **Rung 2 is a one-line gap in `policy.py`**: `self.mapping = info.get("mapping") or dict(FIXED_MAPPING)`. The reset info carries only `{"instruction": ...}` from the server, so the mapping silently falls back to fixed red-left, yellow-center, blue-right. Anything that turns the instruction into a `{colour: bin}` dict and puts it in that field gives rung 2.
- Planner already has: both arms, a finger-direction hint list tried in turn, bin-wall clearance checks, relocation of an object out of the way (`plan_relocation`), a failed-position list so it does not retry the same spot, and sends `done` when nothing sortable is left.
- **No post-placement verification**: nothing checks that the object ended up in the bin. Since scoring is final-state, a verify-and-retry step (camera or wrist check) is an unused lever.
- Not yet read: `kinematics.py`, hand posture tables (`config.yaml` hand section), how long each pick takes per phase (README says 10-20 s per pick, so about 6-12 picks fit in 120 s, enough for 5 objects only if few retries).

## 2026-11-13 ~22:40 | Hand specs and public G1 Dex3 datasets (web, standard search)

### Dex3-1 hand (Unitree's own page, https://unitree.com/Dex3-1, retailers repeat it)
- 7 DoF: thumb 3 joints (-60..60, -35..60, 0..100 deg), index 2 joints (0..90, 0..100), middle 2 joints (same). Fingers only flex, no sideways spread on index/middle, so pinch geometry is limited and the thumb does the opposition work.
- 33 tactile pressure sensors in 9 arrays (palm, finger pads, fingertips), 10-2500 g range. **The challenge interface exposes no force or tactile data** (FAQ), so this cannot be used in a submission.
- Fingertip repeat positioning +-2 mm, 710 g, rated for a 500 g grasp of a 5 cm round hard object. Our objects (45 mm, light printed) are inside that envelope.
- Teams cannot change the hand, so grasp design has to work with flex-only index/middle plus a 3-joint thumb.

### Public G1 Dex3 datasets
- Unitree publishes Dex3 datasets on Hugging Face: ObjectPlacement, BlockStacking, PickBottle, Pouring, CameraPackaging (https://huggingface.co/unitreerobotics). The earlier STARTER_PATHS names (`G1_Stack_Block`, `G1_Dex3_PickGum_Dataset`) were not confirmed by this search. Check each card's `meta/info.json` for action dims and fps before mixing with ours.
- `unitree_lerobot` repo converts to LeRobot with `--robot_type Unitree_G1_Dex3` (https://github.com/unitreerobotics/unitree_lerobot). One community sim dataset is 30 fps, LeRobot v3.0.
- Real action layout is NOT confirmed to be 28-dim in those sets; ours is (arm 14 + hand 14).

## 2026-11-13 ~23:00 | Oracle ceiling and existing instruction parser (repo sim README + oracle.py, read directly)

- **The organisers' own oracle only reaches about 84% on rung 1 in sim**, even with privileged exact object poses and shapes. Their note: "needs >90%: triangle grasps (thumb vs one finger spins the prism)". Source: `aaltonomous/sim/README.md` line 77. So perception is not the bottleneck, the grasp is. A perfect vision stack tops out near 84% until the triangle grasp changes. (Grokbot Research bot reached the same conclusion independently and told us so in its partial reply.)
- **A rung-2 instruction parser already exists in the repo**: `oracle.py` parses the mapping "from the instruction (all training templates)" and falls back to the privileged `env.mapping`. It is in the oracle (a sim-only policy), not in the submitted baseline policy, but the code can be read and copied (check its licence and that it uses no privileged state).
- `scene.resolve_bin` accepts mapping keys by **colour, shape, or (colour, shape)**. That hints the finals may use shape-based or combined instructions, which matches the "harder knobs" in STARTER_PATHS. The baseline `policy.py` only keys on colour (`self.mapping.get(d.color)`).
- The oracle also implements a **relay hand-over**: an object one arm can grasp but not carry to its bin (cross-body) is put down at a free point near the centre line for the other arm. The baseline's README lists "objects near the centre line need the arm on the wrong side" as a failure, so this is a known fix already written, only on privileged state.
- The Research bot could not find the 40/25/20/15 judging split anywhere public and says the repo rules rank finals by real-robot score with sim breaking ties. The split is on the official challenge page (pasted by Oliver), which is behind the Junction login. Both are consistent.

## 2026-11-13 ~23:30 | scene.yaml read directly (geometry, the single source of truth)

Source: `aaltonomous/scene/scene.yaml`. Values marked "(default)" in the file are placeholders until measured.

- Bins: outer 21 x 21 cm (**measured on the real boxes 2026-11-13**). **Wall height 6 cm and wall thickness 3 mm are defaults, not measured.** Objects are 4.5 cm tall, so an object has to be **lifted over a 6 cm rim**. Sweeping or pushing objects into a bin cannot work.
- Side-bin centres sit at r = 0.57 m from the waist, **past the ~0.55 m straight-arm reach**. The file says release in the near, inner part of the side boxes. Centre bin is at r = 0.505 m.
- Object region: pelvis x 0.26 to 0.35 m, y +-0.17 m, farthest corner r = 0.39 m, kept inside the head camera's view. Min spacing 8 cm, yaw -45 to +45 deg, 3 to 5 objects.
- Object size 45 mm for every shape, with 2.5 mm fillets. "Cube side = cylinder diameter = triangle ALTITUDE (flat-to-apex)". An equilateral triangle with a 45 mm altitude has about 52 mm sides. So a pinch across flat-to-apex is 45 mm, a pinch across the corners is wider. Which way the thumb meets the triangle (flat side vs apex) depends on object yaw, and the baseline planner ignores yaw. (Inference.)
- 18 printed objects in total, 2 copies of each of the 9 types.
- Counter: table top 80 cm above floor, pelvis assumed 79 cm, so counter surface at z = +0.01 in the pelvis frame (default, to be measured).
- "Not at rest at episode end counts as counter (0 points) unless it is inside a bin." So a bounce or roll at release time can cost the point.
- Rung 3 lid: sits on bin 'left', 3 mm thick, hole clearance 1 cm default (1.5 cm easy, 3 mm hard).

## 2026-11-13 ~23:50 | Grokbot Research report received

Saved in `docs/humanoid-grokbot-report.md` (condensed, bot claims UNVERIFIED unless marked checked). Verified here against `sim/dr/train.yaml`: the DR defaults (depth noise 1 mm + 0.2%, 1% holes, +-1 cm / +-1 deg head jitter) are as the bot says, and the file itself calls them "(default) guesses until the venue capture".
Decision from Oliver: do NOT use the plain xAI API / `grokbot login`; only the bot channel and normal web search.
Bot's headline matches ours: hybrid, parser for rung 2, grasp is the bottleneck, skip end-to-end VLA with about 20 demos.

## 2026-11-14 ~00:10 | Friction and grasp-heuristic web pass (standard search, thin results)

- **Friction in Isaac Lab:** NVIDIA's gear-insertion tutorial sets object friction 0.75 and robot finger bodies 3.0, restitution 0, via `randomize_rigid_body_material` at startup, and says the values came from iterative visual tuning. An NVIDIA forum answer suggests friction combine mode max or multiply so a high fingertip value is not averaged down by the object's. Isaac Sim tips say coefficients above 1.0 are usually not needed, so 3.0 is a deliberate workaround. Nothing found for PLA or Dex3 fingertips specifically. Source: https://isaac-sim.github.io/IsaacLab/develop/source/policy_deployment/02_gear_assembly/gear_assembly_policy.html
- **Grasp heuristic:** no source for the exact recipe (minAreaRect on the top face, align the pinch axis to a flat-face normal, symmetric-prism check). Closest: a patent and an arXiv paper that fit a plane, check whether its normal aligns with the camera z-axis, then use the minimum bounding box or pinch along an edge (https://arxiv.org/pdf/2009.08618v2.pdf, treat as unchecked). Top-down single-depth-view grasping can fail when the arm occludes the planned grasp.
- **Failed-grasp detection without tactile:** Calli et al. (Yale, ICRA 2018) infer contact modes from actuator states plus vision and find commanded velocities among the most important features. A patent says motor-current sensing "may have a high degree of error under influence of friction". Nothing found on comparing final closure angle to commanded angle. Note: in our interface `hand_q` is the only finger feedback (`FAQ.md:52`), no current or torque.
- Second Research-bot job (triangle-grasp deep dive) is still running at this point.

## 2026-11-14 ~00:30 | Compute lane: where the sim can run (web, standard search, unverified)

- **Verda** (the challenge's compute partner, event credit EUR 200 per person per `hackerpack.md`): RTX PRO 6000 listed at $1.39/h on Verda's own products page (https://verda.com/products), a third-party tracker shows $1.93/h on-demand and $0.675 spot (https://computeprices.com/providers/verda/gpus/rtx-pro-6000). Figures disagree, check Verda's live page. L40S about $1.37/h (one comparison). A6000 about $0.61/h in one ambiguous RunPod comparison. Nothing found on Verda + Isaac Sim + VNC. The only public hint is PR #1 on the starter repo, "Add Verda VNC deployment". At $1.39/h, EUR 200 buys about 140 GPU-hours (about 4 people x 35 h on one GPU each, if the credit converts 1:1, UNVERIFIED).
- **Modal** ($100 per person): third-party mirror lists L40S, A100, H100, A10G, L4, T4; Modal's own guide mentions B200/B300. Isaac Sim needs RT cores: L40S has them (NVIDIA says Isaac Sim runs on AWS G6e with L40S), H100 lacks them (NVIDIA forum), A100/B200/B300 unconfirmed but consistent with the challenge's own warning that A100/H100 do not render. So on Modal use L40S for sim, H100/A100 only for training (https://modal.com/docs/guide/gpu).
- **Friend's A6000**: 48 GB, RT cores, explicitly on the challenge's supported list. Not tested yet. Needs Docker + nvidia-container-toolkit + driver at least 580.65.06 (QUICKSTART requirement), and about 80 GB free disk.
- Plan implication (opinion): sim on the A6000 or one Verda RTX PRO 6000, training on Verda 80 GB or Modal H100. Keep the sim box and the training box separate: sim physics is CPU-bound at 0.5-0.7x real time, so training on the same cores breaks the 27 Hz gate (bot report).

## 2026-11-14 ~01:00 | sim README "Still open" section (read directly, `aaltonomous/sim/README.md` lines ~60-100)

- **The 84% oracle / 76-82% baseline numbers may be pessimistic or distorted by speed.** README: "Earlier tuning ran ~1.8x too fast in physics time (wall pacing at RTF ~0.55); re-check grasp parameters on the sim clock." So the grasp parameters were tuned while motions were effectively 1.8x faster than 30 Hz sim time. Triangle slip could partly be a speed artifact. Unknown until re-run on the new sim clock. Worth asking mentors whether the baseline score was re-measured since.
- **Sim clock fix**: the sim publishes its step clock (zmq PUB on scene port + 1, e.g. 55581) and `AaltoEnv(backend="sim")` paces 30 Hz ticks on it (`interface/backends.py` `SimClock`), so physics sees commands at 30 Hz of sim time whatever the load. Obs `t`, time limits, watchdog and `command_hz` count sim seconds. `policy_hz_rt` (rate the policy's wall latency allows) is logged. A sim started before this has no clock and falls back to wall pacing: restart the sim. `AALTO_SIM_CLOCK=0` forces wall pacing.
- Episodes take 1/RTF longer in wall time: **120 s sim is about 220 s wall** (RTF about 0.55). Evaluation of 20 trials is therefore slow, budget accordingly. Rendering is about 2/3 of the step cost (3 cameras at 30 fps cannot reach 1.0x on that machine).
- Classical `home_arm_q` (wrist pitch 1.0, arms out) collides hand-vs-body in sim: wrist pitch stalls at about 0.6 and wrist yaw is pushed about 0.7 rad off, so every grasp from it misses. The task `ARM_HOME` replaces it (sim profile).
- Camera noise applies to the :55560 head RGB-D only, not the wrist or teleimager streams. Upstream `rt/sim_state` is disabled (4 KB slot overflows with 18 objects), use the scene service.
- Open items the organisers list: wrist camera poses (M7), head pose re-check against the real overlay (M7), real arm home pose, **real Dex3 friction and PD fit (N1)**. That last one means sim finger physics is unfitted to the real hand, so a sim-tuned grasp may not transfer.
