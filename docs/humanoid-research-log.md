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
