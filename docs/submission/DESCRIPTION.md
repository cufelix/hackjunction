# Submission description

Draft for the submission form. Every number has an HTML comment with its source. Check the comments before you paste.
Status line to update on the day: as of the readiness doc, we had zero runs on the real robot.

## Tagline

A sorting policy for the Unitree G1 with five stages you can read, test and switch off one at a time.

## Short description (about 150 words)

Our policy sorts coloured objects from a counter into bins on the Unitree G1 with Dex3 hands. It reads a language instruction, finds objects with colour and depth, picks a grasp by shape, plans arm motion with collision checks, and runs it at a capped speed. There is no learned controller in the loop, so each stage can be inspected and tested alone. It ships as one Docker image that serves a websocket policy on port 8000. It returns joint targets only, and the organisers' runner keeps the safety filter. In the official-style sim, the release candidate placed 33 of 38 objects on the fixed task and 30 of 36 on the language task, with no wrong bins <!-- READINESS.md section 2, rc1 @ 7a4168e, box A -->. When the policy cannot parse an instruction or see an object, it leaves the object on the counter.

## Long description (about 500 words)

The task is to sort coloured objects from a counter into bins on a Unitree G1 humanoid with Dex3 hands. We started in simulation and aimed at the real robot from the first week. Instructions come as text, for example a colour and sometimes a shape.

We built a classical policy and kept every stage visible. The instruction resolver turns text into a colour and shape to bin map. Perception finds objects by colour and reads depth to locate each object's top face in 3D. The grasp is chosen by shape. Arm motion comes from inverse kinematics with collision checks against the bins, and trajectories are time-scaled to speed and acceleration limits. We chose this design because we could test each stage on its own and read the logs when something failed. <!-- READINESS.md section 1 -->

The policy runs in one amd64 Docker image. It serves a websocket and msgpack policy on port 8000. Each tick it receives head RGB-D, joint states and the instruction text. It returns arm and hand joint targets and nothing else. It never opens a camera or the robot network, and every target goes through the organisers' safety filter. <!-- READINESS.md section 1 -->

Most of our work went into the gap between the simulator and the real setup. The organisers told us the Dex3 gains are far softer than in the sim, the objects have different colours, and the bins sit farther away than nominal. We measured these on real frames. We also built an adversarial copy of the sim that changes the world but not what the policy believes. It shifts bins by 5 to 14 cm, moves the camera pose, changes lighting, punches holes in depth and softens the hands. <!-- READINESS.md sections 3 and 6 --> Seeds with seed modulo 4 equal to 3 are never used for tuning. On the hardest profile, our early baseline fell from +3.8 to about +0.2 points per episode. We traced 42 percent of the losses to camera pitch and 42 percent to bin rims read as objects. That result decided which calibration work we did. <!-- READINESS.md section 6 -->

Every behaviour change sits behind a flag that is off by default. A release gate checks that the baked flags match the chosen candidate exactly. With new flags off, the policy reproduces the previous release in 13 of 13 digests. <!-- READINESS.md section 2, tools/equivalence_check.py --> Fault drills, including all-zero depth, a frozen camera, a missing bin, a bad instruction and a NaN joint state, passed 12 of 12. <!-- READINESS.md section 2, felix/fault-drill-rc1 --> On failure the policy holds or abstains. An unparseable instruction leaves the objects on the counter, which scores zero and never loses a point for a wrong bin. <!-- READINESS.md section 5 -->

An optional local language model sidecar (Qwen3-VL-8B on llama.cpp) can run next to the policy. It is off in every release overlay, and the policy keeps working if the sidecar is missing or stops. <!-- READINESS.md sections 1 and 2 -->

We have not yet run on the real robot as of the readiness document. Every score above comes from simulation. We list the open risks, such as soft-finger grip and pad friction, in the readiness document.

## How it works

1. **Instruction resolver.** Text becomes a colour and shape to bin map. Text it cannot parse abstains. <!-- READINESS.md sections 1 and 3 -->
2. **Perception.** HSV colour masks plus head depth give a 3D top-face position per object. Per-episode calibration corrects camera pitch, counter height and bin positions behind flags. <!-- READINESS.md sections 1 and 4; ARCHITECTURE.md section 1 -->
3. **Grasp choice.** The grasp posture depends on the object's shape. <!-- READINESS.md section 1 -->
4. **Planning.** Inverse kinematics on the G1 model with collision checks against the bins. <!-- READINESS.md section 1; ARCHITECTURE.md section 1, kinematics.py -->
5. **Motion.** Trajectories are time-scaled to speed and acceleration limits. A speed cap can only lower speed. The runner's safety filter bounds every per-tick change. <!-- READINESS.md section 5; ARCHITECTURE.md section 1 -->

## What we verified

All results are from simulation unless a row says otherwise.

| Check | Result | Source |
|---|---|---|
| Official-style eval, fixed task | 33/38 objects correct, 0 wrong bins | <!-- READINESS.md section 2 --> `aaltonomous/sim/evaluate.py`, rc1 @ 7a4168e, organiser seeds |
| Official-style eval, language task | 30/36 correct, 0 wrong bins | <!-- READINESS.md section 2 --> same run setup |
| Safety gate | 0 clipped steps, command rate at least 29.99 Hz, no abort, watchdog or crash | <!-- READINESS.md section 2 --> evaluator gate `--gate-max-clips 0 --gate-min-hz 27` |
| Release gate | PASS, baked overlay equals rc1 flag for flag | <!-- READINESS.md section 2 --> `tools/release_gate.py rc1` |
| Offline cold start | 0.77 s to listening with `--network none` | <!-- READINESS.md section 2 --> `docker run --network none` |
| Image size | 1.46 GB | <!-- READINESS.md section 2 --> `hackjunction/policy:rc1-7a4168e` |
| Equivalence with flags off | 13/13 digests identical | <!-- READINESS.md section 2 --> `tools/equivalence_check.py` |
| Unit tests | 602/602 on the newest fix branch | <!-- READINESS.md section 2 --> |
| Fault drills | 12/12 pass, p99 at most 13.5 ms | <!-- READINESS.md section 2 --> `felix/fault-drill-rc1` |
| Language sidecar, optional | 2.8 s to /health, 10.7 GB VRAM, decision latency p50 114 ms, p95 117 ms | <!-- READINESS.md section 2 --> Blackwell GPU, same class as the venue RTX 5090 |
| Adversarial sim, hardest profile | Early baseline fell from +3.8 to about +0.2 per episode | <!-- READINESS.md section 6 --> `real_hard` profile, `tools/adv/` on `oliver/adv-sim` |
| Principal point artefact | Stock sim under-scores rc1 by 1.20 points per episode (se 0.31), 7 wins, 0 losses, 3 ties over 10 paired seeds. With the real camera principal point rc1 placed 38/40 with 0 wrong bins, against 27/40 and 1 wrong bin | <!-- results/1800-oliver-adv-pp/PP.md, rc1 @ 7a4168e9, seeds 1000-1009 --> |
| Real-frame colour detection audit | On 538 visible objects in 40 real `drop_2` episodes (no depth), recall was 0.597 for the baked submission overlay with 1569 unmatched detections, and 0.362 for `finals_r21` with 123 | <!-- results/1625-oliver-recall-ab/RECALL-AB.md, submission @ 9e940de and finals_r21 @ 4540b9b, tools/recall_ab.py --> |

## Built with

- Python, NumPy, OpenCV (perception, HSV masks, depth) <!-- ARCHITECTURE.md section 1; run.py and runner.py use cv2 -->
- Pinocchio on the G1 URDF (forward and inverse kinematics) <!-- ARCHITECTURE.md section 1 -->
- Websocket and msgpack policy server on port 8000 <!-- READINESS.md section 1 -->
- Docker (amd64 image, Wendy packaging) <!-- READINESS.md sections 1 and 7 -->
- Isaac Lab simulator and the organisers' evaluator <!-- README.md, aaltonomous/sim -->
- llama.cpp with Qwen3-VL-8B for the optional sidecar <!-- READINESS.md section 1 -->
- Unitree G1 with Dex3 hands <!-- task description -->

## What's next

- Run on the real robot. This is the biggest open item. <!-- READINESS.md section 8 -->
- Measure soft-finger grip and pad friction, then decide on the cage grasp and staggered release flags, which are off until they win in the soft-hand sim. <!-- READINESS.md section 8 -->
- Raise real-frame detection recall without bringing back bin and hand false detections. The audit above shows the trade-off. <!-- RECALL-AB.md -->
- Verify the Wendy offline transfer on venue hardware. <!-- READINESS.md section 8 -->
