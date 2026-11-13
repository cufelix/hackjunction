# Aaltonomous Humanoid: starter repo notes

Research only, 2026-11-13 evening. Source: `github.com/zupermiez/aaltonomous-humanoid` (brief, RULES_AND_SCORING, STARTER_PATHS read in full). Not yet read: INTERFACE.md, SUBMISSION.md, FAQ.md, baseline README, data card. Nothing is built.

## Task
- Unitree G1 with two Dex3 hands ("Väiski"), behind a counter, stiff lower body.
- 3-5 objects (cube/cylinder/prism, red/yellow/blue, ~45 mm), 3 identical orange bins left/center/right. 120 s per episode.
- Obs every 1/30 s: head RGB-D, two wrist RGB, joint states, text instruction. Action: absolute joint targets for both 7-DoF arms and both 7-DoF hands.
- Rungs: `sort_fixed` (red L, yellow C, blue R), `sort_language` (instruction sets mapping, paraphrases unseen), `insert_shapes` (stretch, sim only).

## Scoring
- Per object: fixed +1 correct; language +2 correct; wrong bin -1; knocked off -1; left on counter 0. Score is a sum.
- Safety abort after more than 30 clipped steps keeps points so far.
- Finals: 3 real-robot episodes: sort_fixed, sort_language known template, sort_language unseen instruction and layout (x1.5). Sim score breaks ties.
- Finalists chosen from sim score on a held-out variant (different textures, lighting, poses, camera jitter, secret seeds).

## Hard gates to reach the robot
- 0 clipped steps. Velocity cap 1 rad/s arms, 3 rad/s hands, plus joint limits, workspace box, counter keep-out.
- Command rate at least 27 Hz, also in wall-clock latency (`policy_hz_rt`). Held ticks do not count. A slow model must return action chunks (k, 14).
- No crash, no timeout (30 s per call), no lost connection.
- Nobody touches the robot. We submit a Docker image (websocket + msgpack, port 8000). No network inside the container unless we accept the risk of external APIs.
- Submissions see only `obs`. Privileged sim state is forbidden in a submission.

## Provided
- Classical baseline: HSV blobs + head depth, Pinocchio IK, Cartesian waypoints, state machine. Rung 1 only. Weak: grasp slips (prisms, cylinders), objects near centre line, 10-20 s per pick.
- About 20 real demos per rung, LeRobot v3.0 and a GR00T copy. Raw, including failures; labels in `meta/aalto_episodes.jsonl`.
- Drop 0 Friday evening (teleop demos), Drop 1 Saturday (venue lighting, depth noise). Links in Junction Discord.
- Sim: Isaac Lab, deliberately rough. Needs RTX-class GPU with 24 GB or more.

## Routes (from STARTER_PATHS)
1. Classical/modular: improve the baseline (yaw-aware grasps, Grounding DINO/SAM 2, cuRobo, both arms).
2. VLM planner over own skills: fits rung 2, but VLM calls take 10-20 s, so plan in a background thread and hold meanwhile.
3. IL/VLA: LeRobot ACT/Diffusion/SmolVLA, openpi, GR00T. Fine-tuning needs 80 GB GPUs (Verda). Inference must fit the 32 GB finals GPU.
4. RL: train one skill with privileged state, distil.
5. Real2sim: re-texture, depth noise, Dex3 friction fitting. Helps every route.

## Open questions
- Which route fits the generality (25%) and originality (20%) criteria best, given only ~20 demos per rung?
- Does rung 2 need a VLM at all, or is a deterministic parser for 8 templates plus paraphrase handling enough?
- GPU: who has one, or do we rent from Verda?
