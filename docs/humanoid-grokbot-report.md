# Grokbot "Research" report: Aaltonomous Humanoid approach research

Condensed from the bot's final reply, 2026-11-13 late evening. It is bot output: **every claim is UNVERIFIED unless marked "checked"**. "Checked" means I confirmed it in the starter repo myself. No plain xAI API was used (Oliver's call), only the bot channel.

## Bottom line (bot)
Go hybrid: parse the instruction once into a colour-to-bin mapping, classical perception from head RGB-D, scripted IK grasping with a small learned piece only where the script fails (triangles). End-to-end VLA on about 20 demos has almost no evidence of working.

## Prior work, ranked by relevance
1. Baseline on this exact sim: 31/38 (82%) in-process, 29/38 (76%) as Docker image, 0 clips, 30 Hz, default scene without randomisation; 6 of 7 misses were triangles. Real robot: "one pick into a bin" per FAQ (conflicts with the baseline README, which says never run). Checked (README).
2. Oracle with true poses tops out near 84%, thumb vs one finger spins the prism. Checked (`sim/README.md` line 77).
3. Open-TeleVision (H1): ACT on 10 demos of colour can-sorting, 92% pick / 74% place, but 10 picks per demo. arXiv 2407.01512.
4. GR00T on real G1: 81 demos, naive N1.6 fine-tune 0/50, engineered recipe 32-42% (DEED, arXiv 2607.20345). NVIDIA N1.5 with 1,000 demos 98.8%. A GitHub issue reports "very low" success at about 600 demos (Isaac-GR00T #414).
5. ACT trained on 150 scripted sim demos (G1 Dex3) got 60% vs 80% for the script (team-Z, self-published, sim only). Policies learned from a script lose to the script.
6. SmolVLA paper, 50-demo colour sort on SO-100: SmolVLA 70, pi0 45, ACT 25 (arXiv 2506.01844). Not a humanoid.
7. No public G1 colour+shape sorting dataset exists. Unitree Dex3 datasets (210-418 episodes) use different camera keys.

## Few-shot recipes and latency
- 20 demos is below every vendor minimum: GR00T wants 100+ (about 30 for very narrow tasks), SmolVLA says 25 was not enough and recommends about 50.
- GR00T N1.6 fits best: organisers ship a GR00T copy (RGB only, relative arm actions, absolute hand actions); `--embodiment-tag NEW_EMBODIMENT`; default recipe (`--max-steps 2000 --global-batch-size 32`) under about 35 GB; tuning LLM or vision layers needs 80 GB; inference 37 ms on RTX 5090 with torch.compile.
- Trap: every GR00T checkpoint loads the gated `nvidia/Cosmos-Reason2-2B` on first use, so bake all weights into the offline image (no network in the container). One user's policy broke after a 1 cm camera shift, and LoRA did worse than full fine-tuning (issue #130).
- pi0/pi0.5: inference over 8 GB, LoRA over 22.5 GB, full fine-tune over 70 GB; pi0 73 ms on RTX 4090; one report says LoRA was 23-40 points below full fine-tuning; a fine-tuned pi0.5 ignored text variations even at about 250 demos (openpi #768), a direct threat to rung 2.
- SmolVLA in LeRobot: about 370 ms per chunk on a 4090 (lerobot #2174).
- 27 Hz is reachable for all of them with chunking; bot estimates chunk length k of about 9 (GR00T), 18 (pi0), 45-99 (SmolVLA) for a synchronous server (its own reading of the runner), or compute in a background thread. Real constraint is 0 clipped steps: per tick the command may change by at most about 0.033 rad (arms) and 0.1 rad (hands) (that is 1 rad/s and 3 rad/s divided by 30, checked arithmetic). Blend chunk seams and add your own rate limiter.

## Sim-to-real
- Sim+real co-training (GR-1, 20 real demos per task): 45.3% to 81.1% average; 99% sim mix best, 1:1 worse; 1,000 sim demos beat 100; a misaligned sim camera cost up to 25 points (arXiv 2503.24361).
- Cheapest synthetic demos: run the classical baseline in randomised scenes and keep successful rollouts.
- Isaac Lab Mimic needs end-effector-space actions and subtask annotation, and has no G1 Dex3 environment. DexMimicGen's 90% vs 0% needed generated trajectories replayed on the real robot.
- **Checked** (`aaltonomous/sim/dr/train.yaml`): DR ranges are "(default) guesses until the venue capture". Depth noise 1 mm + 0.2% of range, 1% holes, head pose jitter +-1 cm / +-1 deg, key light jitter +-8% per channel, hue jitter 0.02 turns (7 deg), layout xy jitter 5 mm, yaw jitter 5 deg, RGB noise std 3.
- Bot's own calculation: D435 spec is up to 2% RMS error, about 10 mm at 0.5 m, so real depth is likely noisier than the sim default. Fit the noise to Drop 1 instead of guessing. Isaac Lab has only generic noise, holes and edge noise are DIY.
- Bot inference, untested: GR00T's default hue jitter (0.08) may blur red/yellow/blue.
- BEHAVIOR 2025 winner: replaced language with learned task embeddings and added a rule-based fix (fully closed gripper = missed grasp, reopen and retry).

## This challenge
- Nothing public outside the repo; the only PR is #1 "Add Verda VNC deployment".
- Rung 2 is 8 colour-only templates today; shape words, negation, distractors are listed as difficulty knobs. Checked.
- Sim runs at 0.5-0.7x real time with physics on CPU, so training on the same cores can break the 27 Hz gate; A100/H100 cannot render.
- Baseline bugs already fixed by organisers: saturation floor too high, centred grasps wedged, cubes hooked on release, home pose collided with the body.

## Hybrid design and failure modes (bot)
1. Instruction: deterministic parser, small local LLM in a background thread as fallback for negation and shape words.
2. Perception: baseline HSV + depth, plus shape and rotation estimates.
3. Grasping: rotation-aware scripted pinch, both arms in parallel (baseline takes 10-20 s per pick), checked offline against `SafetyFilter`.
4. Optional learned grasp correction for triangles only.
Failure modes: prisms spinning (grasp two flat faces, or push them); weak real thumb (one teleop user reports a minimum thumb-index gap of about 5 cm, xr_teleoperate #229); depth holes on the dark mat (fall back to a plane at known object height); camera extrinsic error alone capped one ACT policy at 54% (self-published); one bad tick counts as a clip so plan at 0.95 rad/s or less; hands blocking the head camera; bins beyond comfortable reach.
Note on "push them into the bin": the 6 cm bin wall (unmeasured default) vs 4.5 cm objects makes pushing into a bin implausible, see the scene.yaml notes in `humanoid-research-log.md`.

## Bot's 36-hour plan
1. Tonight: confirm scoring on Discord, get the classical baseline passing the sim gate in Docker, a rate-limiter wrapper as the only output path.
2. Rung 2: parser to colour-to-bin mapping, tested on paraphrases with negation and shape words, plus a quantised local LLM fallback baked into the image.
3. Grasping: rotation-aware thumb-against-one-finger pinch, both arms in parallel, fully-closed-hand check that triggers retry.
4. After Drop 1: refit head extrinsics and depth noise, rerun with hidden-variant-style randomisation, tune on the real demo grasps.
5. Stretch: small learned triangle-grasp correction trained on scripted sim rollouts. Skip end-to-end VLA unless the core is finished.

## Unresolved (bot)
Are the 40/25/20/15 weights real (they are on the official Junction page, which is behind login; the repo says real-robot score ranks finalists with sim as tie-break)? Will finals add shape/negation/distractors, how many objects, how much time? When do wrist calibration and Drop 1 land? How good is the real Dex3 pinch on 45 mm prisms? How does the baseline do on the real robot? Is any ~20-demo result for GR00T / pi0.5 / SmolVLA on G1 Dex3 published (none found)? Official RTX 5090 latency for pi0.5 or GR00T N1.7?
