# Handoff — Oliver's Claude (lane 3: perception + robustness), Sat 14 Nov ~11:30 Finnish

Box addresses and logins are NOT in this file (repo goes public at submission). Ask Oliver or read Warren.

## Team decisions (from Warren, so every agent starts from the same facts)
- Priority: TOP 5 in the sim qualifier first. Real-robot-only work waits until we qualify.
- Sim candidate = `cand_r16_nopp` (submission + retry_vary_slots, no pp_center). Modal seeds 1040-1069:
  submission 105, cand_r16 111, cand_r16_nopp 111, 0 wrong. pp_center adds nothing on top of retry,
  so one build serves sim and real robot.
- Shared architecture (Timi 289f69bf): resolver (Felix, 21af8498, frozen) → optional brain
  (Qwen3-VL-8B via llama.cpp on the finals RTX 5090, `brain.enabled`, OFF in sim) → HSV+depth
  perception → planner (posture by shape, retry_vary_slots) → control/safety.
- Language: Felix's `policy/normalize.py` (sha ea24b7a7) runs BEFORE the resolver:
  `resolve(normalize(s))`. Typos 9/17 → 15/17, 0/336 dev sentences changed. Timi runs the sealed sets.
- Repo `cufelix/hackjunction` is PRIVATE now but flips PUBLIC before submission: never commit keys,
  tokens, box IPs, root logins, join links. Push often.
- Real Dex3 data (Timi, knowledge/humanoid/33): held detector needs `finger_sum` rule; real bins
  ~7 cm farther than nominal (release-pose FK), degenerate with the −7.5° head-cam pitch fit.

## Delivered by lane 3 (all behind flags, default OFF)
- `oliver/lane3-stack` 41c255d = principal-point + reject_orange + counter-plane + bin-detect + hand-calib.
- `oliver/fusion-track` d61560b = multi-frame depth fusion + object tracking (real-robot; parked).
  Offline: xy 3.79 → 3.47 mm, phantoms 12 → 0, jitter 0.19 → 0.04 mm, +0.2 s per look.
- `oliver/pitch-fit` (OpenCode session): head-cam pitch −7.5 ± 1.8°, varies per episode.
- Sealed blind on resolver 21af8498: PASS (v1 143/151, m2o 30/30, v2 50/60, ORDERING 6/8).

## Running / next
1. Timi's assignment (99de02b3): cand_r16_nopp vs submission, seeds 2000-2059, sort_fixed +
   sort_language, boxes F and G (H is Timi's). Paired blocks of 5 seeds. Report per-flag delta +
   loss breakdown (grasp miss / perception / wrong bin).
2. Old lane-3 A/B on box F (lane3_pp_only, lane3_ro_only) finishing; pp result is moot now.
3. Re-blind Felix's fixes when a hash arrives (sealed sets live outside the repo, never commit them).

## Lanes
1 language = Felix · 2 motion + integration = Timi (only integrator) · 3 perception + robustness = Oliver.
OpenCode session of Oliver: drop_2 hue/demo analysis, cross-episode memory module.
