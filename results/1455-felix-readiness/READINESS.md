# Aaltonomous Humanoid: readiness summary for a real-robot deployment

**What it is.** A classical, fully inspectable policy server for the Unitree G1 with Dex3 hands. It takes head-camera
RGB-D and a typed instruction, and outputs joint targets only. The pipeline is perception (colour + depth, 3D top-face
localisation), then an instruction resolver, then a planner (IK, collision checks against the bins), then the safety
filter. It ships as one amd64 Docker image. It runs with `docker run` and no env or args: a baked overlay file is the
only switch.

## Why we think it is safe to run on your robot
- **Joint targets only, rate-limited.** Every plan is pre-checked against the per-joint velocity limits of the safety
  filter. A new `motion.arm_vel_cap` can only *lower* arm speed (min of profile and cap, acceleration scaled with it).
  An A/B sweep picks the fastest cap that costs ≤ 1 object per 20 seeds.
- **The 30 Hz loop never blocks.** Heavy work (calibration, planning, optional model calls) runs off the control
  thread. With `policy.safety_shell`, `act()` never raises and holds the last target on an exception or a deadline
  miss.
- **Abstain beats guessing.** Scoring is +2 / +1 / −1 / 0, so every uncertain case leaves the object on the counter
  (0) rather than risk a wrong bin (−1). This covers unknown instruction words, colour-named bins, shape without depth,
  and an insert/lid task.
- **Hard caps.** A global cap of 18 attempts per episode, a per-object time cap, and a release-over-bin or retract
  rule near the time limit.

## How we know what it does: the engineering process
- **Every change ships behind a flag that defaults OFF.** With all flags off, an equivalence check replays 13 sim
  episodes and must produce byte-identical action digests versus the submission.
- **A release gate diffs every registered flag** between the baked overlay and the chosen release candidate, and it
  refuses known-bad combinations (for example two camera corrections stacked).
- **Testing and review.** The full unit suite (~330 tests) runs 3× before any release-candidate push. Every merge
  gets an independent review. Held-out real frames are never opened.

## Evidence
| Measurement | Result |
|---|---|
| Stock sim, language task (box A) | resolver ON: 54/74 vs 1/74 baseline, **0 wrong bins** |
| Stock sim, Modal 82-object sweep | current candidate 74/82, 0 wrong |
| Our "real-like" sim (`real_hard`: real colours, bin offsets dx 7±3 cm, measured −7.5° camera pitch, lighting and depth noise) | baseline collapses from +3.8 to ≈ +0.2 per episode; losses are 42 % camera pitch, 42 % bin-rim phantoms |

We built that adversarial sim *because* it reproduces the real failure modes we saw in real head-camera frames. Every
fix is A/B-tested against it before it is switched on:
- online camera-pitch calibration from the counter plane;
- a colour gate with a 3D bin footprint, distance-scaled area floor and FK arm mask;
- hand-eye calibration from the robot's own visible forearms;
- a per-bin position vote.

Material: `results/1450-oliver-adv-dm/` (side-by-side head-camera images, loss breakdown).

## What we would bring to the real robot
An image built from a release candidate whose flags are all justified by A/B numbers in the real-like sim. Its default
behaviour on any failure is to stop and hold, or leave the object on the counter. We need no network access and no
operator input; the GPU is optional (classical path).
