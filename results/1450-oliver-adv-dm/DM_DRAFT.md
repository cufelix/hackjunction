Hi erkka! Team hackjunction (Felix, Tim, Oliver) would like to apply for the real-robot deployment via the Wendy agent.

**What we have**
- A classical, inspectable policy: head RGB-D perception, a language resolver, grasp by shape, and a safety shell.
  It sends joint targets only, and every joint is clamped to its limits. When unsure it abstains instead of guessing.
- An amd64 image (CUDA 12.8) that starts with no env/args and listens on :8000 well within 120 s.
- Process: every feature is behind a flag, and flags stay off until a paired A/B passes. With all flags off the
  output is byte-identical to the baseline on 13 episodes. The full test suite runs 3x before a release, and we
  run fault drills (offline cold start, ws drop/reconnect, depth blackout).
- Evidence: sim language tasks 54/74 vs 1/74 for stock, 0 wrong; seeds 1040-1059: 74/82 objects, 0 wrong.

**We took your advice and made our sim match your setup**
- Purple/yellow/orange objects with colours measured from your drop_2 frames, real bin offsets, and the -7.5°
  head-camera pitch we measured in drop_2.
- Lighting/exposure noise and depth holes.
- Result: our baseline drops from +3.8 to about +0.2 per episode. Losses are 42 % camera pitch and 42 % orange bin
  rims read as objects. Every fix (online pitch calibration, colour gate with a 3D bin footprint, hand-eye
  calibration) is A/B-tested against that before it goes on. Images and the table are attached.

**Questions**
1. Which machine runs the Wendy agent: the RTX 5090 x86 PC, or the G1's onboard arm64 computer?
2. How does the image get there without internet? Should we send a tarball / registry in advance? Size ~X GB.
3. Which Wendy agent version and port should we target?
4. Are robot DDS and the cameras reachable with host networking from the container?
5. Could you share a ~10 s RGB-D recording of the finals setup (head cam), so we can check our calibration?
6. Is the GPU available to us (we may run a ~9 GB Qwen3-VL-8B for instruction parsing)?
7. Does the container persist between episodes, or is it restarted for each one?

Thanks! Happy to send anything else you need.
