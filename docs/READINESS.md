# Readiness: running our policy on the real G1

For the organisers and judges. It covers what we have verified and how, what we do about the real robot, and what can
still go wrong. Every number names the commit, image or tool it came from. If this page and the code disagree, the code
wins. The deeper technical reference is `docs/ARCHITECTURE.md`.

## 1. What runs

One amd64 Docker image, built from `tools/submission.Dockerfile`. It serves a websocket + msgpack policy on port 8000.
Each tick it receives head RGB-D, joint states and the instruction text, and it returns arm and hand **joint targets
only**. It never opens a camera, DDS or the robot network. The organisers' runner owns the robot and its `SafetyFilter`,
and every target we send goes through that filter.

The policy is classical and every stage can be inspected:

1. Instruction resolver (text → colour/shape → bin).
2. Colour + depth perception with 3D top-face localisation.
3. Grasp choice by shape.
4. IK planning with bin collision checks.
5. Time-scaled motion.

There is no learned controller in the loop. An optional language sidecar (Qwen3-VL-8B on llama.cpp) can run next to it
on the GPU. It is off in every release overlay, and the policy keeps running if the sidecar is missing or stops.

## 2. What is verified, and how

### Release candidate rc1: `felix/rc1` @ `7a4168e`, image `hackjunction/policy:rc1-7a4168e` (1.46 GB)

All of the following were run on box A, a clean sim box used only for official-style runs, with the exact image above:

| Check | Result | How |
|---|---|---|
| Release gate | PASS: the baked overlay equals `rc1` flag for flag | `tools/release_gate.py rc1` |
| Offline cold start | 0.77 s to listening, with `--network none` | `docker run --network none` |
| sort_fixed (official-style eval) | 33/38 objects correct, 0 wrong bins | `aaltonomous/sim/evaluate.py`, organiser seeds |
| sort_language | 30/36 correct, 0 wrong bins | same |
| Safety gate | 0 clipped steps, command rate ≥ 29.99 Hz, no abort / watchdog / crash | the evaluator's sim gate (`--gate-max-clips 0 --gate-min-hz 27`) |

### Process checks behind every release

- **Flags default to off.** Every behaviour change sits behind a flag in `config.FLAGS`. Only an overlay file turns flags
  on, and the image refuses unregistered keys.
- **Equivalence.** With all new flags off, `tools/equivalence_check.py` reproduces the previous release byte for byte:
  13/13 digests (a 20-render perception digest plus 12 seeded mock episodes).
- **Release gate.** The baked overlay must equal the chosen candidate exactly, and known-bad flag combinations are
  refused. `tools/promote.sh` is the only path that writes the release overlay.
- **Tests.** The full unit suite runs before every candidate push (602/602 on the newest fix branch). An independent
  reviewer passes every merge.
- **Fault drills** (`felix/fault-drill-rc1`): depth all-zero, frozen camera, missing bin, bad instruction, NaN joint
  state, 50 ms stall. 12/12 pass with p99 ≤ 13.5 ms. We also ran a full episode, an offline cold start with every network
  call patched to fail, and a websocket drop and reconnect against the real server. Bad observations sent straight to
  the policy (zero depth, NaN arm_q, missing depth key, no instruction) for 40 ticks each: it never raised, and every
  action was finite.
- **Language sidecar (optional), on Blackwell (same class as the venue RTX 5090):** cold start to /health 2.8 s,
  10.7 GB VRAM, decision latency p50 114 ms / p95 117 ms, works offline. Stopping it leaves the policy healthy.

## 3. Real-robot facts we handle

| Real-robot fact | Where it comes from | What we do |
|---|---|---|
| Dex3 gains are kp 1.5 / kd 0.2, far softer than the sim | organisers | Grasp postures squeeze past the object width. Soft-hand sim profile `real_soft` (§6) to validate the grasp before turning on anything that depends on grip |
| Objects are purple / yellow / orange, not the sim's red / blue / yellow | organisers; measured on the real `drop_2` frames | "Robust" HSV set tuned on real frames. The colour names in instructions are aliased (orange → red class, purple → blue class). The sim is re-coloured to the measured top-face colours for testing |
| Bins sit 5–14 cm farther than nominal; head pitch differs from the URDF by several degrees | measured on `drop_2` | The real-like sim (§6) reproduces both. Online calibration (§4) is A/B-tested against it |
| Finals instructions: colour first, maybe shape, no negation | organisers | The resolver maps (colour, shape) → bin. Anything it cannot parse abstains (object stays on the counter: 0 points, never −1) |
| Compute: x86 PC with an RTX 5090, wired to the robot | organisers | amd64 image. GPU optional (the classical path needs none). Large images announced in advance for pre-download |
| No internet at the venue | our assumption | Offline cold start is a release check (§2). No runtime downloads |

## 4. How calibration works

The policy trusts `calibration.yaml` (factory D435i intrinsics, URDF head extrinsics) and corrects it per episode from
what it sees. Each correction is a flag, off in rc1, and only enabled after it wins a paired A/B in the real-like sim:

- **Camera pitch** from the counter plane in head depth (`perception.online_pitch_cal: table`) or from the orange bin
  rims (`true`).
- **Counter height** from a plane fit over the first looks, clamped to ±`max_dz_m`, falling back to nominal on bad depth
  (`calib.plane.apply_height`).
- **Bin positions** from three votes: depth rim, tape and prior (`bin.vote.enabled`). Every bin consumer then follows
  the voted geometry (`bin.geometry_runtime`).
- **Head pose refinement** from the robot's own forearms and hands in depth against forward kinematics
  (`perception.hand_calibration.enabled`; a dense-silhouette variant is in review).

Calibration runs in a low-priority thread, so the 30 Hz loop never waits for it. If a correction fails its sanity
checks, the nominal value stays.

## 5. Fallbacks and safety behaviour

- **Fallback release:** rc1 is the shipping candidate. The next candidate (rc7, overlay `finals_r22`) replaces it only
  if it scores at least as well as rc1 on the same box-A eval (33/38, 30/36).
- **Flags as kill switches:** any single behaviour can be turned off by rebuilding with an overlay that sets its flag to
  false. The release gate then proves that this is the only difference.
- **On failure the policy holds or abstains, it does not improvise:**
  - unknown or unparseable instruction → abstain;
  - shape without depth → abstain;
  - insert task → pick nothing;
  - hard cap of 18 started picks per episode;
  - a NaN in the joint state or an exception in `act()` → hold the last target (`safety.nan_guard`,
    `policy.safety_shell`, on in the finals overlays). In rc1 the runner's `SafetyFilter` does this job: stale command
    after 0.2 s, latched safe stop after 1.0 s.
- **Speed:** `motion.arm_vel_cap` can only lower arm speed below the profile. The filter bounds every per-tick change.

## 6. Testing against the real setup before we touch it

The stock simulator flatters any policy. We keep a separate "adversarial" copy (`tools/adv/`, branch `oliver/adv-sim`,
extended in `oliver/adv-real2`) that changes the world but not what the policy believes:

- real colours and bin tape;
- bins shifted 5–14 cm;
- camera pitch, yaw and position error;
- lighting / exposure / white balance;
- depth holes and noise;
- soft Dex3 gains;
- counter height and lateral table offsets;
- depth dropout on the black hand links;
- a start pose that is not our home pose;
- the real camera principal point.

Seeds with `seed % 4 == 3` are held out and never used for tuning. Reports give the paired score delta, why each lost
object was lost (not detected / wrong position / phantom / wrong colour / grasp failed / dropped / no plan / timeout),
and the localisation error as median / p95 / max. On `real_hard`, the pre-calibration baseline fell from +3.8 to about
+0.2 per episode: 42 % of losses came from camera pitch and 42 % from bin rims read as objects. That result is why the
calibration and colour-gate work exists.

## 7. How to deploy

```bash
docker build -f tools/submission.Dockerfile -t hackjunction/policy:<tag> .   # repo root as context
docker run -d --gpus all -p 8000:8000 hackjunction/policy:<tag>              # the organisers' form: no args, no env
docker run --rm --entrypoint python3 hackjunction/policy:<tag> tools/flags.py  # shows the baked flags
python -m aaltonomous.deploy.replay --synthetic 100 --policy ws://<host>:8000  # dry run: shapes + command rate
```

Wendy package: branch `timi/wendy-pkg`, `tools/wendy/`. It contains `wendy.json` (5090 PC, amd64, readiness probe on
TCP 8000), an optional `wendy.gpu.json` for the sidecar, an arm64 variant and `smoke.sh` for a local cold-start test.
The full command sequence and an on-the-day checklist are in `tools/wendy/README.md`. If the Wendy agent cannot take
the image, plain `docker run` on the PC is the fallback.

What we need from the venue: the image pulled (or loaded from a tarball) in advance, port 8000 reachable from the
runner, and nothing else. No env vars, no network, no operator input.

## 8. Known risks

| Risk | Status | Mitigation |
|---|---|---|
| **Zero runs on the real robot so far.** Every number above is sim | open | Real-robot slot requested. All defaults hold or abstain |
| Soft real fingers may not hold an object that the stiff sim fingers hold | open, being measured | `real_soft` sim profile. Cage grasp and staggered release are behind flags, off until they win on it |
| Camera pose on the day differs from the URDF | partly covered | Online pitch / plane / bin calibration (finals overlays) |
| Pad friction is unmeasured (rubber) | open | Sim friction is grippy. We assume less grip and close tighter |
| rc1 with an **empty** instruction uses the sim colour → bin map, not the real-robot map | known | The finals overlays set `policy.fixed_mapping` to the real map. Finals always send an instruction, which the resolver maps |
| Wendy offline transfer (`--chunking force`) is unverified on venue hardware | open | Plain `docker run` fallback, or a `docker save` tarball handed over in advance |
| The language sidecar adds a second image (~11-13 GB) | optional | Off in all release overlays. The policy runs without it |
