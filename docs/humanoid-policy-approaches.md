# Which policy architecture can actually hold 27 Hz

**Researched 13 Nov 2026 21:40 CET. Research only, nothing built.** My assigned lane for the
Aaltonomous Humanoid challenge: classical vision + IK vs ACT / diffusion / VLA, and what survives
the hard gate.

**The gate, from the challenge page:** a solution reaching the real robot must produce **at least
27 commands per second**, with no clipped steps and arm velocity under the cap. The policy sees
head camera (colour + depth), both wrist cameras, joint states and the instruction, and returns
joint targets **30 times a second**.

---

## 🔑 The gate is not about model speed. It is about action chunking.

This is the finding that decides the approach. You do **not** run inference at 30 Hz. You infer a
**chunk** of future actions once, then replay it at 30 Hz. Inference latency and control rate are
decoupled, and the only requirement is that the chunk lasts longer than the next inference takes.

That is why Diffusion Policy measures **0.090 s latency but 267 Hz throughput** — the throughput
is the chunk being replayed. Any architecture that emits a chunk can clear 27 Hz; any architecture
that emits one action per forward pass is capped by its own latency.

## Measured rates, and who fails the gate

Figures are from different GPUs, precisions and chunk sizes, so they do not compare cleanly, and
most come from secondary aggregators rather than primary papers. Treat as a shortlist, not a
benchmark.

| Approach | Reported | Verdict against 27 Hz |
|---|---|---|
| **ACT** | **~20 ms per action step**, listed as the fastest of the trainable policies compared | ✅ **passes comfortably (~50 Hz)** |
| **Diffusion Policy** | 0.090 s latency, **267 Hz throughput** (A100, ALOHA table) | ✅ passes via chunking |
| **OpenVLA-OFT** | 71.4 Hz throughput, 0.112 s latency (A100, LIBERO) — parallel decoding + chunking | ✅ passes, but it is the OFT variant |
| **π0** | ~73 ms inference, control rates reported up to 50 Hz; 31.6 Hz on RTX 5090 optimized | ⚠️ marginal; depends on optimization we would not have time for |
| **GR00T N1.7** | **~140 ms per call on RTX A6000** (VLM ~60 ms + 4 DiT denoising steps ~80 ms) → ~7 Hz. 27.9 ms (~36 Hz) only with TensorRT on H100/RTX PRO 6000 | ❌ **fails on RTX-class in plain PyTorch** |
| **OpenVLA (base)** | **~6 Hz on RTX 4090**; ~4.2 Hz on A100 | ❌ fails badly |
| **π0.5** | 485 ms per action step in one catalogue | ❌ fails |

⚠️ **And a warning that matters more than any of these numbers:** inference rate is not control
rate. A 40 Hz inference path does not give a 40 Hz robot loop if the cameras, the network hop or
command execution are slower. The gate is measured on the **whole** path, so the budget has to
include three camera streams and the websocket round trip, not just the model.

## Recommendation: the language model does not belong in the control loop at all

The instruction is **one sentence per episode** — *"Sort the objects: red into right, blue into
left, yellow into center."* Parsing it is a one-shot text task. There is no reason to pay for it
thirty times a second.

So the split that both clears the gate and matches what a three-person team without robotics
experience can build:

1. **Once per episode:** parse the instruction into a colour→box mapping. Any model, any latency,
   it happens before the robot moves.
2. **Once per object:** classify colour and shape from the head camera, and produce a
   **confidence**. This is where the abstention decision lives.
3. **At 30 Hz:** classical vision for the object pose plus inverse kinematics for the reach and
   grasp. Deterministic, debuggable, trivially inside the velocity cap, and it cannot blow the
   command-rate gate because there is no network model call in the loop.

Then, if time allows, replace step 3 with **ACT** trained on the real-robot datasets Aalto
publishes during the event — it is the one learned option that clears the gate with margin, it is
supported in LeRobot, and it was designed for exactly this shape of problem (bimanual
manipulation from camera + joint states, which is a two-armed G1).

**What I would not attempt this weekend:** GR00T or OpenVLA base as the control policy. Both fail
the command-rate gate on the hardware we are told to use, and the fix is TensorRT work, not
modelling work.

## Why this also wins points rather than just passing

Oliver's reading of the starter repo: **fixed +1/−1, language +2/−1, knocked off −1.** So the
expected value of placing an object is `2p − (1−p)` on a language episode, which crosses zero at
**p = ⅓**, and `p − (1−p)` on a fixed episode, crossing at **p = ½**.

**Below those confidences, leaving the object on the counter scores better than placing it.** The
abstention has to be per *object*, not per episode — and the confidence that decides it comes from
step 2 above, which is outside the control loop and therefore cheap to calibrate properly.

That makes the calibration work (Platt or isotonic on the colour and instruction heads, conformal
prediction for the unseen-instruction episode weighted ×1.5) a **scoring component, not
methodology**. And it is a genuinely fresh answer to the 20 % originality criterion: everyone else
will optimise grasp success; a robot that deliberately declines an object it is not sure about,
and can say why, is a different thing.

## To verify before relying on this

- [ ] **Is an unplaced object really 0, or is there an incomplete/timeout penalty?** The whole
      abstention strategy collapses if leaving an object also scores −1. Asked Oliver.
- [ ] Can a misplaced object be moved again, or is the score final on contact with a box?
- [ ] Measure the real end-to-end budget in the provided simulator: three cameras plus websocket,
      before choosing a chunk length.
- [ ] Confirm the ACT 20 ms figure against the primary ALOHA/ACT paper rather than the aggregator.

**Sources** (secondary unless noted; verify before quoting): voxel51.com/blog/vla-models-compared ·
arxiv.org/pdf/2502.19645 (fine-tuning VLAs, ALOHA efficiency table) ·
arxiv.org/pdf/2509.23224 (real-time correction for action chunks) ·
ay-robots.com/blog/en/latency-budget-for-30-hz-robot-control · vla-gallery.is-a.dev
