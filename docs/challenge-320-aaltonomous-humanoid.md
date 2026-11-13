# Challenge 320 — Aaltonomous Humanoid (Aalto University)

**The brief verbatim, read off the platform 13 Nov 2026 21:38 CET and confirmed against Felix's
own copy.** The platform page is behind a login, so this is the repo's authoritative copy — do not
re-fetch it, read it here.

Prizes: **2 000 € + a visit to the Aalto robotics lab.** Top ~5 teams by simulation performance
run live on the real Unitree G1.

| Criterion | Weight | Their words |
|---|---|---|
| Real-world result | **40 %** | How well it works on the real robot |
| Generality | **25 %** | Holds up when things change |
| Originality | **20 %** | Freshness of idea and approach |
| Presentation | **15 %** | Clarity of what you built and why it matters |

⚠️ **The 40 % is gated.** Only the top ~5, chosen in simulation, reach the robot. Miss that cut
and 40 % of the rubric is unavailable to us.

---

## The task, exactly as specified

The G1 (*Väiski*) stands behind a counter, **legs and waist held stiff**, with **two 7-DoF arms
and two Dex3 three-fingered hands**.

- **3 to 5 printed objects** at random positions and angles: **cubes, cylinders, triangular
  prisms**, about **4.5 cm**, in **red, yellow, blue**.
- **Three identical orange boxes, 21 cm wide**, at fixed positions — left, centre, right — in a
  row, **5 cm apart**, **40 cm from the torso**. **No markings; told apart only by position.**
- Everything into the right box **within 2 minutes**.

Three modes:
1. **Fixed sorting** — red→left, yellow→centre, blue→right.
2. **Sort by instruction** — e.g. *"Sort the objects: red into right, blue into left, yellow into
   center."* **"Expect phrasings you haven't seen before."**
3. **Stretch** — each object through the **matching hole in a lid**, i.e. shape matching, not just
   colour.

**The interface:** the policy sees the head camera (**colour + depth**), **both wrist cameras**,
joint states and the instruction, and returns **target positions for all arm and finger joints,
30 times a second**.

## What they hand us

A ready Docker image with **Isaac Sim + Isaac Lab + their exact scene**; GPU from **Verda**;
**datasets recorded on the real robot, published on Hugging Face during the event** (links in the
Junction Discord); **starter code** — robot interface, safety filter, **a classical baseline**,
policy-server and Dockerfile templates, **and an evaluator that scores us in simulation**; mentors
**Erkka Lehtonen and Viktor Solianoi** on site, plus ~30 Aalto Robotics Club volunteers.

Starter repo: `github.com/zupermiez/aaltonomous-humanoid` — guide in
`aaltonomous/cloud/QUICKSTART.md`, rules and interface in `aaltonomous/docs/participants/`.

## The gates, which are hard

To reach the real robot a solution must pass the safety check in simulation: **no crashes or
safety stops, no commands over the joint and speed limits, and at least 27 commands per second.**
Teams never control the robot directly — their runner executes our Docker image through their
safety filter while their team holds the emergency stop.

**GPU:** RTX-class, ≥ 24 GB (RTX Pro 6000, L40S, A6000, 4090/5090). **The simulator ray-traces the
cameras, so it will not render on A100 or H100** — those are fine for training only. **On-demand
instances, not spot**, work on a persistent volume. First simulator start ~5–10 min while shaders
compile; without Docker the setup script is 30–40 min.

---

## 🔑 Two things in the brief that change the approach

### 1. They tell us what "generality, 25 %" means — and it argues for classical vision

Their own wishes, verbatim: *"Solutions that still work when the real world doesn't match the
simulation exactly"*, *"General ideas that could work beyond a single task, rather than ones tuned
to one exact setup"*, and *"Clever ways to do a lot with very little real-world data."*

**That is a quarter of the score spent on the sim-to-real gap and on data efficiency.** A policy
trained on simulator renders learns the simulator's visual statistics and breaks on real cameras —
which is the single most common way this kind of project fails on the day. A geometric pipeline
(colour segmentation in HSV, depth to pose, IK to reach) needs **zero real-world data**, and what
little tuning it needs is a colour threshold calibrated on a handful of real frames.

So **the classical route is not the timid fallback. It is the direct answer to two of their three
stated wishes**, and it is the only route whose failure modes we can explain on stage — which also
feeds presentation at 15 %.

### 2. The scene is far more constrained than it first reads, and that favours geometry

- **Boxes are 21 cm wide and 5 cm apart.** Objects are 4.5 cm. So hitting the right box needs
  roughly **±10 cm** of placement accuracy — coarse, and easy for IK.
- **Box positions are fixed**, at a known 40 cm from the torso. Nothing to detect; they can be
  constants.
- **Three colours, three shapes.** A closed, tiny label set.
- **Legs and waist are stiff.** No locomotion, no balance problem — this is a fixed-base
  bimanual arm task wearing a humanoid costume.

**Consequence: the hard part is the grasp, not the placement, and not the perception.** Effort
should go into picking up a 4.5 cm prism with a three-fingered hand, not into a model that decides
which box to aim at.

## How this connects to the abstention strategy

Oliver's reading of the starter repo gives per-object scoring: **fixed +1/−1, language +2/−1,
knocked off −1.** Expected value of placing is `2p − (1−p)` on a language episode (zero at
**p = ⅓**) and `p − (1−p)` on a fixed one (zero at **p = ½**). Below those, **leaving the object on
the counter scores better than placing it.**

Combined with the point above — the perception problem is small, the grasp problem is large —
the abstention should mostly trigger on **grasp confidence and instruction ambiguity**, not on
colour. And because the instruction is parsed once per episode and each object classified once,
that confidence lives outside the 30 Hz loop, where it can be calibrated properly. See
[`humanoid-policy-approaches.md`](./humanoid-policy-approaches.md).

🔴 **Still unverified and the strategy depends on it:** is an **unplaced** object really 0, or is
there an incomplete/timeout penalty? And can a misplaced object be moved again, or is the score
final on contact? Asked Oliver, who has the repo open.
