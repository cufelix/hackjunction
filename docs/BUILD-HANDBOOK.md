# Build handbook — Aaltonomous Humanoid (challenge 320)

**Written during the research freeze of Fri 13 Nov 2026, no code. Felix's rule: nothing is built
before 02:00; this document is what the night produces, and from 02:00 we build from it.**

How to read it: **§1–3 are facts with a source.** §4 is the build order, and every item names the
fact that puts it in that position. §5 is what we deliberately do *not* build. §8 is what is still
unknown, each tagged with what it blocks. Anything marked *(inference)* is reasoning from the
facts, not something we read — tell them apart, because at 02:00 nobody will have time to re-check.

Sources: `aaltonomous/` starter repo read directly (Oliver, see `humanoid-research-log.md`), the
official challenge page (`challenge-320-aaltonomous-humanoid.md`), and primary vendor docs. Numbers
without a source do not belong here.

---

## 1. Where the points are

This is the only arithmetic that matters, and it decides everything below.

| | per object | 5 objects |
|---|---|---|
| **Rung 1, fixed mapping** | +1 correct / −1 wrong bin / −1 knocked off / 0 on counter | max **5** |
| **Rung 2, language mapping** | **+2** correct / −1 wrong bin / −1 knocked off / 0 on counter | max **10** |

Finals are **3 episodes: fixed + language + an unseen-instruction language episode weighted ×1.5.**
So with 5 objects the final is 5 + 10 + 15 = **30 points, of which 25 — 83 % — come from language
episodes.** Timi and I reached this independently, from the episode format and from the rung weights.

**And the baseline we are handed scores exactly zero on all of it.** `policy.py` contains
`self.mapping = info.get("mapping") or dict(FIXED_MAPPING)`, and the reset info carries only
`{"instruction": ...}` — the mapping is never sent, so it silently falls back to fixed. The
handed-out code sorts 76–82 % of objects on rung 1 and **0 % of the 83 %**.

**Conclusion: the instruction is not a feature of this project. It is the project.**

## 2. Hard gates — fail one and none of the above counts

| Gate | Value | Source |
|---|---|---|
| Command rate | **≥ 27 Hz** wall-clock on ticks carrying a command | challenge page + SUBMISSION |
| Clips | a missing key, wrong shape or NaN is a clip; **>30 clips ends the episode** | INTERFACE |
| Safety | no crash, no safety stop. **Self-collision and fingertip-to-table are NOT checked by the filter — staff e-stop for them** | INTERFACE |
| Limits | within joint and speed limits | challenge page |
| Episode | **120 s**, points so far are kept, no timeout penalty | RULES_AND_SCORING |
| Image | x86_64, CUDA ≥12.8, weights baked in, listening within 120 s, ≤32 GB VRAM, ≤40 GB image, no volumes or args | SUBMISSION |
| GPU for rendering | **RTX-class ≥24 GB. Will not render on A100/H100** (ray-traced cameras). Finals PC: RTX 5090 | challenge page |

**The 27 Hz gate is about action chunking, not model speed** — see
[`humanoid-policy-approaches.md`](./humanoid-policy-approaches.md). Returning a chunk of k actions
per call decouples the command rate from the inference rate; the baseline already does this,
returning 6-tick hold chunks from a low-priority planner thread while it plans. **Held ticks do not
count toward the gate.** Anything that cannot hold the rate *without* chunking is disqualified by
arithmetic, not by tuning.

## 3. The facts that constrain the design

### 3.1 The scene is small, but the bins are *far*

From `scene/scene.yaml`, read directly:

- Bins **21 × 21 cm outer, measured on the real boxes 13 Nov**. Wall height **6 cm** and 3 mm
  thickness are *file defaults, not measured*.
- Bin centres: centre bin at pelvis **r = 0.505 m**, **side bins at r = 0.568 m**.
- Reach: **comfortable ≈ 0.40 m, straight arm ≈ 0.55 m.**
- Objects **45 mm** for every shape, 2.5 mm fillets, in a band at pelvis x 0.26–0.35, |y| ≤ 0.17,
  min 8 cm apart, yaw −45…+45°, **3–5 objects** (`sample_layout` allows 3–6, ≤2 per colour).
- Counter surface at pelvis **z = +0.01** (default, to be measured).

🔴 **This corrects something I wrote earlier today.** In
[`challenge-320`](./challenge-320-aaltonomous-humanoid.md) I argued that because the bins are 21 cm
wide and 5 cm apart, placement needs only **±10 cm** and is therefore easy. **The lateral half of
that is right and the radial half is wrong.**

- **Side-bin centres at 0.568 m are past the 0.55 m straight-arm reach.** The centre of a side bin
  cannot be reached at all. `scene.yaml` says so itself: release in *"the near, inner part"* of the
  side boxes.
- All three bins are past the 0.40 m comfortable reach, so **every release happens beyond
  comfortable extension**, and the side ones at essentially full extension.
- With 3 mm walls the inner region of the left bin spans x 0.403–0.607, y 0.158–0.362. Only the
  **near-inner quadrant** of that satisfies r ≤ 0.55 — roughly an 8–10 cm patch, in one corner.
  *(inference, from the published coordinates.)*
- And the object must clear a **6 cm rim** while an object is only 4.5 cm tall — **nothing can be
  swept or pushed in; it must be lifted over.** At near-full extension.

**So placement tolerance is about ±4–5 cm into a specific corner, not ±10 cm anywhere.** It is
still not the hardest part — but it is not free, and a release aimed at the geometric bin centre
aims at a point the arm cannot reach.

### 3.2 Why the triangle fails, with the numbers

This is the mechanism behind the organisers' own note, and it is derivable.

- `scene.yaml`: *"cube side = cylinder diameter = triangle ALTITUDE (flat-to-apex)"*, all 45 mm.
  An equilateral triangle of altitude 45 mm has sides **s = 2h/√3 ≈ 52 mm**.
- A triangle's grip width depends on approach angle: **45 mm across an altitude** (flat face on one
  side, apex on the other) and **52 mm across the side direction** (an apex on *both* sides).
  *(inference, support widths of an equilateral triangle.)*
- Dex3-1: index and middle **flex only, no sideways spread**; thumb has 3 joints and does the
  opposition. Fingertip repeatability ±2 mm. (Unitree's own page.)
- Baseline README: index and middle tips are **57 mm apart**, a 45 mm object wedges between them,
  so the pinch is **thumb vs ONE finger**.
- Organisers, `sim/README.md:77`: their **oracle, with exact privileged poses and shapes, reaches
  only ~84 % on rung 1**, noting *"needs >90 %: triangle grasps (thumb vs one finger spins the
  prism)"*.

**Put together: the worst grip is 52 mm with a corner contact on both sides — two point contacts,
so the prism spins. The best grip is 45 mm with one flat face. They are 30° apart in wrist yaw, and
the triangle's 3-fold symmetry guarantees that from any orientation the nearest good direction is
at most 30° away.** *(inference.)*

And the input is already there: **perception computes object yaw with `cv2.minAreaRect` and marks
it "info only"; the planner never reads it.** The fix is to set wrist yaw so the pinch axis lies
along an altitude direction.

**This is why perception work and a trained policy are both capped.** A perfect vision stack cannot
pass ~84 % on rung 1. The grasp is the ceiling.

### 3.3 Three shapes, three different failure modes *(inference, from the geometry)*

| Shape | Failure mode | What it demands |
|---|---|---|
| **Cube** | essentially none — flat faces both sides, 45 mm | nothing |
| **Triangle** | **grasp** — corner contact spins it | wrist yaw aligned to an altitude (§3.2) |
| **Cylinder** | **rest** — it rolls, and *"not at rest at episode end counts as counter unless it is inside a bin"* | place it **early**, so it has time to settle |

### 3.4 What the baseline already does, so we do not rebuild it

`planner.py` (460 lines) already has: both arms, a finger-direction hint list tried in turn,
bin-wall clearance, relocation of an object out of the way, a failed-position list so it does not
retry the same spot, `done` when nothing sortable is left, and 6-tick hold chunks that keep the
command rate at 30 Hz. Sim rung 1, seeds 0–7: **76–82 % sorted, 0 clips, 30 Hz, gate PASS.**

Two things it does **not** have: it never reads the instruction (§1), and **there is no
post-placement verification at all** — nothing checks whether the object actually landed in the
bin. With final-state scoring, that is an unused lever, not a missing nicety.

⚠️ **The baseline has never run on the real robot** (*"not run yet (M8)"*); the HSV floor, grasp
point and open posture were only ever checked in sim. Treat 76–82 % as a sim number. Sim also ran
at ~0.65× real time, so expect ±10 % between runs.

## 4. What we build, in order, and why each is in that slot

### 1. Instruction → mapping. **Unblocked, highest value, no GPU.**
Because of §1: this is 83 % of the final and the baseline scores zero on it, and because the wiring
gap is one line.

Design in [`instruction-parsing-six-permutations.md`](./instruction-parsing-six-permutations.md):
the instruction is always a full permutation, so **enumerate the candidate mappings and score each
against the sentence** instead of writing a parser. Decide the key vocabulary first (colour words /
shape words / both — `scene.resolve_bin` accepts all three), then enumerate the 6 permutations over
that key; 12 candidates in total. Three properties a parser cannot have: it cannot emit an
impossible mapping, it **completes the third pair by elimination** from two correct ones, and its
**confidence is the margin** between best and second-best candidate.

`oracle.py` already contains a parser for all 8 training templates. The oracle is a sim-only
privileged policy, but **the instruction string is not privileged** — our policy receives it too —
so that parser is legitimate reference for the known phrasings (check its licence first). The
finals templates are deliberately different, so the enumeration is the robustness layer on top.

The one thing enumeration gets *backwards* is **negation** — *"anything but the left bin"* reads to
a proximity count as support for the pair it denies. Needs an explicit inverting flag, and
`STARTER_PATHS` says harder finals may add negation.

### 2. Yaw-aligned triangle pinch. **Promoted by the oracle number.**
Because of §3.2: ~84 % is the ceiling with perfect perception, the organisers name this as the
blocker, and the yaw we need is already computed and thrown away.

### 3. Post-placement verification and correction.
Because scoring is **final state, not on touch** (`RULES_AND_SCORING`), a misplaced object can be
moved again and fixed, and §3.4 says nothing currently checks. A wrong bin is −1 and a correction
turns it into +1 or +2 — **the largest per-object swing available, on a mistake we have already
made.**

### 4. Cleanup phase, ~10–15 s before the buzzer.
Because *"on counter"* is 0 but *"knocked off"* and *"wrong bin"* are −1, and because
**not-at-rest** can cost the point. End the episode with nothing in the hand, nothing moving, and
any object we are not confident about **left on the counter deliberately** — 0 beats −1. The margin
from item 1 and the grasp confidence from item 2 are what decide which.

### 5. Stretch, in this order: relay hand-over → depth-noise randomisation.
The oracle already implements a **relay hand-over** for objects one arm can grasp but not carry
across the body (it puts them down near the centre line for the other arm) — the baseline README
lists exactly that as a failure. Depth-noise randomisation fitted to the real Drop 1 depth is the
citable **sim-to-real** story for the 25 % generality score, and the organisers' stated wishes for
that criterion are all sim-to-real and data efficiency.

## 5. What we do not build, and why

- **No VLA as the control policy.** GR00T N1.7 measures ~140 ms (~7 Hz) on an RTX A6000 and
  OpenVLA base ~6 Hz on a 4090; both fail the 27 Hz gate without chunking, and with ~20 demos per
  rung there is not enough data to beat a baseline that already sorts ~80 %. Measured rates and who
  fails are in [`humanoid-policy-approaches.md`](./humanoid-policy-approaches.md).
- **No from-scratch learned policy.** Evidence says VLAs on G1 need hundreds to ~1K demos; we get
  ~20 per rung.
- **No perception overhaul.** §3.2: capped at ~84 % on rung 1 regardless.
- **No tactile or force feedback.** The Dex3-1 has 33 pressure sensors, and **the challenge
  interface exposes none of it** (FAQ). Designing around touch is designing around data we cannot
  read.
- **No LLM in the control loop.** The instruction is parsed **once per episode in `reset()`**;
  latency is then irrelevant. A model is the fallback for a zero margin, not the mechanism.
- **Nothing that leans on wrist-camera geometry.** Wrist **extrinsics are null and intrinsics are
  placeholders — uncalibrated today.** Head camera is a RealSense D435i, fx=fy=615, extrinsics from
  URDF (~48° down, unverified).

## 6. Judging, and what the demo has to be

**Real robot 40 % (gated to the top ~5 by sim performance) · generality 25 % · originality 20 % ·
presentation 15 %.** Prize 2,000 € plus an Aalto lab visit. Mentors on site: Erkka Lehtonen,
Viktor Solianoi.

Two consequences:

1. **Sim performance is a qualifier, not a score.** It buys the ticket to the 40 %. So a fragile
   sim-only trick that does not survive the real robot is worth nothing, and the last image pushed
   before the deadline is the one that runs in the finals.
2. **Generality is 25 % and their stated wishes for it are sim-to-real robustness, ideas beyond
   this one setup, and a lot from very little real data.** Our story is a design-level one and can
   be checked in the code rather than taken on trust: *we cannot emit an impossible mapping, and we
   recover the whole instruction from two thirds of it.* That is a property, not a benchmark number
   from our own test set.

## 7. Open questions, tagged with what they block

- [ ] **Does the `< 2 cm/s` at-rest test apply to an object on the counter, or only in a bin?**
      `scene.yaml` says *"not at rest at episode end counts as counter (0) unless it is inside a
      bin"*, while `RULES_AND_SCORING` defines IN as requiring at rest — those two readings
      disagree about a cylinder still rolling **inside** a bin at the buzzer. **Blocks: how early
      cylinders must be placed (§3.3) and the length of the cleanup phase (item 4).** Ask a mentor.
- [ ] **Does an object in the robot's hand at the buzzer count as "on counter" (0) or "knocked off"
      (−1)?** **Blocks: whether the cleanup phase must put a held object down, and when the last
      pick may safely start.** Ask a mentor.
- [ ] **Do the finals instructions key on shape or on (colour, shape)?** `scene.resolve_bin`
      accepts all three and `STARTER_PATHS` mentions shape words. **Blocks: the size of the
      candidate set in item 1** (6 vs 12 vs composite).
- [ ] **Is wall height really 6 cm?** It is a file default, not measured, and it sets the lift
      height at near-full extension (§3.1). **Blocks: the release trajectory.**
- [ ] **Rung 3 exists** — a 3 mm lid on bin `left`, hole clearance 1 cm default (1.5 cm easy, 3 mm
      hard). Not yet assessed at all. **Blocks: nothing yet; do not start it before items 1–4.**
- [ ] **Is `oracle.py` licensed for us to copy from?** **Blocks: whether item 1 reuses its template
      list or re-derives it.**
- [ ] **What is the real end-to-end budget** with 3 camera views plus state over the websocket?
      Published rates are per model call with a single view. **Blocks: chunk length.** Measure in
      the provided sim before choosing.

## 8. How we work, until this changes

- **No code before 02:00.** Research and documents only. Three plans died tonight to three reads;
  that is three afternoons saved for one evening.
- **Claim files in Warren before touching shared code, release when done.** No silent edits.
- **Facts carry a source** — file and line, or a primary doc. Inferences are labelled. This
  document keeps them apart on purpose.
- **Do not delete another session's research.** It happened once today and it was not that
  session's call to make.
