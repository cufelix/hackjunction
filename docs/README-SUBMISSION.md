# Väiski — a classical brain for the Unitree G1

**Junction 2026, challenge 320 "Aaltonomous Humanoid".**

Tell the robot where things go, in one English sentence, and it sorts the objects on the table into
three identical orange bins. No model, no network, no training data, and nothing random — the same
sentence and the same scene always produce the same behaviour.

The policy sees the head RGB-D stream plus the instruction and returns joint targets at **30 Hz**.
It has two minutes per episode.

---

## What it does

```
          "red on the left, yellow in the middle, blue on the right"
                                   │
┌──────────────┐   ┌───────────────▼────────────┐   ┌──────────────┐   ┌──────────┐
│  PERCEPTION  │──▶│          RESOLVER          │──▶│   PLANNER    │──▶│  SAFETY  │
│              │   │                            │   │              │   │          │
│ robust HSV   │   │ stdlib only: re, itertools │   │ top-down     │   │ velocity │
│ + depth      │   │ sentence → rule            │   │ pinch        │   │ filter   │
│   ↓          │   │                            │   │              │   │          │
│ 3D bin cut   │   │ keys: (colour,shape)       │   │ per-shape    │   │ 0 clips  │
│ pelvis frame │   │       colour, shape,       │   │ posture      │   │ ≥27 Hz   │
│ split_blobs  │   │       default              │   │              │   │          │
│ top-face     │   │                            │   │ insert_guard │   │ watchdog │
│ shape + apex │   │ EV rule: guess iff p > 1/3 │   │ (abstain)    │   │          │
│ → objects in │   │ else ABSTAIN               │   │              │   │          │
│   metric 3D  │   │                            │   │              │   │          │
└──────────────┘   └────────────────────────────┘   └──────────────┘   └──────────┘
                                   │
                        joint targets, 30 Hz, chunks of 10 ticks
```

Each stage is allowed to say "I don't know". That is the design idea the whole entry is built on,
and the scoring is why.

---

## Why abstaining is the whole trick

The challenge scores a correct `sort_fixed` placement **+1**, a correct `sort_language` placement
**+2**, and a wrong placement **−1** either way. Declining to place an object scores **0**.

So language sorting is roughly **83 % of the points available**, and a guess is only worth making
when it is better than silence. If the robot believes a bin is right with probability *p*:

```
guess   →  3p − 1
decline →  0
```

> **Guess only when p > 1/3.**

That single inequality runs through every stage. The resolver abstains on a sentence it cannot
parse. The shape classifier abstains below 30 surface points. On the `insert_shapes` task, the
planner's `insert_guard` picks up nothing at all and ends the episode, scoring 0 instead of −1,
because our pick-and-drop would otherwise leave objects sitting on the lid.

It is why our sim runs show **zero wrong placements**.

---

## Reading the instruction

This is the part that pays, so it is the part we measured hardest.

The resolver is **Python standard library only**:

```python
from __future__ import annotations
import itertools
import re
```

It turns the sentence into a mapping keyed by colour, by shape, by a `(colour, shape)` pair, or
`default`. The bin for an object is the first key that matches **in that order**, so
`"red objects left, cubes right"` sends a red cube left — a colour key outranks a shape key.

It handles the constructions people actually write: elimination (`"the remaining colour goes
right"`), many-to-one (`"red and yellow share the centre bin"`), all-but (`"everything except the
cubes"`), bin-first ordering (`"the left bin takes blue"`), compounds (`"red cubes left, blue
cylinders centre"`), ordinals, negation, and self-correction.

### How we know it is not tuned to our own sentences

Every change was scored on **sealed sentence sets the implementer never saw**, by a different team
member, with the prediction **registered before measuring** — so a change that did nothing was
caught instead of celebrated. One change predicted zero sentences would move; exactly zero moved,
across 241.

| Held-out set | Measure | Before | After |
|---|---|---|---|
| Blind set, 358 unseen sentences | expected value | 84.5 % | **96.8 %** |
| | sentences scoring −1 | 239 | **67** |
| | abstentions | 132 | **0** |
| Second corpus, held-out **test** split, 3276 sentences | exact match | 82.2 % | **87.9 %** |

Abstentions fell to zero *while* expected value rose, which means the resolver understood more
sentences — it did not start gambling.

A final independent check scored behaviour over all nine `(colour, shape)` object types rather than
comparing strings, because a mapping can look right and still place objects wrongly:

| sealed set | before | after |
|---|---|---|
| v1 | 141/151 (0.934) | **143/151 (0.947)** |
| many-to-one | 28/30 | **30/30** |
| v2 | 48/60 (0.800) | **50/60 (0.833)** |

No set dropped. Resolver `21af849894f526b1`.

**On the second corpus we quote the test split on purpose.** Counting both halves it reads 92.8 %
expected value and 91.7 % exact match, but the later changes gained more on the development half,
and quoting that number as evidence against overfitting would be circular.

---

## Results

### Simulation — the organisers' evaluator, seeds 0–7

Run on the organisers' Isaac sim and `aaltonomous/sim/evaluate.py`, on the public scene. The
denominators are **objects**, not episodes: 8 episodes of 3–5 objects each.

| | objects in the right bin | wrong placements |
|---|---|---|
| `sort_fixed` | 31–32 / 38 | **0** |
| `sort_language` | 28–29 / 36 | **0** |

The official gate allows **zero clipped steps** and requires **≥ 27 Hz**. Across four builds
(r13, r14, r15b, r16) it measured **0 clipped steps, a minimum command rate of 30.0 Hz, and no
safety stop or crash**.

These were measured on resolver `fd410acb` and `fd410acb` + alias. The shipping resolver
`21af8498` returns **identical output for all 8 `sort_language` templates** in the organisers'
`tasks.py`, so the figures carry over unchanged.

**These are simulation numbers. We have not scored the real robot and we do not claim to have.**

### Shape classification

Colour uses deliberately wide HSV windows (red H 0–16 and 170–180, yellow 18–52 with S ≥ 45, blue
100–145); bin pixels are removed by a 3D cut rather than by hue, and `split_blobs` separates a blob
larger than 1.5× the expected object area using distance-transform peaks.

Shape is decided on the **metric top-face points in the pelvis frame**, not on the 2D colour blob —
the feature is silhouette area ÷ `minAreaRect` area (1.000 cube, π/4 ≈ 0.785 cylinder, 0.500
triangle).

| | 2D pixel blob | metric top face |
|---|---|---|
| cube | 62.9 % | **100 %** |
| cylinder | 100 % | **100 %** |
| triangle | 21.0 % | **100 %** |

On the 2D blob a triangular prism reads as a cylinder 79 % of the time. Mean accuracy goes from
61 % to 100 %, and the feature is invariant to the per-episode camera-pose bias. Below
`MIN_TOP_POINTS = 30` it abstains rather than guess, and the narrow grasp posture is used only for a
triangular prism with `apex_conf ≥ 0.5` **and** a maximum angle ≤ 75°.

*Measured on 105 poses per shape from our own synthetic rasteriser — not an Isaac render, and it
does not model HSV edge error or occlusion.*

---

## Generality

**One image runs both sim and the real robot.** No network at any point — the container is verified
with `--network none`. There are no weights to download and nothing to warm up: the policy is pure
CPU and holds 30 Hz there. (The standard run line still passes `--gpus all`, but nothing in our
stack needs a GPU.)

We tested the perception against **real recorded frames** from the Hugging Face dataset
`zupermiez/g1_pick_cube` — head RGB only, no depth, no calibration, and a different scene from our
sim: foam objects on bare wood, with orange cardboard bins.

**Colour segmentation, 60 frames, 336 hand-annotated objects.** Frames taken at the start, middle
and end of all 20 episodes. An object counts as detected when its centre lies inside a blob of the
right colour.

| | stock thresholds | robust thresholds |
|---|---|---|
| orange objects | **0 / 120** | **120 / 120** |
| yellow objects | 8 / 120 | **119 / 120** |
| violet objects | 96 / 96 | 96 / 96 |
| false positives | 14.4 per frame | **0.3 per frame** |

The stock thresholds fail completely on real orange because the real scene inverts the sim: the
cardboard **bins** sit at hue 2–7 and the **objects** at hue 9–15, the opposite way round, and the
real yellow is over-exposed up to hue ≈ 40. Widening the windows and removing bin pixels with the
3D bin cut — rather than by hue — is what fixes it.

**End-to-end perception, 15 frames, 85 objects.** A later, stricter check through the full
submission pipeline including phantom counting: **79 of 85 objects found with the correct colour
(93 %), 0 colours swapped**, 4 missed, about 7 phantom blobs per frame — mostly the robot's own
hand reading as blue and bin cardboard as red.

That exercise found the single most useful thing we know about the real setup: the real bins image
**8–17 cm farther away and 4–10 cm to the right of nominal**. A later geometric fit says most of
that is not the bins moving at all but the **head camera pitched up by about 7.5°** against the
URDF calibration, varying per episode (≈ −4° to −9°) — fitting it drops the rim error from ~20 cm to
2–3 cm. Either way the lesson is the same and is the reason the mitigations below exist: a fixed
bin layout is wrong on the real robot. Which of the two it is decides the fix — a per-episode
online calibration rather than a baked constant. A fixed bin layout erases objects and
turns bin walls into phantom detections. We built per-episode bin detection, counter-plane fitting
and hand-FK camera self-calibration in response. They ship **switched off** — they are insurance we
can enable once we see the real table, not something we are claiming works there.

---

---

## The language brain, and the guard around it

> **Not in the submitted image.** Everything in this section runs beside the policy, in a separate
> module (`policy/brain_lang.py`), and is **switched off** in what we submitted. The image you
> score contains the deterministic resolver and nothing else. We describe it because it is where
> the generality argument goes next, not because it is doing work in our results above.

The resolver is excellent inside its grammar and helpless outside it. A sentence like *"the object
closest to the robot goes left"* has no colour and no shape to match, so no amount of regex reaches
it. For those we add a **local** Qwen3-VL-8B on llama.cpp, running offline on the finals machine,
as a **closed-set tool chooser**: it is handed the world state, the instruction, and a numbered list
of allowed calls — `pick_place(obj_3, left)`, `skip(obj_3)`, `look()` — and returns an index and a
confidence. A GBNF grammar constrains the output to exactly those indices, so the reply is always
valid JSON and can never name a tool that was not offered.

**The model proposes; it never disposes.** One rule governs every decision:

```
LLM agrees with the classical choice        -> take it
LLM disagrees, confidence < 0.95            -> take the classical choice
LLM disagrees, confidence >= 0.95           -> allowed ONLY where the resolver abstains
neither has an opinion                      -> abstain (0 beats -1)
```

So the model fills gaps. It cannot overrule an answer the resolver is sure of, whatever confidence
it claims and whichever tool it picks. If llama.cpp is slow, crashed or absent, every call returns
"no answer" and the behaviour collapses exactly onto the resolver we submitted.

**We measured how much that guard is actually worth, because a model's confidence is not
calibrated.** Simulating models of known accuracy on our open-vocabulary set:

| model accuracy | confidence | with the guard | without it |
|---|---|---|---|
| 1.00 | 0.99 | 248 pts | 248 pts |
| 0.50 | 0.99 | 107 pts | 95 pts |
| 0.00 | 0.99 | **−103 pts** | −124 pts |
| 0.00 | 0.90 | **14 pts** | −124 pts |

Read the bottom two rows together, because they are the honest part: **the guard protects us where
the resolver has an opinion, and by construction it cannot protect us where it does not.** On
open-vocabulary sentences the resolver abstains 94.4 % of the time, so a confidently-wrong model
walks through. There the only mechanism that bites is the confidence threshold — which is why a
model has to clear roughly **mid-thirties percent accuracy** before switching it on beats leaving
it off, and why we treat enabling it as a measured decision rather than a default.

**What is verified today, with no model running:** the brain's stub backend reproduces the shipped
resolver on **0 of 336 dev sentences differing**, scored by behaviour over all nine (colour, shape)
types rather than by comparing dictionaries. That is the regression guard that proves the new layer
cannot silently change what we already submitted.

**Open-vocabulary dev set.** 80 instructions over three families — spatial, ordinal/size,
relational — each with a synthetic 3–5 object scene in the robot's pelvis frame and a gold answer
**computed from the scene geometry**, never hand-written. On it the resolver abstains on 117 of 124
objects. The set exists to measure the model, and it is deliberately adversarial: it is **not** a
sample of finals instructions, so nothing here is a claim about the resolver's general accuracy.

### What it actually scored

A teammate ran it on the finals-class box (RTX 5090, Qwen3-VL-8B at Q8 through llama.cpp). On the
124 open-vocabulary objects above:

| | resolver alone | with the model |
|---|---|---|
| correct | 7 | **87** |
| wrong | 0 | **24** |
| declined | 117 | 13 |
| **points of a possible 248** | **14** | **150 (60.5 %)** |

Where it decided at all it was right 78 % of the time, against a break-even of 37 %, so switching
it on is worth about **+136 points** on this set. By family, as a percentage of the points
available: spatial 87 %, ordinal 67 %, relational 45 %.

**Two results we would rather not have found, and are reporting anyway:**

**1. The model introduces wrong placements where the resolver made none.** 24 of them. Our "zero
wrong placements" figure belongs to the submitted image and does not survive turning the brain on.
It is a net gain in points, not a strict improvement, and those are different things.

**2. The confidence threshold does nothing.** Replaying the same decisions at every threshold from
0.00 to 0.95 gives the identical 150 points; only 0.99 moves it, and downward, to 130. The model
reports ≥0.95 on essentially everything, including the answers it got wrong — it is not calibrated,
so the number it hands us is not information. That demolishes the tidy version of our own story:
what bounds the damage is **the guard and the resolver**, not the arithmetic on the confidence.
We kept the threshold because it costs nothing and a better-calibrated model would make it bite,
but we are not going to claim it is doing work here.

**22 of the 24 errors are a single failure:** asked which bin an object is nearest, the model
cannot compute Euclidean distance. That is not a language problem and it does not want a bigger
model. We compute the geometry in code — distances to each bin, nearest bin, and the orderings
by depth, lateral position and size — and hand them to the model as fields it only has to read.
**Geometry is code; language is the model.** Re-measurement of that change is in flight at the time
of writing and is not included in the table above.

### Numbers still open

| | |
|---|---|
| The geometry-fields version | **being re-run** — the table above predates it |
| Accuracy on real finals instructions | unknown, and not knowable from our own dev set |
| `find()` grounding accuracy | **unverified** — a 96.5 % / 0 FP figure circulates internally, but we have not seen the method behind it, so we are not putting it in front of you as a result |

Decision latency is 0.23 s at the median text-only and 0.30 s with an image, in 10.8 GB of VRAM,
measured on the same box by the same teammate rather than by us.

## How to run

The submission is a policy server in one container.

```bash
# the submitted image: a websocket policy server on 0.0.0.0:8000
docker run -d --gpus all -p 8000:8000 hackjunction/policy:integ-75e95b2

# smoke test against the organisers' runner
python -m aaltonomous.deploy.runner --backend mock \
    --policy ws://localhost:8000 --team <name> --episodes 2

# it needs no network at any point
docker run -d --network none -p 8000:8000 hackjunction/policy:integ-75e95b2

# the instruction resolver's own test suites (stdlib only, no deps)
python3 -I policy/test_instruction.py     # unit tests
python3 -I policy/test_integration.py     # behaviour over all 9 object types
python3 -I policy/probe_families.py       # 16 sentence families
python3 -I policy/probe_structures.py     # 48 sentences across 6 structures
```

The server must be listening within 120 s of start; ours takes about **0.8 s**. It receives
`{"instruction": ...}` at reset, parses it once, and returns joint targets in chunks of 10 ticks
at 30 Hz.

---

## What we tried that did not work

We kept the measurements that argued against us.

* **Insert push.** `insert_shapes` push variant: **0/15 cylinders inserted, −6 points over 10
  episodes** → kept OFF. We ship `insert_guard`, which abstains: 0 instead of −1.
* **`y_affine`.** An empirical sim-only bias correction. It turned out to be absorbing the sim
  camera's principal-point offset (`cx`/`cy` ≠ calibration), so it is **not used in the real-robot
  image**.
* **`reject_orange`.** A clean fix for orange bins leaking into the red mask — but it also drops
  real orange objects, and in the finals red *is* orange on the lab set. Rejected for the
  submission; kept as a sim-only research flag.
* **The wider grasp.** `pinch_demo` was predicted to win and lost: **51/82 against 60/82** for
  `pinch_v1`, collapsing on cubes (7/25 vs 20/25). No single aperture fits a 30 mm prism chord and a
  45 mm cube face, which is why per-shape posture is forced by geometry rather than chosen.
* **A bijection repair** that forced three colours into three distinct bins. Many-to-one is a normal
  instruction shape, so the repair converted +2 into −1. It fired zero times on real failures and
  was removed.
* **Our own resolver guard, caught by the blind set.** A fix for "blocks with a circular
  cross-section" searched for the shape word up to the next *bin* noun — and in a sentence with no
  bin noun between clauses it reached into the next clause and dropped a key. Blind 358 fell
  96.8 % → 95.9 %. Every adversarial probe we had written happened to contain the word "bin", so our
  probes shared the code's blind spot. Rewritten to require the descriptor be attached to the noun;
  back to 96.8 %.
* **A completion step** predicted to gain 0.78–0.80 on one set moved exactly zero sentences,
  because the unmapped colour is never named in those sentences.
* **An area floor** to catch shapes spoiled by bin-mask cuts: measured useless (0 of 423 wrong
  labels caught) and retracted.

---

## What we have not measured

* **The real robot.** Everything above is simulation or offline analysis of recorded frames.
* Perception on real frames was tested on **RGB only** — the recorded dataset carries no depth, so
  the colour results above say nothing about the metric top-face shape classifier, which stays
  unverified outside simulation. The real-bin offset we measured there is from colour alone and
  could partly be camera tilt rather than the bins having moved; we have not separated the two.
* `many2one` improving 28 → 30 is **not statistically significant** (McNemar p = 0.25, 3 sentences).
  We report it, we do not lean on it.
* Two instruction constructions are known-wrong and documented rather than hidden: a shape
  elimination phrased `"...and the last one goes left"` abstains, and one phrasing of it picks the
  wrong bin. Both predate every change we made this weekend. The obvious fix re-breaks a larger
  category, measured, so we left the shipping resolver alone.

---

## Licence

MIT. Built at Junction 2026.
