# Build handbook — Aaltonomous Humanoid (challenge 320)

**The document we build from.** Written during the research night of 13–14 Nov 2026 and rewritten
at 00:30 on 14 Nov so that corrections sit **where the fact is**, not 400 lines further down. The
provenance and the full correction log are in Appendix A; if you want to know what changed and
why, read that. If you want to build, read §1 to §5 and ignore the appendix.

**How to read a claim in here:**

| Mark | Means |
|---|---|
| ✅ | read directly from the starter-repo clone or measured locally, with the file and line |
| *(inference)* | reasoning from marked facts — correct only as far as the facts are |
| ⚠️ | known uncertainty, with what would settle it |
| 🔴 | changes a design decision |

**Facts carry a source. Numbers without a source do not belong here.** That rule is why three of
this handbook's own numbers were caught and fixed before anyone built on them.

✅ **The starter repo is public and MIT** (`github.com/zupermiez/aaltonomous-humanoid`, `LICENSE`:
*"MIT License / Copyright (c) 2026 DigiTwin, Aalto University"*). It is cloned locally. **MIT
settles a question we had: we may read and copy from `oracle.py`**, with attribution.

Per-fact line references also live in Oliver's [`humanoid-verified-facts.md`](./humanoid-verified-facts.md)
(pinned to starter commit `36daf5b4`) and his [`humanoid-research-log.md`](./humanoid-research-log.md).
Where that file and this one disagree, **the one with the line number wins.**

---

## 1. Where the points are

This is the only arithmetic that matters, and every decision below follows from it.

| | per object | 5 objects |
|---|---|---|
| **Rung 1 — fixed mapping** | +1 correct · −1 wrong bin · −1 knocked off · 0 on counter | max **5** |
| **Rung 2 — language mapping** | **+2** correct · −1 wrong bin · −1 knocked off · 0 on counter | max **10** |
| **Rung 3 — insert shapes** | **+3** inserted · −1 forced · −1 dropped · **0 wrong hole · 0 counter · 0 off** | max 15 |

✅ Values read from `scene.yaml:78-82`. ⚠️ Rung 3 is almost certainly **not** evaluated — see §1.3.

### 1.1 83 % of the final is language, and the handed-out code expects to score zero on it

The finals are **three episodes: fixed + language + an unseen-instruction language episode weighted
×1.5.** Per object the rungs pay 1, 2 and 3, so for **any** object count N the maximum is
1N + 2N + 3N = 6N and the two language episodes are 5N of it — **83 %, independent of N.**
(Timi and I reached this independently, from the episode format and the rung weights.)

> ⚠️ **Correction.** This section used to say *"5 + 10 + 15 = 30 points, of which 25"*. That needs
> N = 5, and **the finals object count is not stated anywhere** — `RULES_AND_SCORING.md:63-68`
> says the finals settings are deliberately unannounced. The ratio is the robust claim; the
> absolute numbers are not. Everything below uses 5 only as a worked example.

✅ **And the baseline is wired to fall back to the fixed mapping on all of it.** `policy.py:73` has

```python
self.mapping = info.get("mapping") or dict(FIXED_MAPPING)
```

and the reset info carries only `{"instruction": ...}` — **the mapping is never sent over the
wire**, so it silently uses FIXED on a language episode too.

> ⚠️ **Correction, and it matters for how we read our own step-0 numbers.** This used to read
> *"the baseline scores zero on all of it"*. That is wrong as a deterministic claim. Scoring a
> language episode at +2 correct / −1 wrong, the baseline scores **+2a − b** where `a` is the
> objects whose true bin FIXED happens to agree with. Over the 6 permutations of the true mapping:
>
> | true mapping vs FIXED | probability | score |
> |---|---|---|
> | identical | 1/6 | **+2N** |
> | one colour right, two swapped | 3/6 | **0** |
> | a derangement (nothing right) | 2/6 | **−N** |
>
> The **expectation** is zero — `(1/6)(2N) + (3/6)(0) + (2/6)(−N) = 0` — but the realised value
> on one episode is anywhere from −N to +2N. So "the baseline scores 0 on language" is right on
> average and wrong on any single run, and **a lucky draw at step 0 will look like a working
> policy.** Measure rung 2 over several seeds or not at all.

The handed-out code sorts 76–82 % on rung 1 and has **no mechanism whatsoever** for the 83 %.

> 🔴 **The instruction is not a feature of this project. It is the project.**

### 1.2 ✅ The ×1.5 weight is prose only and is not implemented

`Scene.score` is a plain `sum()` (`scene.py:251`). Nothing in the repo applies the finals weight,
so **our own evaluation will never show it.** The arithmetic above has to be applied by hand when
we judge our own runs — otherwise a language improvement looks like a third of what it is worth.

### 1.3 ⚠️ Rung 3 pays the most per object and is probably never played

✅ `docs/participants/SUBMISSION.md:134` declares `tasks: [sort_fixed, sort_language]` and
`evaluate` is invoked only with those two (`:106-107`). `insert_shapes` appears nowhere in the
submission path, and the finals format is fixed + language + unseen language.

Note the risk profile anyway, because it is unusual and because finals settings are officially
unannounced (✅ `RULES_AND_SCORING.md:68`): **wrong hole costs 0, counter costs 0, off costs 0.**
Only `forced` and `dropped` are −1. If rung 3 ever appears, *attempting* is nearly free, and the
only thing it needs is knowing each object's shape (§3.3) — no instruction parsing at all, because
the instruction is the constant `INSERT_INSTRUCTION` with mapping `{}` (✅ `tasks.py`).

**It is not a reason to build anything.** It is a reason the shape classifier is cheap insurance.

---

## 2. Hard gates — fail one and nothing above counts

| Gate | Value | Source |
|---|---|---|
| Command rate | **≥ 27 Hz** on ticks carrying a command | challenge page + SUBMISSION |
| Clips | ✅ the sim gate is **`--gate-max-clips 0` — ZERO**, while >30 clips *aborts* the episode | `safety.py:239-245`, `RULES_AND_SCORING.md:55` |
| **Per-tick step** | **≤ ~0.033 rad arms, ~0.1 rad hands, per tick** (1 and 3 rad/s at 30 Hz). Exceeding it clips | arithmetic from `safety.py:49-50` |
| Safety | no crash, no safety stop. ⚠️ **self-collision and fingertip-to-table are NOT checked by the filter** — staff e-stop for them | INTERFACE |
| Episode | **120 s**; points so far are kept. ✅ **no time, retry, touch or per-clip penalty exists anywhere in the scoring code** | `scene.yaml:78-82` |
| Finals settings | ⚠️ **not announced in advance** — objects, layout and time limit all open. Today's code says 3–5 objects / 120 s | ✅ `RULES_AND_SCORING.md:68` |
| Image | x86_64, CUDA ≥12.8, weights baked in, listening within 120 s, ≤32 GB VRAM, ≤40 GB image, no volumes, no args | SUBMISSION |
| Render GPU | **RTX-class ≥24 GB; will not render on A100/H100** (ray-traced cameras). Finals PC: RTX 5090 | ✅ `QUICKSTART.md:10-11` |

### 2.1 🔴 The per-tick cap is the gate people miss, and it is stricter than the rate

> ⚠️ **Retracted, 14 Nov 01:30.** This section used to say *"a chunk can be returned fast enough
> and still clip on every seam… chunk seams must be blended, and we need our own rate limiter in
> front of anything that produces targets."* **That is wrong for this architecture, and building
> that limiter would have been wasted work.** Read end to end:
>
> - `_pop()` (`policy.py:243`) slices **one pre-planned trajectory** into chunks of
>   `motion.chunk`, so a chunk seam is two consecutive samples of the same array — not a
>   discontinuity at all.
> - `time_scale` (`planner.py:58`) says in its own docstring *"no joint ever moves faster than
>   vmax"*, samples at `T/n` with `n = ceil(T/dt)` so the spacing is **≤ dt**, and **excludes
>   `Qs[0]`**.
> - across trajectories, `_pop` sets `arm_cmd = a[-1]` and the next `go_to(arm_from=self.arm_cmd,
>   …)` starts from exactly that value, so the first sample of the new trajectory is one dt of
>   travel away.
> - `_hold()` re-sends the last target — zero delta.
>
> So **the emitted command stream is velocity-continuous by construction.** No blending needed, no
> limiter needed.

**The real exposure is not seams. It is the speed knob, and it is a trap.**

The safety filter clips when a command moves more than `arm_max_vel * dt` **from the last command**
(`safety.py:16`, `:231`). Two different `arm_max_vel` values are in play, and the margin between
them depends entirely on the profile:

| profile | planner `motion.arm_max_vel` | filter `arm_max_vel` | margin |
|---|---|---|---|
| base (robot) | 0.5 | 1.0 | **2×** |
| `sim` | **0.97** | 1.0 | 🔴 **3.1 %** |

Both numbers are verbatim from `config.yaml` — `:81` carries the comment *"safety filter allows
1.0"*, and `:126` in the `sim` profile repeats it at 0.97. ✅ And §3.4.1 establishes that **a
submitted container defaults to the `sim` profile**.

> 🔴 **So the obvious response to a slow step-0 number — raise `motion.arm_max_vel` to buy time —
> crosses 1.0 with 3.1 % of headroom and makes the filter clip on *every move*. With
> `--gate-max-clips` defaulting to **0**, that fails the entire run, and it fails it for a reason
> that looks nothing like "the arm is too fast".** §3.4 says time is the binding constraint, so
> this is precisely the knob someone will reach for. **Do not raise it above 1.0, and treat
> anything above ~0.95 as already unsafe.** Buy time from the trajectory (fewer waypoints, shorter
> approach, `max_joint_step`), never from `arm_max_vel`.

✅ `arm_max_lead` and `hand_max_lead` default to **`None` = off** (`safety.py:51-52`), so the
lead-cap clip source does not fire unless someone enables it. Hand speed has 20 % of margin on the
`sim` profile (2.5 against 3.0), which is tight but not in the same class.

We still need our own guard in front of anything that produces targets **outside** the planner —

✅ **The gate, exactly, read from `evaluate.py`:**

| check | default | source |
|---|---|---|
| `--gate-max-clips` | **0** — *any* clip fails the gate | `evaluate.py` flag |
| `--gate-min-hz` | **27.0** | `evaluate.py` flag |
| `policy_hz_rt >= 27` | **undocumented, wall-clock**, `(ticks − holds) / Σ latencies` | `evaluate.py:137-139` |

> 🔴 **The third one is in neither the rules nor the flags.** The published gate is about sim-time
> command rate; this one is about *our* wall-clock latency, and nothing warns you about it. A
> policy that holds 27 Hz in sim time while thinking too long per call fails here.

Also in the pass condition: `BAD_ENDS = ("aborted", "watchdog", "safe_stop",
"policy_connection_lost", "policy_timeout", "crash", "interrupted")` — any of those ends the run.

> ⚠️ **Correction.** This section used to say *"30 ends the episode"*, as though 30 aborts were
> allowed. There is no such budget. The 30 is `safety.py:74 max_clipped_steps = 30  # abort when
> exceeded` — the safety layer **aborts the episode when the clipped-step count exceeds 30**,
> and the gate itself already fails at the *first* clip. It is marked *"spec N, to be decided"*
> in the source while the rules state it as final, so **build nothing that depends on the exact
> number** — but build nothing that treats it as a budget either.

### 2.2 ✅ How the rate is actually computed — and why a slow policy can still pass

From `deploy/runner.py`: `hold = not queue`, `wait = hold and pending is not None`. A wait re-sends
the last command **as an explicit target** and is counted in `waits`, not `holds`. The summary is

```python
command_hz = (ticks − holds) / span   # policy targets only (spec §8 gate)
```

> **Only the runner's own idle holds are excluded. Everything the policy emits counts — including
> explicit hold targets and re-sent commands.**

That is good news: a policy that needs thinking time can hold the gate by **returning explicit
holds**, which is exactly how the baseline keeps 30 Hz from a low-priority planner thread. What
must stay rare is the *runner* having nothing to send.

**So the 27 Hz gate is about action chunking, not model speed.** Returning k actions per call
decouples command rate from inference rate (see [`humanoid-policy-approaches.md`](./humanoid-policy-approaches.md)).
✅ Chunk size is **10 ticks per `act()`** (`config.yaml:89`); the planner returns 6-tick holds while
planning. Anything that cannot hold the rate *even with* chunking is disqualified by arithmetic,
not by tuning.

### 2.3 ⚠️ Latency has a second, sharper limit inside `reset()`

✅ The safety watchdog has `stop_after_s = 1.0` (`safety.py:69`). A `reset` that takes over a second
risks a latched `safe_stop`. A deterministic instruction resolver runs in microseconds and is safe.
**An LLM call in `reset()` is not** — if we ever add a model fallback it goes in a background thread
and the deterministic answer is returned immediately.

---

## 3. The facts that constrain the design

### 3.1 The scene is small, but the bins are *far*

✅ From `scene/scene.yaml`:

- Bins **21 × 21 cm outer**, measured on the real boxes 13 Nov. ⚠️ Wall height **6 cm** and 3 mm
  thickness are **file defaults, not measured**.
- Bin centres: centre bin at pelvis **r = 0.505 m**, **side bins at r = 0.568 m**.
- Reach: **comfortable ≈ 0.40 m, straight arm ≈ 0.55 m.**
- Objects **45 mm** for every shape, 2.5 mm fillets, spawned at pelvis x 0.26–0.35, |y| ≤ 0.17,
  min 8 cm apart, yaw −45…+45°, **3–5 objects**.
- Counter surface at pelvis **z = +0.01** (default, to be measured).
- ⚠️ `sample_layout` defaults to `n_max=6` while `scene.yaml` says 3–5 — a caller that forgets
  `layout_kwargs()` draws up to **6** objects (`tasks.py:54` vs `scene.yaml:70`).

🔴 **Never aim at a bin centre — and the baseline already does not.** ✅ Measured locally,
14 Nov, from `scene.yaml` plus the baseline's own `drop_inset`:

| bin | centre r | nearest inner point | **+5 cm inset — what the baseline aims at** |
|---|---|---|---|
| left / right | **0.568 m** | 0.433 m | **0.498 m** |
| centre | **0.505 m** | 0.403 m | **0.453 m** |

against a **comfortable reach of 0.40 m** and a **straight arm of 0.55 m**.

- **A side bin's centre at 0.568 m is past the straight arm.** Aiming there aims at a point the arm
  cannot reach, and `scene.yaml` says so itself: release in *"the near, inner part"* of the box.
- ✅ **`drop_inset: 0.05`** (`config.yaml:56`) — *"release this far inside the inner walls **at the box
  point nearest the pelvis** (null = centre)"* — moves the target to **0.498 m for a side bin**, which
  is **inside** the straight-arm reach. The baseline is already doing the right thing here.
- So the usable release zone is a **strip along the near inner wall**, running from the near inner
  edge at 0.403–0.433 m (essentially *at* the comfortable-reach boundary) out to about 0.50 m. That
  is the whole budget: a few centimetres deep, not a 21 cm box.
- And the object must clear a **6 cm rim** while being only 4.5 cm tall: **nothing can be swept or
  pushed in, it must be lifted over.** ✅ The planner's release height is
  `z_drop = counter_z + wall + H/2 + drop_above_rim` = **0.1325 m** in pelvis z
  (`planner.py:342`, `drop_above_rim: 0.04`), carried at **0.170 m**.

#### 3.1.1 ✅ This also settles the brief-vs-code contradiction about where the bins are

The brief says the boxes are *"40 cm from the robot's torso"*; `scene.yaml` puts the centre bin at
pelvis **r = 0.505 m** and the side bins at **0.568 m**. **Both are right, and they are measuring
different things:** the **nearest inner point of the centre bin is at r = 0.403 m.** That is the
brief's 40 cm. The brief quotes the near edge as a human would describe it; `scene.yaml` quotes bin
centres in the pelvis frame, which is the frame IK and the safety filter work in.

> **No mentor question needed. Use the pelvis-frame numbers above, and read the brief's 40 cm as the
> near edge.**

#### 3.1.2 ⚠️ `check_layout.py` reports *"bins: 0 % reachable"* — and that is a frame artefact, not a wall

✅ Running the organisers' own `scene/check_layout.py --samples 40000` locally gives
**`cube region: 100 % of cells seen + reachable`** and **`bins: 0 % of cells seen + reachable`**, and
✅ `test_scene.py`'s corresponding assertion is marked `@unittest.expectedFailure` with
*"KNOWN 2026-11-05: default counter is out of reach -- see check_layout.py"*.

**Do not read that as "the bins cannot be reached."** Three reasons:

1. ✅ `palm_cloud` samples the **`<side>_hand_palm_link`** position, while the planner's IK targets the
   **grasp point**, which `point_in_palm [0.099, -0.084, 0.015]` puts **0.131 m** away from the palm
   link. I measured the nearest palm sample to each release waypoint at **69–88 mm** — *smaller than
   that offset*, so the metric cannot resolve the question it is being asked.
2. ✅ The cloud is sampled with the **waist at neutral** and uniform random arm joints, so it is a
   crude envelope, not an IK result.
3. ✅ **The baseline scores 76–82 % in sim, which means it does release into bins.** That is an
   empirical refutation of the pessimistic reading.

**What the check *does* tell us, and it matters:** the near field is `r` — **reachable but not seen**
— which is the same clipping as §3.6.2 seen from the other side, and the bins are `v`/`b` — **seen
but outside the palm envelope**, i.e. at the edge of what the arm can do. **Treat every release as a
near-limit motion**, which is exactly why §3.1's strip along the near inner wall is the target.

✅ Counterweight to all that: **`inside_margin` is 0.0** (`scene.yaml:75`), so an object resting
exactly on a bin's inner edge **counts as in**. The boundary is generous; the *reach* is not.

✅ **Knocked off** = centre outside the 110 × 60 cm table top, or more than 3 cm below the counter
(`scene.py:229-230`, `off_counter_drop: 0.03`). **No time component** — an object that falls once is
−1 from then on.

### 3.2 The grasp: three ways to pinch a prism, two of them bad

The organisers' own ceiling note is the anchor. ✅ `sim/README.md:77`: their **oracle, with exact
privileged poses and shapes, reaches only ~84 % on rung 1**, noting *"needs >90 %: triangle grasps
(thumb vs one finger spins the prism)"*.

The mechanism is derivable:

- ✅ `scene.yaml`: *"cube side = cylinder diameter = triangle ALTITUDE (flat-to-apex)"*, all 45 mm.
  An equilateral triangle of altitude 45 mm has sides **s = 2h/√3 ≈ 52 mm**.
- ✅ Dex3-1 index and middle **flex only, no sideways spread**; the thumb's 3 joints do the
  opposition. Fingertip repeatability ±2 mm (Unitree).
- ✅ Baseline README: index and middle tips are **57 mm apart**, so a 45 mm object wedges between
  them and the pinch is **thumb vs ONE finger**.

| Pinch | Span | Why |
|---|---|---|
| **two faces** | — | faces meet at 60°, so squeezing **wedges the prism out**. *(grokbot: needs μ ≥ 0.58 even at perfect yaw — unverified)* |
| **two corners** | **52 mm** | point contact both sides → **it spins**. This is the organisers' named failure |
| **one corner vs the opposite face** | **45 mm** | the only good one: one flat contact, one locating apex, narrowest span |

A triangle has no parallel faces, so **flat-against-flat does not exist** — apex-vs-opposite-face is
not a compromise, it is the optimum. *(inference, support widths of an equilateral triangle.)*

> *(inference)* **The best and worst grips are 30° apart in wrist yaw, and 3-fold symmetry
> guarantees the nearest good direction is at most 30° away from any starting orientation.**

✅ **And it is affordable.** 30° = 0.524 rad against a per-tick arm cap of ~0.033 rad is **~16 ticks
≈ 0.53 s at 30 Hz.** Half a second per prism against a 120 s episode is nothing.

#### 3.2.1 ✅ There are TWO grasp failures, and the organisers already fixed one of them

`oracle.py:48-50`, MIT:

```
# Pinch thumb vs ONE finger: the Dex3 index/middle tips are 57 mm apart (URDF), so with the object
# centred between them (classical `auto` grasp point) a 45 mm object wedges between the two fingers
# and is never released (carried back home; seen with cylinders).
# Shift the grasp point 15 mm sideways in the palm (palm z).
```

| failure | symptom | cause | fix |
|---|---|---|---|
| **wedging** | object **never released**, carried back home — *seen with cylinders* | object centred in the 57 mm gap | ✅ **shift the grasp point 15 mm sideways in the palm** — written, MIT, free to take |
| **spinning** | prism rotates out of the pinch | corner → point contact | **wrist yaw onto an altitude — ours, not in the repo** |

**Take the 15 mm offset. Keep the yaw work for the prism.**

#### 3.2.2 🔴 The yaw the baseline computes cannot align a prism, by construction

✅ `perception.py:190-196`:

```python
def _yaw(xy):
    """Principal direction of the top-face points, folded to [-pi/4, pi/4) (square symmetry). Info only."""
    (_, _), (_, _), ang = cv2.minAreaRect(xy.astype(np.float32))
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
```

Two things in that function decide our design, and **the first one was misread in an earlier draft
of this handbook, so read it carefully:**

1. ✅ **It runs on metres, not pixels.** The caller is `_yaw(top[:, :2])` (`perception.py:149`),
   where `top` is the **3D top-face point cloud in pelvis coordinates**, selected by
   `top_percentile` / `top_band` from the depth image. So `minAreaRect` operates on a **metric**
   point set. No projection error, no pixel quantisation, no foreshortening.
2. ✅ **The fold is `% (π/2)` — square symmetry** — and the docstring says *"Info only."* For a
   3-fold prism that modulus is simply wrong (3-fold is 120°; the grasp cares mod 60°), and the
   planner never reads the value anyway.

**`minAreaRect` is still the right primitive, but its raw angle is ambiguous** mod 90° (which axis
is the altitude?) and mod 180° (which way does the apex point?). For an equilateral triangle the
min-area rectangle has one side flush with a triangle side, so its axes are *(along a side)* and
*(along that side's altitude)* — **the altitude direction is one of the two rect axes**, which is
what makes the fix cheap.

> **Both ambiguities are resolved by one number.** The centroid sits h/3 from the flush side and the
> rect centre at h/2, so **the centroid is offset from the rect centre by h/6 = 7.5 mm, pointing
> directly away from the apex.** Apex direction = *centroid → rect centre*; the pinch axis is the
> altitude through it.

✅ **And because `_yaw` works in metres, 7.5 mm is a metric measurement against a 52 mm-sided
triangle** — about 14 % of the side length, read off the top-face cloud. ⚠️ What has to be checked
is therefore **not pixels but point density**: how many depth points land on a 45 mm top face at
0.47–0.53 m, and whether `top_band` keeps enough of them for a stable centroid. That is the one
measurement this fix waits on.

**Do the disambiguation before the rotation.** Rotating to the raw angle is *worse* than not
rotating, because it will confidently pick a corner-to-corner pinch.

#### 3.2.3 The same primitive gives a shape classifier for free — with one guard

Blob area ÷ min-area-rect area, on the **2D** blob:

| Shape | Fill ratio | |
|---|---|---|
| Cube | **1.00** | a square fills its own bounding rect |
| Cylinder | **0.785** | π/4 |
| Triangle | **0.50** | ½·base·height over base·height |

Three well-separated values from quantities already computed. No training, no new sensing.

⚠️ **This one IS a 2D measurement, so the pixel budget applies to it** (§3.6) and so does the
clipping risk: ✅ **the near edge of the spawn region sits exactly on the bottom image border**
(§3.6.2), and **a clipped blob gives a wrong ratio.** The guard is one line — does the blob touch
row 0 / row 479 / column 0 / column 639 — and the response is to **abstain on shape rather than
guess**, or to relocate the object first. ✅ The baseline already has a relocate path
(`test_classical.py:134 test_cross_body_object_is_relocated`).

⚠️ Also: the head camera looks down at ~47.6°, not straight down, so the top face is seen obliquely
and the ratios are foreshortened; objects have 2.5 mm fillets. **Measure all three ratios against
`mock_world.Renderer` before relying on them** — the gaps (0.50 / 0.785 / 1.00) are large, so the
method likely survives, but the thresholds are not ours to guess.

**Why the classifier stays on the list even though rung 3 is probably dead (§1.3):**
1. ✅ `scene.resolve_bin` accepts shape-keyed mappings and `STARTER_PATHS` says harder finals may add
   shape words (§3.7) — being shape-blind would score zero on those objects;
2. **the prism fix needs to know it is a prism** (§3.2.2). The yaw disambiguation is shape-specific.

> 🔴 **Perception work and a trained policy are both capped by this section.** A perfect vision
> stack cannot pass ~84 % on rung 1. **The grasp is the ceiling.**

### 3.3 Three shapes, three different failure modes *(inference, from the geometry)*

| Shape | Failure mode | What it demands |
|---|---|---|
| **Cube** | essentially none — flat faces both sides, 45 mm | nothing |
| **Triangle** | **grasp** — corner contact spins it | wrist yaw aligned to an altitude (§3.2.2) |
| **Cylinder** | **wedging** (✅ the organisers' own note names cylinders) **and rolling** — an object still moving *inside a bin* scores 0 | the 15 mm palm offset (§3.2.1), and **place it early** so it settles |

### 3.4 🔴 Time is the constraint everyone underestimates, for two independent reasons

Both reasons were read from the source, they **stack**, and together they are the most practical
finding of the research night.

#### 3.4.1 The real robot loads the slow motion profile

✅ From `baselines/classical/config.yaml` and `planner.py`:

| | arm vel | arm accel | hand vel | used by |
|---|---|---|---|---|
| **`sim` profile** | **0.97 rad/s** | **8.0 rad/s²** | 2.5 rad/s | the sim runs the 76–82 % was measured on |
| **base (profile `none`)** | **0.50 rad/s** | **1.5 rad/s²** | 1.2 rad/s | ✅ **`run.py --backend real`** (`config.yaml:8`) |
| safety filter ceiling | 1.0 rad/s | — | 3.0 rad/s | `safety.py:49-50` |

The planner's move duration is `ta + S/vmax` with `ta = vmax/amax` (`planner.py:58-73`), so:

- **per-move overhead**: sim `0.97/8 = 0.121 s` → real `0.50/1.5 = 0.333 s`, **2.75× worse**
- **joint travel**: `S/0.97` → `S/0.50`, **1.94× longer**
- **hand ramps** roughly double (2.5 → 1.2 rad/s)

*(inference, from the published gains and the planner's own formula — every input is a read
number.)* The two terms do not scale alike, so the ratio depends on the move. Putting them
together for a typical 0.5 rad joint move: sim `0.121 + 0.515 = 0.637 s`, robot
`0.333 + 1.0 = 1.333 s` — **2.0–2.1× slower end to end**, which is the number to use, not the
1.94× travel-only figure.

> ⚠️ **Correction, and it loosens the schedule rather than tightening it.** This section used to
> put five objects at *"~19 s each, ~97 s of 120 s"*. Recomputing from `ta + S/vmax` over a whole
> pick-and-place cycle gives **≈31 s for five objects on the fast `sim` profile and ≈61 s on the
> robot's base gains**, plus policy latency → **≈76 s of the 120 s budget**, not 97 s. Time is
> still the binding constraint and there is still no room for several retries, but the earlier
> number overstated it and would have bought panic optimisation we do not need.

> 🔴 **The profile a submitted container actually gets is the FAST one.** `config.yaml:10` is
> `profile: sim`, `baseline.Dockerfile` sets no `AALTO_PROFILE`, and the runner passes no `-e`.
> `AALTO_PROFILE=none` is what selects the robot's base gains. So **our sim numbers are on the
> fast clock unless we deliberately ask for the slow one**, and the stage runs the slow one.

#### 3.4.2 And the sim score itself was measured on a clock the organisers call 1.8× wrong

✅ `sim/README.md` ("Still open"): the earlier grasp tuning *"ran ~1.8x too fast in physics time
(wall pacing at RTF ~0.55); re-check grasp parameters on the sim clock."* The budget is denominated
in **sim seconds** (*"`command_hz` count sim seconds"*, *"120 s sim ~ 220 s wall"*), and RTF is 0.70
alone, ~0.59 with two sims, 0.50–0.53 with three.

✅ **Confirmed by the organisers directly** (Discord, via Oliver, 14 Nov): *"Oracle on sim clock:
NOT re-measured since the clock fix. The ~84 % was pre-fix; triangle grasps were the main miss. Our
76–82 % is in the same range."* And ✅ the oracle's own `ORACLE_OVERRIDES` carry
`motion.arm_max_vel 0.97, arm_max_acc 8.0, hand_max_vel 2.5` (`oracle.py:38-48`) — the fast profile.

> 🔴 **So the 84 % ceiling and the 76–82 % baseline are both fast-profile, fast-clock numbers.
> Nobody — including the organisers — knows either score on the clock we are scored on.**

This is §3.4.1's effect *in the sim*, stacking on top of it rather than replacing it: that one is
about which gains the real backend loads, this one about the sim's own pacing.

⚠️ What remains open: whether the published *"10–20 s per pick"* was wall or sim time. **This is
what step 0 measures** (§5), and it decides whether 5 objects fit in 120 s at all.

#### 3.4.3 ✅ And when it runs out of time, the baseline gives up completely

```python
if t + plan.duration > self.time_limit:
    ...  self.state = "done"
```
`policy.py:220-223`

**One failed time check ends the baseline for the rest of the episode.** Candidates are ranked
**nearest-bin first** (`policy.py:191-204`) with **no fallback to a shorter one**. The moment the
cheapest-looking remaining plan does not fit, it stops — even when another object would have fitted
in the time left.

> ✅ **The cheapest point in the whole challenge after the instruction:** rank the remaining
> candidates by **estimated duration**, try the **shortest that fits**, and stop only when *nothing*
> fits. The estimate already exists — `plan.duration` is what the failing check reads.

#### 3.4.4 ✅ The sim gives a 3-second settle grace that the robot does not

```
sim:  evaluate.py  --settle  default 3.0   "max s to wait for objects at rest before scoring"
real: scored at the instant the episode ends
```
`evaluate.py:199-200`, default at `:314`; rules line 25 *"at rest at episode end"*.

**The sim waits up to 3 sim-seconds for objects to settle before scoring.**

> ⚠️ **UNVERIFIED, and it was stated as fact here.** *"The real robot does not"* is **not in any
> code we have read.** `evaluate.py` is the sim path only; there is no real-robot scorer in the
> repo to check. It is an inference from rules line 25 (*"at rest at episode end"*) plus the
> absence of any stated grace — reasonable, and still an inference. What would settle it: ask the
> organisers whether the stage run waits for rest. **Until then plan for no grace, but do not
> quote this as a measured fact to anyone.**

On that assumption an object still rolling at the buzzer **scores 0 on stage and often +1 or +2 in
sim** — a bias in the exact direction that hurts: it inflates the number that qualifies us and not
the number that pays.

> **Rule: finish with every object settled and the hand clear, and re-score our own runs with
> `--settle 0` to see the real-robot number.** The gap between the two is our rolling-object debt.

### 3.5 The hand: force is not a lever, so the grip must be geometric

#### 3.5.1 ✅ The gains are fixed and identical for every team

✅ **Organisers (Discord, via Oliver, 14 Nov):** *"Dex3 gains: real runner uses **kp 1.5 / kd 0.2,
fixed, same for all teams**. We only send joint targets."* This matches the code: `hand_kp=1.5,
hand_kd=0.2, hand_hz=100.0` (`interface/backends.py:202`).

> 🔴 **There is nothing to tune. Grip force is not a design variable for anyone in this
> competition** — we send joint targets into a fixed controller.

That reduces a question this handbook spent a long time on to a prediction problem. For the record,
because it still shapes what to expect: the sim's implicit finger actuators are **stiffness 100,
damping 10, effort_limit 300** with **self-collisions off** (`unitree_sim_isaaclab robots/unitree.py`,
not overridden in the Aalto `env_cfg.py`), against real limits of **2.45 N·m for thumb_0 and 1.4 N·m
elsewhere** (`interface/joint_spec.yaml`). ⚠️ Whether stiffness 100 means a **67×** gap or a
**~1.2×** one depends on Isaac Lab issue **#4918** (implicit-actuator joints reported 50–57× softer
than stated, a maintainer calling it plausible; NVIDIA separately warns USD and PhysX stiffness
units differ by 180/π). **Never quote "the sim hand is 120–210× stronger" as a fact, and never on
stage.** The honest line: *the sim's force ceiling is two orders of magnitude above the robot's, and
whether it is ever reached depends on a gain whose units are unresolved.*

A **step test** — command a known position error against a blocked finger, compare `hand_q`
response — settles it. It is now **informational, not blocking**, precisely because no gain is ours
to change. Do it after the score measurement, not before.

> **The design rule holds under every reading: command an aperture, not a squeeze.** Close until the
> finger stops tracking and hold just past contact, rather than commanding full closure and letting
> the solver invent force. NVIDIA's own warning that with an implicit drive **extra stiffness does
> not make the drive provide more force** means turning a gain up is not a fix even where it looks
> like one.

#### 3.5.2 🔴 The real robot grips WORSE than the sim

✅ **Organisers (Discord, via Oliver, 14 Nov):** *"Pad friction: not measured. Pads are rubber. Sim
uses static 1.2 / dynamic 1.0 (grippy) → **expect a bit LESS grip on the real robot**."*

And ✅ objects use PhysX combine mode **"max" with static friction 1.2** (`scene_cfg.py:228`), so
**a pad material below 1.2 changes nothing** in sim — the object's value wins.

> 🔴 **This is the single most load-bearing sentence we have about the grasp.** Any grip that holds
> because of friction margin degrades **on the exact run that pays**. The only defence is
> **geometric stability** — the 45 mm apex-against-flat-face pinch of §3.2 — not a firmer squeeze,
> which §3.5.1 says we cannot command anyway.

One forum user reports slips at μ = 10 in Isaac; sim grasp success is fragile in both directions.

⚠️ **Self-collisions are off in sim**, the safety filter does not check them, and **staff e-stop for
them** — a gate failure with no warning anywhere in our loop. Same for fingertip-to-table.

#### 3.5.3 ✅ The free grasp sensor, and it does not depend on the gain question at all

The Dex3-1 has 33 pressure sensors and ✅ **the interface exposes none of them** (FAQ). But the
observation **does** carry **`hand_q` (14)** — measured finger positions — while we send
`hand_q_target`.

> **`hand_q_target` − `hand_q` is a contact signal, for free, every tick.** Fingers stall short of
> target → something is between them. Fingers reach target → we closed on air, or the object is gone.

With no new sensing and nothing in the way of the 27 Hz gate, that gives:

- **grasp success / failure detection** — the thing tactile data would have given us;
- **a drop detector** mid-carry (the lag disappears while we are still holding);
- **a good-grip check**: a prism pinched along its altitude stalls at ~45 mm, across the corners at
  ~52 mm, so **the stall aperture says whether we got the good grip or the bad one — before we
  lift** (§3.2).

✅ **The organisers built the hook and left it off:** the safety filter already has a
target-vs-measured mechanism, **`hand_max_lead`, default `None`** (`safety.py:52`, applied at
`:228-230`), and **nothing in the baseline reads `hand_q` except to seed the command**
(`policy.py:105`).

⚠️ Two cautions. **A blocked finger lags its target whatever the gain is**, so the signal's
*existence* survives the #4918 question — only its *size* moves; but it must be detectable above the
solver's tracking error, so **log `hand_q_target − hand_q` during step 0 and look before building on
it.** And a relayed lead says **real `hand_q` reads slightly past zero at rest and can glitch**, so
any lag code needs a clamp and a median filter. ⚠️ A second lead says `hand_max_lead` **may be
enabled in the finals** (*"final safety values will not be looser"*) — if it is on, a deliberate
squeeze-past-contact posture **clips**, and the gate wants zero clips. **Another argument for
aperture over squeeze.**

### 3.6 Perception: the numbers are measured, and the margins are thin

#### 3.6.1 ✅ Use `Calibration.load()`, never a literal

`interface/calibration.yaml` is the source of truth and **`INTERFACE.md`'s fx = 615 is stale**:

```yaml
head:
  intrinsics: {width: 640, height: 480, fx: 604.827, fy: 604.298, cx: 325.042, cy: 250.927}
  # D435I s/n 332522071513 factory colour 640x480 (rs2, 2026-11-12)
  T_pelvis_optical: [... 0.053660], [... 0.017530], [... 0.473870]
```

**Measured from the actual serial number on 2026-11-12**, not a placeholder, and the sim reads the
yaml. The camera sits at pelvis **(0.054, 0.018, 0.474)** looking down **47.6°**, so the object band
is **0.47–0.53 m from the lens**. A 45 mm object is therefore **51–58 px**, and 7.5 mm is ~9.7 px,
or **~7 px after foreshortening** — relevant to the 2D shape classifier (§3.2.3), **not** to the
yaw, which works in metres (§3.2.2).

#### 3.6.2 🔴 The near edge of the spawn region sits exactly on the bottom image border

✅ Measured locally, 14 Nov: `scene/test_scene.py:224 test_cubes_and_bins_visible` — whose class
docstring states *"With the default calibration (URDF head pose), **every object must be in the head
image**"* — **fails** on the shipped calibration. The four points outside are the **bottom corners
(z = 0)** of an object at the **near-y edge** of the region (`region_y = [0.12, 0.19]`,
`region_x = [0.28, 0.62]`). Via `cal.project` they land at **v = 480.2 in a 480-px-high image** —
out by two tenths of a pixel. The far corners sit 21 px from the left border, 22 px from the right.

Object **tops** are comfortably in frame; the **bottom edge of a near object is clipped**.

*(inference)* `sim/dr/train.yaml` jitters the head camera pose by ±0.01 m and **±1.0°**. At
fy ≈ 604 one degree is **≈10.5 px**. A 0.2 px margin under ±10 px of jitter means the near row of
the region is clipped on some episodes and clear on others.

✅ **`perception.py` has no border check of any kind** — verified by grep: no `border`, `edge`,
`clip` or image-shape test anywhere in it.

> **Guard: one line.** Does the blob touch row 0 / row 479 / column 0 / column 639? If yes, **abstain
> on shape** rather than guess, or relocate the object first (the relocate path exists).

#### 3.6.3 ✅ The real DR numbers — and the colour panic was overstated

| | range | note |
|---|---|---|
| object **hue** jitter | **±0.02 turn = ±7.2°**, per object | small |
| object sat / val scale | ×[0.8, 1.15], clipped | red can drop to S≈0.68, V≈0.63 |
| object mass / friction | ×[0.7, 1.5] of 50 g · ×[0.75, 1.1] | |
| key / dome light | ×[0.6, 1.4], ±8 % · ×[0.5, 1.5], ±5 % | |
| counter / bin colour | ±20 % / ±8 % per channel | **bin colour moves, bin geometry never does** |
| head cam pose | **±0.01 m, ±1.0°** | the dominant error term, see below |
| head RGB | gain ×[0.9, 1.1] per seed, noise σ = 3.0 (0–255) | |
| head depth | σ = 1.0 mm + 0.002·depth, **1 % of pixels zeroed**, invalid = **0** | ≈2 mm at 0.5 m |
| layout | ±0.005 m disc, ±5° yaw | |

✅ Base colours (`scene.yaml:34-36`) are red H≈0°, yellow H≈50°, blue H≈226°, so with ±7.2° of
jitter **no two of them can collide** — they stay 36°, 127° and 160° apart. **Hue thresholds on the
objects are fine.** Three real risks replace the imagined one:

1. 🔴 **The orange bins sit between red and yellow.** Bins are H≈24° (`scene.yaml:53`); the
   baseline's red window is OpenCV hue 0–8 (plus 170–180) against bins at ≈12 — **a margin of about
   4 OpenCV units, the narrowest gap in the whole colour system.** Exclude bins by **position**,
   never by hue alone. *(inference from the two read values.)*
2. **Low saturation and value corners**, not hue, are where a red goes dull-brick. Tune against
   drawn `train` samples, never one nominal render.
3. ✅ **Camera pose jitter beats perception error.** Stated perception error is **≈4 mm** after the
   HSV fix (`baselines/classical/README.md:32`), while ±1 cm and ±1° together are worth **≈15 mm**.
   **The camera pose is the dominant error term** — which is direct support for Timi's
   self-calibration-from-the-bins idea: the bins are at known fixed positions and **their geometry
   is never randomised**, so they are a free calibration target in every frame.

✅ **Wrist cameras are not randomised at all**, and they are uncalibrated presets with *"no measured
mount yet"* (`aalto_task/scene_cfg.py:11`) — null extrinsics, placeholder intrinsics. **Unusable.**

#### 3.6.4 ⚠️ The bin-rim false positive — real, but narrower than it first looked

`baselines/classical/perception.py:100-118`:

```python
keep = self._bin_hulls_mask(...)          # 2D projected bin hulls
if depth is None: return keep
if valid.mean() < min_valid_frac: return keep
keep[:] = 255    # with depth: holes stay "unknown" (kept); valid pixels are judged in 3D
ok = (h >= 0.25) & (h <= c["height_range"][1])
half = outer_size/2 + c["bin_pixel_margin"]
for ctr in bins.centers: ok &= ~(inside that footprint)
```

1. ✅ **With valid depth the 2D bin mask is thrown away** (`keep[:] = 255`).
2. ✅ **The height test does not reject a bin wall.** `height_range: [0.4, 2.0]` is in object sizes
   (`config.yaml:29`, comment claims it *"rejects hands, bin walls"*); a 6 cm rim is
   **6/4.5 = 1.33 sizes — inside the band.**
3. ✅ **`bin_pixel_margin` is 0.004** — a 4 mm footprint margin against the ≈15 mm camera-pose error
   of §3.6.3.
4. ✅ **Depth holes are kept as "unknown" by design**, and real depth has holes near bin walls.

✅ **But a second guard survives the depth path:** `_reject` (`perception.py:170-187`) runs **after**
3D reconstruction and rejects any detection whose top centre lands inside a bin footprint +
`bin_exclusion_margin`, returning `in_bin:<name>`. **That one is not discarded.**

> **So the honest exposure is narrow: a rim blob whose reconstructed centre lands *outside* the
> footprint + margin — the outer face of the wall.** Not "the robot will try to pick up a bin".
> Still worth one rendered frame, and the fix is free: keep the 2D hull mask as a **union** with the
> 3D test instead of discarding it, and widen `bin_pixel_margin` to cover real jitter.

✅ Minor, same file: the code hardcodes the lower bound as `h >= 0.25` while `config.yaml` says
`height_range: [0.4, 2.0]`. **The configured lower bound is ignored.**

✅ And one behaviour to design around: **a blob matching no configured colour is never formed at
all** — perception loops only over configured colours (`perception.py:51-58`), so unknown-coloured
pixels are silently dropped rather than reported as an unknown object. ⚠️ A relayed lead says **a
"distractor object" is the named harder-layout knob** and that finals episode 3 is unseen
instruction *and* layout. A distractor is therefore either invisible to us or mistaken for cargo —
**worth a deliberate "unknown object" branch.**

#### 3.6.5 ⚠️ Object depth holes are unmodelled in `train.yaml` — and the numbers I first quoted were wrong

**An earlier version of this section claimed the organisers ship a model of real camera noise that
is harsher than the training DR. That was wrong and it is retracted.** Oliver's session caught it.
✅ `data/drop1/capture.py:133` says what it is in its own docstring:

```python
class MockSource:
    """Synthetic static scene for offline tests: grey counter + coloured objects at their projected
    positions, Gaussian RGB noise, depth = counter plane / object tops with noise and random holes."""
    def __init__(self, seed=0, rgb_sigma=2.0, depth_sigma_mm=3.0, hole_p=0.02, obj_hole_p=0.2):
```

It is a **test fixture**. `test_drop1.py:22` passes `3.0 / 0.02 / 0.25` so the test can check that
`stats.py` recovers the values it injected. Those numbers are **plausible placeholders chosen to be
measurable, not measurements of the D435i.** Treating them as a camera model was exactly the
over-read §9 exists to prevent.

**What actually survives, and only this:**

1. ✅ **`train.yaml` has no object-specific depth-hole parameter at all**, while the capture tooling
   has the *concept* of one — `holes = g.random(dep.shape) < np.where(self.obj_mask, self.obj_hole_p,
   self.hole_p)` (`capture.py:180`). The **structural** asymmetry is real; the magnitude is unknown.
2. ✅ **`stats.py` computes per-object `hole_frac` and `hole_frac_always`** (`:142`), i.e. the
   organisers built a measurement for object holes — which says they expect them to matter. It does
   not say how large they are.
3. ✅ `train.yaml` does model **1 % holes with `invalid = 0`**, so missing depth exists in training
   already, just not concentrated on objects.

> **The design consequence is unchanged, and it never depended on the numbers:** §3.2.2's prism fix
> reads the **centroid of the top-face depth cloud**, so it must tolerate missing points whatever
> their rate — a **trimmed or median centroid** rather than a mean, an explicit **minimum point
> count** below which we abstain, and the area-ratio shape classifier (§3.2.3) read from the **2D
> colour blob**, which has no holes, rather than from the depth mask. All three cost nothing and are
> right under any hole rate, which is why they stay.

⚠️ **The real hole rate on a matte PLA prism is now an open question with no number** (§8), and the
only thing that answers it is a real drop or a render measured on the GPU box. ✅ One line from
`stats.py` is worth having in mind before any colour tuning, though: *"Matte PLA under unknown light:
wide hue bands, saturation floor."*

#### 3.6.7 🟢 The noise model that counts, and why the camera POSE is the whole problem

`sim/dr/train.yaml` is the organisers' actual noise model, and the finals run in their sim with
their DR, so this is the distribution we are graded against. ✅ Verbatim:

```yaml
camera:
  rgb_noise_std: 3.0        # 0-255
  rgb_gain: [0.9, 1.1]
  depth_noise_mm: 1.0
  depth_noise_rel: 0.002    # D435 ~0.2 % of range   <- their comment
  depth_hole_frac: 0.01
  pose_jitter_m: 0.01       # head extrinsic error (spec: +-1 cm / +-1 deg)
  pose_jitter_deg: 1.0
```

**The two halves behave completely differently, and that is the finding.** ✅ `dr.py:102-105`
samples `pos_offset` and `rot_offset_rad` **once, into the episode's params dict**:

```python
"pos_offset": [float(rc.uniform(-pj, pj)) if pj else 0.0 for _ in range(3)],
"rot_offset_rad": [float(rc.uniform(-aj, aj)) if aj else 0.0 for _ in range(3)],
```

whereas `corrupt(rgb, depth, cam, rng)` at `dr.py:131` draws the pixel noise **fresh every frame**:
`d += rng.normal(0.0, 1.0, d.shape) * (dn + dr * d)`.

| Error source | Size at the 0.47-0.53 m object band | Behaviour |
|---|---|---|
| depth noise | `1.0 + 0.002 x 500` = **2.0 mm**, per pixel, zero mean | per **frame** — averages away over a ~50x50 px top face and over ticks |
| camera position offset | **±10 mm**, uniform, per axis | sampled **once per episode** — a constant bias, averaging does nothing |
| camera rotation offset | ±1° → `0.5 x tan(1°)` = **±8.7 mm** lateral at 0.5 m | same: constant for the episode |

⚠️ So the dominant error is **5-10x the depth noise, systematic, and immune to every filter**, and
note it is `uniform`, not Gaussian — ±10 mm is a hard bound, not a sigma. Against §3.2's measured
grasp slack (a 45 mm pinch inside a 57 mm hand gap, so ~6 mm per side) **the pose bias alone can
exceed the slack that makes the grasp work.**

> **🟢 The lever, and it is free: the bins are at known positions.** `scene.yaml` gives the bin
> geometry exactly, and §3.1 measured it. The episode's camera offset is one constant 6-DoF
> transform, and the bins are large, fixed, always in frame. Observe them, compare to the known
> layout, solve for the offset, and apply the correction to every object detection for the rest of
> the episode. That turns the largest error term into a per-episode calibration computed from data
> already in the first frame — and the baseline does none of it (`perception.py` trusts
> `calibration.yaml` as given).
>
> ⚠️ Unverified: whether the bins are visible enough to constrain all 6 DoF, or only the 3
> translations. Fit translation first; it is the ±10 mm term and the easy one. Measure on the GPU
> box before trusting a rotation fit.

What this replaces: the retracted §3.6.5 reached for a fixture to argue the DR was optimistic. It
did not need to — the DR's own structure says where the error lives, and it is not in the pixels.

#### 3.6.6 ⚠️ We have NO real-robot data — correction to an assumption we were all making

✅ `aaltonomous/data/` is **416 kB and contains no frames.** `data/drop1/` holds the capture kit only
(`capture.py`, `package.py`, `stats.py`, `CHECKLIST.md`, `checklist.yaml`); `test_drop1.py` runs
against `MockSource` — *"mock backend: no DDS, no cameras, no sim"* — and `DATA_CARD.md` is a
**template** with placeholders (*"Drop \<N\>"*, `aalto/<name>`, *"\<camera frame drops ... see
warnings\>"*).

⚠️ Combined with the organisers' *"no major drop tonight"* (venue wifi unusable for goggles teleop),
**the honest position is: there is no Drop 1 to fit anything to.** The expected format when one
arrives is ✅ **LeRobot v3.0** written by `lerobot==0.7.0`, with `observation.images.head_depth` as a
480×640 HEVC 12-bit lossless depth video, **linear 0–4.095 m at 1 mm/code, 0 = invalid**.

**What can be done now, with no data:** read `stats.py`, write down exactly which statistics it
produces, and prepare the **mapping from each of them to a `train.yaml` knob** — including the
`obj_hole_p` knob that does not exist yet. Then the first drop is actionable in minutes instead of
hours.

### 3.7 ✅ The instruction space, exactly

| | |
|---|---|
| Bins | `BINS = ("left", "center", "right")` — ✅ and the brief confirms *"no markings; they are told apart only by their position"* |
| Fixed mapping | `{"red": "left", "yellow": "center", "blue": "right"}` |
| Training templates | **8**, two word orders (colour-then-bin in six; bin-then-colour in *"The {b0} bin is for {c0} objects"* and *"Use the {b0} bin for {c0}"*) |
| Mappings | `rng.permutation(COLORS)` zipped with `rng.permutation(BINS)` → **6** |
| Total training sentences | 8 × 36 = **288** |

#### 3.7.1 🔴 `bin_phrase` is an empty seam, and it tells us what the finals will do

```python
def bin_phrase(b, rng=None):
    """Bins are always named by position ('left'); they all look the same."""
    return b
```

✅ **The function takes an `rng` and does not use it**, and it is called as `bin_phrase(b, rng)` when
the instruction is built. The organisers wrote a hook whose entire purpose is to vary how a bin is
worded and shipped it as the identity function in the training set.

> 🔴 **Match bin *phrases*, not bin words.** *"the one on the left"*, *"leftmost"*, *"the left-hand
> bin"*, *"the far left"* must all resolve to `left`. A matcher keyed on the exact token `left`
> passes every training episode and can fail the ×1.5 final — which is exactly the shape of this seam.

#### 3.7.2 🔴 One of the eight templates never says the word "bin"

```
"Sort the objects: {c0} into {b0}, {c1} into {b1}, {c2} into {b2}."
```

> **Never require the token `bin` to find a pair.** The anchor is colour-to-position **adjacency**,
> not the noun.

#### 3.7.3 ✅ `resolve_bin` accepts four key forms, in order

```python
for k in ((color, shape), color, shape, "default"):
    if k is not None and k in mapping: return mapping[k]
```
`scene.py:46-49`

So a mapping may be keyed on **`(colour, shape)` composites**, colour, shape, or a **`"default"`
catch-all**.

#### 3.7.4 ✅ What the organisers say the finals will actually test

✅ **Organisers (Discord, via Oliver, 14 Nov):** *"Finals instructions: **colour first**. If several
teams handle colour, **shape-based instructions may also be tested**. **No negation**."*

Three consequences:

1. **Colour permutations (6) are the primary target.** Build that first and build it well.
2. **Shape-keyed is a contingent second tier** — and we should assume several teams handle colour,
   so **assume shape words appear.** This is the argument that keeps the shape classifier (§3.2.3)
   on the list now that rung 3 looks dead: going into the finals shape-blind scores **zero** on
   those objects.
3. **No negation.** This removes the one case enumeration reads *backwards* — *"anything but the
   left bin"* looks to a proximity count like support for the pair it denies. The inverting flag and
   the *"one default + one exception"* candidate family both leave the critical path.

⚠️ **Provenance caution, and it is load-bearing:** this is Discord relayed through two hops, not a
line of code. *"No negation"* is a **relaxation**, so if it is wrong we lose points. Treat it as
binding for **priority** (what we build first) and not as licence to break robustness that costs us
nothing — enumeration cannot emit an impossible mapping with or without it.

### 3.8 🔴 Two targets, and a sim fix can cost us the one that gates the prize

**We cannot change the organisers' sim for the held-out evaluation.** Every physics fix —
`solve_articulation_contact_last`, torsional friction, capping `effort_limit`, self-collisions on —
applies only to **our** training and testing copy.

| | runs on | what it decides |
|---|---|---|
| **The qualifier** | the **organisers'** unmodified sim | whether we are in the finalists at all |
| **The prize** | the **real robot** | ✅ the brief: *"The best real-robot score wins."* |

**The trap:** *(inference)* making our copy more realistic makes it **harder**, and a grasp tuned
against our harder copy can score **worse** on theirs — and theirs picks the finalists.

> **The rule: use the realistic copy to choose the design, validate the submitted score on the
> organisers' unmodified sim. Never submit a policy whose numbers only exist against our modified
> physics.**

A geometry-stable grasp (§3.2) is exactly the kind of choice that holds on both, because it does not
depend on which physics is more forgiving.

### 3.9 ✅ Scoring details that decide the endgame

#### 3.9.1 At rest applies to **bins only**

```python
b = self.bin_of(p)
if b is None or speed > float(self.cfg["success"]["at_rest_speed"]): return "counter"
```
`scene.py:239-241`

An object **on the counter is "counter" whatever its speed** — motion costs nothing there. The
velocity test only ever decides whether something **in a bin** counts as in it.

> **An object still moving *inside a bin* loses the whole point (it becomes 0). An object rattling
> on the counter was already 0 and loses nothing.** So the cleanup phase protects **placements**, it
> does not tidy the counter.

✅ `scene.yaml:83`'s *"unless it is inside a bin"* is simply wrong; `RULES_AND_SCORING.md:38`
(*"or still moving at the end"*) agrees with the code. **Where prose and code disagree, the code is
what scores us.**

#### 3.9.2 An object in the hand is judged only by where its centre is

✅ There is **no code path for a held object at all** — nothing in the repo mentions the hand when
scoring (`scene.py:237-242`):

| Centre at the buzzer | Score |
|---|---|
| in the hand, above the bin rim | **0** (counter) |
| past the 110 × 60 cm table top, or >3 cm below the counter | **−1** (off) |
| **inside a bin box, below the rim, at rest** | **counts as in that bin**, held or not |

> **Releasing a last object late over the correct bin beats carrying it to the buzzer.**

⚠️ But the real-robot score is read **by eye**, and a human judge may not count an object still in
the gripper. **Design for the release; treat held-in-bin as a fallback we do not advertise.** (Open
question for a mentor, §8.)

---

## 4. ✅ What the baseline already does, so we do not rebuild it

`planner.py` (460 lines) already has: both arms, a finger-direction hint list tried in turn,
bin-wall clearance, relocation of an object out of the way, a failed-position list so it does not
retry the same spot, `done` when nothing sortable is left, and 6-tick hold chunks that keep the
command rate at 30 Hz. ✅ Sim rung 1, seeds 0–7: **76–82 % sorted, 0 clips, 30 Hz, gate PASS.**

**Three things it does not have**, and each is an item in §5:

1. ✅ **It never reads the instruction** (§1.1) — 83 % of the final.
2. ✅ **There is no post-placement verification at all.** Nothing checks whether the object landed in
   the bin. With final-state scoring that is an unused lever, not a missing nicety.
3. ✅ **It gives up on the first failed time check** (§3.4.3).

⚠️ **The baseline has never run on the real robot** (*"not run yet (M8)"*); the HSV floor, grasp
point and open posture were only ever checked in sim. **Treat 76–82 % as a sim number**, and expect
±10 % between runs.

### 4.1 ✅ Free tuning values from the oracle, and the relay hand-over

`ORACLE_OVERRIDES` (`oracle.py:38-48`) are the organisers' own tuned numbers and differ from the
baseline's: `motion.counter_clearance 0.02`, `motion.max_joint_step 0.6`, `ik.max_tilt_deg 35`,
`ik.random_restarts 4`, `motion.close_hold_s 0.25`, `motion.settle_s 0.3`.

✅ **The relay hand-over is a config trick with nothing privileged in it** (`oracle.py:251-261`): it
deep-copies the scene config, zeroes `wall_height` and replaces the bin centres with hand-over
points, then runs the **stock** planner against that fake scene — so the whole bin-placement
machinery places an object gently at counter level for the other arm. The privileged parts of
`oracle.py` are cleanly fenced into `PrivilegedPerception` and the `privileged_mapping` callable;
**everything else in that file is ordinary MIT code we may read and copy.**

### 4.2 ✅ Grasp config, raw, so nobody re-reads it

`point_in_palm [0.099, -0.084, 0.015]`, approach axis `[0, -1, 0]`, `height_frac 0.5`, fingertip
clearance `0.012`, three finger hints (`[1,0,0]`, `[1,-1,0]`, `[0,-1,0]`) at weight `0.3`; hand open
`[0, -0.5, 1.5, -0.4, -0.4, -0.4, -0.4]`, closed `[0, -0.1, 1.2, -1.0, -0.6, -1.0, -0.6]`, release
`[0, -0.5, 1.5, -0.05, -0.05, -0.05, -0.05]`. Baseline HSV windows (OpenCV units): red `0–8` and
`170–180`, yellow `18–35`, blue `100–130`, with S,V floors 70/70, 70/90, 70/50.

⚠️ A relayed lead: **real arm sag**, no gravity feedforward, estimated **1.5–3.5 cm** against the
12 mm `fingertip_clearance`. Real-robot only, so the qualifier will never show it.

---

## 5. The build order

Each item says **which lane it needs** — and that split is itself a finding: ✅ **all 12 test files
in the repo are CPU-only** (none import `isaac` or `omni`), and `deploy/replay.py` is too. 93 tests
ran on a GPU-less laptop; 79 passed. **Only physics and rendering need the GPU**, so the two lanes
run in parallel and the GPU box is never the bottleneck for logic work.

| # | What | Lane | Why it sits here |
|---|---|---|---|
| **0** | **Measure the unmodified baseline on the sim clock** | 🖥 GPU | Three unknowns, one run. Everything else is measured against it |
| **1** | ✅ **Instruction → mapping** — **DONE** `ae2416f` | 💻 CPU | 83 % of the final; the baseline has no mechanism for it; immune to every clock and physics question |
| **2** | **Take the oracle's free constants** | 🖥 GPU to verify | A one-line fix for a named failure, already written and MIT |
| **3** | **Time triage: shortest-fitting candidate, never give up** | 💻 CPU logic, 🖥 verify | A few lines against a silent episode truncation |
| **4** | **Endgame: settle, and release rather than carry** | 💻+🖥 | The sim's 3 s grace does not exist on stage |
| **5** | **Yaw-aligned prism grasp** | 🖥 GPU | The organisers' own named blocker, and the real robot grips *worse* |
| **6** | **Post-placement verification** | 🖥 GPU | Largest per-object swing available: −1 → +2 |
| **7** | ✅ **Shape classifier** — **DONE on CPU** `c81360f`; **my module superseded by Oliver's `shape_id.py`**, the border guard is his (`77715e8` measured why) | 💻 CPU | Insurance against shape-keyed finals instructions. The win was finding the feature was on the wrong data: **61 % → 100 %**, and that a bin-mask cut inverts it: **100 % → 21.7 %** |
| 8 | Stretch: relay hand-over · DR fitted to Drop 1 | both | Generality story, 25 % |

### 0. Measure the baseline on the sim clock 🖥 **before anything else**

**Why first:** §3.4.2 — every number we would improve against was measured on a clock the organisers
say was 1.8× wrong, and they have ✅ confirmed they never re-measured it. Without this reference,
every "improvement" is measured against an unknown and we cannot tell a gain from a clock change.

`sim.evaluate --task sort_fixed --trials 20`, unmodified. Record score, clip count, command rate,
**and wall vs episode time per pick**. Three free add-ons on the same run:

- **re-score with `--settle 0`** → the real-robot number (§3.4.4). The gap is our rolling-object debt.
- **log `hand_q_target − hand_q` per tick** → decides whether §3.5.3 gives us a grasp sensor at all.
- **`aalto check`** (8 PASS / 0 FAIL expected) and `scene/check_layout.py` → settles §8's reach
  question directly.

**Done when:** we have a number for rung 1 on the sim clock, with and without settle, and a measured
seconds-per-pick. **Owner: Timi** (Verda RTX PRO 6000, running).

### 1. Instruction → mapping 💻 ✅ **DONE, 14 Nov 00:50** — `policy/instruction.py`, commit `ae2416f`

**Measured:** 288/288 regenerated training sentences (8 verbatim templates × 6 colour orders ×
6 bin orders), 294/294 of Oliver's held-out corpus, 23/23 adversarial sentences written against
this file by an audit agent.

> ⚠️ **The 294/294 is NOT an out-of-sample score.** That corpus found two real bugs in this file —
> `"blocks"` parsed as a shape word, and `"the remaining bin"` read as the catch-all when it means
> the exact opposite — so the resolver has been tuned against it. A **second blind batch** has been
> requested from Oliver (different seed, categories he has not used, not shown to us) for a real
> out-of-sample number. Until that lands, treat 294/294 as a regression suite, not a result.

**What it handles that the baseline cannot:** ordinals and numbered bins (`bin 1/2/3`,
`first/second/third`, `box A/B/C`), `"the one in between"`, many-to-one (`"red and yellow go in the
left bin"` — `resolve_bin` is a dict lookup, so the mapping is a **function, not a bijection**, and
two colours sharing a bin has to survive), the catch-all (`"everything else"`), naming a bin by
**elimination** (`"the remaining bin"`), and shape-keyed mappings.

**Architecture:** the sentence is split into **clauses** and each is resolved on its own, because a
single global min-distance pairing breaks on four real sentence shapes (all four found by the
audit, all four in the commit message). Two rules decide where a clause ends — both sides must bind
a key and a bin, so a comma inside a list does not cut the list; and a clause with more keys than
bins keeps reading when what remains has more bins than keys, which is what separates many-to-one
from a list cut short.

**It never abstains**, and that is deliberate: at +2/−1/0 a random bijection is worth
(1/3)(+2) + (2/3)(−1) = 0 per object, exactly what abstaining is worth, so a guess is never worse
in expectation. See §3.9.

The original reasoning for why this item is first:


**Why:** §1.1. This is 83 % of the final, the baseline scores zero, and the wiring gap is one line.

**The design** ([`instruction-parsing-six-permutations.md`](./instruction-parsing-six-permutations.md)):
the instruction is always a full permutation, so **enumerate the candidate mappings and score each
against the sentence** instead of writing a parser. Decide the key vocabulary first (colour / shape /
both, §3.7.3), then enumerate the 6 permutations over that key.

Three properties a parser cannot have:
- it **cannot emit an impossible mapping** (two colours into one bin);
- it **completes the third pair by elimination** from two correct ones;
- its **confidence is the margin** between the best and second-best candidate — a real number we can
  act on in §4's endgame.

**Scope, from ✅ the organisers (§3.7.4):** colour first; assume shape words appear too; **no
negation**. So the candidate set is 6 colour permutations, extended to shape when shape words are
detected. Build the bin-**phrase** matcher of §3.7.1, not a token match, and never require the word
`bin` (§3.7.2).

**The test set is free and exhaustive.** ✅ `oracle.py:196-218` builds `_parse_table()` — all 8
templates × 36 orderings = **288 sentences with ground-truth mappings**, generated by the
organisers' own code. **Any design scoring less than 288/288 on it is broken before we reach
held-out phrasings.** One import, no GPU, runs in milliseconds.

🔴 **And note what that table tells us about the competition:** the oracle "parser" is **exact string
equality** against those 288 sentences; on any held-out phrasing it returns `None` and the oracle
falls back to the privileged mapping it is given (`:291-294`). **The organisers' own reference
solution does not solve the unseen-instruction episode — it cheats past it with state we do not
have.** Nobody is handing us a solution to copy. This is the differentiator.

**Done when:** 288/288 on the generated table, plus a held-out set we write ourselves with
paraphrased bin phrases, and the mapping is actually threaded into `policy.py`.

### 2. Take the oracle's free constants 🖥 to verify

✅ `oracle.py:48-50` is the organisers' own fix for a failure we had not separately identified:
**wedging** — a 45 mm object centred in the 57 mm index-middle gap is *never released* and gets
carried back home, **seen with cylinders**. The fix is **shift the grasp point 15 mm sideways in the
palm**. It is a constant, it is MIT, and it costs one config value.

Also available: `ORACLE_OVERRIDES` (§4.1) — the organisers' tuned IK and motion values.

**Done when:** rung 1 with the offset is measured against step 0's number, and cylinders stop coming
home in the hand.

### 3. Time triage 💻 logic, 🖥 verify

**Why here:** §3.4.3. **One failed time check ends the baseline's episode.** Rank remaining
candidates by `plan.duration`, attempt the **shortest that fits**, stop only when *nothing* fits.

**And if step 0 confirms §3.4.1, this stops being a tidy-up and becomes a budget.** At ~19 s a pick,
120 s does not fit 5 objects, so the policy must **choose which objects to attempt**: cheapest shape
first (cube), the known-hard prism last, and on a language episode the objects whose mapping we are
**confident** about (item 1's margin) before the ones we are not. **Order of attempts becomes a
scoring decision, not a scheduling detail.**

**Done when:** an episode that previously stopped early now keeps sorting, measured on the same seeds.

### 4. Endgame 💻+🖥 🔴 **CLOSED 14 Nov 01:49 — worth zero points, nothing built**

Correction 18 cut this to one surviving case and left an FK question. Correction 19 answered it:
the palm sits at `pelvis_x` **0.2185 m** at home, the footprint starts at **0.09**, so a wedged
object carried to home is still *over the counter* — `counter` = 0, never `off` = −1. There is no
remaining case. Kept below only so nobody re-derives it.

#### The original item, for the record

I wrote this item believing that being cut off mid-carry was expensive. It is not. The scoring
rule, read in full (`scene.py classify_cube`, which `layout.py:110` calls):

```python
p = self.pelvis_to_mat(pos_pelvis)
if not self.on_counter(p):      return "off"        # -1
b = self.bin_of(p)
if b is None or speed > at_rest_speed:  return "counter"   # 0
return "correct" if resolve_bin(mapping, color, shape) == b else "wrong"
```

And `on_counter` is a **2D footprint plus a floor, with no ceiling**:

| | pelvis frame |
|---|---|
| x | **0.09 … 0.69 m** (= `0.14 + mat_y`) |
| y | **−0.55 … 0.55 m** (= `−0.45 + mat_x`) |
| z | **> −0.02 m — a floor only, no upper bound** |

*(computed from `scene.yaml` `mat_size [0.90, 0.50]`, `counter_size [1.10, 0.60]`,
`mat_origin_in_pelvis [0.14, −0.45, 0.01]`, `off_counter_drop 0.03`.)*

> 🔴 **So an object still in the gripper, held in the air over the counter, scores `counter` = 0 —
> exactly the same as one deliberately set down.** "Release rather than carry" buys **nothing**
> there. Setting it down gently and being cut off mid-air are worth the same, and so is simply
> holding. The whole premise of this item, as I originally wrote it, was wrong.

**What IS worth points, and it is one narrow case.** `off` is −1, and it fires when the object ends
**outside** that footprint. The near edge is `pelvis_x = 0.09` — **9 cm in front of the pelvis** —
and `home_arm_q` is *"arms down, elbows bent"*. In the normal path nothing is held by then
(`hand("open", …)` runs before `retreat`, and `move("home", …, hand_to=hand_open)` opens again). But
a **wedged** object — ✅ the organisers' own named cylinder failure, §3.5 — survives the open phase
and gets carried through `retreat` and `home`.

> ⚠️ **OPEN, and it decides whether this item is worth building at all:** does the home pose put the
> palm inside `pelvis_x ∈ [0.09, 0.69]`? If **inside**, a wedged object is 0 and this item is worth
> nothing — close it. If **outside**, every wedge is **−1 instead of 0**, and the guard below is
> worth one point per wedge. **This is an FK evaluation, not an opinion.** Oliver's session already
> has pinocchio FK driving the G1 URDF, so it is minutes of work there and should not be guessed.

**If the FK says "outside", the whole item is this:** before `retreat`, if the gripper still holds
something (hand did not reach the open posture, or a detection persists at the palm), open the hand
while the palm is still over the footprint. Nothing else. No settle logic, no release-before-buzzer,
no time margin for carrying — those are all score-neutral.

⚠️ The time **margin** on the plan-time check is a separate question and still live, but note what
it is now worth: not starting a plan we cannot finish is valuable because it **keeps the time for an
object we can** — which is item **#3**, Oliver's. The margin itself only avoids `counter` = 0, and
not starting is also 0. See `policy.py:220` and §3.4.1 for why `plan.duration` (= `len(arm)/30`,
perfect tracking, zero latency) is an optimistic estimate.

### 4b. Original scope, kept for the record 💻+🖥

Three separate facts converge on the last five seconds of an episode:

- §3.4.4 — the sim grants 3 s of settle before scoring, **the robot grants none**;
- §3.9.1 — an object still moving **inside a bin** scores 0; one rattling on the counter costs
  nothing. **Protect placements, do not tidy the counter;**
- §3.9.2 — **release over the correct bin beats carrying to the buzzer**, and an object we are not
  confident about is better **left on the counter deliberately** (0) than put in a wrong bin (−1).

That last one is where item 1's confidence margin and item 5's stall aperture get spent.

**Done when:** `--settle 0` and `--settle 3` give the same score on our runs. That equality *is* the
success criterion.

### 5. Yaw-aligned prism grasp 🖥

**Why it matters more than the ordering suggests:** ✅ the organisers' oracle caps at ~84 % on rung 1
**with perfect privileged perception**, and names this as the blocker. And ✅ **the real robot grips
*worse* than the sim** (§3.5.2) while **grip force is not tunable by anyone** (§3.5.1) — so
**geometric stability is the only lever that exists**, and it pays on the run that decides the prize.

**Do the disambiguation before the rotation** (§3.2.2): the raw `minAreaRect` angle is ambiguous mod
90° and mod 180°, and the centroid-to-rect-centre offset (h/6 = 7.5 mm, pointing away from the apex)
fixes both. **Rotating to the raw angle is worse than not rotating** — it will confidently pick a
corner-to-corner pinch. Work in **metres on the top-face cloud**, which is where `_yaw` already
works. ⚠️ The open measurement is **point density on a 45 mm top face at 0.47–0.53 m**, not pixels.

Cost: ≤30° = ~16 ticks ≈ **0.53 s** per prism. Against a 120 s episode, nothing.

**Done when:** prism success on rung 1 rises against step 0's number on the same seeds, with zero
clips.

### 6. Post-placement verification 🖥

Scoring is **final state, not on touch**, so a misplaced object **can be moved again and fixed**, and
✅ nothing checks today. A wrong bin is −1 and a correction turns it into +1 or +2 — **the largest
per-object swing in the challenge, on a mistake we have already made.**

### 7. Shape classifier + border guard 💻 ✅ **DONE on CPU, 14 Nov 01:15**

§3.2.3 and §3.6.2. Free (area ÷ rect area), no training, and it is insurance against ✅ the
organisers' *"shape-based instructions may also be tested"*.

✅ **Oliver's session got there first** (`prism_pose.classify_blob`, branch `feat/grasp-yaw`) and
picked the same discriminator: silhouette area ÷ minAreaRect area, which is **1.000 for a square,
π/4 = 0.785 for a circle, 0.500 for a triangle** — scale-free and rotation-free. So the work left
was not a second implementation. It was finding that **the feature is computed on the wrong data.**

> 🔴 **A colour blob is not a top face.** `classify_blob` is fed the 2D colour blob, and the head
> camera's optical axis is **42.4° off vertical** (`calibration.yaml` head `T_pelvis_optical`:
> 47.6° below horizontal). The visible **side faces add area** the ratio was never meant to see.

Measured on the real pipeline — rasterise → z-buffer → `train.yaml` depth noise and holes →
`erode_px 2` → `top_percentile 92` / `top_band 0.006` → deproject — over the real sampling region
(`scene.yaml` layout: pelvis x 0.26–0.35, y ±0.17, yaw ±45°), **105 poses per shape**:

| shape | 2D pixel blob | → | metric top-face points | → |
|---|---|---|---|---|
| cube | 0.806–0.982 | **62.9 %** | 0.942–0.987 | **100 %** |
| cylinder | 0.822–0.859 | **100 %** | 0.792–0.858 | **100 %** |
| triangle | 0.624–0.730 | **21.0 %** | 0.542–0.608 | **100 %** |

**A triangle reads as a cylinder 79 % of the time on the blob.** On the metric points all three are
clean and the bands separate with room to spare. Mean **61 % → 100 %**, same feature, right input.

✅ **The fix is ~3 lines and the array already exists.** `_from_depth` computes `top`, the metric
top-face points it hands to `_yaw` (`perception.py:149`). Deprojected to the pelvis frame those are
the **true top face in metres** — no perspective, no shear. Keep `classify_blob` for the
`_from_plane` fallback, where there are no 3D points at all and 61 % beats nothing.

**Thresholds 0.70 / 0.90**, the measured gap midpoints, so both boundaries carry equal margin.
Oliver's 0.66 / 0.88 also separate the measured bands but leave the **cylinder only 0.022 of
headroom** against 0.092 on the triangle side, and the held-out DR is harsher than `train.yaml` by
an unknown amount (A.2 correction 11). Module owner's call.

Two properties worth having on the record:

- ✅ **This feature is immune to the dominant error term.** The per-episode camera pose bias
  (§3.6.7, ±10 mm and ±1° drawn once per episode at `dr.py:102-105`) is a **rigid transform**, and
  an area **ratio** is invariant under rigid transforms. It is the one quantity in the perception
  stack that the camera-pose problem can neither hurt nor help — so it is independent of Timi's
  `xy_offset` work, confirmed by him.
- ✅ **Blobs should not merge.** `scene.yaml` `layout.min_spacing` is 0.08 m against 0.045 m
  objects — ~35 mm of gap, ~43 px at this range. Touching objects would break any shape feature;
  the scene generator does not produce them.

**Border guard.** 🔴 `perception.py:74` reads **only `CC_STAT_AREA`** — there is **no border check
anywhere in the baseline today**. `connectedComponentsWithStats` already returns
`CC_STAT_LEFT/TOP/WIDTH/HEIGHT` in the same row, so the check is free. Oliver's abstain inside
`classify_blob` is right as far as it goes, but the flag belongs on the `Detection`: a clipped blob
biases the **position** as well as the shape, and a biased grasp target is **−1** while refusing the
object is **0**.

> ⚠️ **Flag only, no reject, until the rate is measured.** How often an object actually clips the
> frame is unknown, and rejecting one costs a guaranteed 0. Queued as an offline job on a GPU box.

**Code:** `policy/shape_from_top.py`, `scripts/measure_shape_feature.py`,
`artifacts/item7-shape-feature.log` (commits `c81360f`, `eb9b823`).

> ⚠️ **The 100 % is from our own synthetic rasteriser, not an Isaac render.** It uses the real
> intrinsics, extrinsics, object sizes, sampling region, yaw range, erode/top_band/top_percentile
> and `train.yaml` noise; it does **not** model HSV segmentation error on the mask boundary,
> object–object occlusion, or renderer artifacts. The 61 → 100 gap is far too large to be an
> artifact of those, and the *direction* is arithmetic from `T_pelvis_optical` — but **the absolute
> 100 % is not verified** until it runs against a real render.

> 🔴 **And `mock_world` cannot do that verification.** Originally Oliver's finding, relayed by
> Timi: **`mock_world` draws the triangular prism as a box.** So the obvious cheap check —
> `mock_world.Renderer`, which the organisers ship and `test_classical.py:56` already drives with
> depth noise and 15 % holes — would report a clean pass on cube and cylinder and tell us **nothing
> about the one shape the classifier most needs to get right.** Worse, a box-shaped "triangle"
> would score ratio ≈ 0.97 and be labelled a cube, so a naive mock test would look like a *failure*
> of a correct classifier. **Verify on an Isaac render only.** Timi can supply frames from box A
> with `aalto.sh snap` now the cameras are fixed — that is the cheapest real check and it is the
> next thing this item needs.

### 8. Stretch

**Relay hand-over** (§4.1) for objects one arm can grasp but not carry across the body — ✅ the
baseline README lists exactly that as a failure and the oracle already implements it, config-only.

**DR fitted to real data — and 🔴 there is NO real data yet, so this is preparation, not fitting.**

⚠️ §3.6.6: `aaltonomous/data/` is **416 kB and contains no frames** — `data/drop1/` is the capture
kit only, `test_drop1.py` runs against `MockSource`, and `DATA_CARD.md` is a template. ⚠️ And the
organisers (14 Nov) say **no major drop tonight**: teleoperation runs over wifi from the VR goggles
and the venue wifi is unusable with the hall full. **Do not plan on having Drop 1.**

**So the deliverable here is the prepared path, not the fit:** `docs/dr-fit-plan.md` — every
statistic `data/drop1/stats.py` produces, mapped to the `train.yaml` knob it should set, **including
the `obj_hole_p` knob that does not exist in `train.yaml` at all** (§3.6.5). No data and no GPU
needed, and it makes the first drop actionable in minutes instead of hours.

**When a drop does arrive**, fitting our depth-noise and colour randomisation to real frames is the
citable **sim-to-real** story for the 25 %, it runs on CPU, and it needs **zero** demonstration
rollouts — which is itself the answer to *"clever ways to do a lot with very little real-world
data."* Expected format: ✅ **LeRobot v3.0** (`lerobot==0.7.0`), `observation.images.head_depth` as a
480×640 HEVC 12-bit lossless depth video, **linear 0–4.095 m at 1 mm/code, 0 = invalid.**

---

## 6. What we do not build, and why

- **No VLA as the control policy.** GR00T N1.7 measures ~140 ms (~7 Hz) on an RTX A6000, OpenVLA
  base ~6 Hz on a 4090 — both fail the 27 Hz gate without chunking, and with ~20 demos per rung
  there is not enough data to beat a baseline that already sorts ~80 %.
- **No from-scratch learned policy.** The data-efficiency numbers are brutal and specific to this
  robot: on a **real G1 with 81 demos** a naive GR00T N1.6 fine-tune scored **0/50**; an engineered
  recipe reached 32–42 % (DEED, arXiv 2607.20345); NVIDIA's own 98.8 % needed **1,000 demos**, and a
  GitHub issue reports *"very low"* success at ~600 (Isaac-GR00T #414). **We get ~20 per rung**, and
  ⚠️ tonight we get **no new drop at all**. ⚠️ Via the grokbot report — confirm DEED against the
  paper before quoting it on stage.
- ⚠️ **If anyone tries GR00T anyway, two traps are known:** every checkpoint loads the **gated
  `nvidia/Cosmos-Reason2-2B`** on first use and the submission container has **no network**, so all
  weights must be baked in; and its **default hue jitter (0.08) may blur red/yellow/blue**, which in
  this task is not augmentation, it is destroying the label.
- **No perception overhaul.** §3.2: capped at ~84 % on rung 1 regardless. The grasp is the ceiling.
- **No tactile or force feedback.** 33 pressure sensors exist and ✅ the interface exposes none of
  them. But see §3.5.3 — the **finger-position lag** gives us most of what we wanted from touch.
- **No grip-force tuning.** ✅ §3.5.1: the gains are **fixed and identical for every team.** There is
  nothing to tune.
- **No LLM in the control loop.** The instruction is parsed **once per episode in `reset()`**, and
  ⚠️ §2.3: `stop_after_s = 1.0` means a slow `reset()` risks a latched `safe_stop`. A model is the
  fallback for a zero margin, in a background thread, never the mechanism.
- **Nothing that leans on wrist-camera geometry.** ✅ Null extrinsics, placeholder intrinsics,
  *"no measured mount yet"*, and not randomised in DR.
- **Nothing that depends on rung 3.** §1.3.

---

## 7. Judging, and what the demo has to be

⚠️ **Two sources disagree about how winning works, and we build so that both hold.**

| Source | Says |
|---|---|
| Platform rubric | real robot **40 %** · generality **25 %** · originality **20 %** · presentation **15 %** |
| ✅ The brief | *"We evaluate submissions in simulation, on a version of the scene you can't train on, and pick the finalists. They run on the real robot on Sunday. **The best real-robot score wins.**"* |

**We follow the rubric, because it is the more specific of the two, and we note that nothing in the
build order trades one against the other:** every item that raises the real-robot score (geometric
grasp, time triage, settle before the buzzer) is also the generality story. The only genuine
trade-off is **hours into the pitch versus hours into the robot**, and that is decided by what the
sim tells us on Saturday, not now.

Prize 2,000 € plus an Aalto lab visit. Mentors on site: **Erkka Lehtonen, Viktor Solianoi**.

**Two things follow for the demo:**

1. **Sim performance is a qualifier, not a score.** It buys the ticket. A fragile sim-only trick that
   does not survive the real robot is worth nothing, and **the last image pushed before the deadline
   is the one that runs in the finals.**
2. ✅ **Generality is 25 % and the brief states its three wishes verbatim:** *"solutions that still
   work when the real world doesn't match the simulation exactly"*, *"general ideas that could work
   beyond a single task"*, *"clever ways to do a lot with very little real-world data."* Our story
   answers all three **as a property of the design, checkable in the code** rather than a number from
   our own test set: *we cannot emit an impossible mapping, and we recover the whole instruction from
   two thirds of it.*

### 7.1 The one idea underneath all of it

Four of our design decisions turned out to be the same decision. It is worth saying as one thing
rather than four tricks — for the 25 %, for the 20 % originality, and because a single idea is what
a judge remembers from a three-minute pitch.

> **Use relative and ordinal structure, never absolute thresholds.**

| Where | The absolute version (fragile) | Ours (relative) |
|---|---|---|
| **Instruction** | parse the sentence and hope | **rank all six permutations**; the margin is the confidence |
| **Colour** | fixed HSV thresholds | the scene always holds red, yellow and blue — **order the blobs' hues** and assign by position on the hue circle *(inference)* |
| **Grasp** | tune a squeeze force | **compare the stall aperture** — 45 mm means the altitude pinch, 52 mm means the corners |
| **Shape** | train a classifier | **area ÷ bounding-rect area**, three separated ratios |

Every row is a quantity that **moves with the perturbation instead of against it** — which is
precisely what the held-out sim and the real robot do to us. ✅ None of the four needs a GPU,
training data, or a place in the 30 Hz loop.

**And it is checkable rather than taken on trust:** *we never ask "is this value above a number we
picked?" — we ask "which of the possible answers fits best, and by how much?"*

---

## 8. Open questions, with who can close them

| | Question | Blocks | Who |
|---|---|---|---|
| 🔴 | **Was the published 10–20 s per pick wall or physics time, and what does a pick cost on the corrected clock?** | the whole time budget, item 3, and how to read any score | **step 0** |
| ✅ | ~~**Where are the bins, really?**~~ **CLOSED 14 Nov 00:30, from our own files.** The brief's *"40 cm from the torso"* is the **near inner edge** of the centre bin (measured r = 0.403 m); `scene.yaml`'s 0.505 / 0.568 m are bin **centres** in the pelvis frame. Both correct. And `check_layout.py`'s *"bins: 0 % reachable"* is a **frame artefact** — it samples the palm link, 0.131 m from the grasp point the IK targets | — | — see §3.1.1 and §3.1.2 |
| | **Does an object in the robot's hand at the buzzer count as counter (0) or knocked off (−1)?** ✅ The code has no path for it (§3.9.2), but the real score is read by eye | whether the endgame must put a held object down, and when the last pick may start | mentor |
| | **Does the at-rest test apply on the counter or only in a bin?** ✅ The code says bins only (§3.9.1); `RULES_AND_SCORING` reads more broadly | how early cylinders must be placed | mentor |
| | **How dense is the depth cloud on a 45 mm top face at 0.47–0.53 m?** | item 5's centroid stability | 🖥 one rendered frame |
| | **Is wall height really 6 cm?** ✅ A file default, not measured, and it sets the lift height at near-full extension | the release trajectory | mentor / measure at the venue |
| | **Does the second object in a bin get lost?** A relayed lead: an object resting on another has its centre at z ≈ 0.0705, **above the 6 cm rim**, so it scores as counter. The baseline uses one drop point per bin | the payoff of many-to-one mappings in item 1 | 🖥 or `scene.py` arithmetic — no physics needed |
| | **What is the real end-to-end budget** with 3 camera views plus state over the websocket? Published rates are per model call with a single view | chunk length | 🖥 measure before choosing |

---

## 9. How we work

- **Facts carry a source** — file and line, or a primary doc. **Inferences are labelled.** This
  document keeps them apart on purpose, and that rule is why three of its own numbers were caught.
- **Relayed information is marked as relayed.** Discord via a teammate is authoritative about
  *intent* and not a substitute for a line of code. Where a relay **relaxes** a requirement
  (*"no negation"*), it sets priority, never licence to drop robustness that costs nothing.
- 🔴 **Nobody looks for the held-out configuration.** ✅ `RULES_AND_SCORING.md:87-88` bans *"trying to
  extract or reverse-engineer held-out configs, seeds or templates"* and *"training or tuning on the
  held-out sim variant"*. Only `dr/train.yaml` exists; `eval_heldout` is named in `run_sim.sh:9` but
  **has no file** and crashes on load. Not in git history, not in reverted merges, not by probing
  their sim. **If anyone stumbles on held-out values, that is a leak to report to the organisers,
  not material to train on.** This applies to subagents too.
  ⚠️ **One live instance:** a held-out DR parameter appears to be quoted in a comment in the public
  `baselines/classical/config.yaml`. It has not been opened or reproduced here. **Position: we do not
  use it, and we raise it with a mentor.** Anything already derived from it is re-derived from
  `train.yaml` plus our own margins.
- **Claim a file in Warren (or post to `@room`) before touching shared code; release when done.**
- **Do not delete another session's research.** It happened once and it was not that session's call.
- **Over-communicate.** Post on starting, on changing, on being blocked. At a hackathon silence
  costs more than noise.

---

## Appendix A — provenance, and every correction this handbook has made

The value of §1–§9 is that the wrong versions were caught **before** anyone built on them. They are
listed here, not inline, so the body reads as fact and this reads as history.

### A.1 Sources

| Source | What came from it |
|---|---|
| The starter repo, cloned locally (MIT, `github.com/zupermiez/aaltonomous-humanoid`) | everything marked ✅ with a file and line |
| Locally executed tests, 14 Nov 00:00–00:20 | the 93-test run, the visibility failure, `_yaw`'s input, the `_reject` guard |
| ✅ The official challenge brief | reach discrepancy (§8), *"best real-robot score wins"* (§7), *"no markings… told apart only by their position"*, rung 3 as *"stretch goal"* |
| Organisers via Discord, relayed by Oliver, 14 Nov | fixed Dex3 gains · less grip on the real robot · colour-first / shape-maybe / no negation · oracle never re-measured · no data drop tonight |
| Oliver's [`humanoid-verified-facts.md`](./humanoid-verified-facts.md) (starter commit `36daf5b4`) and [`humanoid-research-log.md`](./humanoid-research-log.md) | per-fact line references — **if they disagree with this file, the one with the line number wins** |
| grokbot reports | marked *(bot, unverified)* wherever used |

### A.2 Corrections, in the order they were made

| # | The claim | What it actually is | Where it lives now |
|---|---|---|---|
| 1 | *"The sim hand is 120–210× stronger."* | That is the **effort-ceiling** ratio. The force gap is bimodal (67× or ~1.2×) pending Isaac Lab #4918 — and ✅ it turned out moot: the gains are fixed for every team | §3.5.1 |
| 2 | *"We can run the organisers' held-out DR."* | **We cannot and must not.** Only `train.yaml` exists; `eval_heldout` has no file; `RULES_AND_SCORING.md:87-88` forbids it | §9 |
| 3 | *"Absolute HSV thresholds are the wrong shape of answer."* | Overstated — hue jitter is only ±7.2° and the colours stay 36–160° apart. The real risks are the orange bins, low sat/val corners, and camera-pose jitter | §3.6.3 |
| 4 | *"Nothing moving at the end."* | Too broad. ✅ At rest applies to **bins only** | §3.9.1 |
| 5 | Pixel arithmetic using fx 615 at 0.35 m | ✅ fx **604.8**, object band **0.47–0.53 m**, so **51–58 px**, not 79 | §3.6.1 |
| 6 | *"Held ticks do not count toward the gate."* | Backwards. ✅ Only the **runner's idle holds** are excluded; policy-returned holds count | §2.2 |
| 7 | Rung 3 used to promote the shape classifier | ✅ `SUBMISSION.md:134` lists only `[sort_fixed, sort_language]`. The classifier now stands on shape-keyed instructions instead — which ✅ the organisers have since confirmed as a real possibility | §1.3, §3.7.4 |
| 8 | *"The baseline Dockerfile may not build — blocks everything."* | ✅ Not a bug. `baseline.Dockerfile:8-9` says the setup script clones the URDF; `setup_instance.sh:24` pins `XRTELEOP_REV=817fb00c`. A sparse checkout of `assets/g1` is 74 MB. **Nothing was blocked** | §A.3 |
| 9 | *"`minAreaRect` runs on the 2D pixel blob."* | ✅ It runs on **metric top-face points** (`perception.py:149`). The pixel budget never applied to the yaw — only to the 2D shape classifier. **This made the prism fix easier, not harder** | §3.2.2 |
| 10 | *"The bin-rim false positive is the most concrete bug we found."* | ✅ Overstated. `_reject`'s `in_bin` guard survives the depth path. The real exposure is the outer wall face only | §3.6.4 |

| 12 | *"The finals are 30 points, of which 25 are language."* | ✅ Needs N = 5, and the finals object count is **not stated** — the settings are deliberately unannounced (`RULES_AND_SCORING.md:63-68`). The robust claim is **83 % for any N** | §1.1 |
| 13 | *"The baseline scores zero on the language episodes."* | ✅ Wrong as a deterministic claim. It always applies FIXED, so it scores **+2a − b**: +2N if the true mapping happens to match (1/6), 0 on a single swap (3/6), −N on a derangement (2/6). **Expectation zero, realised value −N to +2N** — so a lucky step-0 draw can look like a working policy | §1.1 |
| 14 | *"Zero clips is the gate; 30 ends the episode."* | ✅ There is no 30-abort budget. 30 is `safety.py:74 max_clipped_steps` and **exceeding it aborts**. Also found: an **undocumented wall-clock `policy_hz_rt >= 27`** check at `evaluate.py:137-139`, in neither the rules nor the flags | §2.1 |
| 15 | *"Five objects at ~19 s each is ~97 s of 120 s."* | ✅ Overstated. Recomputed from `ta + S/vmax`: **≈31 s fast profile, ≈61 s on base gains, ≈76 s with latency.** Still tight, not as tight as claimed. Also found: a submitted container defaults to the **fast** `sim` profile | §3.4.1 |
| 18 | *"Endgame: settle, and release rather than carry"* (my own build item #4) | ⚠️ **Mostly worth zero.** `on_counter` is a 2D footprint plus a floor with **no ceiling**, so an object held in the air over the counter scores `counter` = 0 — identical to setting it down. Release-rather-than-carry buys nothing there. The only −1 case is ending **outside** `pelvis_x ∈ [0.09, 0.69]`, which a **wedged** object carried to home might do. Whether it does is an **FK question, not yet computed** | §5 item 4 |
| 17 | *"Chunk seams must be blended, and we need our own rate limiter in front of anything that produces targets."* | ✅ **Wrong, and it would have been wasted work.** `_pop` slices one pre-planned trajectory, `time_scale` guarantees ≤ vmax and excludes `Qs[0]`, every `go_to` starts from the last emitted target, and a hold is a zero delta. The stream is **velocity-continuous by construction**. The real exposure is the **3.1 % margin between the `sim` profile's `arm_max_vel 0.97` and the filter's 1.0** | §2.1 |
| 16 | *"The real robot grants no settle time."* | ⚠️ **UNVERIFIED and was stated as fact.** Not in any code we have read — `evaluate.py` is the sim path only. An inference from rules line 25, not a measurement. Plan for no grace; do not quote it as fact | §3.4.4 |
| 11 | *"The organisers ship a model of real camera noise harsher than the DR — depth σ 3 mm, 20 % holes on the object."* | ✅ Those are **test-fixture values**. `capture.py:133` `MockSource` is *"Synthetic static scene for offline tests"* and `test_drop1.py:22` injects them so `stats.py` can be checked against them. Caught by Oliver's session. What survives: `train.yaml` has **no object-hole parameter** and `stats.py` **measures one**, so the structural gap is real and the magnitude is unknown | §3.6.5 |
| 19 | *"#4's one surviving case is worth building"* (correction 18's open question) | ✅ **Closed: it is not.** FK on the planner's own `ArmKinematics` gives the palm at **pelvis_x = 0.2185 m** at `home_arm_q`, and the counter footprint starts at **0.09**. The palm at home is *inside* the footprint, so even a wedged object scores `counter` = 0, never `off` = −1. **Item #4 is dead in full; nothing was built** | §5 item 4 |
| 20 | *"A hull-area floor will catch a blob the bin mask cut"* (my own first fix for #7's border defect) | ✅ **Measured and wrong, retracted before building.** A cube cut by 10 % has median hull area **1844 mm²**, an honest triangle **1158–1296 mm²**, so a floor under the thinnest triangle (1042 mm²) catches **0 of 423** wrong cube labels. Also found: the cut is **not in image space** on the depth path at all — `perception.py:109` throws the 2D hull mask away and bins are excluded in **3D** at `:115-118`, so an image-border guard cannot see it by construction. And the damage is an *inversion*, not a degradation: a 10 % cut takes cubes from **100 % → 21.7 %**, labelling them cylinders. The guard belongs in `_from_depth` next to `top` and must **abstain** | §5 item 7 |

### A.3 What ran locally on a GPU-less laptop, 14 Nov

✅ The simulator **cannot** run on an Intel iGPU — `QUICKSTART.md:10-11` requires an RTX-class card
with ≥24 GB and ray-traced cameras (*"A100 / H100 have no RT cores: the sim will not render"*).

But ✅ **all 12 test files are CPU-only.** With the baseline Dockerfile's own pins (numpy 2.2.6,
scipy 1.15.2, opencv-headless 4.10.0.84, pyyaml, msgpack, websockets, pin 3.3.1) and a sparse
checkout of the G1 URDF, `unittest discover` ran **93 tests: 79 passed**, 9 skipped, 1 expected
failure, 2 errors (missing `zmq`, not in the baseline pins), **2 real failures**:

- ✅ `test_cubes_and_bins_visible` — the 0.2-pixel finding, now §3.6.2;
- ⚠️ `deploy/test_deploy.py:134 test_stuck_policy_latency_cap` — wanted a stuck policy handled in
  under 8.0 s, took **11.12 s** on a machine with 24 of 30 GB RAM in use. Probably host load, **but
  it is a timing test of the watchdog — re-run it on the GPU box before calling it environmental.**

✅ `deploy/replay.py` imports no Isaac: `--synthetic N` drives the real policy server over the
websocket and measures the command-rate gate, and `load_obs(log)` replays an `episode_NNN.npz`
recorded by the runner. **Episodes captured on the GPU box can be re-examined on any laptop** — so
GPU hours go to things that genuinely need a GPU.
