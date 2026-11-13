# The language rung collapses to six options, and that makes it provable

**Written 13 Nov 2026 21:48 CET. Design, not code — the build freeze holds.** My lane, following
Oliver's read of `aaltonomous/interface/tasks.py`.

## The fact that changes everything

> `sort_language` instructions are **always a full permutation** — each colour goes to a different
> bin. Bins are only ever named **left / center / right** (US spelling). 8 training templates, two
> word orders: colour-then-bin (*"red into left"*) and bin-then-colour (*"The left bin is for red
> objects"*). **Finals templates are deliberately different from the training ones.**

So the entire output space of the language problem is **6 mappings**. Not 27. Three colours
assigned bijectively to three bins:

```
RLB? no - the six are:
red→L yellow→C blue→R     red→L yellow→R blue→C
red→C yellow→L blue→R     red→C yellow→R blue→L
red→R yellow→L blue→C     red→R yellow→C blue→L
```

**This is why the language rung is worth the most and costs the least.** It carries +2 per object
against the fixed rung's +1, the unseen-instruction final episode is weighted ×1.5, and the
classical baseline we are handed scores **zero** on it because it never reads the instruction.

## The design: score all six, don't parse one

Do not write a parser that extracts a mapping and hopes. **Enumerate all six permutations and score
each against the sentence**, then take the best. It is a handful of lines, it is deterministic, and
it gives three things a parser cannot.

For each of the 6 candidate mappings, count the evidence in the sentence that supports it: for each
(colour, bin) pair in the candidate, does the sentence contain that colour word and that bin word
in proximity, in either word order? Sum the matches. Take the argmax.

### Why this is better than parsing

**1. It is self-checking, by construction.** A parser can emit *"red→left, yellow→left"*, which is
impossible, and then silently act on it. Enumeration **cannot produce an invalid mapping**, because
every candidate is already a permutation. The invalid region of the output space does not exist.

**2. It completes itself from partial evidence.** If an unfamiliar phrasing lets us locate only two
of the three pairs, the third is **determined by elimination** — there is exactly one permutation
consistent with two correct pairs. So we survive a sentence where one third of the wording is
unrecognisable. This is the direct answer to *"expect phrasings you haven't seen before"*, and it is
a logical guarantee rather than a hope about generalisation.

**3. It yields a real confidence for free, as a margin.** Compare the best score with the
second-best:
- clear margin → act on it
- **two candidates tied** → that is genuine ambiguity, not a low number we invented. The sentence
  really does fail to distinguish them.

That margin is what should drive the end-of-episode decision in
[`challenge-320`](./challenge-320-aaltonomous-humanoid.md): with scoring on final state, an object
we are unsure about is better **left on the counter (0) than placed in the wrong bin (−1)**.

## Consequences

- **No GPU, no training data, no model in the loop.** This runs in microseconds, once per episode,
  before the robot moves. It cannot threaten the 27 Hz gate because it is not in the control loop
  at all — see [`humanoid-policy-approaches.md`](./humanoid-policy-approaches.md).
- **A model becomes the fallback, not the mechanism.** If the margin is zero, *then* ask an LLM to
  pick among the six (a 6-way choice, which is the easiest possible shape for a model, and
  `gemini-decision-1`-style scoring fits it exactly: six options, calibrated probabilities, no
  generated text). But the deterministic path handles the overwhelming majority.
- **It argues for the 25 % generality score with a proof rather than a benchmark.** We can say on
  stage: *"we cannot emit an impossible mapping, and we recover the whole instruction from two
  thirds of it."* That is a property of the design, checkable by a judge in the code, not a number
  from our own test set.

## What to verify in the repo before building

- [ ] Confirm the permutation claim directly in `aaltonomous/interface/tasks.py` — everything here
      rests on it. If some instruction leaves a colour unassigned, or sends two colours to one bin,
      the enumeration set is wrong.
- [ ] Are there exactly 3 colours in every language episode, or can an episode contain objects of
      only two colours? (Affects whether elimination from two pairs is always valid.)
- [ ] Check the training templates for negation or exclusion phrasings (*"anything but the left
      bin"*) — a proximity count handles those badly, and if they exist in the held-out set the
      scoring needs a negation guard.
- [ ] US spelling is stated for bins (`center`); confirm whether `centre` ever appears, and accept
      both regardless.

---

## 🟠 Amendment, 21:52 CET — six is the right number, but not only over colours

Oliver, reading the repo: **`scene.resolve_bin` accepts mapping keys by colour, by shape, or by
`(colour, shape)`**, and `STARTER_PATHS` says harder finals **may add shape words and negation**.
So the output space above is the *colour-keyed* case, not the whole space.

The enumeration survives this, and it generalises cleanly, because **the permutation structure is
what matters, not the colour vocabulary**:

1. **Decide the key first.** Which vocabulary does the sentence use — colour words
   (red/yellow/blue), shape words (cube/cylinder/prism/triangle), or both?
2. **Enumerate the 6 permutations over that key.** Three shapes assigned bijectively to three bins
   is the same 6-element set with a different alphabet. Colour-keyed and shape-keyed is **12
   candidates total**, still trivially scorable.
3. **Only if the sentence names a colour *and* a shape for the same bin** do we need the composite
   `(colour, shape)` form, and then the candidate set is built from the pairs actually mentioned
   rather than enumerated blind.

The three properties all hold unchanged: no invalid mapping can be emitted, the third pair still
completes by elimination, and the margin is still the confidence.

⚠️ **Negation is the part enumeration does not fix by itself.** *"Anything but the left bin"* is
scored wrongly by a proximity count — it will read as support for the very pair it denies. Scoring
needs an explicit negation flag that **inverts** a pair's contribution, and if the flag fires on a
sentence we cannot otherwise resolve, that is exactly the case to treat as low-margin.

### Also: a parser already exists in the repo, and it is readable

`oracle.py` contains a rung-2 parser covering all 8 training templates. The oracle is a sim-only
privileged policy, but **the instruction string is not privileged** — our policy receives it too —
so that parser is legitimate reference for the training phrasings. It does not solve our problem
(the finals templates are deliberately different), but it means **the 8 known templates are free**
and our work is only the robustness layer on top. Enumeration is that layer.

### And the priority order changes, because of the oracle number

Oliver verified in `sim/README.md:77` that the organisers' **oracle — exact privileged poses and
shapes — reaches only ~84 % on rung 1**, with their own note: *"needs >90 %: triangle grasps (thumb
vs one finger spins the prism)"*.

**This is hard evidence for what [`challenge-320`](./challenge-320-aaltonomous-humanoid.md) argued
from the geometry: perception is not the bottleneck, the grasp is.** A perfect vision stack cannot
beat 84 % on rung 1. So:

1. **Instruction parsing** — rung 2 is `self.mapping = info.get("mapping") or FIXED_MAPPING` with
   no mapping ever passed, so it silently scores zero on the rung that pays double. Unchanged as
   the first job.
2. **Triangle grasp** — promoted, and it is cheaper than it looked: perception **already computes
   object yaw** (`cv2.minAreaRect`) and marks it *"info only"*. The input to an aligned grasp is
   already in the pipeline; the planner just ignores it.
3. **Post-placement verification** — Oliver: none exists. With final-state scoring, checking the
   bin after a release and correcting a misplacement is free points on a mistake we already made.
4. **Cleanup phase** — nothing in hand, nothing moving, ~10–15 s before the buzzer.
