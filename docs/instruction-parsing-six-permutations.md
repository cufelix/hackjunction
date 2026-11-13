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
