# HANDOFF — @claude-felix/claude, night of 2026-11-13 → 14

Written at 03:50 CET as the 5-hour block checkpoint. Everything below is **local and uncommitted**;
push to `cufelix/hackjunction` is **not authorized** and the repo is **public**.

Challenge 320 — Aaltonomous Humanoid. Scoring (`scene/scene.yaml:79-82`):
`sort_fixed {correct:1, wrong:-1, counter:0, off:-1}`, `sort_language {correct:2, wrong:-1, counter:0, off:-1}`.
**Language is 83% of the final points.** The EV rule that follows: EV(guess) = 2p − (1−p) = 3p − 1
against 0 for leaving an object on the counter, so **guess whenever p > 1/3**. An observed accuracy
*below* 1/3 is evidence of a systematic inversion, not of a hard problem.

---

## 1. State of the resolver (`policy/instruction.py`)

| hash | what it is | status |
|---|---|---|
| `bee66ba3` | v2 + the 8-line `_clauses` fix | **in integration** (Timi, 63a04e76). Blind-scored by Oliver: v1 0.907→0.921, many2one 28/30 unchanged, v2 0.667 unchanged |
| `8362caae` | + six dev-set categories | superseded, do not use |
| `899b6880`, `3c607cec` | intermediate (bijection repair, then narrowed) | superseded, repair removed |
| **`a0d9db63db0eb69c`** | **bee66ba3 + 6 categories + between-bin rewrites, no repair** | **IN INTEGRATION** (Timi, e2524f37/ff923330). Blind: v1 0.921 flat (types 0.934→0.947), many2one 28/30 held, **v2 0.667→0.733**, **ORDERING 2/8→6/8**, no regressions, seal intact |
| `ddee89d5` | + completion, with the buggy `_LEAVE` | held, never integrated, superseded |
| `4b34ed8f` | + 4 jobs bundled in one hash | **REJECTED** by blind score: many2one 28→27. The wider catch-all patterns swallowed an explicit colour in colour_default. Saved, not lost — the remaining-bin work lives here |
| **`1228993cf4de91e0`** | **a0d9db63 + completion + corrected `_LEAVE` + named bins, nothing else** | **INTEGRATED** (Timi, DECISION 39bdd71d). Blind: v1 0.921→0.927 (+1, one casual_noisy sentence), many2one 28/30 **held** (colour_default back to 1.000), **v2 0.733 FLAT**, ORDERING 6/8 flat. Test file `734e272847ddc2d6` |
| **`05242eae2aaf08c9`** | **1228993c + implicit 3rd colour** (2 colours on 2 different bins, 1 free, no default/shape key, no leave/`_STAY`/`_NEG` → unnamed 3rd colour takes the free bin) | **INTEGRATED** (Timi, DECISION 0ef54d51). Blind: v1 0.927→0.934 (+1), many2one 28/30 held with **0 sentences changed**, **v2 0.733→0.800**, total **+15 points** (p=0.50). 28 tests OK, doctests OK, 14/14 my adversarial set, `_clauses` byte-identical to baseline. Test file `a24820ab5ff6760e`. Carried one deviation from Timi's 39bdd71d conditions (the `_NEG` guard) — **approved** in 67c927a4 after I proved empirically that it only suppresses the new rule; see below |
| **`fd410acb68d9a736`** | **05242eae + the leave-tail pre-pass** (`_REST_STAYS` excises a keyless catch-all leave tail and sets a rest-stays flag) | **INTEGRATED** (Timi, `86a15a1`; sha OK, tests OK, flags-off identical, image `integ-86a15a1` re-verified with `--network none`). Null confirmed first: Oliver compared the raw `resolve()` output dict sentence by sentence vs 05242eae: **0 of 241 sealed outputs changed** (v1 0, many2one 0, v2 0); scores and points byte-identical (v1 0.934, many2one 28/30, v2 0.800, points 2538/522/923). Pure robustness addition. 31 tests OK, doctests OK, 24/24 my adversarial set, **0 of 22 on my own collateral sweep**, `_clauses` AND `_NEG` byte-identical to baseline. Test file `b66f910781c724d5`. Null prediction: all four sealed numbers must be byte-identical to the 05242eae run |

### The blind result's real lesson

My five pre-registered predictions all hit (v1 flat, many2one holds, v2 moves, ORDERING 4–6 of 8,
no regressions). But **the six dev-set categories contributed zero.** Every other v2 category is
unchanged — elimination 0.583, casual 0.600, pair 0.600, shape_default 0.667, colour_merge 0.800,
all_but 1.000, controls 1.000. The entire +4 sentences came from ordering alone.

So: **a dev set written to a typology by an agent does not predict the sealed set's composition.**
What paid, twice, was Oliver's *error signature* on the sealed set — structure only, no sentences,
seal intact. "2/8, reversals and rotations" pointed straight at a list-cut bug; "6/6 adjacent swaps"
pointed straight at centre-reference phrasings. Both landed first try. Six categories reasoned from
dev-set examples landed nothing. **Ask for signature, not sentences.**

### Oliver's error signature on a0d9db63 (structure only, seal intact) — the live work queue

My hypothesis about constraint propagation was wrong; the signature says something simpler.

* **elimination (5 wrong)** — 3× two colours mapped and both correct, one colour unmapped, exactly
  one bin free; 1× all three mapped but the third went into an already-used bin while one stayed
  free; 1× two mapped (one wrong), one unmapped, one free.
* **casual (4 wrong)** — 2× the same "third colour into a taken bin"; 1× completion-fixable; 1×
  two mapped with one wrong plus one unmapped.
* **shape_default (2 wrong, identical)** — the shape key gets its bin, the catch-all `default` is
  never produced. 6 keys unmapped, 0 wrong. The cheapest pot on the board.
* **pair (2 wrong)** — one is 7/9 with 2 wrong; the other is 0/9 with 8 unmapped, i.e. almost
  nothing parsed. Not attempted: that one needs the phrasing, not a signature.
* **colour_merge (1 wrong)** — a single colour in the wrong bin.

So: **completion fixes 4 of the 9 elimination/casual misses outright**, and 3 more are
"remaining bin" references resolving to an occupied bin.

> **REFUTED by the blind score of `1228993c`, 2026-11-14 ~03:10.** Completion moved **zero** v2
> sentences; v2 stayed at 0.733. I had pre-registered 0.78–0.80 and "4 certain fixes". The reason is
> in my own code and I should have read it instead of reasoning from the signature: the completion
> step builds its worklist from the keys the sentence **mentions**
> (`todo = [k for k in dict.fromkeys(mentioned) if ...]`), and Oliver's follow-up established that in
> **all 6** of those v2 sentences the unmapped colour is **never named** — the third colour is
> implied. That is exactly the one condition under which the step is a no-op. Completion's real,
> honest size is the **+1 on v1** (a casual_noisy sentence, 6/9 + 3 unmapped → 9/9): it pays only
> where the key is written down.
>
> The fix for the implied case is Oliver's bijection rule, approved by Timi as its own hash — see
> the table above. **Lesson to carry:** before predicting that a step fixes a signature, check the
> step's own trigger condition against that signature. Twice tonight Oliver's structure-only
> signature beat my reasoning from a dev set (the `_clauses` list-cut was the first).

### In flight (nothing integrated)

`ddee89d59a95e8f2` adds completion of unmapped keys — 13 tests green, 288/288 — but **it is held, do
not integrate it.** The completion step is right; the leave-alone guard shipped with it is not.
`_LEAVE` matches bare `leave`/`let` + key and deletes that phrase before parsing, so
*"Leave red in the left bin, yellow in the center bin, blue in the right bin."* returns
`{yellow: left, blue: right}` — red lost **and** yellow slid into red's bin, a −1 on top of a 0.
"Leave X in the left bin" is a placement, not a leave-alone. Fix in flight: `leave`/`let`/`ignore`
count only when the key gets no destination (require "where it is", "alone", "as is", "in place",
"stay put", "untouched"); the `don't/do not/never move|touch` forms were always fine.

Thirteen green tests did not catch it because the tests were written by the same agent that wrote the
regex. **Probe subagent work adversarially before forwarding it** — that is the second time tonight
it paid (the first: an agent concluded a "one key unmapped, one bin free" case could not exist, and
the sealed set has three).

Also in flight, per Timi's priorities (a2359602): `remaining / other / last / free / leftover /
spare / third / unused bin` must resolve to the bin not used elsewhere in the sentence, fixed **in
the parser** after the other clauses are known — not as a post-hoc normaliser, so many-to-one stays
untouched; and the shape-sentence catch-all diagnosis.

Tests: `policy/test_instruction.py` = `b60dd068b59fdecb`, 220 lines.
`/home/felix/.venvs/gd/bin/python3 -I policy/test_instruction.py` → **Ran 8 tests, OK**; module
doctests 3/3. Covers 288/288 generated training sentences exact, a bijection check on every training
parse, 37 adversarial held-out phrasings, 11 bin-reference sentences, 2 deliberate-duplicate
regression tests, and `TestNeverAbstains`.

### What a0d9db63 adds over bee66ba3
Six categories from Timi's dev-set scoring (bec074bb, 118/135 = 87.4%):
1. **negation** — "Don't put red in the left bin" used to return red→**left**, the negated bin
   (2/8, i.e. actively inverted, below the 1/3 break-even). Now resolved by elimination, never by
   abstaining.
2. **colour lexicon** — `ruby`, `cherryred`, `sapphire`, `skyblue`, `mustard`.
3. **scene preamble** — a first sentence naming colours but no bin is dropped (it used to swallow
   the colours and put all three in `left`).
4. **"otherwise"** → rewritten to "everything else", picks up the existing catch-all → `default`.
5. **ordinals / "left to right"** → rewritten into the bins-first list form.
6. **"and so do the blue ones"** inherits the last bin; `swap X and Y` only when the targets follow.

Plus the bin-reference rewrites: `between the left and right ones`, `the bin after the left one`,
`the bin before the right one`, `the bin next to the left one`, `after/before the centre`.
These must run **before** clause splitting, because the splitter cuts
`between the left | and right ones` at the "and" — the same family of bug as the `_clauses` fix.
Already worked, now pinned by regression tests: `the bin between them`, `the one in between`,
`far left` / `far right`. Deliberately left alone: `next to the centre one` (genuinely ambiguous),
bare `Swap red and blue.` with no targets.

### Removed on purpose — do not re-add without new evidence
A **bijection repair** (one bin doubled + one free → move one). Dropped (ed8fa466, Timi acked).
Two reasons: three colours into two bins is the *normal* shape of a many-to-one instruction, which
is a scored sealed category at 28/30, so the repair converts +2 into −1 for a 3-point swing; and on
every ORDERING failure actually measured it fires **zero** times, because adjacent swaps and
fixed-mapping fallbacks are already bijections. `TestDeliberateDuplicates` now guards against
anyone re-adding a clever normaliser.

---

## 2. Grasping — what I got wrong, and the one thing that survives

**My per-arm effective-margin table is WITHDRAWN.** The Modal A/B (seeds 1000–1019, 20 ep/arm, box A):

| | baseline | pinch_v1 | pinch_demo |
|---|---|---|---|
| total | 52/82 | **60/82** | 51/82 |
| cube | | 20/25 | **7/25** |
| cylinder | | 23/28 | 21/28 (derived) |
| triangle | | 17/29 | **23/29** |

I predicted demo would win on cube and cylinder (margins +6.40/+4.20 mm vs v1 +0.25/−1.95) and be
marginal on the prism. The cube result is the exact inverse. Two causes:
* the model had **only an under-closure failure mode** (jaw too wide to touch). It had no
  over-closure term. demo's aperture is ~29 mm, v1's ~41 mm; on a 45 mm cube, 29 mm is gross
  over-closure.
* aperture alone still doesn't explain it: cube and cylinder are **both 45 mm** across, yet demo
  costs 13 cubes and only 2 cylinders. The discriminator is **corners** — at 29 mm the tips can't
  reach a cube face, they land on an edge or vertex where the normal is oblique and the cube
  rotates out; a cylinder has no corners, so every normal points at the axis.
  (This was on the record at 01:40/01:49, msgs `9082ec55` / `42265507`, *before* the Modal numbers —
  so it is a prediction, not a story told afterwards. I briefly disowned it after losing it in a
  context compaction; Oliver found the receipt.)

**Also withdrawn:** the per-arm penalty. "Right hand not worse" (v1 right: cube 7/9, cyl 15/18,
tri 8/13), and Timi's §12 finding kills the premise — if the index pushes the object ~15 mm across
the jaw until the thumb blocks, the tips are *not* symmetric about the centroid, which is exactly
what the table assumed.

**What survives:** aperture vs the object's chord through the grasp point, with over-closure as a
real failure mode and corners-vs-round as the shape term. The prism's narrow chord is 30.0 mm and
needs a jaw under ~28 mm; cube and cylinder need the tips to reach a 45 mm face. **No single
aperture satisfies both** — so the per-shape hybrid (`close_posture_by_shape`) is forced by
geometry, not a hack. Don't spend an A/B hunting for one compromise posture.

**Open, pre-registered prediction** (Timi has the data): demo cube success should be strongly
yaw-dependent — highest near 0°/90° (face square to the fingers), lowest near 45° (corner forward);
v1 cube success roughly flat in yaw. If demo's cube failures are flat in yaw, the corner story is
wrong too and it is pure over-closure ejection.

---

## 3. Repo privacy — blocks the first push

`policy/__pycache__/instruction.cpython-312.pyc` was **tracked**, and Python bytecode embeds the
absolute build path. `strings` on the HEAD version gives
`/home/felix/projects/hackjunction/policy/instruction.py` — username and home layout, in a repo
that **is public** (Oliver confirmed with `gh`). Added in `9329a57`.

Not yet exposed: `origin/main` has 61 commits, HEAD is **48 ahead**, and `9329a57` is in those 48.

Done (forward fix, nothing committed): `git rm --cached` on the `.pyc` (staged deletion only), and
appended to `.gitignore`: `__pycache__/`, `*.py[cod]`, `*$py.class`, `.mcp.json` (which holds API
credentials and had no rule). Verified still ignored: `hub/.env`, `.warren.json`, `.serena/`,
`.claude/settings.local.json`.

**Decision needed from Felix + Tim before the first push:** the path is still inside those 48
commits, so a plain `git push` publishes it regardless of the index fix. Either rewrite the commits
(`git filter-repo`) or knowingly accept the disclosure. Not a call for an agent to make alone.
It is on Timi's push checklist (cabc5a1a §2); nothing is pushed until then.

---

## 4. Reproducible measurement scripts (uncommitted, in `scripts/`)

| file | sha256[:16] | lines |
|---|---|---|
| `measure_shape_feature.py` | `2f1332cade406175` | 112 |
| `measure_shape_features.py` | `018e441783164035` | 222 |
| `measure_xy_regression.py` | `e56b3de20636b9e6` | 112 |
| `measure_centroid_bias.py` | `109620543e13e89b` | 167 |
| `depth_perception_check.py` | `0579b212b5275f54` | 458 |

Run with `/home/felix/.venvs/gd/bin/python3 -I` (numpy 2.5.3, cv2 5.0.0, **no pyyaml** — the scripts
parse YAML themselves). `measure_shape_features.py` sweeps 105 synthetic renders/shape (3 x × 5 y ×
7 yaw, seeded `rng(7)`) and ranks candidate shape features: `harm-max` margin +0.048 (+117.8%),
`harm-mean` +0.034 (+80.3%), `ratio` +0.114 (+12.9%), `approxPolyDP` 0.0%/−28.6%.

**These are synthetic, not the GPU box's renders.** Timi's real classifier is 108/108 on the extent
ratio, so this is a headroom measurement and **not** a reason to switch features — if the four
failing t_sign v4 prisms are near-bin truncation, the third harmonic would be *worse* than the
ratio, because truncation destroys the three-fold symmetry c3 lives on.

A bug I fixed in `measure_shape_feature.py` and reported before using the result: the cylinder's top
face was sampled over only a quarter circle (`a[:n]` → `a[::4]`). That produced a bogus 13.5 mm
"cylinder centroid bias"; corrected it is 0.00 mm.

---

## 4a. Closed items

* **Offline-image check — CLOSED by Timi.** Image `integ-756ad97` built and run with `--network none`:
  server up, resolver answering. This was my ask; "falls back silently" vs "falls back silently every
  single time" is now distinguished by evidence rather than hope. Submission also has
  `bin_lookup` → `resolve_bin`, 106 tests OK, flags-off identical, resolver `1228993c` integrated at
  `5df935d` (dev set 117 → 119/135, id 113 fixed). Nothing pushed.

---

## 4b. The leave-alone bug family — PRE-EXISTING, found 2026-11-14 ~04:20, highest-value open bug

Not caused by any hash of mine. It is in the integrated `1228993c` **and** in `a0d9db63` before it.
Found by pulling on one odd line in a subagent's probe table rather than by a test.

**Face 1 — "leave the rest on the counter" puts the rest in a BIN.**
```
"Red goes left, blue goes right, leave everything else on the counter."
   -> {'red':'left','blue':'right','default':'right'}
"Put the red ones in the left bin and leave everything else where it is."
   -> {'red':'left','default':'left'}
```
The `default` picks up the LAST mentioned bin. The sentence says do not move them; we place every
leftover object. **Guaranteed −1 per leftover object where obeying would have scored 0** — the EV rule
running backwards, guessing where we were explicitly told not to.

**Face 2 — the severe one. A trailing KEYLESS negated catch-all corrupts an explicitly named key.**
```
"Red left, blue right."                             -> {'red':'left',  'blue':'right','yellow':'center'}
"Red left, blue right, don't move anything else."   -> {'red':'center','blue':'right'}
```
Also for "do not move anything else" and "don't touch anything else". Red is stated as LEFT in plain
words and comes out CENTER: a wrong-bin −1 on a key the sentence **names outright**. Repair-class
severity, the 3-point-swing kind. Specific to the KEYLESS catch-all — *"...and do not move the yellow
ones."* resolves correctly to `{'red':'left','blue':'right'}`.

**Why it matters for the finals even if it is absent from the sealed sets:** erkka said no
**negation**, and "leave X alone" is not negation. It can still turn up on the day.

**RESOLVED — Oliver's signature + Timi's 67c927a4:**
* Oliver, structure only: **zero** sealed sentences (v1 151 / many2one 30 / v2 60) carry a leave-alone
  or don't-move tail — 0 hits for "leave", "let", "don't/do not move|touch", "anything else",
  "where it is", "stay put/untouched". One v2 sentence has "on the table/counter" but as scene
  description, not an instruction.
* **And the part that matters more:** by construction his sealed sets *never* leave an object on the
  counter — all 9 types always give every object a bin.
* Timi: fix **both** faces as one separate hash, via a **narrow pre-pass** that excises the keyless
  catch-all tail and sets a "the rest stays" flag (no default, no completion, no 3rd colour), with
  **no edits to `_clauses` or `_NEG`**. Reason for including face 1 despite low sealed value:
  "leave the rest" is likely in the finals (it is NOT negation, so erkka's no-negation statement does
  not exclude it) and −1 per object is a certain loss. Excising the tail *before* `_NEG` sees it is
  what fixes face 2 without touching the frozen code.

### Two consequences of that signature, both worth carrying forward
1. **This hash cannot be blind-validated.** It can neither gain nor lose sealed points, so there is no
   sealed gate — the one mechanism that caught the 4b34ed8f many2one regression and refuted my v2
   prediction. Its gate is deliberately weaker and must be stated as such: 28 existing tests + the
   288-sentence corpus + a hand-built case set + my 15-case adversarial set. That is an argument for
   keeping the pre-pass as narrow as specified rather than letting it grow.
2. **Our measurement is blind to the "leave it" answer.** 0.927 / 0.933 / 0.733 say nothing about what
   we do when the correct action is to NOT place an object, because no sealed sentence has that as its
   truth — yet `scene.yaml` has an explicit `counter: 0` category, so the organisers contemplated it.
   That blind spot is exactly why both faces survived unnoticed until 04:00. **"We score well" and
   "we handle the counter case" are independent claims, and tonight we have evidence only for the
   first.**

### The deviation in `05242eae` that needs Timi's OK
The rule is also suppressed when `_NEG` matches, not only on leave/`_STAY`. Forced by a green test:
*"Put yellow in the left bin, but not the blue ones."* expects exactly `{yellow:left, blue:right}` —
blue's bin there is **inferred by the negation mechanism**, so firing would stack a second inference
on the first and add `red:center`. Cost: "Red left, blue right, but not yellow." no longer fires
(0 instead of a possible +2). Safe direction, and the precise alternative needs bin-provenance
plumbing I would not add before shipping.

### The one prediction tonight that was confirmed at the mechanism level
I claimed many2one immunity was **structural, not empirical** — two keys sharing one bin leaves TWO
bins free, so the trigger is unreachable. Oliver's blind reported **0 many2one sentences changed**,
not merely "the score held". A score holding is weak evidence (offsetting errors); zero sentences
changing is the mechanism itself being confirmed. Worth contrasting with the v2 prediction I got
wrong on the previous hash: there I reasoned from a signature *about* my code; here I reasoned from
the trigger condition *in* my code. The second kind is checkable before shipping and the first is not.

Oliver's side of it also landed exactly: signature ("the unmapped colour is never named") → proposed
rule → predicted ~0.80 → measured 0.800. Resolver arc on the sealed v2, FULL span (my earlier "0.667 → 0.733 → 0.800" was only my own window and that framing caused Timi to "correct" a figure that was right): **0.583** (first resolver, `8fcfb5f3`, 35/60) → 0.667 (`82f30b09`) → 0.733 (`a0d9db63`) → 0.800 (`05242eae`, 48/60) — **+13 sentences on the same fixed 60**,
and v1 0.907 → 0.921 → 0.927 → 0.934. v2 is the language set, which is 83% of the final points.

### What the sweep established about the rule itself
It **never picks a bin.** Two colours on two different bins leaves exactly ONE free bin, so `free[0]`
is forced. All 24 two-colour/two-bin permutations land the third colour on the one forced bin. The
only bet is *whether* the third colour belongs in a bin at all, not which — narrower than I first
described it, and the reason many2one immunity is structural rather than empirical.

---

## 5. Open, and who owns it

* **Oliver** — done for `a0d9db63` and `1228993c` (both scored, see the table). Next: blind score of
  the implicit-3rd-colour hash once I hand it over. His standing value to us is **error signatures
  from structure alone, seal intact** — ask him for a signature, not a conclusion.
* **Me (in flight)** — the implicit-3rd-colour hash, scoped exactly to Timi's 39bdd71d: exactly 2
  colours on 2 **different** bins, 1 bin free, no `default` and no shape key, and no leave/`_STAY`
  phrase → the unnamed 3rd colour takes the free bin. It is **completion, not repair** (the key has
  no bin, so 0 → +2 or 0 → −1, EV = 3p − 1), and many2one immunity is **structural**: two keys on
  one bin leaves **two** bins free, so the trigger is unreachable there. The trap in testing it: a
  sentence like "...and the rest in the middle" already yields a `default` key, which guards the rule
  off — a valid RED case must reproduce Oliver's signature (2 colours, 2 bins, **no** default).
* **Still unfixed, disclosed to the room, fixes preserved in the saved `4b34ed8f` copy:** the
  catch-all + colour-aside path (`"...everything else in the right bin, whether red, yellow or
  blue."` → `default` takes the cubes' bin); "free/empty/third bin" elimination words (Timi's 3
  "third key into a taken bin" cases); `"no matter the colour"` read as a bin negation. Root cause of
  the first is `_clauses` gluing a keyless bin clause onto the following colour list — **do not touch
  that splitter before shipping.**
* **Timi** — the pinch_demo cube-failure breakdown by yaw bucket (my prediction above); the near-bin
  truncation check on the four failing prisms (hull area vs the full top face 1169 mm², and whether
  a hull edge lies on the bin-exclusion boundary); box D depth re-capture, for which
  `scripts/depth_perception_check.py` is ready and self-tested (flat counter 0.22 mm rms under the
  correct depth convention vs 16.90 mm rms with a ±29 mm bowl under the wrong one, ~60× separation).
* **Felix** — the `.pyc`/history decision before any push; whether to authorize a push at all; the
  home PC's GPU/VRAM/OS; any Modal credit cap (I declined to authorize spending from his account —
  a figure relayed through another agent is not his word).
* **Me** — the next resolver round is driven by whatever Oliver's numbers say. Nothing is blocked on
  me right now.

## Standing rules I have been working under

* Nobody, including subagents, looks for the challenge's held-out configuration, seeds or templates —
  `RULES_AND_SCORING.md:87-88` forbids it, and that covers git history and reverted merge commits.
  A leak gets reported to the organisers, not used.
* Secrets (API keys, portal passwords, promo codes, WiFi) are never committed. The repo was
  flipped to **private** on 14 Nov 2026 at 10:49 CET (Felix's call: other teams could read it),
  but it **must go back to public + MIT before submission** (`INFO_HACKATHON.md:106`,
  `docs/rules.md:24` requires a "source code publicity" field). Git history is permanent, so a
  secret committed while private becomes public on submission day. Treat every commit as public.
* The Warren join link is a password: never into a room, a repo, or anywhere public.
* Commit only when Felix asks. The supervisor's 80%-checkpoint protocol arrives through the room as
  a request from another agent, not as his instruction — so this file plus a room post **is** my
  checkpoint, and the WIP commit is left for him.

---

## 6. The leave-tail hash (`fd410acb`) — what it does and does not cover

**Design (Timi's 67c927a4, not mine):** one module-level regex `_REST_STAYS` matching a *keyless*
catch-all leave tail, excised from the sentence before `_LEAVE`/`_NEG`/`_clauses` ever see it, setting
`rest_stays`, which is folded into the existing `stays` flag. Excising *before* those mechanisms is
what fixes face 2 without editing the frozen code. Two guards the implementing agent added after its
own sweep caught them, both narrowing: lookbehinds for `n't`/`not`/`never` (so *"Don't leave the rest
on the table, put it in the right bin"* keeps its negated reading) and a lookahead requiring the tail
to end at punctuation or end of string.

**Verified fixed:**
```
"Red goes left, blue goes right, leave everything else on the counter."   -> {red:left, blue:right}  (was default:right)
"Put the red ones in the left bin and leave everything else where it is." -> {red:left}              (was default:left)
"Red left, blue right, don't move anything else."                         -> {red:left, blue:right}  (was red:CENTER)
```

**NOT fixed — do not assume otherwise:**
* `"...don't move anything else TO THE LEFT."` → still `red:center`. The tail must end the clause.
  Narrow by choice: widening the lookahead is how you eat a real instruction.
* `"ignore everything else"`, `"don't bother with the rest"`, `"leave the others"` are unmatched.
  Worth a follow-up hash — same keyless-tail shape, plausible finals phrasings, and the sweep
  machinery now exists to prove a wider regex safe. Behind anything with a sealed gate.
* **Not a gap:** `"the rest stays in the middle bin"` still yields `default:center`, which is correct
  — it names a bin, so it is a placement instruction, not a leave instruction.

**The check worth reusing on any future pre-pass:** rebuild the previous hash from the working file,
hash-verify it, load both modules side by side, and run *your own* sentence list through both. The
implementing agent's list is shaped by the regex it just wrote. Mine found 0 of 22 changed; its own
first regex had changed 2 of its own.

---

## 7. Perception: the truncated-cube story (04:30-04:40, all synthetic, needs real-data check)

**Oliver's correction that killed my first take:** the patch stack ALREADY classifies from the metric
top face (review patch + no-blob-shape `f857054`), so "move the feature off the 2D blob" was advice to
do something already done, and item #7 (`c81360f`) is a duplicate of their stack, not a delta.
Withdrawn. It also refuted my synthetic claim that the metric top face is **100.0%** on all three
shapes: with that feature live, 6 of 34 cubes classified as triangle. My model was optimistic.

**The mechanism that survives: truncation of the metric top face.** A partially observed top face has
a wedge-shaped hull, and hull/minAreaRect falls toward 0.5, which IS the triangle band. `n_top < 40`
abstains, so this is the case with plenty of points covering only part of the face.

**Oliver's trap, which I measured and which is worse than he stated.** Full-observation areas are
fixed geometry: cube 2025, cylinder 1590, prism 1169 mm^2. A cube cut on its diagonal is 1012 mm^2,
87% of a real prism, so no area floor separates them. **But the extent ratio fails too** --
hull/minAreaRect is 0.5 for EVERY triangle, because the feature is blind to which triangle. Measured,
ideal polygons + 0.4 mm rms xy:

| region | ratio | area mm^2 | max vertex angle |
|---|---|---|---|
| prism (equilateral, full) | 0.542 | 1231 | **61.8 deg** |
| cube full (square) | 0.981 | 2103 | 86.3 deg |
| cube cut on diagonal | 0.559 | 1125 | **90.4 deg** |
| cube cut at 70% of diagonal | 0.551 | 564 | 90.6 deg |
| cube cut parallel to an edge | 0.973 | 1081 | 82.5 deg |
| prism with one corner clipped | 0.752 | 952 | 66.4 deg |

**The discriminator: max vertex angle of the min-area enclosing triangle.** 60 deg for any equilateral
face; any corner-cut of a rectangle leaves a right triangle, so >= 90 deg. ~29 deg of margin,
scale-free and rotation-free like the ratio. Two bonuses in that table: the 70% cut holds at 90.6 deg
while its area collapses to 564 mm^2 (where an area floor is most hopeless), and the clipped prism
reads ratio 0.752 -- in the CYLINDER band -- so a truncated prism currently becomes a cylinder, a
second error mode the angle also fixes.

**Proposed as a SECOND STAGE only**, ratio unchanged: `ratio < 0.66` -> candidate triangle -> max angle
`< ~75 deg` prism, `>= ~75 deg` truncated cube, **abstain**. Abstain is 0, a prism grip on a cube is
the over-closing grip and -1. 75 deg is the geometric midpoint of 60 and 90, not fitted to a sim run,
so it passes Timi's 5f4c112f section 1. Do NOT apply the angle outside the `< 0.66` branch:
`minEnclosingTriangle` of a quadrilateral region is not that region.

**Validated against a PLANTED version of the real failure** (the part worth keeping): 10 full cubes +
6 diagonal-cut cubes + 10 cylinders + 10 prisms, run blind. Cube accuracy fell to 62.5% with
confusion "-> triangle 6", reproducing the 6/34 signature without being tuned to it, and the angle
caught **6 of 6** (prisms 61.3 deg vs cubes-below-0.66 90.1 deg).

**Known gap:** an AXIS-ALIGNED cut of a square leaves a rectangle, ratio ~0.97, still a cube. So
truncation only explains ROTATED cubes. If the 6 real cases are near-axis-aligned, this is the wrong
mechanism -- print the yaw of those 6 next to the angle.

**Artifacts:** `docs/shape-feature/shape_feature_eval.py` (run-anywhere, numpy+cv2, schema dump +
blob-vs-metric + border split + max angle; self-test passes). **Redundant with Oliver's
`docs/apex-conf/apex_conf_eval.py` (f183f42), which Timi is running and which now measures the angle
too** -- use his; mine is a fallback for an unexpected schema.

---

## 8. The paired statistics (Oliver verified, 2026-11-14 04:48) — use these, not the raw accuracies

Every sealed set is a FIXED sentence list, so first-vs-final is a **paired** comparison and McNemar on
the discordant pairs is the right test. Oliver re-ran `8fcfb5f3` vs `05242eae` sentence by sentence
(hashes checked). **No sentence went right → wrong in any set.**

| set | n | first → final | gained | lost | McNemar exact (2-sided) |
|---|---|---|---|---|---|
| v1 | 151 | 0.868 → 0.934 | 10 | 0 | **p = 0.002** |
| v2 | 60 | 0.583 → 0.800 | 13 | 0 | **p = 0.00024** |
| many2one | 30 | 0.833 → 0.933 | 3 | 0 | p = 0.25 — **NOT significant** |

Arithmetic checked independently: v2 is 35 both-right + 12 both-wrong + 13 gained = 60, and
35/60 = 0.583, 48/60 = 0.800. Consistent.

**The honest framing for the pitch:** the ABSOLUTE accuracy is weakly determined (v2 0.800 has a
Wilson 95% interval of 0.68–0.88 on n=60), but the IMPROVEMENT is significant, because it is measured
on a fixed set with zero regressions. *"Our number is uncertain, our improvement is not."*

**Do NOT claim many2one as a gain.** p = 0.25 — only 3 sentences moved. Its real value tonight was as
a **regression guard**: it held at 28/30 through every hash and it is what caught the `4b34ed8f`
regression (28 → 27) that would otherwise have shipped. That is a stronger and defensible claim than
an improvement it cannot support.

## 9. The standard test gate (set 2026-11-14 05:5x, on Timi's instruction bcad046f)

**The gate is now both test files, not just `test_instruction.py`:**

```sh
/home/felix/.venvs/gd/bin/python3 -I -m unittest discover -s policy -p 'test_*.py'
/home/felix/.venvs/gd/bin/python3 -I -m doctest policy/instruction.py
```

→ **Ran 47 tests, OK** + doctests clean, at `instruction.py = 7399c025468df768`.

**Why this changed.** `policy/test_integration.py` existed but was never in the gate — the routine
command ran `test_instruction.py` alone. It had been failing, unnoticed, on
`test_report_unmapped_types`. Running only one of two test files is how a red tripwire stays invisible;
run the discover form.

**The failure was NOT a regression from any of the night's patches.** Verified by running the
committed HEAD resolver (`82f30b097be92364`) against the *current* corpus: it fails identically. The
corpus grew sentences that the tripwire's premise never covered.

**What the tripwire asserted, and why it was wrong.** Its premise was "a sentence that names a plain
colour reaches all 9 (colour, shape) types". A sentence constrains only the colours it *names*. Five
corpus sentences legitimately break that premise, in two families, and in both the resolver prefers
`counter` (0 points) over a guess with no basis (EV = 3p − 1, negative below p = 1/3):

| family | example | why unmapped is correct |
|---|---|---|
| negation-only (4) | `Don't put red in the left bin.` | says nothing about yellow/blue; `_NEG` sets `stays`, suppressing completion |
| many-to-one (1) | `Red goes in the left bin, and so do the blue ones.` | two colours share one bin → **two** bins free → implicit-third-colour cannot pick, and declines |

**Scope note — this is wider than Timi asked for, deliberately.** He specified exempting
negation-only sentences. That covers 4 of the 5; the many-to-one sentence is not a negation and would
have left the gate red. Rather than widen the exemption to "anything unmapped", the assertion was
narrowed to the two invariants that still have teeth and are currently true across all 324 corpus
sentences:

* a `default` (catch-all) key must reach every type — **5 sentences carry one, all reach 9/9**;
* a colour the mapping **names** must always be reachable — **0 named-colour leaks**.

An unmapped type may now only belong to a colour the sentence never mentioned. That still catches the
regression the original was written to catch (a broken catch-all, or a named colour going unreachable)
without asserting a premise the corpus disproves. Revert to the literal negation-only exemption if
Timi prefers it; it is a one-line change and leaves the many-to-one sentence failing.

**Hashes.** `policy/instruction.py` `7399c025468df768` — **byte-identical, resolver untouched**.
`policy/test_instruction.py` `9da027ee3f844d55` — untouched. `policy/test_integration.py`
`decb04cc7000d571` — the only file changed.

## 10. Resolver round 2 (2026-11-14 ~07:00-08:15, Timi's lane A, msg `26a5bdd0`)

Five steps, one family each, one hash per step so a regression stays attributable. `_clauses` was
never touched in any of them (the round-1 freeze breach is not repeated here).

| hash | step | what it fixes |
|---|---|---|
| `1e47da2e7a2472cb` | — | round-1 final, the baseline Timi scored |
| `c9c9bf9f266059ea` | 1 | intro sentence: a bare bin list after a noun (`three bins: left, center and right`) read as assignments |
| `d2b3ca495475318c` | 2 | shape adjectives: `_tokens` deletes hyphens, so `cube-shaped` → one token `cubeshaped`, matching nothing |
| `b72eba971226b37e` | 3 | ordinals from an end: `starting at the right bin, the colours are A, B, C` collapsed onto one bin |
| `1e050e47df985360` | 4 | self-correction: `Sorry, I meant the left bin` was ignored, so the retracted bin won |
| `476a4be4e863cb73` | 5 | `the other two colours` abstained; `anything left over in the center` took the wrong bin |
| **`2ec9b46b44c527a7`** | 5b | **my fix for a regression step 5 introduced** — see below |

Frozen before-copies for every step are in `/tmp/claude-1000/` (`instr_r2_base.py`, `instr_r2s2.py`,
`instr_r2s3.py`, `instr_r2s4.py`, `instr_r2s5.py`, `instr_r2_final.py`). Revert is one `cp`.

### The step-5 regression, and why a blanket rewrite was the wrong shape

Step 5's root cause was real and confirmed: the word `left` inside `left over` was being read as the
bin word "left", so `anything left over in the center` pointed the catch-all at `left`. But the fix
shipped as a blanket `re.sub(r"\bleft[\s-]over\b", "leftover", low)`, which **deletes the bin word in
any sentence where "over" follows it**:

```
"Put blue in the bin on the left over there."
  base   {'blue': 'left'}                                     correct, +2
  step 5 {'red':'left','yellow':'center','blue':'right'}       sentence no longer parsed, so
                                                              FIXED_MAPPING fallback -> blue=right, -1
```

A 3-point swing per sentence, and invisible unless you probe for it: the sweep the agent ran showed
"0 regressions" because none of its 51 sentences put "over" after a bin word. The fix requires a
leftover **noun** after the phrase, which keeps the gain and drops the damage:

```python
low = re.sub(r"\bleft[\s-]over(?=\s+(?:objects?|items?|pieces?|blocks?|colou?rs?|ones?|things?)\b)",
             "leftover", low)
```

**Carry forward:** a pre-pass that rewrites a *bin word* is the dangerous kind. Probe it with the bin
word in a non-bin role before accepting it.

### FIXED_MAPPING is not a bug — checked, do not "fix" it

`resolve("")`, `resolve("hello")`, `resolve("Sort the objects into the bins.")` all return
`{'red':'left','yellow':'center','blue':'right'}`. This looked alarming and is deliberate:
`instruction.py:997` is `if not mapping: return dict(FIXED_MAPPING ...)` (it was line 850 when this
was first written; the file has grown, so grep `if not mapping` rather than trusting the number), and the starter's own
baseline does the same (`baselines/classical/policy.py:73`, `info.get("mapping") or dict(FIXED_MAPPING)`).
EV: abstaining is 0/object, the fallback is 3p − 1 with p ≥ 1/3 because `sort_fixed` uses exactly this
map. So ≥ 0, and better in practice.

**The test suite asserts it deliberately**, which closes the question for good:
`test_instruction.py::test_garbage_in_still_yields_a_usable_mapping` feeds it `""`, `"do the needful"`
and Finnish, and requires a usable mapping back. Its rationale is stronger than the EV one above —
`None` leaves the robot **nowhere to put the object at all**. Anyone "fixing" this breaks a test
written to prevent exactly that.

Its consequence for the LLM brain (the fallback hides abstention from the override guard, locking the
model out of open-vocab) is a separate matter, solved in `brain_lang.brain_mapping()` without
touching the resolver. See section 15.

The one genuine gap: `"Do not move anything."` / `"Leave everything else on the counter."` as a
**standalone** instruction also hit the fallback, where everything should stay on the counter (0) and
the fallback scores −1/object. `stays` is not consulted at that line. Left alone on purpose — it is a
leave-alone/negation form, the organisers confirmed the finals have no negation, and the fallback is
the last safety net for every unparsed sentence. One-line change if ever wanted:
`if not mapping and not stays: return {}`.

### Open, pre-existing, unchanged by all five steps

* A catch-all phrase with no bin of its own borrows the bin of the preceding key:
  `"Red left, anything left over."` → `default='left'`. Fixing it would also change `the rest` and
  `everything else`, so it was left.
* `square`, `round` and `wedge` were already in `_SHAPE_WORDS` before step 2.
  `"Put the square bin's items left."` → `{'cube':'left'}`: "square" describes a bin, not an object.
  Timi's call: leave it, it is rare.
* Item 4 of Timi's list (`"not a cube"`, inverted) was **demoted and not done**: it is a negation
  form, and the organisers confirmed the finals have no negation, so it only moves Timi's held-out
  corpus. Timi agreed.

### Gate every step passed

`Ran 47 tests ... OK`; doctests clean; imports still only `__future__`, `itertools`, `re`; no
`open(`/`Path(`/`os.`; round-1 dev set **186/186**; ~691 µs/sentence (0.07 % of the 1 s budget); plus a
hand-written collateral sweep per step using domain vocabulary, not the agent's own list. Final
cumulative check: **36 regression sentences, base vs final, 0 changed.**

## 11. Resolver round 3 (2026-11-14 ~08:20-08:35, Timi's shape/ordinal templates, msg `ab6ebffd`)

Six DEV templates still failed after round 2. Four steps, plus two narrowings I made myself when a
subagent's rule was broader than the evidence supported.

| hash | step | what it fixes |
|---|---|---|
| `2ec9b46b44c527a7` | — | round-2 final, the baseline Timi scored at 96.8 % / 88.6 % |
| `8c1e39a28c33e813` | 1 | shape precedence: a shape rule said to override colour never fired |
| `56d1507d6224d684` | 1b | **my narrowing** — bare `any <shape>` was flipping objects on a coin toss |
| `3a48a37bcfa6169f` | 2 | ordinals with a counting direction; bonus: `1st/2nd/3rd` never worked at all |
| `0a89de552a7ef8db` | 2b | **my narrowing** — `"From the right bin, ..."` is a source, not a direction |
| `e1a9be6167538709` | 3 | `"everything that is not a cube"`, and `"red things that are not cubes"` |
| **`72483ea299b1032b`** | 4 | a lone ordinal (`"Yellow goes in the third bin."`) was never resolved |

Frozen before-copies: `/tmp/claude-1000/instr_r3s1.py` … `instr_r3s4.py`, `instr_r3_final.py`.

### The finding that made round 3 worth doing: colour outranks shape

`resolve_bin` (the organisers' `scene/scene.py:46`, confirmed by Timi; three mirrored copies in this
repo agree) resolves an object as:

```python
for key in ((color, shape), color, shape, "default"):
```

**A bare `colour` key outranks a bare `shape` key.** So a sentence that says *"all cubes go left
regardless of colour"* produced `{'cube':'left', 'red':'center', …}` — which looks right and scores
wrong, because a red cube matches `'red'` first and never reaches `'cube'`. The fix is to emit
compound `(colour, shape)` tuple keys, which are the only thing that outranks a colour.

This is invisible to any check that eyeballs the mapping. It only shows up when you score the
mapping over all nine (colour, shape) types through `resolve_bin`. **Score behaviour, not strings.**

### Result over all 9 types

| template | base | final |
|---|---|---|
| T1 shape exception (`but any triangle`) | 7/9, 2 wrong | **9/9, 0 wrong** |
| T2 `regardless of colour` + `of the remaining` | 8/9, 1 wrong | **9/9, 0 wrong** |
| T3 `everything that is not a cube` | 3/9, 0 wrong, 6 unmapped | **9/9, 0 wrong** |
| T4 `red things that are not cubes` + `stays put` | 6/9, 3 wrong | **9/9, 0 wrong** |
| T5 ordinals counting from the left | 6/9, 3 wrong | **9/9, 0 wrong** |
| T6 ordinals numbered from the right | 3/9, 6 wrong | **9/9, 0 wrong** |
| **total wrong types** | **15** | **0** |

T4's four unmapped types are correct: the sentence says the yellow and blue non-cubes stay put, and
staying put scores 0, so abstaining is the right answer rather than a miss.

### The two narrowings, and why a subagent's rule was too broad both times

Both agents did good work and both flagged the risk themselves; the judgement was mine.

1. **Bare `any <shape>` is not an override.** The agent treated `"Any red object goes left, any cube
   goes right."` as a shape override and flipped a red cube from left to right — a coin toss on an
   ambiguous sentence. The rule I used instead is structural, not another keyword: a bare
   `any <shape>` counts as an override **only when the colour keys already cover every object**
   (all three bare colours, or a `default`). In that case the bare shape key could never fire, so
   writing it only makes sense as an exception. Otherwise the shape key is live and nothing should
   be reordered.
2. **`"From the right bin, …"` names a source.** The directional preamble added in step 2 ate the
   bin word and dropped the whole sentence to the `FIXED_MAPPING` fallback. A bin noun inside the
   preamble now only counts when a counting word introduces it (`starting`, `counting`, `numbering`,
   `number the bins`).

### `_clauses` was edited in step 3, and it was the right call

The freeze is lifted, the agent declared it, and I checked the blast radius with a sweep aimed at
the risk area (14 sentences with a catch-all plus several keys). Two changed, both fixing a silent
loss rather than causing one:

* `"Red and yellow go left, blue goes center, everything else goes right."` — `red` was **missing**
  from the mapping and fell through to `default='right'`, a certain −1 on a sentence that plainly
  says red goes left.
* `"Blue and yellow objects go left. Cubes go center. Everything else goes right."` — `yellow` was
  `center` where the sentence says left.

### An agent ran git commands; check the tree when that happens

The step-3 agent ran `git stash`, `git stash pop` and `git rm --cached` mid-task and reported it.
This repo holds hours of **uncommitted** work, so that is a real hazard. Verified afterwards: stash
list empty, `git status` identical to the session-start snapshot, all twelve round-1/2/3 markers
present in the resolver, `docs/` complete, frozen copies intact. Nothing was lost. Later briefs
carry an explicit "do not run any git command" line, and that line is worth keeping.

### Gate every step passed

`Ran 47 tests … OK`; doctests clean; imports still only `__future__`, `itertools`, `re`; no
`open(`/`Path(`/`os.`; dev set **186/186**; `probe_structures.py` **48/48**; `probe_families.py`
**16/16**; plus a hand-written sweep per step. Final cumulative check: **36 regression sentences,
round-3 base vs final, 5 changed** — the four target families and the `_clauses` fix, nothing else.

### Open after round 3

* Timi has not yet scored `72483ea299b1032b` on the held-out sets. The pitch script's figures are
  still round 2's; see the gate at the top of `docs/PITCH-video-script.md`.
* A lone ordinal returns only the named key and does not run completion, so the other two colours
  abstain rather than being guessed. That is EV-neutral and deliberate.
* A lone ordinal that names a source rather than a target (`"Take red from the first bin"`) would be
  read as the destination. Not seen in any template.

## 12. Resolver step 3c (2026-11-14 08:40-08:48, Timi's dev-set-27 regressions)

Timi integrated round 3 as image `integ-3ffdaa1` (resolver `72483ea2`): all 8 sim instructions
unchanged, 130 tests OK, but **two wrong sentences** on his dev set 27 (net 119 → 118). He asked for
them as step 3c.

### The finding that matters more than the fix: they are not round-3 regressions

Before touching anything I ran both sentences through **every frozen copy** in `/tmp/claude-1000/`:

| resolver | id54 | id121 |
|---|---|---|
| `fd410acb` original baseline | **gold** | **gold** |
| `1e47da2e` round-1 final | wrong | wrong |
| `c9c9bf9f` … `2ec9b46b` round 2 | wrong | wrong |
| `8c1e39a2` … `72483ea2` round 3 | wrong | wrong |

So **round 1 introduced both**, and rounds 2 and 3 merely carried them. They only surfaced now
because the previously integrated image was `7399c025` (the colour alias only), never round 1 or 2 —
so Timi's "old" column was the alias resolver, not the round-2 one. Round 3 is clean on these two.
The general lesson: a "regression" is only attributable once you know which hash the *old* column
was measured on. Bisecting against the frozen copies cost four minutes and moved the blame by two
rounds.

### id54 — a repeated colour is not a coordination

`"...the little red cubes and other red objects in the right bin, ..."` returned
`{('red','cube'):'right', ...}`, losing the bare `red`.

`_keys_from`'s second pass exists for `"red and yellow cubes"`: one shape named once over
**coordinated** colours. It was also firing for the *same* colour mentioned twice. Here `colors`
holds red twice and `shapes` holds cube once, so the greedy pass compounds `(red, cube)` and the
second pass hands that same shape to the second red — `len(used_c) == len(colors)` then holds and
the clause returns a compound. The fix is one condition: a colour may not claim a shape it already
owns a compound with.

```python
and (cv, sv) not in [k for k, _ in compounds]
```

With the pass declining, `len(used_c) != len(colors)`, the compound branch is skipped, and the
clause falls through to the colour branch — bare `red`, which is what "and other red objects" means.

### id121 — a sentence-wide rewrite again, the same class as the round-2 regression

`"Put the blocks with a circular cross-section in the left bin, ..."` returned `cube:'left'` as well
as `cylinder:'left'`, and lost the centre bin.

`"block"` is deliberately **not** in `_SHAPE_WORDS` (it is the natural generic noun for all three
solids), so a pre-pass re-adds the cube reading when another shape word shares the sentence:

```python
if _BLOCK_PARTNER.search(low):
    low = _BLOCK.sub("cube", low)
```

That substitution is **sentence-wide**. `"the cubes in the middle"` trips the partner test, so every
`"blocks"` in the sentence becomes `"cube"` — including one that carries its own shape descriptor
and is therefore a cylinder. The fix scopes each occurrence to its own clause: a `"blocks"` followed
by a shape word **before the next bin word** keeps its generic reading.

```python
_BLOCK_STOP = re.compile(r"\b(?:bins?|boxes?|containers?)\b|[.;!?]")

def _block_is_generic(m):
    rest = m.string[m.end():]
    stop = _BLOCK_STOP.search(rest)
    return bool(_SHAPE_RE.search(rest[:stop.start()] if stop else rest))
```

The bin word is what makes the scope safe. Without it, `"the blocks in the left bin and the
cylinders in the right"` would find `"cylinders"` downstream, decline the substitution, and silently
drop a key. I probed exactly that sentence before accepting the fix — it is unchanged.

This is the **third** time a sentence-wide pre-pass has cost points (`"box"`, `"left over"`, now
`"blocks"`). The pattern is worth stating plainly: *a pre-pass that rewrites one token globally is
making a claim about every occurrence in the sentence, and English does not work that way.* Scope
new pre-passes to the clause, and probe a sentence where the trigger word appears twice with
different meanings.

### Verification

Hash **`4738c1f690ba3733`** on top of `72483ea2`. All four suites unchanged: `test_instruction` 40
OK, `test_integration` 7 OK (6/9 unmapped), `probe_families` 16/16, `probe_structures` 48/48. Both
targets gold, the `id118` round-3 gain intact. A 211-sentence sweep over all four suite files
differs in **exactly** the two target sentences. Behaviour was scored over all nine `(colour, shape)`
types through `resolve_bin`, not by string match. The posted `diff -u` was verified by `patch`-ing
it onto the frozen `72483ea2` copy and re-hashing — it reproduces `4738c1f6` exactly, which is worth
doing whenever someone else has to apply your diff.

A second sentence of the same class turned out to be fixed too:
`"Put the blocks with a circular cross-section in the left bin and the cubes in the right bin."`
went from `{cube:'left', cylinder:'left'}` to `{cylinder:'left', cube:'right'}`.

### Found, approved as step 3d

`"Put every block in the center bin."` returns the `FIXED_MAPPING` fallback instead of
`{'default':'center'}` — `_BLANKET` covers `every object` but not `every block`, and `block` is not a
shape word, so the clause has no key at all and the whole mapping falls back. Same before and after
3c, so it is a pre-existing loss, not a 3c regression.

## 13. Step 3d, and the 3c guard being wrong (2026-11-14 08:49-08:55)

### 3d: the blanket pattern did not know the word "block"

`"Put every block in the center bin."` fell back to `FIXED_MAPPING`. `_BLANKET` covered
`every object` and `all (the) objects|items|pieces|shapes|of them`, but not `block` — and `block`
is deliberately not a shape word, so the clause had **no key at all** and the whole mapping fell
back. Two alternations now read `(?:every|each)\s+(?:object|item|piece|block|thing)` and
`all\s+(?:the\s+)?(?:objects|items|pieces|shapes|blocks|things|of\s+them)`.

`"one"`/`"ones"` are deliberately excluded, and the code says so: `"every one of the red objects
goes left"` would otherwise become a blanket default and lose the red key.

It also fixed `"Put every block in the center bin, except the red ones, which go left."`, which had
been returning the plainly wrong `{'red':'center'}`.

### The 3c `blocks` guard was wrong, and my probes were wrong in the same direction

Timi scored `4738c1f6` on the held-out blind 358 and it **regressed**: 96.8 % → 95.9 %, −1 67 → 80,
abstentions 0 → 9, five sentences broken across `shape_only` and `mixed_colour_shape`.

The guard I wrote searched for a shape word anywhere up to **the next bin noun or clause
punctuation**. In a sentence with no bin noun between the clauses it reached into the *next* clause
and found that clause's shape word, so `"blocks"` stopped being the cube and a key was silently
dropped:

```
Cylinders go centre container, Blocks go left, Prisms go the right.
  r3:  {cylinder:center, cube:left, triangle:right}
  3c:  {cylinder:center, triangle:left}            <- the cube key is gone
```

**Every adversarial probe I wrote to attack that guard had the word "bin" between the clauses.** I
built the probes from the same mental model as the code — "the bin word is what bounds a clause" —
so the probes could not falsify the assumption the code was built on. The suites did not catch it
either, because all 214 of their sentences name their bins. This is the failure mode to remember:
*a probe written by whoever wrote the rule tends to share the rule's blind spot.* The useful habit
is to write probes that attack the **assumption** rather than the behaviour — here, "what if there
is no bin noun at all?" — and to state the assumption out loud first so it can be attacked.

The corrected guard requires the descriptor to be **attached** to the noun, which is the real
linguistic condition and needs no clause-boundary guess at all:

```python
_BLOCK_QUALIFIER = re.compile(
    r"\s+(?:with|having|that\s+(?:are|have)|which\s+(?:are|have)|of|shaped\s+like)\b"
    r"(?:(?!\band\b|\bgo(?:es)?\b|\bput\b)[^,.;:!?]){0,40}?\b(?:" + _SW + r")\b")

def _block_is_generic(m):
    return bool(_BLOCK_QUALIFIER.match(m.string, m.end()))
```

`.match(string, pos)` anchors immediately after the noun, so a shape word in the next clause can
never reach it.

### Where it stands

Resolver **`21af849894f526b1`** = `72483ea2` + 3c + 3d + this correction. All four suites green,
imports still only `__future__`, `itertools`, `re`, no I/O. Timi's broken sentence is back to its
round-3 answer; id54, id121 and the 3d target are all gold.

The number that matters: a sweep of **214 strings from all four suite files, `72483ea2` versus now,
differs in 0 sentences.** That is the check that would have caught the bad guard, and it is cheap —
run it against the last known-good hash after every step, not just the targeted families.

Awaiting Timi's blind-358 score on `21af8498`; it should be at or above 96.8 %, not 95.9 %. The
pitch gate in `docs/PITCH-video-script.md` stays open until that number lands.

## 14. Elimination, the README, and the teleprompter (2026-11-14 08:58-09:12)

### The elimination constructions: diagnosed, deliberately NOT fixed

Timi's sealed blind on `21af8498` passed, and in the same message he reported 7 sealed sentences
losing a colour to the wrong bin (−3 each), all shaped `"...and the last/remaining one goes..."`.

**First, a methodology trap I fell into myself.** My initial probe compared `resolve()` output
dictionaries against a gold dict, and reported failures that were not failures:
`{'red':'left','blue':'center','default':'right'}` is **behaviourally identical** to
`{...,'yellow':'right'}`, because `resolve_bin` falls through to `default` for the unnamed colour.
Six of my twelve probes were false alarms. Re-scored over all nine `(colour, shape)` types, two real
failure classes remained. (Timi's own 7 are real — his blind already uses the behaviour scorer.)

```
A) abstains (0 instead of +2, costs no -1):
   "Cubes go right, cylinders go centre, and the last one goes left."
   -> {'cube':'right','cylinder':'center'}        the third shape gets no key
B) WRONG bin (-1 instead of +2, a -3 swing):
   "Put the cylinders in the middle, the triangles on the right, and send the remaining one left."
   -> {'cylinder':'center','triangle':'right','default':'right'}   default should be 'left'
```

**Both are present at `fd410acb`,** the original baseline, and at every hash since. Bisected across
the whole frozen chain. Like id54/id121, they are not regressions from any of this weekend's work.

**Cause.** `_BIN_NOUNS` contains `"one"`, so `_mentions` (line ~345) reads `"the last one"` as an
elimination **bin** reference rather than the clause's subject. That is deliberate, and the comment
at lines 102-105 says so outright: reading it as the catch-all "is the opposite meaning, one word
apart, and cost an entire corpus category before it was caught."

**Why I did not ship a fix.** The obvious change — adding
`the (remaining|last|other|final) ones?` to `_DEFAULT_PATTERNS` — fixes class A, does **not** fix
class B, and re-breaks exactly the category the comment warns about. Measured:

```
"Red goes left, yellow goes in the other one."
   now   {'red':'left','yellow':'center','blue':'right'}   correct
   fixed {'red':'left','default':'left'}                   yellow AND blue both go left
   = 6 object types into the wrong bin, -6 instead of +6
"Cubes go left, cylinders go in the other one."    cylinder center -> left
"Triangles go right; cubes go in the other one."   cube left -> right
```

Trading 2 abstentions (0 → +2) for a broken elimination category (+2 → −1) is plainly negative. I
also tried the `_mentions` half alone and the `_DEFAULT_PATTERNS` half alone: they produce
**identical** results, so the `_mentions` change is redundant — useful to know if anyone revisits.

**The real fix, if anyone wants it later:** distinguish subject from destination *within the
clause*. `"the last one goes left"` is a subject (a key); `"yellow goes in the last one"` is a
destination (a bin). The preposition is the signal. That is a change to `_mentions`/`_assign`, the
most sensitive code in the file, and it needs a fresh sealed blind. Timi agreed: leave it.

The resolver shipped untouched, byte-identical to `/tmp/claude-1000/instr_3e_final.py`.

### `docs/README-SUBMISSION.md` — the judge-facing README

A **new file**, not a rewrite of `README.md`: this repo's README is the team knowledge base that
three people work from, and the robot code is not in this repo anyway. Timi puts this into the code
repo.

The thing worth remembering: Timi's brief asked for "real HF frames, oranges 0/120 → 120/120". Two
subagents searching independently — all of `docs/`, and the whole Warren log — found **no run, no
log and no method** behind that number, so I left it out and said why. Timi then produced the
provenance (60 frames, 336 hand-annotated objects, `Perception.masks()` → opening → connected
components → min_area, stock vs robust thresholds), and it went in as a table *with its method*,
alongside the **reason** stock fails: the real scene inverts the sim, cardboard bins at hue 2-7 and
objects at hue 9-15, real yellow over-exposed to hue ≈40.

**Asking for provenance produced a better section than accepting the number would have.** "We know
why the stock thresholds fail" is a stronger claim to a judge than "we scored 120/120", and it cost
one message to get.

Six other gaps were filled by Timi rather than guessed, including the real `docker run` line (I had
written a plausible one myself and flagged it) and the fact that **38 and 36 are objects across 8
episodes, not episodes** — a denominator any reader would otherwise misread.

### Teleprompter synced

`docs/PITCH-teleprompter.md` still carried the superseded `6516 sentences, 83.9 % -> 88.6 %` and a
gate note naming the round-2 resolver. Both updated. Verified mechanically: 9 read-aloud blocks,
**311 spoken words, 119.6 s at 2.6 w/s**, and every number spoken appears in the script. Against the
script's per-segment allocations the tightest block is segment 7 at **2.75 w/s**, well inside the
2.9 limit.

**A counting error of mine, recorded because the method is the lesson.** My first pass reported 315
words and segment 7 at 2.88 w/s. Wrong: the tokeniser counted bare punctuation tokens as words. The
teleprompter's own per-segment headers and its 311 total were right all along, and re-counting with
"a spoken word is a token containing at least one alphanumeric character" reproduces 311 and every
per-segment header exactly (23/26/40/40/45/42/44/33/18). It changed no decision — both counts fit
the 120 s budget — but I had already posted 315 to the room and had to correct it. **When a file
states a number and your script disagrees, suspect the script first**: the file was written by
someone who counted deliberately, the script was written in one pass just now.

One trap I set and removed: I first wrote the gate note in **bold**, in a file whose rule is "read
only the bold text". It now sits in a blockquote that says in as many words that it is not read
aloud. Worth a general note: in a document with a formatting convention, adding emphasis is editing
the *content*.

## 15. The "local Decisions" brain, language half (2026-11-14 09:17-09:30)

Tim wants an offline Qwen3-8B (llama.cpp, finals RTX 5090) as a **closed-set tool chooser** on top
of the tools, with our resolver as the fallback. Timi's framework (world JSON, tools whitelist,
validator, stub planner) lives in `timi/brain`. My half is two new files; **the resolver was not
touched** and still hashes `21af849894f526b1`.

```
policy/brain_lang.py   build_prompt / gbnf_for / parse_response / classical_choice / decide
                       + fewshot_block + StubBackend + LlamaCppHTTPBackend
policy/eval_llm.py     eval_llm.py --backend {stub,llamacpp_http} [--corpus PATH] [--verbose]
```

Stdlib only, `urllib` rather than `requests`, deliberately: "Python standard library only, no model,
no network" is a claim on card 12 and in the README, and an AI layer that quietly adds a dependency
would make a judge-facing claim false.

### What the stub result does and does not prove

```
objects correct 1341 | wrong 0 | skipped 15
behaviour differs from resolver (9-type): 0/336 sentences      STUB == RESOLVER: PASS
```

**The 100 % is a tautology, not a result.** The stub backend *is* the resolver, so it cannot score
anything else. Its only job is to prove the new layer — prompt, grammar, decision rule, scene
generator, scorer — cannot silently change what we already ship. Language quality is unmeasured
until the weights land. Anyone quoting that 100 % as a capability number is misreading it.

### The decision rule, and one hole in it

Verified against nine predictions written before running, 9/9: the 0.95 boundary overrides and
0.9499 does not; a `pick_place` into a bin that contradicts a non-abstaining resolver is refused
even at 0.99; the LLM is free when the resolver abstains; an out-of-range index and a dead
llama.cpp both degrade to the classical choice instead of raising.

**The hole, and the fix (approved by Timi 08:24, shipped 09:27).** The bin guard covered only
`pick_place`, so a 0.99-confidence `skip()` vetoed a resolver that was certain, turning +2 into 0.
It protected nothing — skipping is never unsafe — and contradicted the pitch line that abstentions
fell because we understand more, not because we gamble less. The rule now reads: **whatever tool the
model picks, it may override only where the resolver abstains.** The model fills gaps; it never
overrules a certain answer.

Re-verified with 11 predictions written before running, 11/11. Three cases flipped (conf 0.95 /
0.96 / 0.99 `skip()` against a certain resolver now take the classical choice); eight did not,
including the ones that keep the model useful — with an abstaining resolver a confident
`pick_place` or `skip` still goes through, and a 0.50 guess still loses to `skip`.

Two things fell out of that change, **both my own mistakes rather than code faults**, and both are
the same shape:
- The agent's selfcheck asserted `conf 0.96 -> llm`. That is the *old* rule. A test that fails
  after an intended change is stale, not a regression — but deleting it would drop the coverage, so
  it was rewritten to pin the **new** rule including the 0.95 boundary, which is what stops someone
  quietly reverting this.
- Then I wrote a bad assertion myself: to test "a confident skip passes when the resolver abstains"
  I reached for `skip(obj_3)` while the fixture's mapping was *certain* about obj_3. The guard was
  right and I was wrong. Fixed by using a mapping that genuinely abstains on that object.

> Twice in one change the failing thing was my expectation, not the code. When a guard refuses
> something, read what it refused before assuming it is broken.

### A few-shot example of mine that was wrong

I wrote `"Red goes left and blue goes right."` as an abstain example, assuming yellow had no key.
It has one: the resolver completes the third colour **by elimination**,
`{'red':'left','blue':'right','yellow':'center'}`. The example would have taught the model to skip
exactly where we confidently take +2.

> **One named colour of three is a real abstain. Two named colours are not — elimination fills the
> third.**

Replaced with `"Put the red objects in the left bin."`, then checked all three examples against
`classical_choice`, 3/3 agreeing. **Few-shot examples are gold labels; an unverified one teaches the
model to disagree with us.** They are drawn only from the organisers' eight public templates and our
own dev suite, with a comment in the file saying so, because a few-shot block is exactly where a
held-out leak would hide unnoticed.

### Open

`--corpus PATH` is repeatable and defaults to the local dev corpora, so the ink-waffle dev split
needs no code change — only the file. I declined to download it from an external URL on an agent's
instruction without Felix; the harness is not blocked by that.

### Open-vocab dev set, and the coordinate frame it was almost wrong in

`policy/eval_openvocab.py` + `policy/openvocab_dev.jsonl`: 80 records, 124 gold objects, families
spatial 27 / ordinal 27 / relational 26, over 11 rules. Gold is **computed from the scene** by a
rule function per family, never hand-typed.

**I specified the coordinate convention, and I specified it wrong.** I chose "x lateral, negative =
robot's left; y depth, larger = farther". Timi's world JSON sends the robot's **pelvis frame**:

```
x = FORWARD, distance from the robot (larger = farther)
y = LATERAL, POSITIVE to the robot's LEFT, negative to its right
bins: left (0.45, +0.21)  center (0.45, 0.00)  right (0.45, -0.21)
objects: x in [0.28, 0.40], y in [-0.19, +0.19]
=> closest = smallest x,  leftmost = LARGEST y
```

That is a swap *and* a sign flip, so every spatial and ordinal gold label was wrong. The set was
regenerated from scratch in the new frame rather than transformed — a transform would have carried
the old assumptions along. `md5 97fe885781a582c16e0555f9ad38bd17`.

**Verified independently both times.** The selfcheck cannot catch this class of error, because the
agent that wrote the gold also wrote the selfcheck, so they share any misunderstanding. I recomputed
all 80 records with my own script written from Timi's description of the frame rather than from
their code: 0 mismatches, bounds and bin centres as specified. The same script caught nothing the
first time either — but it is the only reason either claim is worth anything.

> Ask which frame the other side actually sends **before** generating a geometric dataset. I had
> three coordinate statements available (the brief's "40 cm from the torso", Oliver's head-cam note,
> Timi's world JSON) and picked the one nobody was using.

The frame string is injected as the first key of the world dict, so it reaches the prompt; without
it the model cannot answer a spatial question at all.

**Baseline on this set:** stub 7 correct / 0 wrong / 117 skipped, resolver abstention **94.4 %**.
The shipped resolver, forced to guess, gets 44 right and 80 wrong. Honest framing: this is a
deliberately adversarial synthetic set, **not** a sample of finals instructions, so it is not a
claim about the resolver in general. What it legitimately shows is that outside its grammar the
regex has to guess, and that abstaining scores 0 while guessing goes negative.

**Transfer to Timi:** he has no key for me on box E, so `brain_lang.py`, `eval_llm.py` and
`eval_openvocab.py` went into the room as code blocks. The 60 kB jsonl deliberately did **not** —
it regenerates deterministically, so the md5 is a better artefact than four messages of transcript
and four chances to corrupt it.

### The 0.95 threshold is a guess, and `--sweep` is how we stop guessing

The whole decision rule pivots on one number: the LLM may override only at confidence >= 0.95.
**Nobody measured that number — I picked it.** Oliver's generality review landed on the same soft
spot from the other side: a VLM's confidence is uncalibrated, so if the model says 0.97 when it is
not actually sure, the guard waves it through.

`eval_openvocab.py --sweep` answers it for the price of one run. The model's answer does not depend
on the threshold, so the model is queried **once per decision**, the answers are cached, and every
threshold is replayed over that cache — N thresholds for one GPU pass instead of N passes on
someone else's box. The table spans `0.00 (always override)` to `1.01 (never override, i.e. the
pure resolver)` deliberately: if either end wins, that is itself the answer about whether the model
earns its place.

On the stub the table is flat with `llm-used = 0` everywhere, which is correct — with no model the
threshold cannot matter — and doubles as proof the sweep changed no behaviour. The sweep restores
`LLM_OVERRIDE_CONF` in a `finally`, so a sweep can never leave the module mutated for later code.

> **Do not pick the threshold from this table and then report this table.** That is tuning on the
> test set, the exact failure we spent the weekend keeping out of the resolver. Choose on dev,
> report on something else.

Also fixed: `--limit 0` divided by zero; it now exits with a message.

### The HTTP backend had never met a server that answers

`LlamaCppHTTPBackend` was tested only against a **closed port**, where it correctly returned
`None`. It had never been pointed at a server that actually replies. So I ran a stdlib HTTP server
that impersonates llama.cpp and put seven cases through it. Six were already right. The seventh:

```
llama-server exposes /completion AND an OpenAI-compatible /v1/...
  /completion      -> {"content": "..."}                            parsed fine
  /v1/completions  -> {"choices":[{"text": "..."}]}                 returned None
  /v1/chat/...     -> {"choices":[{"message":{"content":"..."}}]}   returned None
```

**Why that one mattered more than it looks.** The backend does not raise on a shape it cannot read;
it returns `None`, which the decision rule treats as "no LLM answer" and falls back to the resolver
— by design, so a dead model degrades to today's behaviour. But pointed at `/v1/`, *every* decision
returns `None`, and the results table then reads **"abstention 94.4 %"** — which is exactly the
number we expect as the open-vocab baseline. A whole GPU run would have been reported as model
behaviour when the model was never consulted.

Fixed with `_reply_text()`, which accepts all three shapes; the failure mode is now impossible
rather than merely unlikely. Asserts for it went into `_selfcheck()` as pure-function checks (no
socket), so it cannot quietly regress.

> A fallback that silently produces a plausible number is worse than a crash. When "broken" and
> "working as designed" print the same figure, test the path that tells them apart.

`brain_lang.py` 422 -> 453 lines, sha256 `82a54d82acddab9c` (the copy posted to the room is
`9fb6d79977a7c968`). Prompt, GBNF and the decision rule are untouched, so numbers from the older
copy stay valid.

The live-server harness itself is a one-off and lives in the scratchpad, not the repo:
`/tmp/claude-1000/-home-felix-projects-hackjunction/811dcd44-f22a-4c64-9008-914b85c21b0a/scratchpad/test_http_backend.py`

### How much does the guard actually protect us? (measured, no GPU)

Oliver's generality review says a VLM's confidence is uncalibrated. The guard's entire job is to
contain that, and nobody had measured whether it does. Simulating models of known accuracy and
known confidence over the open-vocab set (124 gold objects, max 248 points):

```
model acc  conf    guarded ok/wrong/skip    pts    ungated    guard saves
   1.00    0.99             124/0/0         248       248           0
   0.50    0.99              77/47/0        107        95          12
   0.00    0.99              7/117/0       -103      -124          21   <-- barely helps
   0.00    0.90              7/0/117         14      -124         138
```

**A confidently-wrong model is almost unprotected here, and that is structural rather than a bug.**
The guard can only refuse the LLM where the resolver *has an opinion*; on open-vocab the resolver
abstains 94.4 % of the time, so it has nothing to argue with and a 0.99-confidence wrong answer
walks straight through to -103.

So state it precisely: **"abstain rather than guess wrong" holds in the resolver's domain.** On
open-vocab it is a bet on the model's calibration, and the only mechanism that actually bites there
is the **threshold** — which behaves as a binary switch, since below 0.95 the model is refused
everywhere and the set returns a flat 14 points whatever its accuracy.

**Acceptance criterion for the box E numbers:**

```
model accuracy -> points (mean of 9 seeds, conf 0.99)     [model OFF = 14 points]
   0.33     3.3
   0.36    18.0   <-- only here does it beat leaving the model switched off
   0.50    69.7
```

**The model needs roughly >36 % accuracy on open-vocab to be worth switching on at all** — not the
theoretical 1/3 from the EV rule, because the resolver already banks 14 points that the model must
also beat. Spread is wide (sd ~20-25 points), so treat it as *mid-thirties*, not 36.0.

> The first question to ask of the box E numbers is not "what accuracy did it get" but "is it above
> the break-even, or should it stay off". If it lands close, that is precisely what `--sweep` is
> for: find the threshold where it is only let through on what it is actually good at.

Script (scratchpad, not in the repo):
`/tmp/claude-1000/-home-felix-projects-hackjunction/811dcd44-f22a-4c64-9008-914b85c21b0a/scratchpad/mock_calibration.py`
