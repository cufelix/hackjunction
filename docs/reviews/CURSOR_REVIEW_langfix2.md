# Adversarial review: timi/langfix2

Scope is `ea92c79` against `team/timi/langzip` (`01eecd4`): `normalize.py`, `policy.py`, `test_instruction_langfix2.py`. The new path runs only when `instruction.strict` and `instruction.structured` are both on (`policy.py:167`). `finals_r21` does not set `structured`. With the flag off the diff is not on the call path. This commit does not touch `act()`, joint commands, retries, the 120 s budget, or reset generation.

Scoring reminder used below: a correct bin is +2, a wrong bin is -1, abstain is 0. Bugs are wrong bins or tests that cannot fail.

## 1. High — `doprav` / `naprav` is rewritten to the right bin

`aaltonomous/baselines/classical/normalize.py:870`

The right-hand Czech rule is `\b(?:doprav|naprav|zprav)\w*`. That is the adverb *doprava* / *napravo* (“to the right”), but `\w*` also eats the verbs *dopravte* / *dopravit* (“deliver”) and *napravte* (“correct, fix”).

Failure: `dopravte červené kostky doleva` (“deliver the red cubes to the left”) normalises to `right red cube left`. `structured_final` returns `{("red", "cube"): "right"}`. Every red cube is placed on the right. Same for `dopravte červené kostky doleva a modré válce doprava`: red cubes go right, blue cylinders go right. The red ones are -1 each. *Napravte červené doleva a modré doprava* does the same to every red object.

Minimal fix: match the adverbs only, e.g. `\b(?:doprava|dopravo|napravo|zprava|vpravo)\b`, and keep the `pravý` inflection alternative. Do not use `doprav\w*` / `naprav\w*` / `zprav\w*`.

## 2. High — a lone `a` becomes `and` and inherits the next bin

`aaltonomous/baselines/classical/normalize.py:877` (applied at `:926`)

`_S_CZ_AND` rewrites every standalone `a`/`A` once any Czech cue is present. `a` is also the left-bin ordinal (`instruction.py` `_ORDINAL_BINS["a"] = "left"`) and the English article. The resolver already accepts a lone `a` as a list conjunction in `yellow left a blue right`, so the rewrite is not what makes the listed Czech sentence parse.

Failure: `modré do a, žluté doprava` (“blue into A, yellow to the right”) becomes `blue do and, yellow right`. Strict mapping is `{blue: "right", yellow: "right"}`. Every blue object goes right (-1). The same text with `a` left in place is `{blue: None, yellow: "right"}` (blue abstains). The substitution is what turns the abstain into a wrong bin.

Second consequence, 0 instead of +2: `a yellow cube goes doleva and a blue cube goes doprava` becomes `and yellow cube goes left and and blue cube goes right`, and strict abstains both colours. `Žluté do a, modré do b a červené do c` loses the A/B/C ordinals the same way.

Minimal fix: delete the `_S_CZ_AND` pass. Re-check `Žluté dejte doleva, fialové doprostřed a oranžové doprava` — after the direction rewrite, `Yellow dejte left, purple middle a orange right` already resolves to yellow left, blue center, red right.

## 3. High — amber’s shape key is applied to blue, and the catch-all is cleared

`aaltonomous/baselines/classical/normalize.py:997`

`_s_amber` sets `yellow` and `red` to None and forces `default` to None. It leaves a bare shape key in place. `xamber` is not a colour, so `xamber cubes left` has already been parsed as `cube -> left` (every cube). `resolve_bin` stops on a None colour, so yellow and red abstain, but blue has no colour key and still hits `cube`.

Failure: `Put amber cubes left. Everything else right.` Core mapping, before `_s_amber`, is `{cube: "left", default: "right"}`. After `_s_amber`: `{cube: "left", default: None, yellow: None, red: None}`. Blue cubes go left. They are not amber, and the sentence sends everything else right. -1 per blue cube. `amber cylinders left, the rest right` puts blue cylinders on the left the same way.

The three amber sentences in `test_instruction_langfix2.py` never give amber a shape word and a catch-all together, so they stay green.

Minimal fix: in `_s_amber`, drop bare shape keys (they are the widened amber phrase) and do not assign `default = None`. A None `yellow` / `red` already stops those colours from taking the catch-all, so blue still receives “everything else”. On the sentence above that yields blue right and yellow/red abstained. A sentence that also has a real shape-wide rule (“all cubes go left, and amber items …”) would abstain that rule rather than place it; that is 0, not -1.

## 4. Medium — the near-word guard is applied backwards

`aaltonomous/baselines/classical/normalize.py:901`

```python
c = _s_colour_typo(w) if w.lower() in PROTECTED or w.lower() not in _ins._KNOWN else None
```

`PROTECTED` is the list of words that must not be rewritten (`normalize.py:17`, including `_NEAR_REAL`). The condition repairs them. `_s_colour_typo` only rejects `VOCAB` and `_KNOWN`, not `PROTECTED`. Words that flip, same first letter, distance ≤1 or 2 with the same last letter: `people`, `purpose`, `puddle` → purple; `violent` → violet.

Failure: `people cubes go left, yellow cylinders go right` normalises to `purple cubes go left, ...`. Blue cubes go left. Flag-off `normalize` leaves `people` unchanged. `purpose cubes belong in the left bin` and `violent prisms go left` do the same to blue cubes and blue triangles. `puddle cubes go to the right bin` sends blue cubes right.

Minimal fix: call `_s_colour_typo` only when `w.lower() not in PROTECTED`. Unknown typos such as `pupre prism` are not in `PROTECTED` and still repair. Add one assertion that `structured_normalize("people cubes go left")` does not contain `purple`.

## 5. Medium — “colour blocks” plus a keyed complement never narrows to the cube

`aaltonomous/baselines/classical/normalize.py:1011` and `:968`

`_s_block_scope` is documented to map only the cube cell, because “yellow blocks” is either cubes or every solid of that colour. It returns immediately unless `default` is a real bin (`:983`). `structured_final` skips it entirely when `_s_complement` returns a dict (`:1011`).

Failure: `Yellow blocks go center. For the remaining shapes, red goes left and blue goes right.` Complement treats the first half as `yellow -> center` and copies that onto every yellow shape. Yellow cylinders and yellow triangles go center. Under the cube reading those two were not given a bin (the second clause never says yellow), so that is -1 each. The existing block tests use “everything else <one bin>”, which has no keys in the second half, so complement returns None and the cube-only path runs.

Minimal fix: when the first half matches `_S_COLOUR_BLOCKS`, store only `(colour, "cube")` from that half and leave the other shapes of that colour unnamed so the second half can fill them or they stay None. Apply that inside `_s_complement` before the per-object copy, not only in `_s_block_scope`.

## 6. Low — one test is true of any return value; another cannot see a wrong bin

`aaltonomous/baselines/classical/test_instruction_langfix2.py:110`

`test_unparsed_half_defers_to_the_unchanged_path` compares `run(text)` to `structured_final(structured_normalize(text))`. `run` is that same call (`:17`). The assertion is true for every string, including one that places every object in the wrong bin.

`test_orange_can_variants` (`:43`) only checks that exactly three of the nine object slots are non-None. A swap of those three bins, or the three slots, still passes. The current parser happens to fill the right three slots; the test would not notice if it stopped.

Minimal fix: compare the unparsed sentence to `strict_mapping` / `_s_core` on the same normalised text (or assert the whole mapping is `{default: None}`), and assert the bin of each named object in the can sentences.
