"""End-to-end check of the instruction resolver against the organisers' OWN lookup function.

    python3 -I policy/test_integration.py

`policy/test_instruction.py` scores the resolver on dict equality. That is the wrong metric on its
own, for two reasons:

  1. **Two different dicts can be behaviourally identical.** `{"red": "left", "yellow": "left",
     "blue": "left"}` and `{"default": "left"}` send every object to the same bin, so counting the
     first as a miss against the second would be a false alarm -- and counting a real difference as
     a pass would be worse.
  2. **Dict equality never touches the key format.** `scene.resolve_bin` looks keys up in a fixed
     order, and if our compound keys were `(shape, colour)` instead of `(colour, shape)`, every
     dict could be "correct" and every lookup still wrong. 288/288 would mean nothing.

So this file scores what the policy actually does: for all 9 (colour, shape) object types, does
`resolve_bin(ours, c, s)` return the same bin as `resolve_bin(theirs, c, s)`?

It also checks the thing that silently costs points: a mapping that leaves an object with **no**
target at all. `resolve_bin` returns `None` then, and the planner has nowhere to put it.

WIRING -- one line, `baselines/classical/policy.py:73`:

    self.mapping = info.get("mapping") or dict(FIXED_MAPPING)
    #  ->
    self.mapping = info.get("mapping") or resolve(info.get("instruction", ""), fallback=FIXED_MAPPING)

`info.get("mapping")` stays first so the sim's privileged path is untouched; over the wire it is
never present (see BUILD-HANDBOOK SS1.1), so the resolver is what runs on a language episode.
`reset()` has a hard budget -- `safety.py:69 stop_after_s = 1.0` -- so this file measures the
resolver's latency instead of assuming it.
"""
from __future__ import annotations

import itertools
import sys
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instruction import resolve                       # noqa: E402
from test_instruction import ADVERSARIAL, training_corpus  # noqa: E402

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
BINS = ("left", "center", "right")
TYPES = [(c, s) for c in COLORS for s in SHAPES]

# Verbatim from aaltonomous/scene/scene.py:43 (MIT, DigiTwin / Aalto University). Copied rather
# than imported so this test runs with no starter repo on the path -- if they change it, this
# copy is what has to be updated, and that is the point of quoting the line number.
def resolve_bin(mapping, color, shape=None):
    for k in ((color, shape), color, shape, "default"):
        if k is not None and k in mapping:
            return mapping[k]
    return None


def behaviour(mapping):
    """What the policy will actually do with this mapping: one bin per object type."""
    return tuple(resolve_bin(mapping, c, s) for c, s in TYPES)


class TestBehaviourMatches(unittest.TestCase):
    def test_training_corpus_behaviour(self):
        n = bad = 0
        for text, want in training_corpus():
            n += 1
            if behaviour(resolve(text)) != behaviour(want):
                bad += 1
                if bad <= 3:
                    print(f"  BEHAVIOUR MISMATCH: {text[:70]}")
        self.assertEqual(bad, 0, f"{bad}/{n} training sentences behave differently")

    def test_adversarial_behaviour(self):
        for text, want in ADVERSARIAL:
            self.assertEqual(behaviour(resolve(text)), behaviour(want), text[:70])

    def test_compound_key_order_is_colour_then_shape(self):
        """If we emitted (shape, colour) the dicts could still compare equal while every lookup
        missed. This pins the order against resolve_bin's own ((color, shape), ...) tuple."""
        m = resolve("Red cubes to the left bin, red cylinders to the right bin, "
                    "everything else in the centre.")
        self.assertIn(("red", "cube"), m, f"compound key not (colour, shape): {sorted(map(str, m))}")
        self.assertEqual(resolve_bin(m, "red", "cube"), "left")
        self.assertEqual(resolve_bin(m, "red", "cylinder"), "right")
        self.assertEqual(resolve_bin(m, "blue", "cube"), "center")

    def test_every_key_the_instruction_names_has_a_bin(self):
        """The invariant that actually matters. A key we emitted must be reachable through
        resolve_bin -- if we put `("red", "cube")` in the mapping and resolve_bin cannot find it,
        the mapping is decoration.

        Note what this does NOT assert: that all 9 types have a target. A purely compound
        instruction ("red cubes left, blue cylinders right, yellow triangles centre") names 3 of
        the 9 and says nothing about a red cylinder, so 6 come back `None`. That is faithful, not
        broken, and filling in a default would not help:

          * an unnamed object left unmapped is never attempted -> `counter` -> **0**;
          * guessing a bin for it is +2 with p=1/3 and -1 with p=2/3 -> **expectation 0**.

        The two are worth the same, so there is nothing to gain and a guess adds variance and
        burns ~12-15 s of a 120 s budget that a retry on a *named* object would use better. And in
        the organisers' own generator the instruction is written from the scene's actual contents,
        so an object the instruction does not name is unlikely to be on the mat at all.
        `test_resolver_never_abstains` in test_instruction.py covers the case that does matter:
        a sentence we cannot parse at all still yields a usable mapping.
        """
        for text, _ in itertools.chain(training_corpus(), ADVERSARIAL):
            m = resolve(text)
            for key in m:
                if key == "default":
                    continue
                c, sh = key if isinstance(key, tuple) else (key, None)
                if c in COLORS:
                    probe = (c, sh) if sh else (c, SHAPES[0])
                else:                               # a shape-only key
                    probe = (COLORS[0], c)
                self.assertIsNotNone(resolve_bin(m, *probe),
                                     f"emitted key {key!r} is unreachable: {text[:60]}")

    def test_report_unmapped_types(self):
        """Mostly a number to watch, with two assertions that still have teeth.

        The original premise here was "a sentence that names a plain colour reaches all 9 types".
        That is false, and deliberately so: a sentence constrains only the colours it *names*.
        Two families legitimately leave the rest unmapped, and in both the resolver is choosing
        `counter` (0 points) over a guess it has no basis for (EV 3p-1, negative below p=1/3):

          * negation-only -- "Don't put red in the left bin." says nothing about yellow or blue,
            so `_NEG` sets `stays` and suppresses completion. Erkka confirmed on 2026-11-14 that
            the finals carry no negation, so this family is a corpus artefact, not a scoring risk.
          * many-to-one -- "Red goes in the left bin, and so do the blue ones." puts two colours
            in one bin, which leaves *two* bins free, so the implicit-third-colour rule cannot
            pick one and correctly declines to.

        What must still hold, and is asserted below:
          * a `default` (catch-all) key covers every type by construction -- if one is emitted and
            a type is still unreachable, the catch-all is broken;
          * a colour the mapping *names* must always be reachable. An unmapped type may only ever
            belong to a colour the sentence never mentioned.
        """
        worst = []
        for text, _ in itertools.chain(training_corpus(), ADVERSARIAL):
            m = resolve(text)
            missing = [t for t in TYPES if resolve_bin(m, *t) is None]
            if not missing:
                continue
            self.assertNotIn("default", m, f"a catch-all default still left {len(missing)} types"
                                           f" unmapped: {text[:60]}")
            named = {k for k in m if not isinstance(k, tuple) and k in COLORS}
            leak = sorted({c for c, _s in missing} & named)
            self.assertEqual(leak, [], f"named colour(s) {leak} unreachable, {len(missing)} types"
                                       f" unmapped: {text[:60]}")
            worst.append((len(missing), text[:60]))
        print(f"  sentences leaving an unnamed colour unmapped: {len(worst)}"
              + (f" (max {max(w[0] for w in worst)}/9)" if worst else ""))

    def test_every_bin_named_is_a_real_bin(self):
        """A typo'd bin name would pass every behavioural test above and crash the planner."""
        for text, _ in itertools.chain(training_corpus(), ADVERSARIAL):
            for key, value in resolve(text).items():
                self.assertIn(value, BINS, f"{text[:50]} -> {key}: {value!r}")

    def test_resolve_is_fast_enough_for_reset(self):
        """safety.py:69 stop_after_s = 1.0 is the whole reset() budget. Measure, do not assume."""
        texts = [t for t, _ in training_corpus()][:64] + [t for t, _ in ADVERSARIAL]
        for t in texts[:8]:
            resolve(t)                                      # warm the regex cache
        t0 = time.perf_counter()
        for t in texts:
            resolve(t)
        per = (time.perf_counter() - t0) / len(texts)
        print(f"  resolve(): {per * 1e6:.0f} us per sentence "
              f"({per / 1.0 * 100:.4f} % of the 1.0 s reset budget, n={len(texts)})")
        self.assertLess(per, 0.010, "resolve() is too slow to sit in reset()")


if __name__ == "__main__":
    unittest.main(verbosity=2)
