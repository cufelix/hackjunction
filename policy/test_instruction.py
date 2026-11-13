"""Tests for policy/instruction.py. Pure stdlib, no simulator, runs in under a second.

Two corpora:
  * the 288 training sentences, regenerated here from the organisers' own 8 templates
    (aaltonomous/interface/tasks.py, MIT) x 6 colour orders x 6 bin orders -- the sanity floor,
    where anything below 100% is a bug;
  * an adversarial set written by hand for the phrasings the finals can legitimately use:
    British "centre", bin-first word order, shape keys, compound (colour, shape) keys, and
    "everything else" -> the "default" key.
"""
from __future__ import annotations

import itertools
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from instruction import FIXED_MAPPING, resolve  # noqa: E402

# Verbatim from aaltonomous/interface/tasks.py::TEMPLATES (MIT, DigiTwin / Aalto University).
TEMPLATES = (
    "Put the {c0} objects in the {b0} bin, the {c1} objects in the {b1} bin and the {c2} objects in the {b2} bin.",
    "{C0} goes to the {b0} bin, {c1} to the {b1} bin, {c2} to the {b2} bin.",
    "Sort the objects: {c0} into {b0}, {c1} into {b1}, {c2} into {b2}.",
    "Place every {c0} object in the {b0} bin. {C1} objects belong in the {b1} bin, and {c2} ones in the {b2} bin.",
    "The {b0} bin is for {c0} objects, the {b1} bin for {c1}, the {b2} bin for {c2}.",
    "Move {c0} objects to the {b0} bin, {c1} objects to the {b1} bin, and {c2} objects to the {b2} bin.",
    "Drop the {c0} ones into the {b0} bin, the {c1} ones into the {b1} bin, the {c2} ones into the {b2} bin.",
    "Use the {b0} bin for {c0}, the {b1} bin for {c1} and the {b2} bin for {c2} objects.",
)
COLORS = ("red", "yellow", "blue")
BINS = ("left", "center", "right")
FIXED_INSTRUCTION = ("Put the red objects in the left bin, the yellow objects in the center bin "
                     "and the blue objects in the right bin.")


def training_corpus():
    """[(sentence, mapping)] for all 8 x 6 x 6 = 288 training instructions."""
    out = []
    for tpl in TEMPLATES:
        for colors in itertools.permutations(COLORS):
            for bins in itertools.permutations(BINS):
                w = {f"c{i}": c for i, c in enumerate(colors)}
                w.update({f"C{i}": c.capitalize() for i, c in enumerate(colors)})
                w.update({f"b{i}": b for i, b in enumerate(bins)})
                out.append((tpl.format(**w), dict(zip(colors, bins))))
    return out


# (sentence, expected mapping). Held-out shapes of instruction, not held-out content.
ADVERSARIAL = [
    # British spelling: the trap, because "centre" is never in the training set.
    ("Put the red objects in the centre bin, the blue objects in the left bin and the yellow "
     "objects in the right bin.", {"red": "center", "blue": "left", "yellow": "right"}),
    ("Sort them: blue to the centre, yellow to the right, red to the left.",
     {"blue": "center", "yellow": "right", "red": "left"}),
    # "middle" for the centre bin.
    ("Yellow in the middle bin, red on the left, blue on the right.",
     {"yellow": "center", "red": "left", "blue": "right"}),
    # Bin named first throughout.
    ("The right bin takes red, the left bin takes yellow, the middle one takes blue.",
     {"red": "right", "yellow": "left", "blue": "center"}),
    # Imperative, terse, no word "bin" at all.
    ("red left, blue centre, yellow right", {"red": "left", "blue": "center", "yellow": "right"}),
    # Leftmost / rightmost.
    ("Place blue in the leftmost bin, red in the rightmost bin, and yellow in the centre.",
     {"blue": "left", "red": "right", "yellow": "center"}),
    # Shape keys instead of colour keys.
    ("Put the cubes in the left bin, the cylinders in the center bin and the triangles in the right bin.",
     {"cube": "left", "cylinder": "center", "triangle": "right"}),
    ("Triangular prisms go left, cylinders go right, cubes go in the middle.",
     {"triangle": "left", "cylinder": "right", "cube": "center"}),
    # Compound (colour, shape) keys -- resolve_bin's most specific case.
    ("Put the red cubes in the left bin, the blue cylinders in the center bin and the yellow "
     "triangles in the right bin.",
     {("red", "cube"): "left", ("blue", "cylinder"): "center", ("yellow", "triangle"): "right"}),
    # A catch-all: one named colour, everything else elsewhere.
    ("Put the red objects in the left bin and everything else in the right bin.",
     {"red": "left", "default": "right"}),
    ("All blue items go in the centre bin; the rest go in the left bin.",
     {"blue": "center", "default": "left"}),
]


class TestTrainingCorpus(unittest.TestCase):
    def test_all_288_exact(self):
        bad = [(s, g, resolve(s)) for s, g in training_corpus() if resolve(s) != g]
        self.assertEqual(bad[:5], [], f"{len(bad)}/288 training sentences misparsed")

    def test_fixed_instruction(self):
        self.assertEqual(resolve(FIXED_INSTRUCTION), FIXED_MAPPING)

    def test_every_training_parse_is_a_bijection(self):
        for s, _ in training_corpus():
            m = resolve(s)
            self.assertEqual(sorted(m.values()), sorted(BINS), s)


class TestAdversarial(unittest.TestCase):
    def test_held_out_phrasings(self):
        bad = [(s, g, resolve(s)) for s, g in ADVERSARIAL if resolve(s) != g]
        for s, g, got in bad:
            print(f"\nMISS {s!r}\n  want {g}\n  got  {got}")
        self.assertEqual(len(bad), 0, f"{len(bad)}/{len(ADVERSARIAL)} adversarial sentences misparsed")


class TestNeverAbstains(unittest.TestCase):
    """A wrong guess and an abstention are both worth 0 in expectation; None is worth less than both,
    because the policy would have nowhere to put the object."""

    def test_garbage_in_still_yields_a_usable_mapping(self):
        for junk in ("", "   ", "please sort the objects", "do the needful", "put things away",
                     "aseta punaiset vasempaan laatikkoon"):
            m = resolve(junk)
            self.assertTrue(m, junk)
            self.assertTrue(all(v in BINS for v in m.values()), (junk, m))


if __name__ == "__main__":
    unittest.main(verbosity=2)
