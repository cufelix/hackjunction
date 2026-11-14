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
from instruction import FIXED_MAPPING, _BIN_WORDS, resolve  # noqa: E402

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
    # Parallel lists (colours and bins as two lists, zipped by position), both orders.
    ("Red, yellow and blue go to the left, center and right bins.",
     {"red": "left", "yellow": "center", "blue": "right"}),
    ("The left, center and right bins take red, yellow and blue.",
     {"red": "left", "yellow": "center", "blue": "right"}),
    ("The right, left and middle bins are for blue, red and yellow.",
     {"blue": "right", "red": "left", "yellow": "center"}),
    ("Red, blue and yellow go in the center, left and right bins respectively.",
     {"red": "center", "blue": "left", "yellow": "right"}),
    ("Red, yellow and blue go to the left, centre and right bins.",
     {"red": "left", "yellow": "center", "blue": "right"}),
    ("The left, centre and right bins take red, yellow and blue.",
     {"red": "left", "yellow": "center", "blue": "right"}),
    ("red, yellow, blue -> left, centre, right", {"red": "left", "yellow": "center", "blue": "right"}),
    ("Red and blue go in the left and right bins, everything else in the centre.",
     {"red": "left", "blue": "right", "default": "center"}),
    # Trap: looks like a list but is item-by-item pairing; the list fix must not touch it.
    ("Red to the left bin, yellow and blue to the right bin and the centre bin respectively.",
     {"red": "left", "yellow": "right", "blue": "center"}),
    # 1. Negation: resolve by elimination, never invert. One bin excluded -> guess among the rest.
    ("Don't put red in the left bin.", {"red": "center"}),
    ("Red doesn't go in the center bin.", {"red": "left"}),
    ("No red in the right bin.", {"red": "left"}),
    ("Put yellow in the left bin, but not the blue ones.", {"yellow": "left", "blue": "right"}),
    ("Put red in the left bin and yellow in the right bin, but blue does not go in the left bin.",
     {"red": "left", "yellow": "right", "blue": "center"}),
    # 2. Colour lexicon: hyphenated, exotic and spaced compound colour words.
    ("Put the cherry-red ones in the right bin, the mustard ones in the left bin and the "
     "sky-blue ones in the center bin.", {"red": "right", "yellow": "left", "blue": "center"}),
    ("Ruby goes to the right bin, sapphire to the left bin, mustard to the center bin.",
     {"red": "right", "blue": "left", "yellow": "center"}),
    ("Cherry red in the center bin, sky blue in the left bin, yellow in the right bin.",
     {"red": "center", "blue": "left", "yellow": "right"}),
    # 3. Scene preamble: a descriptive first sentence names colours but no bins.
    ("There are red, yellow and blue objects on the table. Put red in the left bin, yellow in the "
     "center bin and blue in the right bin.", {"red": "left", "yellow": "center", "blue": "right"}),
    ("I can see three objects. Red goes left, yellow center, blue right.",
     {"red": "left", "yellow": "center", "blue": "right"}),
    # 4. "otherwise" is the catch-all.
    ("Put red in the left bin, otherwise use the right bin.", {"red": "left", "default": "right"}),
    ("Red objects go left; otherwise the center bin.", {"red": "left", "default": "center"}),
    # 5. Ordinal / positional bin names.
    ("Left to right: red, yellow, blue.", {"red": "left", "yellow": "center", "blue": "right"}),
    ("Bin 1 red, bin 2 yellow, bin 3 blue.", {"red": "left", "yellow": "center", "blue": "right"}),
    ("The first bin takes red, the second yellow, the third blue.",
     {"red": "left", "yellow": "center", "blue": "right"}),
    # 6. References: "so do the blue ones" inherits the bin; a swap with explicit targets.
    ("Red goes in the left bin, and so do the blue ones.", {"red": "left", "blue": "left"}),
    ("Yellow to the center. Swap red and blue: red right, blue left.",
     {"yellow": "center", "red": "right", "blue": "left"}),
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


# Bin references built from OTHER bins: "between the left and right ones" names the centre bin even
# though it contains the words left and right. (sentence, expected mapping)
BIN_REFERENCES = [
    ("The bin between the left and right ones is for yellow; red goes left, blue right.",
     {"yellow": "center", "red": "left", "blue": "right"}),
    ("Yellow goes in the bin between the left and right bins; red goes left, blue right.",
     {"yellow": "center", "red": "left", "blue": "right"}),
    ("Yellow goes between the left and right ones, red goes left, blue goes right.",
     {"yellow": "center", "red": "left", "blue": "right"}),
    ("Yellow goes in the bin between them, red goes left, blue goes right.",
     {"yellow": "center", "red": "left", "blue": "right"}),
    ("Red goes left, blue goes right, yellow goes in the one in between.",
     {"red": "left", "blue": "right", "yellow": "center"}),
    ("Red goes left, blue goes right, yellow goes in the bin after the left one.",
     {"red": "left", "blue": "right", "yellow": "center"}),
    ("Yellow goes in the bin after the left one; red goes left, blue goes right.",
     {"yellow": "center", "red": "left", "blue": "right"}),
    ("Red goes left, yellow goes in the bin before the right one, blue goes right.",
     {"red": "left", "yellow": "center", "blue": "right"}),
    ("Blue goes in the bin before the right one; red goes left, yellow goes right.",
     {"blue": "center", "red": "left", "yellow": "right"}),
    # The left bin has exactly one neighbour, so "next to the left one" can only be the centre.
    ("Yellow goes in the bin next to the left one, red goes left, blue goes right.",
     {"yellow": "center", "red": "left", "blue": "right"}),
    ("Red goes in the far left bin, yellow in the middle, blue in the far right bin.",
     {"red": "left", "yellow": "center", "blue": "right"}),
]


class TestBinReferences(unittest.TestCase):
    def test_bin_named_by_other_bins(self):
        bad = [(s, g, resolve(s)) for s, g in BIN_REFERENCES if resolve(s) != g]
        for s, g, got in bad:
            print(f"\nMISS {s!r}\n  want {g}\n  got  {got}")
        self.assertEqual(len(bad), 0, f"{len(bad)}/{len(BIN_REFERENCES)} bin references misparsed")


class TestDeliberateDuplicates(unittest.TestCase):
    def test_deliberate_duplicate_bin_is_preserved(self):
        # Two bin words named, a doubled bin was asked for (the sealed many2one shape).
        self.assertEqual(resolve("Red goes right, yellow goes right, blue goes left."),
                         {"red": "right", "yellow": "right", "blue": "left"})

    def test_within_clause_many_to_one_is_preserved(self):
        self.assertEqual(resolve("Red and yellow go in the left bin, blue goes in the right bin."),
                         {"red": "left", "yellow": "left", "blue": "right"})


class TestCompletion(unittest.TestCase):
    """A key the sentence names but that got no bin scores 0; completion only ever touches those."""

    def test_one_key_unmapped_one_bin_free_gets_it(self):
        m = resolve("Cubes go left, cylinders go right, and triangles are sorted last.")
        self.assertEqual(m, {"cube": "left", "cylinder": "right", "triangle": "center"})

    def test_two_keys_unmapped_get_free_bins_in_sentence_order(self):
        self.assertEqual(resolve("Red goes left; yellow and blue go in the remaining bin."),
                         {"red": "left", "yellow": "center", "blue": "right"})
        self.assertEqual(resolve("Cubes go left, cylinders go in the remaining bin, triangles go in the other bin."),
                         {"cube": "left", "cylinder": "center", "triangle": "right"})

    def test_explicit_leave_alone_is_not_completed(self):
        for s, left, rest in (
                ("Put blue in the left bin, yellow in the right bin, and leave the red ones where they are.",
                 "red", {"blue": "left", "yellow": "right"}),
                ("Don't move the blue ones; red goes left, yellow goes right.", "blue", {"red": "left", "yellow": "right"}),
                ("Let the yellow ones stay; red goes left, blue goes right.", "yellow", {"red": "left", "blue": "right"}),
                ("Red goes left, blue goes right, and leave yellow where it is.", "yellow", {"red": "left", "blue": "right"})):
            m = resolve(s)
            self.assertNotIn(left, m, s)
            for k, v in rest.items():
                self.assertEqual(m[k], v, s)

    def test_fully_mapped_sentences_are_untouched(self):
        for s, want in training_corpus()[::37]:
            self.assertEqual(resolve(s), want, s)

    def test_all_keys_bound_duplicate_does_not_complete(self):
        self.assertEqual(resolve("Red goes right, yellow goes right, blue goes left."),
                         {"red": "right", "yellow": "right", "blue": "left"})


class TestLeaveVsPlacement(unittest.TestCase):
    """leave/let/ignore only mean "leave alone" when the key gets no destination."""

    def test_leave_and_let_with_a_destination_are_placements(self):
        want = {"red": "left", "yellow": "center", "blue": "right"}
        for s in ("Let red go left, yellow center, blue right.",
                  "Leave red in the left bin, yellow in the center bin, blue in the right bin."):
            self.assertEqual(resolve(s), want, s)

    def test_leave_alone_phrasings_still_leave_alone(self):
        for s in ("Red goes left, blue goes right, and leave yellow alone.",
                  "Red goes left, blue goes right, and leave yellow on the table.",
                  "Red goes left, blue goes right, and let yellow stay put.",
                  "Red goes left, blue goes right, ignore yellow."):
            m = resolve(s)
            self.assertNotIn("yellow", m, s)
            self.assertEqual((m["red"], m["blue"]), ("left", "right"), s)


class TestCatchAllKeepsExplicitColour(unittest.TestCase):
    """Regression: a wider catch-all swallowed the explicit colour, leaving only {'default': X}."""

    def test_colour_key_survives_a_catch_all(self):
        m = resolve("Put the red objects in the left bin and everything else in the right bin.")
        self.assertIn("red", m)
        self.assertEqual(m, {"red": "left", "default": "right"})


class TestBinLabelDefinitions(unittest.TestCase):
    def test_partial_definition_gives_undefined_labels_the_unused_bins_in_order(self):
        self.assertEqual(resolve("Box A is the right box. Red goes in box A, yellow in box B, blue in box C."),
                         {"red": "right", "yellow": "left", "blue": "center"})
        self.assertEqual(resolve("Bin 3 is the left bin. Red goes in bin 1, yellow in bin 2, blue in bin 3."),
                         {"red": "center", "yellow": "right", "blue": "left"})
    def test_definition_overrides_the_abc_default(self):
        self.assertEqual(resolve("Bin A is the left bin, bin B is the right bin and bin C is the middle one. "
                                 "Put red in bin C, yellow in bin A and blue in bin B."),
                         {"red": "center", "yellow": "left", "blue": "right"})

    def test_reverse_copula(self):
        self.assertEqual(resolve("The left bin is bin C, the centre bin is bin A, the right bin is bin B. "
                                 "Put red in bin A, yellow in bin B, blue in bin C."),
                         {"red": "center", "yellow": "right", "blue": "left"})

    def test_partial_definition_and_no_definition_keep_defaults(self):
        want = {"red": "left", "yellow": "center", "blue": "right"}
        self.assertEqual(resolve("The box on the left is box A. Red goes in box A, yellow in box B, blue in box C."), want)
        self.assertEqual(resolve("Put red in bin A, yellow in bin B, blue in bin C."), want)

    def test_contradictory_definition_still_usable(self):
        m = resolve("Bin A is the left bin, bin B is the left bin. Put red in bin A, yellow in bin B, blue in bin C.")
        self.assertTrue(m)
        self.assertTrue(all(v in BINS for v in m.values()), m)


class TestImpliedThirdColour(unittest.TestCase):
    """Two colours on two different bins, one bin free, no leave phrase: the unnamed third colour
    takes the free bin. Every guard below is a sentence where the rule must stay out."""

    def test_implied_third_colour_gets_the_free_bin(self):
        for s in ("Red goes left, blue goes right, and the last one goes in the middle.",
                  "The red objects go in the left bin, the blue objects in the right bin, the third colour in the middle one.",
                  "Red goes left, blue goes right, and the other one goes in the middle."):
            self.assertEqual(resolve(s), {"red": "left", "blue": "right", "yellow": "center"}, s)

    def test_no_third_colour_reference_fires_deliberately(self):
        # Documented outcome: a two-colour, two-bin sentence implies the third colour takes the last bin.
        self.assertEqual(resolve("Red goes in the left bin and blue goes in the right bin."),
                         {"red": "left", "blue": "right", "yellow": "center"})

    def test_leave_alone_does_not_fire(self):
        self.assertEqual(resolve("Put red in the left bin and blue in the right bin, and leave the yellow ones alone."),
                         {"red": "left", "blue": "right"})
        self.assertNotIn("yellow", resolve("Red goes left, blue goes right, leave everything else on the counter."))

    def test_default_key_does_not_fire(self):
        self.assertEqual(resolve("Red in the left bin, blue in the right bin, and the rest in the middle."),
                         {"red": "left", "blue": "right", "default": "center"})

    def test_shape_key_does_not_fire(self):
        self.assertEqual(resolve("Red goes left, cubes go right."), {"red": "left", "cube": "right"})

    def test_many_to_one_does_not_fire(self):
        self.assertEqual(resolve("Red goes right, blue goes right."), {"red": "right", "blue": "right"})
        self.assertEqual(resolve("Red and blue go in the left bin."), {"red": "left", "blue": "left"})

    def test_deliberate_duplicates_untouched(self):
        # Three keys already: the rule's two-key precondition is unreachable here.
        m = resolve("Red goes right, yellow goes right, blue goes left.")
        self.assertEqual(len(m), 3)
        self.assertEqual(m, {"red": "right", "yellow": "right", "blue": "left"})


class TestRestStays(unittest.TestCase):
    """A keyless "leave/don't move everything else" tail means the leftovers stay on the counter (0):
    no default bin, no completed third colour, and it must not corrupt an explicitly named key."""
    RL = {"red": "left", "blue": "right"}

    def test_leave_tail_creates_no_default(self):
        for s in ("Red goes left, blue goes right, leave everything else on the counter.",
                  "Put red in the left bin and blue in the right bin, leave everything else on the counter.",
                  "Put red in the left bin and blue in the right bin and leave the rest on the counter.",
                  "Red left, blue right, everything else stays on the counter."):
            self.assertEqual(resolve(s), self.RL, s)
        self.assertEqual(resolve("Put the red ones in the left bin and leave everything else where it is."),
                         {"red": "left"})

    def test_negated_tail_does_not_corrupt_named_key(self):
        for s in ("Red left, blue right, don't move anything else.",
                  "Red left, blue right, do not move anything else.",
                  "Red left, blue right, don't touch anything else.",
                  "Put red in the left bin, blue in the right bin, don't move anything else."):
            self.assertEqual(resolve(s), self.RL, s)

    def test_keyed_negation_and_real_catch_all_untouched(self):
        self.assertEqual(resolve("Put the red objects in the left bin and the blue objects in the right bin, "
                                 "and do not move the yellow ones."), self.RL)
        self.assertEqual(resolve("Put the red objects in the left bin and everything else in the right bin."),
                         {"red": "left", "default": "right"})
        self.assertEqual(resolve("Red left, blue right."), {"red": "left", "blue": "right", "yellow": "center"})


class TestColourAliases(unittest.TestCase):
    """Lab wording: purple/violet/lilac/magenta -> blue, orange -> red (but a bin's orange is left alone)."""

    def check(self, sentence, want):
        self.assertEqual(resolve(sentence), want, sentence)

    def test_purple_family_is_blue(self):
        for w in ("purple", "violet", "lilac", "magenta"):
            self.check(f"Put the {w} objects in the left bin.", {"blue": "left"})

    def test_all_three_families(self):
        self.check("Put violet in the left bin, yellow in the center, red in the right.",
                   {"blue": "left", "yellow": "center", "red": "right"})
        self.check("Put purple in the left bin, yellow in the center bin, orange in the right bin.",
                   {"blue": "left", "yellow": "center", "red": "right"})

    def test_two_aliases_one_key(self):
        self.check("Magenta goes left, lilac goes left.", {"blue": "left"})

    def test_orange_object_is_red(self):
        self.check("Put the orange objects in the left bin.", {"red": "left"})
        self.check("Put the yellow ones in the left bin and the blue ones in the right bin, orange goes center.",
                   {"yellow": "left", "blue": "right", "red": "center"})
        self.check("Blue goes right, yellow goes left, and the orange, middle.",
                   {"blue": "right", "yellow": "left", "red": "center"})

    def test_orange_bin_is_not_a_key(self):
        self.check("Put the red objects in the left orange box", {"red": "left"})  # the guard
        self.check("Put the purple blocks in the orange bin on the right.", {"blue": "right"})
        for noun in ("box", "boxes", "bin", "bins", "container", "containers", "crate", "crates",
                     "basket", "baskets"):
            self.check(f"Put the red objects in the left orange {noun}.", {"red": "left"})

    def test_orange_object_and_orange_bin(self):
        self.check("Put the orange objects in the left orange box.", {"red": "left"})

    def test_orange_bin_first(self):
        # No bin position is named, so there is nothing to resolve: the module's documented fallback.
        self.assertEqual(resolve("Orange boxes are for the purple ones."), FIXED_MAPPING)

    def test_box_is_still_not_a_bin_word(self):
        self.assertNotIn("box", _BIN_WORDS)

    def test_inert_without_alias_words(self):
        self.check("Put the red objects in the left bin and the blue objects in the right bin.",
                   {"red": "left", "blue": "right", "yellow": "center"})


class TestNeverAbstains(unittest.TestCase):
    """Guessing beats leaving the object on the counter whenever p(right) > 1/3: a wrong bin costs -1
    against the counter's 0, so a uniform guess (3p - 1 = 0) merely ties it. Only a systematically
    inverted parse (p < 1/3) is worse than abstaining; None is worse still (nowhere to put the object)."""

    def test_garbage_in_still_yields_a_usable_mapping(self):
        for junk in ("", "   ", "please sort the objects", "do the needful", "put things away",
                     "aseta punaiset vasempaan laatikkoon"):
            m = resolve(junk)
            self.assertTrue(m, junk)
            self.assertTrue(all(v in BINS for v in m.values()), (junk, m))


if __name__ == "__main__":
    unittest.main(verbosity=2)
