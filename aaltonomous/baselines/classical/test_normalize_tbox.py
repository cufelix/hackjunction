"""T-BOX: clauses that send objects to a colour-named bin are dropped before the resolver (0 instead of -1)."""
import unittest

from .instruction import resolve
from .normalize import drop_colour_bin_clauses, name_foreign, normalize


class TestColourBinClauses(unittest.TestCase):
    def test_wrong_bin_guesses_become_abstains(self):
        cases = {
            "Red objects go to the yellow box, blue to the left.": {"blue": "left"},
            "Put the red cubes in the orange bin and the blue ones in the left box.": {"blue": "left"},
            "Blue things in the right bin, red things in the orange basket.": {"blue": "right"},
        }
        for s, want in cases.items():
            got = resolve(normalize(s), fallback={})
            self.assertEqual({k: v for k, v in got.items() if k != "default"}, want, s)

    def test_subject_colour_bin_clause_is_dropped(self):
        got = resolve(normalize("Yellow box gets the blue cubes; red cubes go right."), fallback={})
        self.assertNotIn("yellow", got)
        self.assertNotIn("blue", got)
        self.assertEqual(set(got.values()) - {None}, {"right"})

    def test_correct_sentences_unchanged(self):
        for s in ("Sort by colour: red to the left box, yellow to the middle box, blue to the right box.",
                  "Put red in the left bin, yellow in the middle bin and blue in the right bin.",
                  "Put the orange cubes in the left bin.",
                  "The box on the left is orange; put the red cubes there.",
                  "Put the blue box in the left bin.",
                  "blue boxes go to the left"):
            self.assertEqual(drop_colour_bin_clauses(s), s, s)

    def test_position_word_keeps_a_colour_named_bin_clause(self):
        # the real bins ARE orange: "the orange box on the left" is a position, not a colour-named bin
        cases = {"Put the purple cubes in the orange box on the left.": {("blue", "cube"): "left"},
                 "Purple objects go in the orange box on the right, yellow ones in the orange box on the left "
                 "and orange ones in the middle orange box.": {"blue": "right", "yellow": "left", "red": "center"}}
        for s, want in cases.items():
            self.assertEqual(resolve(normalize(s), fallback={}), want, s)

    def test_everything_dropped_returns_the_original_sentence(self):
        # an emptied sentence = the resolver's FIXED fallback guess; the raw sentence is never worse than raw
        s = "Put the red cubes in the orange bin."
        self.assertEqual(drop_colour_bin_clauses(s), s)
        self.assertTrue(normalize(s).strip())

    def test_ordinal_bins_counted_from_a_side(self):
        cases = {"Lilac pieces must be placed in the third bin (left to right).": {"blue": "right"},
                 "Put the yellow things in the third box from the left.": {"yellow": "right"},
                 "Orange ones go in the first bin from the right.": {"red": "right"},
                 "Purple goes into the second one from the left.": {"blue": "center"},
                 "Put yellow in the first bin, counting from the left.": {"yellow": "left"}}
        for s, want in cases.items():
            got = resolve(normalize(s), fallback={})
            self.assertEqual({k: v for k, v in got.items() if k in want}, want, (s, normalize(s), got))
        for s in ("Put the first cube in the left bin.", "Yellow goes in the third bin."):
            self.assertEqual(normalize(s), s)   # no side word: untouched

    def test_count_from_right_preamble_mirrors_third_to_left(self):
        s = ("Count the bins from the right. Red objects go in the third bin, yellow objects in the second bin, "
             "blue objects in the first bin.")
        self.assertEqual(resolve(normalize(s), fallback={}),
                         {"red": "left", "yellow": "center", "blue": "right"})

    def test_finals_colour_names_and_typos(self):
        # real objects: purple / yellow / orange (resolver keys blue / yellow / red)
        cases = {"Sorting: pruple -> right; yellow -> leeft; oange -> centre.": {"blue": "right", "yellow": "left", "red": "center"},
                 "Put the lavender cubes left, amber ones right and lemon ones in the middle.":
                     {("blue", "cube"): "left", "red": "right", "yellow": "center"},
                 "violetti left, oranssit right, keltaiset middle": {"blue": "left", "red": "right", "yellow": "center"},
                 "fialová vlevo: fialové left, oranžové right, žluté center": {"blue": "left", "red": "right", "yellow": "center"}}
        for s, want in cases.items():
            got = resolve(normalize(s), fallback={})
            self.assertEqual({k: v for k, v in got.items() if k in want}, want, (s, normalize(s), got))
        self.assertEqual(normalize("That was violent, people."), "That was violent, people.")


class TestNormalizerRules(unittest.TestCase):
    """Fixes on top of the T-BOX normaliser: bare ordinals, regardless-of-colour, shape synonyms, Finnish/Czech words."""

    @staticmethod
    def _bins(s, keys):
        got = resolve(normalize(s), fallback={})
        return {k: v for k, v in got.items() if k in keys}

    def test_bare_ordinals_follow_the_anchor_from_the_left(self):
        s = ("Put the red cubes in the third bin from the left, the yellow cylinders in the second, "
             "and the blue triangles in the first.")
        self.assertEqual(normalize(s), "Put the red cubes in the right bin, the yellow cylinders in the middle bin, "
                                       "and the blue triangles in the left bin.")
        want = {("red", "cube"): "right", ("yellow", "cylinder"): "center", ("blue", "triangle"): "left"}
        self.assertEqual(self._bins(s, want), want)

    def test_bare_ordinals_follow_the_anchor_from_the_right(self):
        s = ("Put the red cubes in the first bin from the right, the yellow cylinders in the second, "
             "and the blue triangles in the third.")
        want = {("red", "cube"): "right", ("yellow", "cylinder"): "center", ("blue", "triangle"): "left"}
        self.assertEqual(self._bins(s, want), want)

    def test_bare_ordinal_negatives(self):
        # a bare ordinal that names no bin ("the first red object") is not a bin: untouched
        s = "Put the first red object in the left bin, and the blue cubes in the third bin from the left."
        self.assertIn("the first red object", normalize(s))
        # "at first" is not a place: untouched
        s = "Put the red cubes in the third bin from the left, and at first the yellow cylinders go to the middle."
        self.assertIn("at first", normalize(s))
        # anchors pointing both ways, first anchor from the left: the resolver handles the sentence, no rewrite
        s = "Put the red cubes in the first bin from the left, the yellow cylinders in the third bin from the right."
        self.assertEqual(normalize(s), s)

    def test_regardless_of_colour_phrasings(self):
        for form in ("whatever their colour", "regardless of their colour", "no matter their colour",
                     "no matter the colour", "regardless of colour"):
            s = f"Put all the cubes in the left bin, {form}."
            self.assertEqual(normalize(s), "Put all the cubes in the left bin, regardless of colour.", form)
        self.assertEqual(resolve(normalize("Put all the cubes in the left bin, whatever their colour."), fallback={}),
                         {"cube": "left"})

    def test_regardless_negative(self):
        s = "Whatever shape it is, the red ones go left."
        self.assertEqual(normalize(s), s)

    def test_shape_synonyms_the_resolver_missed(self):
        cases = {"Red hexahedra go left, and yellow things go right.": {("red", "cube"): "left", "yellow": "right"},
                 "Red columns go left, and yellow things go right.": {("red", "cylinder"): "left", "yellow": "right"},
                 "Red prismatic go left, and yellow things go right.": {("red", "triangle"): "left", "yellow": "right"},
                 "Red cuboids go left, and yellow things go right.": {("red", "cube"): "left", "yellow": "right"}}
        for s, want in cases.items():
            self.assertEqual(resolve(normalize(s), fallback={}), want, s)

    def test_shape_synonym_negative(self):
        # "roll" as a verb is not a shape: the singular is not mapped, so the sentence is untouched
        s = "Blue ones roll to the left."
        self.assertEqual(normalize(s), s)

    def test_finnish_and_czech_words(self):
        self.assertEqual(normalize("Kuutiot vasemmalle, sylinterit oikealle."),
                         "Cube left, cylinder right.")
        self.assertEqual(normalize("Kolmiot keskelle."), "Triangle middle.")
        self.assertEqual(normalize("Laatikko vasemmalla, kostka oikealla."), "Bin left, cube right.")
        self.assertEqual(normalize("Vlevo válec, vpravo hranol, uprostřed koš."), "Left cylinder, right triangle, middle bin.")

    def test_foreign_words_are_whole_words_only(self):
        for s in ("Put the kosher cubes in the left bin.", "The prisms go right.", "Keep the kostrel out of it.",
                  "Oikeus decided the matter."):
            self.assertEqual(name_foreign(s), s, s)
        self.assertEqual(normalize("Put the prisms in the left bin."), "Put the prisms in the left bin.")


class TestParaStressFixes(unittest.TestCase):
    """R48 (para): a typo'd compound bin word stays a bin clause; "and <colour> can" after a conjunction is the noun."""

    def test_typo_leftmost_is_a_bin_word(self):
        self.assertEqual(normalize("Cylinders go to the centre bin; scratch that, cylinders go to the lfetmost bin."),
                         "Cylinders go to the centre bin; scratch that, cylinders go to the leftmost bin.")
        m = strict("Cylinders go to the centre bin; scratch that, cylinders go to the lfetmost bin.")
        self.assertIsNone(m.get("cylinder"))           # two different bins for one key: abstain, not the first

    def test_conjunction_colour_can_is_the_cylinder_noun(self):
        self.assertEqual(normalize("Put the red can in the left bin and blue can into the right bin."),
                         "Put the red cylinder in the left bin and blue cylinder into the right bin.")
        self.assertEqual(normalize("Red goes left and blue can go right."), "Red goes left and blue can go right.")
        self.assertEqual(resolve(normalize("Orange triangle goes left, yellow cuboid into the middle bin, "
                                           "and blue can into the rightmost bin."), fallback={}),
                         {("red", "triangle"): "left", ("yellow", "cube"): "center", ("blue", "cylinder"): "right"})


def strict(text):
    from .normalize import strict_mapping
    n = normalize(text)
    return strict_mapping(n, resolve(n, fallback={}))

if __name__ == "__main__":
    unittest.main()
