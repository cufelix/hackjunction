"""T-BOX: clauses that send objects to a colour-named bin are dropped before the resolver (0 instead of -1)."""
import unittest

from .instruction import resolve
from .normalize import drop_colour_bin_clauses, normalize


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


if __name__ == "__main__":
    unittest.main()
