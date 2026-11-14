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


if __name__ == "__main__":
    unittest.main()
