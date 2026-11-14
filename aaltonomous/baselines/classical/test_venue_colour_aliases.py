"""instruction.venue_colour_aliases: venue colour words in EN/FI/CS (inflections included) resolve in strict mode.

Run: PYTHONPATH=. python -m unittest aaltonomous/baselines/classical/test_venue_colour_aliases.py
"""
import unittest

from .config import load_config
from .normalize import normalize, strict_mapping, venue_colour_aliases
from .instruction import resolve
from .policy import ClassicalPolicy

# (language, colour word, internal class). Class is the resolver key: orange -> red, purple -> blue.
COLOURS = [
    ("en", "purple", "blue"), ("en", "violet", "blue"), ("en", "lilac", "blue"),
    ("en", "lavender", "blue"), ("en", "magenta", "blue"),
    ("en", "orange", "red"), ("en", "amber", "red"), ("en", "tangerine", "red"),
    ("en", "yellow", "yellow"), ("en", "gold", "yellow"),
    ("en", "red", "red"), ("en", "blue", "blue"),
    ("fi", "violetti", "blue"), ("fi", "violetin", "blue"), ("fi", "laventeli", "blue"),
    ("fi", "laventelin", "blue"), ("fi", "sininen", "blue"), ("fi", "sinistä", "blue"),
    ("fi", "oranssi", "red"), ("fi", "oranssia", "red"), ("fi", "meripihka", "red"),
    ("fi", "mandariini", "red"), ("fi", "punainen", "red"), ("fi", "punaisia", "red"),
    ("fi", "keltainen", "yellow"), ("fi", "keltaisen", "yellow"), ("fi", "kultainen", "yellow"),
    ("fi", "kultaisen", "yellow"),
    ("cs", "fialová", "blue"), ("cs", "levandulová", "blue"), ("cs", "modrá", "blue"),
    ("cs", "modrého", "blue"), ("cs", "oranžová", "red"), ("cs", "oranžového", "red"),
    ("cs", "jantarová", "red"), ("cs", "mandarinková", "red"), ("cs", "červená", "red"),
    ("cs", "červeného", "red"), ("cs", "žlutá", "yellow"), ("cs", "žluté", "yellow"),
    ("cs", "zlatá", "yellow"),
]
# (language, bin word, internal bin)
BINS = [
    ("en", "left", "left"), ("en", "right", "right"), ("en", "center", "center"),
    ("fi", "vasemmalle", "left"), ("fi", "oikealle", "right"), ("fi", "keskelle", "center"),
    ("cs", "vlevo", "left"), ("cs", "vpravo", "right"), ("cs", "uprostřed", "center"),
]
# Near misses: colours the venue does not have, or words that only look like one. Must abstain.
NEGATIVES = ["pink left", "green right", "the left one", "white center", "vaaleanpunainen vasemmalle",
             "zelená vlevo", "vihreä oikealle", "sinister left", "červenec vpravo", "the left one, please"]


def _strict(text, flag):
    cfg = load_config(overlay="finals_r22_coloralias" if flag else "finals_r21")
    return ClassicalPolicy(config=cfg, log=None)._instruction_mapping({"instruction": text})


class TestVenueColourTable(unittest.TestCase):
    def test_every_language_colour_bin_triple_resolves_with_flag_on(self):
        for (cl, colour, cls), (bl, bin_word, bin_) in (
                (c, b) for c in COLOURS for b in BINS if c[0] == b[0]):
            with self.subTest(lang=cl, colour=colour, bin=bin_word):
                text = f"{colour} {bin_word}"
                self.assertEqual(_strict(text, flag=True), {cls: bin_}, text)

    def test_sentence_form_with_flag_on(self):
        self.assertEqual(_strict("Purple goes left, orange goes right, yellow goes center.", True),
                         {"blue": "left", "red": "right", "yellow": "center"})
        self.assertEqual(_strict("Amber goes right.", True), {"red": "right"})

    def test_amber_is_orange_with_flag_on_and_normalize_off(self):
        # With normalize on (finals_r21) amber is already orange via normalize's colour names. The resolver alone reads
        # amber as yellow, so the flag is what makes amber orange when normalize is off.
        cfg = load_config(overlay="finals_r21")
        cfg["instruction"]["normalize"] = False
        cfg["instruction"]["venue_colour_aliases"] = True
        mapping = ClassicalPolicy(config=cfg, log=None)._instruction_mapping({"instruction": "amber right"})
        self.assertEqual(mapping, {"red": "right"})
        self.assertEqual(_strict("amber right", False), {"red": "right"})

    def test_near_misses_still_abstain_with_flag_on(self):
        for text in NEGATIVES:
            with self.subTest(text=text):
                self.assertEqual(_strict(text, flag=True), {"default": None}, text)

    def test_venue_colour_aliases_is_idempotent_on_canonical_words(self):
        for s in ("purple left", "orange right", "yellow center", "red left", "blue right"):
            self.assertEqual(venue_colour_aliases(s), s)

    def test_flag_off_by_default_and_off_in_the_finals_base(self):
        self.assertFalse(load_config(overlay="none")["instruction"]["venue_colour_aliases"])
        self.assertFalse(load_config(overlay="finals_r21")["instruction"]["venue_colour_aliases"])
        self.assertTrue(load_config(overlay="finals_r22_coloralias")["instruction"]["venue_colour_aliases"])

    def test_resolve_alone_is_unchanged_by_the_module(self):
        # The alias pass lives only in the policy path; the resolver itself is not modified.
        self.assertEqual(resolve("purple left", fallback={}), {"blue": "left"})


if __name__ == "__main__":
    unittest.main()
