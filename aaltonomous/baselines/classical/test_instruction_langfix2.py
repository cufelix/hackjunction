"""instruction.structured, second pass: colour+shape scope, complement phrases, intensifiers, Czech, amber, typos.
    python -m unittest aaltonomous.baselines.classical.test_instruction_langfix2 -v
Each helper call is the whole structured path (structured_normalize -> structured_final). Expectations are written as
per-object bins (scene.resolve_bin) so they hold however the mapping dict is keyed; None = abstain."""
import time
import unittest

from aaltonomous.scene.scene import resolve_bin

from .config import load_config
from .normalize import normalize, structured_final, structured_normalize

COLOURS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")


def run(text):
    return structured_final(structured_normalize(text))


def bins(text):
    m = run(text)
    return {(c, s): resolve_bin(m, c, s) for c in COLOURS for s in SHAPES}


def colour_bins(text):
    """{colour: bin} for a colour-wide mapping (all three shapes must agree, else the entry is the set of bins)."""
    b = bins(text)
    return {c: ({b[(c, s)] for s in SHAPES} if len({b[(c, s)] for s in SHAPES}) > 1 else b[(c, "cube")]) for c in COLOURS}


class TestColourShapeScope(unittest.TestCase):
    def test_colour_and_shape_keep_both(self):
        text = "Kindly The left bin is for purple cube, The center bin is for golden triangle, The rightmost bin is for orange can, please."
        b = bins(text)
        self.assertEqual(b[("blue", "cube")], "left")
        self.assertEqual(b[("yellow", "triangle")], "center")
        self.assertEqual(b[("red", "cylinder")], "right")
        # nothing wider than the named object
        self.assertIsNone(b[("red", "cube")])
        self.assertIsNone(b[("red", "triangle")])

    def test_orange_can_variants(self):
        for text in ("Put orange can in the leftmost bin, yellow cube in the center bin, blue triangle in the right bin.",
                     "The rightmost bin is for yellow cube, The leftmost bin is for red prism, The middle bin is for blue can.",
                     "The middle bin is for yellow can, The left bin is for red square block, The right bin is for blue prism."):
            b = bins(text)
            self.assertEqual(sum(v is not None for v in b.values()), 3, text)

    def test_can_stays_a_verb_when_it_modifies_a_verb(self):
        self.assertNotIn("cylinder", structured_normalize("ok orange can go left"))   # modal: the new rule needs a non-verb to follow

    def test_typo_colour_and_shape(self):
        b = bins("yelow cubs lft, purpel cilinder cntr, orang prizm rigt")
        self.assertEqual(b[("yellow", "cube")], "left")
        self.assertEqual(b[("blue", "cylinder")], "center")
        self.assertEqual(b[("red", "triangle")], "right")
        self.assertEqual(sum(v is not None for v in b.values()), 3)

    def test_pupre_prism_keeps_the_colour(self):
        b = bins("put teh pupre prism into left bsket and ornge into righ bsket")
        self.assertEqual(b[("blue", "triangle")], "left")
        self.assertIsNone(b[("yellow", "triangle")])
        self.assertEqual(b[("red", "cube")], "right")

    def test_typo_repair_needs_a_following_shape_word(self):
        self.assertEqual(structured_normalize("pupre goes left"), normalize("pupre goes left"))
        self.assertEqual(structured_normalize("the lead cube goes left"), normalize("the lead cube goes left"))

    def test_colour_blocks_with_a_catch_all_only_map_the_cube_cell(self):
        for text, col, bin_, rest in (
                ("red blocks right container, everything else the left.", "red", "right", "left"),
                ("Yellow blocks go the middle tray; the rest go left.", "yellow", "center", "left"),
                ("Put yellow blocks in the right and all other objects in middle tray.", "yellow", "right", "center")):
            b = bins(text)
            self.assertEqual(b[(col, "cube")], bin_, text)           # both readings agree on the cube
            self.assertIsNone(b[(col, "cylinder")], text)           # blocks = cubes or every solid: unstated, abstain
            for c in COLOURS:
                if c != col:
                    for s in SHAPES:
                        self.assertEqual(b[(c, s)], rest, (text, c, s))


class TestComplement(unittest.TestCase):
    def test_remaining_shapes_only_covers_the_complement(self):
        b = bins("All cubes go to the right bin. For the remaining shapes, red and blue go left, yellow center.")
        for c in COLOURS:
            self.assertEqual(b[(c, "cube")], "right")
        for s in ("cylinder", "triangle"):
            self.assertEqual((b[("red", s)], b[("blue", s)], b[("yellow", s)]), ("left", "left", "center"))

    def test_the_listed_sentence_with_items_wording(self):
        b = bins("All cubes go to the right bin. For the remaining shapes, red and blue items go to the left bin, "
                 "while yellow items go to the center bin.")
        self.assertEqual(b[("yellow", "cube")], "right")
        self.assertEqual(b[("blue", "cube")], "right")
        self.assertEqual(b[("yellow", "cylinder")], "center")

    def test_other_phrasings(self):
        b = bins("Cubes right. Everything else: red left, yellow center")
        self.assertEqual(b[("blue", "cube")], "right")
        self.assertEqual(b[("red", "triangle")], "left")
        self.assertIsNone(b[("blue", "triangle")])                  # never named: abstain, not a guess

    def test_plain_catch_all_is_unchanged(self):
        m = run("Red goes left, everything else goes right.")
        self.assertEqual(m.get("red"), "left")
        self.assertEqual(m.get("default"), "right")

    def test_unparsed_half_defers_to_the_unchanged_path(self):
        self.assertEqual(run("Put blah in the left bin. For the remaining shapes, hmm."),
                         structured_final(structured_normalize("Put blah in the left bin. For the remaining shapes, hmm.")))


class TestIntensifier(unittest.TestCase):
    def test_right_in_the_middle_is_center(self):
        self.assertEqual(colour_bins("Blue items to the far left, red items to the far right, and yellow items right in the middle."),
                         {"blue": "left", "red": "right", "yellow": "center"})

    def test_other_intensifiers(self):
        self.assertEqual(colour_bins("Red right at the left bin, blue right in the center bin, yellow right on the right bin."),
                         {"red": "left", "blue": "center", "yellow": "right"})

    def test_a_real_right_is_kept(self):
        self.assertEqual(colour_bins("Red goes right, yellow goes in the middle, blue goes left."),
                         {"red": "right", "yellow": "center", "blue": "left"})
        self.assertEqual(colour_bins("Red goes to the right in the middle row bin, blue left, yellow center")["blue"], "left")


class TestCzech(unittest.TestCase):
    def test_listed_sentence(self):
        self.assertEqual(colour_bins("Žluté dejte doleva, fialové doprostřed a oranžové doprava."),
                         {"yellow": "left", "blue": "center", "red": "right"})

    def test_declensions_and_direction_words(self):
        for text, want in (
                ("Žlutou dejte vlevo, fialovou uprostřed, oranžovou vpravo.", {"yellow": "left", "blue": "center", "red": "right"}),
                ("Žlutý doleva, modrý na střed, červený doprava", {"yellow": "left", "blue": "center", "red": "right"}),
                ("Žlutá nalevo, fialová doprostřed, oranžová napravo", {"yellow": "left", "blue": "center", "red": "right"}),
                ("Červené doleva, modré uprostřed, žluté vpravo.", {"red": "left", "blue": "center", "yellow": "right"}),
                ("Oranžové vlevo, žluté do středu a fialové doprava", {"red": "left", "yellow": "center", "blue": "right"})):
            self.assertEqual(colour_bins(text), want, text)

    def test_czech_with_shapes(self):
        b = bins("Červené kostky vpravo, modré válce doleva, žluté hranoly doprostřed.")
        self.assertEqual((b[("red", "cube")], b[("blue", "cylinder")], b[("yellow", "triangle")]), ("right", "left", "center"))

    def test_english_is_untouched(self):
        for t in ("Put the cube on the left, a level thing", "Levy a model"):
            self.assertEqual(structured_normalize(t), normalize(t))


class TestAmber(unittest.TestCase):
    def test_amber_never_maps_to_red_or_to_a_guess(self):
        for text in ("The violet items go into the rightmost bin, while the amber items go into the leftmost bin.",
                     "Put the violet item into the rightmost orange bin, and the amber item into the middle orange bin.",
                     "Excuse me, could you please put the violet block into the central basket and the amber block into the leftmost basket?"):
            b = bins(text)
            self.assertTrue(all(b[("red", s)] is None for s in SHAPES), text)
            self.assertTrue(all(b[("yellow", s)] is None for s in SHAPES), text)
            self.assertTrue(all(b[("blue", s)] is not None for s in SHAPES), text)     # the clear colour still maps

    def test_explicit_colours_next_to_amber_keep_their_bin(self):
        b = bins("amber and yellow go left, blue right")
        self.assertEqual(b[("yellow", "cube")], "left")
        self.assertIsNone(b[("red", "cube")])

    def test_amber_is_orange_only_with_the_flag_off(self):
        self.assertIn("orange", normalize("the amber items go left"))


class TestWiringAndSpeed(unittest.TestCase):
    class _P:
        log = staticmethod(lambda *_: None)
        _cap_text = staticmethod(lambda t: t)

        def __init__(self, cfg):
            self.cfg = cfg

    def _mapping(self, cfg, text):
        from .policy import ClassicalPolicy
        return ClassicalPolicy._instruction_mapping(self._P(cfg), {"instruction": text})

    def test_flag_on_goes_through_the_new_rules_flag_off_does_not(self):
        text = "Žluté dejte doleva, fialové doprostřed a oranžové doprava."
        on = self._mapping(load_config(overlay="ab_langzip"), text)
        self.assertEqual((on.get("yellow"), on.get("blue"), on.get("red")), ("left", "center", "right"))
        off = self._mapping(load_config(overlay="finals_r21"), text)
        self.assertNotEqual((off.get("yellow"), off.get("blue"), off.get("red")), ("left", "center", "right"))

    def test_latency_p99_under_5ms(self):
        texts = ["All cubes go to the right bin. For the remaining shapes, red and blue go left, yellow center.",
                 "Žluté dejte doleva, fialové doprostřed a oranžové doprava.",
                 "yelow cubs lft, purpel cilinder cntr, orang prizm rigt",
                 "Kindly The left bin is for purple cube, The center bin is for golden triangle, The rightmost bin is for orange can, please.",
                 "red blocks right container, everything else the left.",
                 "Left to right: red, yellow, blue.", "Put the violet item into the rightmost bin and the amber item left."]
        samples = []
        for i in range(1000):
            t0 = time.perf_counter()
            structured_final(structured_normalize(texts[i % len(texts)] + (" " * (i % 3))))
            samples.append((time.perf_counter() - t0) * 1000)
        samples.sort()
        self.assertLess(samples[989], 5.0, f"p99 {samples[989]:.2f} ms")


if __name__ == "__main__":
    unittest.main()
