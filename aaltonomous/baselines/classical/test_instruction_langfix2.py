"""instruction.structured, second pass: colour+shape scope, complement phrases, intensifiers, Czech, amber, typos.
    python -m unittest aaltonomous.baselines.classical.test_instruction_langfix2 -v
Each helper call is the whole structured path (structured_normalize -> structured_final). Expectations are written as
per-object bins (scene.resolve_bin) so they hold however the mapping dict is keyed; None = abstain."""
import time
import unittest

from aaltonomous.scene.scene import resolve_bin

from .config import load_config
from .normalize import _s_core, normalize, structured_final, structured_normalize

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
        # The bin of each named object, not merely "three slots filled": a swap of those bins must fail.
        cases = (
            ("Put orange can in the leftmost bin, yellow cube in the center bin, blue triangle in the right bin.",
             {("red", "cylinder"): "left", ("yellow", "cube"): "center", ("blue", "triangle"): "right"}),
            ("The rightmost bin is for yellow cube, The leftmost bin is for red prism, The middle bin is for blue can.",
             {("yellow", "cube"): "right", ("red", "triangle"): "left", ("blue", "cylinder"): "center"}),
            ("The middle bin is for yellow can, The left bin is for red square block, The right bin is for blue prism.",
             {("yellow", "cylinder"): "center", ("red", "cube"): "left", ("blue", "triangle"): "right"}),
        )
        for text, want in cases:
            b = bins(text)
            for key, bin_ in want.items():
                self.assertEqual(b[key], bin_, text)
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

    def test_protected_near_words_are_not_colours(self):
        self.assertNotIn("purple", structured_normalize("people cubes go left").lower())
        self.assertNotIn("purple", structured_normalize("purpose cubes belong in the left bin").lower())
        self.assertNotIn("purple", structured_normalize("puddle cubes go to the right bin").lower())
        self.assertNotIn("violet", structured_normalize("violent prisms go left").lower())
        # the word is not a colour: abstain that shape, keep a different object the sentence does name
        b = bins("people cubes go left, yellow cylinders go right")
        self.assertIsNone(b[("blue", "cube")])
        self.assertIsNone(b[("red", "cube")])
        self.assertIsNone(b[("yellow", "cube")])
        self.assertEqual(b[("yellow", "cylinder")], "right")
        b = bins("violent prisms go left")
        self.assertTrue(all(b[(c, "triangle")] is None for c in COLOURS))
        b = bins("puddle cubes go to the right bin")
        self.assertTrue(all(b[(c, "cube")] is None for c in COLOURS))

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
        # An unparsed complement half abstains. Comparing the helper to itself would pass for any mapping.
        text = "Put blah in the left bin. For the remaining shapes, hmm."
        norm = structured_normalize(text)
        self.assertEqual(run(text), _s_core(norm))
        self.assertEqual(run(text), {"default": None})

    def test_colour_blocks_with_a_keyed_complement_only_map_the_cube(self):
        # "blocks" is cubes or every solid. The other shapes of that colour were not given a bin.
        b = bins("Yellow blocks go center. For the remaining shapes, red goes left and blue goes right.")
        self.assertEqual(b[("yellow", "cube")], "center")
        self.assertIsNone(b[("yellow", "cylinder")])
        self.assertIsNone(b[("yellow", "triangle")])
        self.assertEqual(b[("red", "cylinder")], "left")
        self.assertEqual(b[("red", "triangle")], "left")
        self.assertEqual(b[("blue", "cylinder")], "right")
        self.assertEqual(b[("blue", "triangle")], "right")


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

    def test_deliver_and_repair_verbs_are_not_right(self):
        # dopravte = deliver, napravte = correct. The adverbs are doprava / napravo.
        b = bins("dopravte červené kostky doleva")
        self.assertEqual(b[("red", "cube")], "left")
        self.assertTrue(all(b[(c, s)] is None for c in COLOURS for s in SHAPES if (c, s) != ("red", "cube")))
        b = bins("dopravte červené kostky doleva a modré válce doprava")
        self.assertEqual(b[("red", "cube")], "left")
        self.assertEqual(b[("blue", "cylinder")], "right")
        self.assertIsNone(b[("red", "cylinder")])
        self.assertIsNone(b[("blue", "cube")])
        b = bins("Napravte červené doleva a modré doprava")
        self.assertEqual(b[("red", "cube")], "left")
        self.assertEqual(b[("blue", "cube")], "right")
        self.assertIsNone(b[("yellow", "cube")])

    def test_lone_a_is_not_rewritten_to_and(self):
        # "a" is the left ordinal and the English article. Rewriting it makes the next bin leak backwards.
        norm = structured_normalize("modré do a, žluté doprava")
        self.assertIn(" do a", norm)
        b = bins("modré do a, žluté doprava")
        self.assertIsNone(b[("blue", "cube")])
        self.assertIsNone(b[("blue", "cylinder")])
        self.assertEqual(b[("yellow", "cube")], "right")
        art = structured_normalize("a yellow cube goes doleva and a blue cube goes doprava")
        self.assertNotIn("and and", art)
        self.assertTrue(art.lower().startswith("a "))
        b = bins("a yellow cube goes doleva and a blue cube goes doprava")
        self.assertIsNone(b[("yellow", "cube")])
        self.assertIsNone(b[("blue", "cube")])


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

    def test_amber_shape_does_not_paint_blue(self):
        # "amber cubes left, everything else right": amber abstains, blue takes the catch-all.
        b = bins("Put amber cubes left. Everything else right.")
        self.assertIsNone(b[("yellow", "cube")])
        self.assertIsNone(b[("red", "cube")])
        self.assertIsNone(b[("yellow", "cylinder")])
        for s in SHAPES:
            self.assertEqual(b[("blue", s)], "right")
        b = bins("amber cylinders left, the rest right")
        self.assertEqual(b[("blue", "cylinder")], "right")
        self.assertEqual(b[("blue", "cube")], "right")
        self.assertIsNone(b[("yellow", "cylinder")])
        self.assertIsNone(b[("red", "cylinder")])
        # a real shape-wide rule beside amber is abstained (0), not applied to blue
        b = bins("all cubes go left, and amber items go right")
        self.assertTrue(all(v is None for v in b.values()))


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
        # Exercise the pass-2 Czech token rewrites themselves; the full-finals multilingual grammar is a separate path.
        on = self._mapping(load_config(overlay="finals_r22_langzip",
                                       overrides={"instruction.multilingual": False}), text)
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


class TestBugsweepLeftovers(unittest.TestCase):
    def test_right_in_the_middle_bin_is_centre(self):
        for t in ("Put red cubes to the right in the middle bin", "Red cubes go to the right in the middle bin.",
                  "red cubes right in the middle bin"):
            self.assertEqual(bins(t)[("red", "cube")], "center", t)
        self.assertEqual(bins("Red cubes to the right bin")[("red", "cube")], "right")   # plain right unchanged

    def test_clause_leading_modal_can_is_all_objects_of_that_colour(self):
        for t in ("Red can go left.", "Red can go left and yellow can go right", "Yellow cubes right. Then red can be in the left bin"):
            b = bins(t)
            self.assertTrue(all(b[("red", s)] == "left" for s in SHAPES), (t, b))
        b = bins("Red can go left and yellow can go right")
        self.assertTrue(all(b[("yellow", s)] == "right" for s in SHAPES))
        # noun reading unchanged: "can" before a preposition / "goes" is still the cylinder
        self.assertEqual(bins("red can in the left bin")[("red", "cylinder")], "left")
        self.assertNotEqual(bins("red can in the left bin")[("red", "cube")], "left")
        self.assertNotIn("objects", structured_normalize("put the red can in the right bin"))


if __name__ == "__main__":
    unittest.main()
