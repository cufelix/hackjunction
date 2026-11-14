"""instruction.strict: the mapping abstains rather than guesses (normalize.strict_mapping). Run:
    python -m unittest aaltonomous.baselines.classical.test_instruction_strict -v
"""
import unittest

from .instruction import resolve
from .normalize import normalize, strict_mapping


def strict(text):
    n = normalize(text)
    return strict_mapping(n, resolve(n, fallback={}))


class TestStrictMapping(unittest.TestCase):
    def test_unparsed_text_abstains_on_everything_not_fixed(self):
        self.assertEqual(strict("Put it somewhere sensible, maybe."), {"default": None})
        self.assertEqual(strict("qqq zzz"), {"default": None})

    def test_negation_abstains_the_clause_keys_only(self):
        m = strict("Put red in the left bin, but not the blue ones.")
        self.assertEqual(m["red"], "left")
        self.assertIsNone(m.get("blue"))

    def test_condition_abstains_its_clause(self):
        m = strict("Put red in the left bin unless it is a cube.")
        self.assertIsNone(m.get("red"))

    def test_table_placement_abstains_its_clause(self):
        m = strict("Put the yellow cubes on the table and red in the left bin.")
        self.assertIsNone(m.get("yellow"))
        self.assertEqual(m["red"], "left")

    def test_relation_superlative_abstains_its_clause(self):
        m = strict("Put the red cubes in the bin nearest to you, and yellow in the right bin.")
        self.assertIsNone(m.get("red"))
        self.assertEqual(m["yellow"], "right")

    def test_swap_abstains_the_whole_sentence(self):
        self.assertEqual(strict("Red goes left, then swap, yellow goes right."), {"default": None})

    def test_move_from_bin_abstains_the_clause(self):
        m = strict("Move the red cubes from the left bin to the right bin.")
        self.assertIsNone(m.get("red"))
        self.assertIsNone(m.get(("red", "cube")))

    def test_quantity_abstains_the_clause(self):
        m = strict("Only one red cube goes left; the yellow ones go right.")
        self.assertIsNone(m.get("red"))
        self.assertIsNone(m.get(("red", "cube")))
        self.assertEqual(m["yellow"], "right")

    def test_one_object_with_two_simultaneous_destinations_abstains(self):
        m = strict("Put the yellow object into both the left bin and the right bin at the same time.")
        self.assertIsNone(m.get("yellow"))

    def test_catch_all_keeps_default(self):
        self.assertEqual(strict("Red goes left, yellow goes right, everything else goes centre."),
                         {"red": "left", "yellow": "right", "default": "center"})

    def test_trigger_drops_default(self):
        m = strict("Everything except the red ones goes right.")
        self.assertNotIn("default", m)
        self.assertIsNone(m.get("red"))

    def test_unnamed_colour_is_not_inferred_by_elimination(self):
        m = strict("Red goes left and yellow goes right.")
        self.assertEqual(m, {"red": "left", "yellow": "right"})
        self.assertNotIn("blue", m)

    def test_correct_sentences_are_unchanged(self):
        s = "Put all red cubes in the left bin, yellow cylinders in the center bin, and blue triangles in the right bin."
        self.assertEqual(strict(s), resolve(normalize(s)))

    def test_colour_named_bin_is_not_an_object_key(self):
        m = strict("Yellow objects must be placed in the orange box in the middle.")
        self.assertEqual(m, {"yellow": "center"})

    def test_bin_colour_clause_keeps_the_object_key(self):
        m = strict("Golden pieces should go into the purple box on the left.")
        self.assertEqual(m.get("yellow"), "left")
        self.assertIsNone(m.get("blue"))

    def test_never_raises_on_garbage(self):
        for text in ("", "   ", None, "\U0001F642" * 200, "x" * 5000, "red " * 2000):
            m = strict_mapping(text, {} if text is None else {"red": "left"})
            self.assertIsInstance(m, dict)
        self.assertEqual(strict_mapping("", {}), {})
        self.assertEqual(strict_mapping("\U0001F642", {}), {"default": None})
        self.assertEqual(strict_mapping("x" * 5000, {}), {"default": None})


class TestStrictFlag(unittest.TestCase):
    @staticmethod
    def _policy_mapping(cfg, text):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy.__new__(ClassicalPolicy)
        pol.cfg, pol.log = cfg, lambda *_: None
        pol._cap_text = ClassicalPolicy._cap_text.__get__(pol)
        return ClassicalPolicy._instruction_mapping(pol, {"instruction": text})

    def test_flag_off_is_the_normalize_path_and_flag_on_is_strict(self):
        from .config import load_config
        text = "Put the cans in the right receptacle and the bricks in the left receptacle."
        for overlay, flag in (("rc1", False), ("rc1_strict", True)):
            cfg = load_config(overlay=overlay)
            self.assertEqual(cfg["instruction"]["strict"], flag)
            m = self._policy_mapping(cfg, text)
            n = normalize(text)
            self.assertEqual(m, strict_mapping(n, resolve(n, fallback={})) if flag else resolve(n))

    def test_flag_off_policy_mapping_identical_to_rc1(self):
        from .config import load_config
        texts = ["Put all red cubes in the left bin.", "Put it somewhere.", "Red goes left, but not blue."]
        base = load_config(overlay="rc1")
        off = load_config(overlay="rc1")
        off["instruction"]["strict"] = False
        for t in texts:
            a = self._policy_mapping(base, t)
            b = self._policy_mapping(off, t)
            self.assertEqual(a, b, t)

    def test_empty_text_truthily_abstains_in_strict_mode(self):
        from .config import load_config
        cfg = load_config(overlay="rc1_strict")
        self.assertEqual(self._policy_mapping(cfg, ""), {"default": None})


if __name__ == "__main__":
    unittest.main()
