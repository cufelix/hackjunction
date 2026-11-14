"""instruction.strict: the mapping abstains rather than guesses (normalize.strict_mapping). Run:
    python -m unittest aaltonomous.baselines.classical.test_instruction_strict -v
"""
import unittest

from .instruction import resolve
from .normalize import normalize, strict_mapping, structured_mapping


def strict(text):
    n = normalize(text)
    return strict_mapping(n, resolve(n, fallback={}))


def structured(text):
    n = normalize(text)
    parsed = structured_mapping(n)
    return parsed if parsed is not None else strict_mapping(n, resolve(n, fallback={}))


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


class TestStructuredMapping(unittest.TestCase):
    def test_respectively_and_in_that_order_zip_equal_lists(self):
        cases = {
            "Red, yellow and blue go left, center and right respectively.":
                {"red": "left", "yellow": "center", "blue": "right"},
            "Put purple, yellow, orange into right, left, center, in that order.":
                {"blue": "right", "yellow": "left", "red": "center"},
            "Left, center and right receive cubes, cylinders and triangles, respectively.":
                {"cube": "left", "cylinder": "center", "triangle": "right"},
            "Red cubes, yellow cylinders and blue triangles belong in right, center and left, respectively.":
                {("red", "cube"): "right", ("yellow", "cylinder"): "center",
                 ("blue", "triangle"): "left"},
        }
        for text, want in cases.items():
            self.assertEqual(structured(text), want, text)

    def test_zip_must_be_complete_and_unambiguous(self):
        for text in (
            "Red, yellow and blue go left and right respectively.",
            "Red and red go left and right respectively.",
            "Red and yellow go left, center and right in that order.",
            "Red, banana and yellow go left and right respectively.",
            "Red and yellow belong near left and right respectively.",
            "Red and yellow go from left and right respectively.",
            "Left and right receive no red and yellow respectively.",
            "Left and right go to red and yellow respectively.",
            "Red and yellow receive left and right respectively.",
            "Put left and right into red and yellow respectively.",
            "Red and yellow and go left and right respectively.",
        ):
            self.assertEqual(structured_mapping(normalize(text)), {"default": None}, text)

    def test_directional_lists_work_from_either_side(self):
        self.assertEqual(structured("Left to right: red, yellow, blue."),
                         {"red": "left", "yellow": "center", "blue": "right"})
        self.assertEqual(structured("Reading from right to left: cubes, cylinders, triangles."),
                         {"cube": "right", "cylinder": "center", "triangle": "left"})
        self.assertEqual(structured_mapping(normalize("Left to right: red goes to yellow, blue.")),
                         {"default": None})
        self.assertEqual(structured_mapping(normalize("Left to right: red, yellow, blue and")),
                         {"default": None})

    def test_colon_lists_include_key_first_bin_first_and_multiple_sentences(self):
        cases = {
            "Left: red. Center: yellow. Right: blue.":
                {"red": "left", "yellow": "center", "blue": "right"},
            "Red: right; yellow: center; blue: left.":
                {"red": "right", "yellow": "center", "blue": "left"},
            "In the left bin: red cubes; in the center bin: yellow cylinders; in the right bin: blue triangles.":
                {("red", "cube"): "left", ("yellow", "cylinder"): "center",
                 ("blue", "triangle"): "right"},
        }
        for text, want in cases.items():
            self.assertEqual(structured(text), want, text)

    def test_verb_less_positional_rows_can_span_sentences(self):
        self.assertEqual(structured("Left bin, red. Center bin, yellow. Right bin, blue."),
                         {"red": "left", "yellow": "center", "blue": "right"})

    def test_colon_list_abstains_when_one_entry_has_no_partner(self):
        self.assertEqual(structured_mapping(normalize("Left: red. Center: yellow. Right:")), {"default": None})
        self.assertEqual(structured_mapping(normalize("Left: red. Right: red.")), {"default": None})
        self.assertEqual(structured_mapping(normalize("Left: red center: yellow right:")), {"default": None})

    def test_conditionals_bind_the_condition_key_to_the_action_bin(self):
        text = ("If an object is red, put it in the left bin. If it is yellow, put it in center. "
                "If it is blue, put it right.")
        self.assertEqual(structured(text), {"red": "left", "yellow": "center", "blue": "right"})
        self.assertEqual(structured("If it is a red cube, send it right. If it is a yellow cylinder, send it left."),
                         {("red", "cube"): "right", ("yellow", "cylinder"): "left"})

    def test_conditional_with_multiple_possible_keys_abstains(self):
        for text in ("If it is red or blue, put it left.",
                     "If it is red, move it away from the left bin.",
                     "If it is yellow, take it out of the right bin.",
                     "If it is blue, put it beside the center bin.",
                     "If it is red, move the left bin.",
                     "If it is yellow, take the right bin.",
                     "If it is blue, put the center bin in it.",
                     "If it is red, move it and the left bin.",
                     "If it is red, move it and the red object left.",
                     "If it is red, it and the red object goes left."):
            self.assertEqual(structured_mapping(normalize(text)), {"default": None}, text)

    def test_ordinal_lists_count_from_either_side(self):
        cases = {
            "Count the bins from the right. Red goes in the third bin; yellow in the second bin; blue in the first bin.":
                {"red": "left", "yellow": "center", "blue": "right"},
            "Number the bins starting from the left. Cubes go in the 3rd bin; cylinders in the 2nd bin; triangles in the 1st bin.":
                {"cube": "right", "cylinder": "center", "triangle": "left"},
        }
        for text, want in cases.items():
            self.assertEqual(structured(text), want, text)

    def test_ordinal_list_abstains_on_a_missing_or_competing_position(self):
        for text in ("Count the bins from the right. Red goes in the third bin; yellow goes somewhere.",
                     "Count the bins from the right. Red goes in the third bin on the left.",
                     "Count the bins from the left. Red avoids the first bin; yellow avoids the second bin."):
            self.assertEqual(structured_mapping(normalize(text)), {"default": None}, text)

    def test_structured_lists_reject_stay_avoid_and_forbidden_semantics(self):
        for text in ("Left to right: red should stay put, yellow remains on the table, blue goes nowhere.",
                     "Red: avoid left. Yellow: center.",
                     "Left bin, red is forbidden. Center bin, yellow.",
                     "Left: red. Right: blue. Actually reverse that.",
                     "Left bin, red. Right bin, blue. Ignore those rules."):
            self.assertEqual(structured_mapping(normalize(text)), {"default": None}, text)

    def test_unrecognised_plain_sentence_defers_to_the_existing_strict_layer(self):
        self.assertIsNone(structured_mapping(normalize("Red goes left and yellow goes right.")))


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

    def test_structured_flag_is_default_off_and_ab_overlay_enables_it(self):
        from .config import load_config
        self.assertFalse(load_config(overlay="none")["instruction"]["structured"])
        self.assertFalse(load_config(overlay="finals_r22")["instruction"]["structured"])
        cfg = load_config(overlay="finals_r22_langzip")
        self.assertTrue(cfg["instruction"]["structured"])
        self.assertEqual(self._policy_mapping(cfg, "Left: red. Center: yellow. Right: blue."),
                         {"red": "left", "yellow": "center", "blue": "right"})

    def test_structured_flag_off_preserves_the_existing_resolution_path(self):
        from .config import load_config
        cfg = load_config(overlay="finals_r22")
        text = "Red, yellow and blue go left, center and right respectively."
        n = normalize(text)
        self.assertEqual(self._policy_mapping(cfg, text), strict_mapping(n, resolve(n, fallback={})))


if __name__ == "__main__":
    unittest.main()
