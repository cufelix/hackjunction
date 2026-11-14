"""Regressions for instruction.hazard_guard and the parser-derived brain guard."""
from __future__ import annotations

import unittest

from aaltonomous.scene.scene import resolve_bin

from .config import load_config
from .normalize import structured_final, structured_normalize


COLOURS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")


def mapping(text: str, guarded: bool = True) -> dict:
    return structured_final(structured_normalize(text, hazard_guard=guarded), hazard_guard=guarded)


def bins(text: str, guarded: bool = True) -> dict:
    got = mapping(text, guarded)
    return {(colour, shape): resolve_bin(got, colour, shape)
            for colour in COLOURS for shape in SHAPES}


class TestHazardGuard(unittest.TestCase):
    def test_flag_registration_and_wiring(self):
        self.assertFalse(load_config(profile="none", overlay="none")["instruction"]["hazard_guard"])
        self.assertFalse(load_config(overlay="submission")["instruction"]["hazard_guard"])
        self.assertTrue(load_config(overlay="finals_r22_langzip")["instruction"]["hazard_guard"])

        from .policy import ClassicalPolicy

        class Policy:
            log = staticmethod(lambda *_: None)
            _cap_text = staticmethod(lambda text: text)

            def __init__(self, overlay):
                self.cfg = load_config(overlay=overlay)

        text = {"instruction": "Red shouldnt go left."}
        on = ClassicalPolicy._instruction_mapping(Policy("finals_r22_langzip"), text)
        off = ClassicalPolicy._instruction_mapping(Policy("finals_r22"), text)
        self.assertIsNone(resolve_bin(on, "red", None))
        self.assertEqual(resolve_bin(off, "red", None), "left")

    def test_one_key_cannot_name_two_bins(self):
        cases = (
            ("Put red cubes in the left and right bins. Put blue in center.", ("red", "cube")),
            ("Red goes left or right. Put blue in center.", ("red", "cylinder")),
            ("Either red goes left or right. Put blue in center.", ("red", "triangle")),
        )
        for text, key in cases:
            with self.subTest(text=text):
                got = bins(text)
                self.assertIsNone(got[key])
                self.assertEqual(got[("blue", "cube")], "center")
        got = mapping("Red goes left or right.")
        self.assertIsNone(resolve_bin(got, "red", None))

    def test_anchored_ordinals_keep_rewrites_when_anchors_conflict(self):
        got = bins("Red goes to the second bin from the left. "
                   "Blue goes to the second bin from the right.")
        self.assertTrue(all(got[("red", shape)] == "center" for shape in SHAPES))
        self.assertTrue(all(got[("blue", shape)] == "center" for shape in SHAPES))

    def test_unsupported_conditionals_abstain(self):
        for marker in ("when", "provided that", "as long as", "in the event"):
            text = f"Put the red objects in the left bin {marker} a blue object is present."
            with self.subTest(marker=marker):
                got = bins(text)
                self.assertTrue(all(got[(colour, shape)] is None
                                    for colour in ("red", "blue") for shape in SHAPES))
                self.assertIsNone(resolve_bin(mapping(text), "red", None))
        supported = bins("If it is red, put it in the left bin.")
        self.assertTrue(all(supported[("red", shape)] == "left" for shape in SHAPES))

    def test_rest_keeps_instance_and_dimension_scope(self):
        got = bins("Put one red cube in the left bin. Put the rest of the red cubes in the right bin.")
        self.assertIsNone(got[("red", "cube")])
        for determiner in ("a", "an", "the", "single"):
            text = f"Put {determiner} red cube in the left bin. Put the rest of the objects in the right bin."
            self.assertIsNone(bins(text)[("red", "cube")], text)

        got = bins("Red cubes go right. "
                   "For the remaining shapes, red and blue go left, yellow goes center.")
        self.assertEqual(got[("red", "cube")], "right")
        self.assertIsNone(got[("blue", "cube")])
        self.assertIsNone(got[("yellow", "cube")])
        for shape in ("cylinder", "triangle"):
            self.assertEqual(got[("red", shape)], "left")
            self.assertEqual(got[("blue", shape)], "left")
            self.assertEqual(got[("yellow", shape)], "center")

        got = bins("Red cubes go right, and for the remaining shapes red goes left.")
        self.assertEqual(got[("red", "cube")], "right")
        self.assertEqual(got[("red", "cylinder")], "left")
        self.assertEqual(got[("red", "triangle")], "left")

        got = bins("Red cubes go right. For the remaining colours, cubes go left, "
                   "cylinders go center, triangles go left.")
        self.assertEqual(got[("red", "cube")], "right")
        self.assertIsNone(got[("red", "cylinder")])
        self.assertIsNone(got[("red", "triangle")])
        for colour in ("blue", "yellow"):
            self.assertEqual(got[(colour, "cube")], "left")
            self.assertEqual(got[(colour, "cylinder")], "center")
            self.assertEqual(got[(colour, "triangle")], "left")

    def test_broad_narrow_conflict_abstains_overlap(self):
        for text in ("Red cubes go left. All red objects go right.",
                     "All red objects go right. Red cubes go left.",
                     "Red cubes go left. All red objects go right. Everything else goes center.",
                     "All red objects go left. All cubes go right."):
            with self.subTest(text=text):
                got = bins(text)
                self.assertIsNone(got[("red", "cube")])
                if "All red objects go right" in text:
                    self.assertEqual(got[("red", "cylinder")], "right")
                    self.assertEqual(got[("red", "triangle")], "right")

    def test_misspelled_control_words_abstain_affected_key(self):
        cases = (
            ("Everything exept red goes left.", "red"),
            ("Everything exluding blue goes right.", "blue"),
            ("Red shouldnt go left.", "red"),
            ("Red mustnt go left.", "red"),
            ("Red isnt assigned left.", "red"),
            ("Red arent assigned left.", "red"),
            ("Red cant go left.", "red"),
            ("Red wont go left.", "red"),
            ("Red dont go left.", "red"),
        )
        for text, colour in cases:
            with self.subTest(text=text):
                got = mapping(text)
                self.assertTrue(all(resolve_bin(got, colour, shape) is None for shape in SHAPES))
                self.assertIsNone(resolve_bin(got, colour, None))

        # The new checks are genuinely flag-gated: the old structured path remains unchanged.
        self.assertEqual(bins("Red shouldnt go left.", guarded=False)[("red", "cube")], "left")


class TestFinalizedBrainMapping(unittest.TestCase):
    class _Policy:
        brain = object()
        log = staticmethod(lambda *_: None)
        _cap_text = staticmethod(lambda text: text)

        def __init__(self):
            self.cfg = load_config(overlay="finals_r22_langzip")

    def test_brain_guard_uses_final_multilingual_mapping(self):
        from .policy import ClassicalPolicy

        info = {"instruction": "Lege die gelben Würfel in den linken Behälter."}
        policy = self._Policy()
        policy.mapping = ClassicalPolicy._instruction_mapping(policy, info)
        brain_mapping, source = ClassicalPolicy._resolver_mapping(policy, info)
        self.assertEqual(brain_mapping, policy.mapping)
        self.assertEqual(resolve_bin(brain_mapping, "yellow", "cube"), "left")
        self.assertEqual(source, "resolver")


if __name__ == "__main__":
    unittest.main()
