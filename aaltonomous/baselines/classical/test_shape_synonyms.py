"""Shape words in finals instructions: cylinder / triangle / cube synonyms, plurals, adjectives, colour+shape pairs.

Each sentence goes through the policy path (normalize -> resolve) and must resolve to exactly the mapping listed.
Run: python -m unittest aaltonomous.baselines.classical.test_shape_synonyms -v
"""
import unittest

from .instruction import resolve
from .normalize import normalize, strict_mapping


class TestShapeSynonyms(unittest.TestCase):
    def _check(self, cases):
        for s, want in cases.items():
            with self.subTest(sentence=s):
                self.assertEqual(resolve(normalize(s), fallback={}), want, (s, normalize(s)))

    def _strict(self, sentence):
        text = normalize(sentence)
        return text, strict_mapping(text, resolve(text, fallback={}))

    def test_cylinder_synonyms(self):
        self._check({
            "Red cans go left.": {("red", "cylinder"): "left"},
            "Put the can in the left bin.": {"cylinder": "left"},
            "The red can goes left.": {("red", "cylinder"): "left"},
            "Red rolls go left.": {("red", "cylinder"): "left"},
            "The red roll goes left.": {("red", "cylinder"): "left"},
            "Red rod goes left.": {("red", "cylinder"): "left"},
            "Red rods go left.": {("red", "cylinder"): "left"},
            "Yellow pipes go right.": {("yellow", "cylinder"): "right"},
            "Blue pipe goes right.": {("blue", "cylinder"): "right"},
            "Red tubes go left.": {("red", "cylinder"): "left"},
            "Yellow barrels go right.": {("yellow", "cylinder"): "right"},
            "Red round blocks go left.": {("red", "cylinder"): "left"},
            "Red round block goes left.": {("red", "cylinder"): "left"},
            "Yellow cylindrical ones go right.": {("yellow", "cylinder"): "right"},
            "Put the red cylinders in the left bin.": {("red", "cylinder"): "left"},
        })

    def test_triangle_synonyms(self):
        self._check({
            "Red roof goes left.": {("red", "triangle"): "left"},
            "Yellow roofs go right.": {("yellow", "triangle"): "right"},
            "Blue ridges go center.": {("blue", "triangle"): "center"},
            "Red wedges go left.": {("red", "triangle"): "left"},
            "Orange wedge goes left.": {("red", "triangle"): "left"},
            "Red triangular prisms go left.": {("red", "triangle"): "left"},
            "Yellow triangular prism goes right.": {("yellow", "triangle"): "right"},
            "Blue triangles go center.": {("blue", "triangle"): "center"},
            "Red prism goes left.": {("red", "triangle"): "left"},
        })

    def test_cube_synonyms(self):
        self._check({
            "Red boxes go left.": {("red", "cube"): "left"},
            "Put the blue box in the left bin.": {("blue", "cube"): "left"},
            "Yellow box goes right.": {("yellow", "cube"): "right"},
            "Red cuboids go left.": {("red", "cube"): "left"},
            "Red square blocks go left.": {("red", "cube"): "left"},
            "Red square block goes left.": {("red", "cube"): "left"},
            "Purple squares go right.": {("blue", "cube"): "right"},
            "Blue cubic ones go left.": {("blue", "cube"): "left"},
            "Yellow cubical ones go right.": {("yellow", "cube"): "right"},
            "Cubes go left.": {"cube": "left"},
            "Move the red box to the left bin.": {("red", "cube"): "left"},
            "Put the boxes in the left bin.": {"cube": "left"},
        })

    def test_colour_and_shape_pairs(self):
        self._check({
            # "blocks" beside another named shape is the cube (resolver rule); alone it is the colour's generic objects
            "The yellow cylinders go left and the purple blocks go right.": {("yellow", "cylinder"): "left", ("blue", "cube"): "right"},
            "The yellow cylinders go left and the purple things go right.": {("yellow", "cylinder"): "left", "blue": "right"},
            "Orange wedges go left and purple cubes go right.": {("red", "triangle"): "left", ("blue", "cube"): "right"},
            "Purple blocks go in the middle.": {"blue": "center"},
            "Put the orange wedge in the left bin.": {("red", "triangle"): "left"},
        })

    def test_bin_positions_and_negatives_are_not_shapes(self):
        self._check({
            "Blue ones roll to the left.": {"blue": "left"},
            "You can put the red ones left.": {"red": "left"},
            "Anything red can go left.": {"red": "left"},
            "Put the box on the left.": {},
            "Put the red cubes in the box.": {},
            "The box is orange; put the red cubes there.": {},
            "Sort by colour: red to the left box, yellow to the middle box, blue to the right box.":
                {"red": "left", "yellow": "center", "blue": "right"},
            "Yellow box gets the blue cubes; red cubes go right.": {("red", "cube"): "right"},
        })

    def test_box_copular_bin_description_does_not_create_a_cube_key(self):
        sentence = "The left one is the orange box, the middle one is yellow, the right one is purple."
        text, mapping = self._strict(sentence)
        self.assertEqual(text, sentence)
        self.assertNotIn(("red", "cube"), mapping)
        self.assertNotIn("red", mapping)
        self.assertEqual(mapping, {"yellow": "center", "blue": "right"})

    def test_box_bin_descriptions_both_orders_and_labels_abstain(self):
        cases = (
            "The orange box is the left one; yellow goes center.",
            "The left one is the orange box; yellow goes center.",
            "Box A is orange; yellow goes center.",
            "The orange one is box A; yellow goes center.",
        )
        for sentence in cases:
            with self.subTest(sentence=sentence):
                text, mapping = self._strict(sentence)
                self.assertNotIn("cube", text.lower())
                self.assertNotEqual(mapping.get("red"), "left")

    def test_ambiguous_shape_scene_language_abstains_only_that_key(self):
        cases = (
            "The red roof is above the left bin; put yellow cubes right.",
            "The red rod points to the left bin; put yellow cubes right.",
            "The red pipe is over the left bin; put yellow cubes right.",
            "The red ridge is with the left bin; put yellow cubes right.",
        )
        for sentence in cases:
            with self.subTest(sentence=sentence):
                text, mapping = self._strict(sentence)
                self.assertNotIn(("red", "triangle"), mapping)
                self.assertNotIn(("red", "cylinder"), mapping)
                self.assertIsNone(mapping.get("red"))
                self.assertEqual(mapping.get(("yellow", "cube")), "right")
                self.assertEqual(text.split()[2].lower(), sentence.split()[2].lower())

    def test_leading_question_can_is_not_a_cylinder(self):
        text, mapping = self._strict("Can red go left?")
        self.assertEqual(text, "Can red go left?")
        self.assertNotIn(("red", "cylinder"), mapping)
        self.assertIsNone(mapping.get("red"))


if __name__ == "__main__":
    unittest.main()
