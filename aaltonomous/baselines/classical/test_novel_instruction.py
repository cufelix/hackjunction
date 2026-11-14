"""Deterministic grounding of novel-object instructions, and brain-pick grounding (no simulator needed)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from .novel_objects import match_instruction
from .policy import ClassicalPolicy


def det(color, shape, uv=(0.0, 0.0)):
    return SimpleNamespace(novel=True, color=color, shape=shape, uv=uv, label="")


class MatchInstructionTest(unittest.TestCase):
    def test_middle_bin_is_center(self):
        m = match_instruction("put the pan in the middle bin", [det("grey", "with-handle")])
        self.assertIsNotNone(m.matched)
        self.assertEqual(m.target_bin, "center")

    def test_gray_equals_grey(self):
        m = match_instruction("put the gray mug in the left bin", [det("grey", "with-handle")])
        self.assertIsNotNone(m.matched)
        self.assertEqual(m.target_bin, "left")

    def test_polite_filler_and_right_bin(self):
        m = match_instruction("could you put the frying pan into the right bin please", [det("grey", "with-handle")])
        self.assertIsNotNone(m.matched)
        self.assertEqual(m.target_bin, "right")

    def test_two_candidates_for_shape_noun_is_ambiguous(self):
        pool = [det("grey", "with-handle"), det("red", "with-handle")]
        self.assertIsNone(match_instruction("put the pan in the middle bin", pool).matched)

    def test_spatula_is_elongated(self):
        m = match_instruction("put the spatula in the middle bin", [det("black", "elongated")])
        self.assertIsNotNone(m.matched)
        self.assertEqual(m.target_bin, "center")

    def test_plural_shape_noun(self):
        m = match_instruction("put the pans in the left bin", [det("grey", "with-handle")])
        self.assertIsNotNone(m.matched)

    def test_unknown_noun_never_matches(self):
        m = match_instruction("put the toaster in the middle bin", [det("grey", "with-handle")])
        self.assertIsNone(m.matched)
        self.assertEqual(m.unresolved_noun, "toaster")


class BrainPickGroundingTest(unittest.TestCase):
    def policy(self, grounded_flag, find_points):
        pol = ClassicalPolicy.__new__(ClassicalPolicy)
        pol.cfg = {"brain": {"grounded_picks": grounded_flag}}
        pol._last_find_items = [{"point": p} for p in find_points]
        return pol

    def test_pick_needs_flag_and_find_point_on_det(self):
        d = det("grey", "with-handle", uv=(100.0, 100.0))
        self.assertTrue(self.policy(True, [(103.0, 98.0)])._novel_grounded(d))

    def test_no_find_point_on_det_is_not_grounded(self):
        d = det("grey", "with-handle", uv=(100.0, 100.0))
        self.assertFalse(self.policy(True, [(400.0, 400.0)])._novel_grounded(d))
        self.assertFalse(self.policy(True, [])._novel_grounded(d))

    def test_flag_off_is_never_grounded(self):
        d = det("grey", "with-handle", uv=(100.0, 100.0))
        self.assertFalse(self.policy(False, [(100.0, 100.0)])._novel_grounded(d))


if __name__ == "__main__":
    unittest.main()
