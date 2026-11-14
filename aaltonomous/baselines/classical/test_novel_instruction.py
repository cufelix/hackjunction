"""Deterministic grounding of novel-object instructions, and brain-pick grounding (no simulator needed)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from .mock_world import make_mock_env
from .novel_objects import match_instruction
from .policy import ClassicalPolicy
from .test_universal import MOCK_OV, OVERLAY


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


def make_pol(instruction, task=None):
    env = make_mock_env(seed=0, n_objects=3)
    info = {"instruction": instruction, "time_limit": 120.0}
    if task:
        info["task"] = task
    pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=120.0, overlay=OVERLAY, overrides=dict(MOCK_OV))
    pol.reset(info)
    return pol


class KnownTaskGuardTest(unittest.TestCase):
    def test_known_red_ones_is_inert(self):
        pol = make_pol("Drop the red ones into the right bin.")
        self.assertTrue(pol._universal_inert)
        self.assertFalse(pol.perception.novel_objects)

    def test_named_novel_noun_is_not_inert(self):
        self.assertFalse(make_pol("put the pan in the middle bin")._universal_inert)

    def test_inert_episode_never_binds_novel_dets(self):
        pol = make_pol("Drop the red ones into the right bin.")
        fakes = [det(c, s) for c, s in [("orange", "elongated"), ("black", "elongated"), ("brown", "elongated"),
                                        ("white", "elongated"), ("red", "with-handle"), ("grey", "box-like"),
                                        ("red", "with-handle")]]
        pol._match_novel(fakes, obs=None)
        self.assertTrue(all(d.novel_target_bin is None and not d.novel_matched for d in fakes))
        self.assertEqual(pol._novel_brain_candidates, [])


if __name__ == "__main__":
    unittest.main()
