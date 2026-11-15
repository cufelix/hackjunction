"""Deterministic grounding of novel-object instructions, and brain-pick grounding (no simulator needed)."""
from __future__ import annotations

import unittest
from types import SimpleNamespace

from .mock_world import make_mock_env
from .novel_objects import match_instruction, novel_noun_named
from .policy import ClassicalPolicy
from .test_universal import MOCK_OV, OVERLAY


def det(color, shape, uv=(0.0, 0.0)):
    return SimpleNamespace(novel=True, color=color, shape=shape, uv=uv, label="")


class MatchInstructionTest(unittest.TestCase):
    def test_middle_bin_is_center(self):
        m = match_instruction("put the pan in the middle bin", [det("grey", "with-handle")])
        self.assertIsNotNone(m.matched)
        self.assertEqual(m.target_bin, "center")

    def test_left_box_is_left_bin(self):
        m = match_instruction("put the pan in the left box", [det("grey", "with-handle")])
        self.assertIsNotNone(m.matched)
        self.assertEqual(m.target_bin, "left")

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


class BrainFindLongObjectTest(unittest.TestCase):
    """A find point on a long object (uv at its centre, the point 60 px off) resolves to the existing detection by 3D blob."""

    def policy(self, blob_top_xy):
        import numpy as np
        from unittest import mock
        pol = ClassicalPolicy.__new__(ClassicalPolicy)
        pol.cfg = {"brain": {"find": True, "find_to_detection": True, "novel_max_objects": 3}}
        pol.brain, pol._universal_inert, pol.perception = object(), False, None
        pol.log = lambda *a, **k: None
        pol._novel_store, pol._brain_reg, pol._brain_n = [], [], 0
        elong = SimpleNamespace(novel=True, color="black", shape="elongated", uv=(100.0, 100.0),
                                top=np.array([0.5, 0.0, 0.02]), label="screwdriver")
        blob = SimpleNamespace(uv=(160.0, 100.0), color="black", footprint=0.1, height_m=0.02, ungraspable=False,
                               top=np.array([blob_top_xy[0], blob_top_xy[1], 0.02]))
        pol._brain_find_fn = lambda rgb, q: [{"point": (160.0, 100.0), "bbox": None}]
        self.blob_patch = mock.patch("aaltonomous.baselines.classical.brain.novel.novel_detection", return_value=blob)
        self.blob_patch.start()
        self.addCleanup(self.blob_patch.stop)
        obs = {"head_rgb": np.zeros((4, 4, 3), np.uint8), "head_depth": np.zeros((4, 4))}
        return pol, [elong], obs

    def test_blob_two_cm_from_det_matches_it(self):
        pol, dets, obs = self.policy((0.52, 0.0))
        found, all_dets, ids = pol._brain_find_ids("screwdriver", obs, dets)
        self.assertEqual(found, [ids[0]])
        self.assertEqual(len(all_dets), 1)          # no duplicate novel entry for the same object

    def test_blob_ten_cm_away_is_not_the_det(self):
        pol, dets, obs = self.policy((0.6, 0.0))
        found, all_dets, ids = pol._brain_find_ids("screwdriver", obs, dets)
        self.assertNotIn(ids[0], found)


def make_pol(instruction, task=None):
    env = make_mock_env(seed=0, n_objects=3)
    info = {"instruction": instruction, "time_limit": 120.0}
    if task:
        info["task"] = task
    pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=120.0, overlay=OVERLAY, overrides=dict(MOCK_OV))
    pol.reset(info)
    return pol


KNOWN_PARAPHRASES = [
    "The red cube goes in the left bin.", "Red goes left.", "The blue one belongs in the right bin.",
    "Put the yellow cube into the center bin.", "The purple pieces should end up in the middle bin.",
    "Drop the red ones into the right bin.", "Move every orange block to the left bin.",
    "All blue objects go in the right bin.", "Place the yellow triangle in the centre bin.",
    "The red cylinder needs to go into the left bin.", "Could you put the blue cubes in the right bin please?",
    "Sort the red items into the left bin.", "Take the orange one and drop it in the middle bin.",
    "The yellow things belong in the right bin.", "Toss the purple prism into the left bin.",
    "Every red piece ends up in the right bin.", "Bring the blue cylinder to the center bin.",
    "Red ones go to the right-hand bin.", "Put all the orange shapes in the left bin.",
    "I want the yellow cube in the right bin.",
    "Put the red cube in the left box.", "The blue ones go in the right box.",
    "Drop the yellow cylinder into the middle crate.",
]
NOVEL_NOUN_SENTENCES = [
    "put the pan in the middle bin", "The frying pan goes in the left bin.", "Put the mug into the right bin.",
    "The ball belongs in the center bin.", "Drop the cups in the left bin.",
]


class KnownParaphraseGuardTest(unittest.TestCase):
    def test_known_paraphrases_name_no_novel_noun_and_stay_inert(self):
        for s in KNOWN_PARAPHRASES:
            with self.subTest(sentence=s):
                self.assertFalse(novel_noun_named(s))
                self.assertTrue(make_pol(s)._universal_inert)

    def test_novel_noun_sentences_are_not_inert(self):
        for s in NOVEL_NOUN_SENTENCES:
            with self.subTest(sentence=s):
                self.assertTrue(novel_noun_named(s))
                self.assertFalse(make_pol(s)._universal_inert)


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
