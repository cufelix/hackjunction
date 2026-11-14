"""policy.insert_guard (rung 3, insert_shapes): pick nothing instead of releasing red objects onto the lid.

CPU only, no URDF:  python3 -m pytest aaltonomous/baselines/classical/test_insert_guard.py
"""
from __future__ import annotations

import itertools
import unittest

import numpy as np

from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.instruction import resolve
from aaltonomous.baselines.classical.planner import lookup_bin
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.interface.tasks import (BINS, COLORS, FIXED_INSTRUCTION, FIXED_MAPPING, INSERT_INSTRUCTION, SHAPES,
                                         TEMPLATES)
from aaltonomous.scene.scene import Scene


class Stub:
    def __init__(self, on):
        self.cfg = {"policy": {"insert_guard": on}}
        self.logged = []
        self.log = self.logged.append


def guard(on, info):
    return ClassicalPolicy._insert_guard_mapping(Stub(on), info)


def all_sentences():
    yield FIXED_INSTRUCTION
    for t in TEMPLATES:
        for cs, bs in itertools.product(itertools.permutations(COLORS), itertools.permutations(BINS)):
            w = {f"c{i}": c for i, c in enumerate(cs)}
            w.update({f"C{i}": c.capitalize() for i, c in enumerate(cs)})
            w.update({f"b{i}": b for i, b in enumerate(bs)})
            yield t.format(**w)


class TestInsertGuard(unittest.TestCase):
    def test_without_guard_the_lid_bin_gets_the_red_objects(self):
        """The bug this flag fixes: the hole sentence resolves to FIXED_MAPPING, red -> left = the lid bin."""
        self.assertEqual(resolve(INSERT_INSTRUCTION), FIXED_MAPPING)
        self.assertEqual(Scene.load().lid_bin, FIXED_MAPPING["red"])

    def test_stock_release_point_in_the_lid_bin_is_solid_lid(self):
        """drop_xy(left) (pelvis (0.453, 0.208)) is not over any hole -> the object rests on the lid = forced (-1)."""
        sc = Scene.load()
        cfg = load_config(overlay="none")
        lo, hi = sc.bin_inner_box(sc.lid_bin)
        a = sc.mat_to_pelvis(np.r_[lo[:2], 0.0])[:2]
        b = sc.mat_to_pelvis(np.r_[hi[:2], 0.0])[:2]
        drop = np.minimum(a, b) + float(cfg["grasp"]["drop_inset"])      # = Planner.drop_xy for a side bin at y > 0
        p_mat = sc.pelvis_to_mat(np.r_[drop, 0.0])
        self.assertIsNone(sc.lid_hole_at(p_mat[:2]))
        z = sc.lid_z[1] + sc.object_height() / 2                            # resting on the lid
        self.assertEqual(sc.classify_insert(sc.mat_to_pelvis(np.r_[p_mat[:2], z]), 0.0, "cube"), "forced")

    def test_off_is_the_old_behaviour(self):
        self.assertIsNone(guard(False, {"instruction": INSERT_INSTRUCTION, "task": "insert_shapes"}))

    def test_on_detects_the_sentence_and_the_task_id(self):
        for info in ({"instruction": INSERT_INSTRUCTION}, {"instruction": INSERT_INSTRUCTION.upper()},
                     {"instruction": "", "task": "insert_shapes"}, {"instruction": FIXED_INSTRUCTION, "task": "insert_shapes"}):
            self.assertEqual(guard(True, info), {"default": "abstain"}, info)

    def test_on_never_touches_the_sort_tasks(self):
        for s in all_sentences():
            self.assertIsNone(guard(True, {"instruction": s, "task": "sort_language"}), s)
        self.assertIsNone(guard(True, {"instruction": FIXED_INSTRUCTION, "task": "sort_fixed"}))
        self.assertIsNone(guard(True, {"instruction": "Put the red objects in the left bin, the whole lot.", "task": None}))
        self.assertIsNone(guard(True, None))

    def test_abstain_mapping_picks_nothing_with_either_lookup(self):
        m = guard(True, {"instruction": INSERT_INSTRUCTION})
        for mode in ("colour", "resolve_bin"):
            cfg = {"policy": {"bin_lookup": mode}}
            for c, s in itertools.product(COLORS, (None,) + SHAPES):
                self.assertIsNone(lookup_bin(cfg, m, c, s), (mode, c, s))

    def test_never_raises(self):
        st = Stub(True)
        st.cfg = None                                                       # broken cfg: must not end the episode
        self.assertIsNone(ClassicalPolicy._insert_guard_mapping(st, {"instruction": INSERT_INSTRUCTION}))

    def test_submission_turns_it_on(self):
        self.assertTrue(load_config(overlay="submission")["policy"]["insert_guard"])
        self.assertFalse(load_config(overlay="none")["policy"]["insert_guard"])


if __name__ == "__main__":
    unittest.main()
