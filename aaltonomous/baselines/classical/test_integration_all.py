"""Combined-interaction suite for overlays/finals_r23_all (every reviewed flag group ON together).

Mock-world episodes (random layouts: cubes / triangular prisms / cylinders, 10 seeds), soft and stiff hand variants.
Asserts: no exception out of act()/reset(), every action finite / 14-wide / inside the joint limits and rate caps
(test_fault_drills.violations), act() p99 CPU < 37 ms, no object placed in a WRONG bin. The mean score (objects in the
right bin) is printed against submission and finals_r21 / finals_r22.

Mock limits (documented, not hidden): the mock has no real depth plane, no visible arm for the hand-ICP, no lying
prisms, no displaced bins, and a contact-less gripper. The calibration group (calib.hand_icp_primary,
counter_after_hand, refit_per_look, wide_counter_window) and apply_height/bin.vote therefore cannot complete here and
are pinned off by MOCK_CAL (they are exercised in the Isaac sim, see INTEG.md). Everything else stays ON.
"""
from __future__ import annotations

import statistics
import unittest

import numpy as np

from . import test_fault_drills as D

MOCK_CAL = {"calib.hand_icp_primary": False, "calib.counter_after_hand": False, "calib.refit_per_look": False,
            "calib.wide_counter_window": False}
SEEDS = range(10)
OVERLAYS = ("submission", "finals_r21", "finals_r22", "finals_r23_all")
STIFF = {"grasp.soft_hand": False, "grasp.soft_squeeze": False, "held.soft_detector": False}
LIMIT_S = 0.037


def episode(overlay, seed, extra=None):
    ov = dict(MOCK_CAL if overlay == "finals_r23_all" else {})
    ov.update(extra or {})
    run, pol, env = D.drive(overlay, overrides=ov or None, sim_s=90.0, max_acts=3000, seed=seed)
    outcomes = env.backend.outcomes(pol.mapping) if pol.mapping else []
    return run, pol, env, list(outcomes)


class TestIntegrationAll(unittest.TestCase):
    results = {}

    @classmethod
    def setUpClass(cls):
        for ov in OVERLAYS:
            cls.results[ov] = [episode(ov, s) for s in SEEDS]
        cls.results["r23_stiff"] = [episode("finals_r23_all", s, STIFF) for s in SEEDS]

    def test_no_exceptions_and_safe_actions(self):
        for name in ("finals_r23_all", "r23_stiff"):
            for seed, (run, pol, env, _) in zip(SEEDS, self.results[name]):
                self.assertEqual(run.exceptions, [], (name, seed))
                v = D.violations(run, pol, env, strict_accel=True)
                self.assertEqual(v, [], (name, seed, v[:3]))
                self.assertGreater(len(run.rows_a), 3, (name, seed))

    def test_act_p99_under_37_ms(self):
        for name in ("finals_r23_all", "r23_stiff"):
            for seed, (run, *_rest) in zip(SEEDS, self.results[name]):
                self.assertLess(run.p99("cpu"), LIMIT_S, (name, seed, run.p99("cpu")))

    def test_no_wrong_bin_placements(self):
        for name in ("finals_r23_all", "r23_stiff"):
            for seed, (_, _, _, out) in zip(SEEDS, self.results[name]):
                self.assertNotIn("wrong", out, (name, seed, out))

    def test_score_comparison_is_reported(self):
        def mean(name):
            return statistics.mean(sum(o == "correct" for o in out) for *_x, out in self.results[name])

        table = {n: mean(n) for n in (*OVERLAYS, "r23_stiff")}
        print("\n[integ-all] mean correct objects per episode (10 mock seeds):", table)
        # r23_all must not be worse than the finals overlay it extends (the mock gripper is contact-less, so this is
        # a regression tripwire, not a strength claim)
        self.assertGreaterEqual(table["finals_r23_all"] + 1e-9, table["finals_r22"] - 0.5)

    def test_all_new_flag_groups_are_actually_on(self):
        from .config import effective_flags, load_config
        on = {k for k, _, e, _, _ in effective_flags(load_config(overlay="finals_r23_all")) if e}
        for k in ("perception.sat_floor_adaptive", "grasp.pregrasp_relocalise", "policy.verify_landing",
                  "grasp.cage_mode", "policy.done_confirm", "planner.per_target_guard", "policy.score_order",
                  "perception.bins_from_image", "grasp.soft_hand", "perception.novel_objects", "calib.hand_icp_primary"):
            self.assertIn(k, on, k)
        for k in ("perception.area_floor_geometric", "perception.partial_ok_near_hand", "perception.depth_blob_recover"):
            self.assertNotIn(k, on, k)


if __name__ == "__main__":
    unittest.main()
