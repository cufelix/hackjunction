"""Skill-memory bandit, persistence fallback, and policy ordering invariants."""
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from .policy import ClassicalPolicy
from .skill_memory import ARMS, SkillMemory


KEY = ("cube", "red", "left")


class TestSkillMemory(unittest.TestCase):
    def make(self, directory, seed=0, log=None):
        return SkillMemory(str(Path(directory) / "skills.sqlite"), seed=seed, log=log or (lambda _: None))

    def test_flag_off_keeps_variant_choices_identical(self):
        policy = ClassicalPolicy.__new__(ClassicalPolicy)
        policy.skill_memory = None
        variants = [{"posture": "closed_demo", "yaw": None},
                    {"posture": "closed_tight", "yaw": 1.0}]
        ordered = policy._ordered_skill_variants(None, [{}], variants)
        self.assertEqual(ordered, [(None, variants[0]), (None, variants[1])])
        self.assertIs(ordered[0][1], variants[0])
        self.assertIs(ordered[1][1], variants[1])

    def test_prior_keeps_default_without_data(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = self.make(directory, seed=17)
            default = ["stock", "yaw_positive", "yaw_negative"]
            self.assertEqual(memory.choose(KEY, default), default)
            memory.record(KEY, "yaw_positive", "miss", 0.2)
            self.assertEqual(memory.choose(KEY, default), default)  # fewer than two observations
            memory.close()

    def test_successful_arm_beats_three_misses(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = self.make(directory, seed=0)
            for _ in range(3):
                memory.record(KEY, "stock", "miss", 1.0)
                memory.record(KEY, "alternate_posture", "placed", 1.0)
            self.assertEqual(memory.choose(KEY, ["stock", "alternate_posture"])[0], "alternate_posture")
            self.assertEqual(memory.summary()["cube|red|left"]["best_arm"], "alternate_posture")
            memory.close()

    def test_corrupt_or_unwritable_path_falls_back_in_memory(self):
        warnings = []
        with tempfile.TemporaryDirectory() as directory:
            memory = SkillMemory(directory, seed=0, log=warnings.append)  # a directory cannot be opened as SQLite
            for _ in range(3):
                memory.record(KEY, "stock", "miss", 0.1)
                memory.record(KEY, "yaw_positive", "placed", 0.1)
            self.assertEqual(memory.summary()["cube|red|left"]["counts"], {"stock": 3, "yaw_positive": 3})
            self.assertEqual(len(warnings), 1)
            memory.close()

    def test_choose_never_returns_an_arm_outside_closed_set(self):
        with tempfile.TemporaryDirectory() as directory:
            memory = self.make(directory)
            result = memory.choose(KEY, ["invented_motion", "stock", "stock", "yaw_negative", "also_bad"])
            self.assertEqual(result, ["stock", "yaw_negative"])
            self.assertTrue(set(result) <= set(ARMS))
            memory.close()

    def test_same_seed_is_reproducible(self):
        orders = []
        with tempfile.TemporaryDirectory() as first, tempfile.TemporaryDirectory() as second:
            memories = [self.make(first, seed=91), self.make(second, seed=91)]
            for memory in memories:
                for arm, outcome in (("stock", "miss"), ("yaw_positive", "placed"),
                                     ("yaw_negative", "held"), ("stock", "miss")):
                    memory.record(KEY, arm, outcome, 0.5)
                orders.append([memory.choose(KEY, ARMS) for _ in range(8)])
                memory.close()
        self.assertEqual(orders[0], orders[1])

    def test_policy_can_promote_a_learned_existing_variant_on_first_try(self):
        class Planner:
            def __init__(self):
                self.vary = None

            def plan(self, det, arm, hand, **kw):
                variant = dict(self.vary or {})
                posture = variant.get("posture") or "closed_tight"
                yaw = 0.0 if variant.get("yaw") is None else variant["yaw"]
                plan = SimpleNamespace(ok=True, side="left", posture=(posture, None, 1.0, "metric"),
                                       waypoints={"grasp": det.top.copy()})
                plan.heading = yaw
                return plan

            @staticmethod
            def finger_heading(plan):
                return plan.heading

        with tempfile.TemporaryDirectory() as directory:
            memory = self.make(directory, seed=0)
            key = ("unknown", "red", "left")
            for _ in range(3):
                memory.record(key, "stock", "miss", 1.0)
                memory.record(key, "alternate_posture", "placed", 1.0)
            policy = ClassicalPolicy.__new__(ClassicalPolicy)
            policy.cfg = {"policy": {"retry_vary_grasp": True, "retry_assoc_radius": 0.06,
                                     "retry_yaw_deg": {"none": 90, "default": 90}, "retry_max_variants": 3},
                          "grasp": {"close_posture": "closed_tight"}}
            policy.skill_memory = memory
            policy.planner = Planner()
            policy.arm_cmd = policy.hand_cmd = np.zeros(14)
            policy.vary_hist = []
            policy.log = lambda _: None
            det = SimpleNamespace(color="red", shape=None, top=np.array([0.3, 0.1, 0.8]))
            plan = policy._plan_pick(det, {})
            self.assertEqual((plan.skill_arm, plan.posture[0]), ("alternate_posture", "closed_demo"))
            memory.close()


if __name__ == "__main__":
    unittest.main()
