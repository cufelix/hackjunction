"""planner.skill == 'use' pushes a target with the tool after the lift; None leaves the pick byte-identical."""
from __future__ import annotations

import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

from .brain.planner import validate_plan
from .config import hand_postures, home_arm, load_config
from .perception import Detection
from .planner import Planner
from .test_brain_planner import WORLD

SC, CAL = Scene.load(), Calibration.load()
USE = ["use_approach", "use_lower", "use_push", "use_twist", "use_back"]


class TestUseSkill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(overlay="none")
        cls.post, cls.home = hand_postures(cls.cfg), home_arm(cls.cfg)
        cls.spec = JointSpec()
        L = SC.cfg["layout"]
        cls.top = SC.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]), SC.object_height()])

    def _det(self):
        det = Detection("yellow", self.top.copy(), 0, 0, "depth")
        det.yaw, det.footprint = 0.0, (0.20, 0.04)   # tool axis = pelvis +x, 20 cm long
        return det

    def _plan(self, skill, target_xy=None):
        planner = Planner(self.cfg, SC, self.post, self.home)
        planner.skill = skill
        planner.skill_target_xy = target_xy
        return planner.plan(self._det(), self.home, self.post["open"])

    def test_use_phases_after_lift_then_carry(self):
        target = self.top[:2] + np.array([0.15, 0.0])
        plan = self._plan("use", target)
        self.assertTrue(plan.ok, plan.reason)
        names = [n for n, _, _ in plan.phases]
        i = names.index("lift")
        self.assertEqual(names[i + 1:i + 1 + len(USE)], USE)
        self.assertEqual(names[i + 1 + len(USE)], "carry")
        lengths = {n: e - s for n, s, e in plan.phases}
        print("use phase lengths:", {n: lengths[n] for n in USE})

    def test_twist_moves_only_the_wrist_roll(self):
        plan = self._plan("use", self.top[:2] + np.array([0.15, 0.0]))
        self.assertTrue(plan.ok, plan.reason)
        ir = self.spec.arm_names.index(f"{plan.side}_wrist_roll_joint")
        s, e = next((s, e) for n, s, e in plan.phases if n == "use_twist")
        others = [j for j in range(plan.arm.shape[1]) if j != ir]
        self.assertLess(np.abs(plan.arm[s:e, others] - plan.arm[s - 1, others]).max(), 1e-6)
        self.assertGreater(np.ptp(plan.arm[s:e, ir]), 0.4)

    def test_hand_ends_push_closer_to_target_along_axis(self):
        target = self.top[:2] + np.array([0.15, 0.0])
        plan = self._plan("use", target)
        self.assertTrue(plan.ok, plan.reason)
        push, lift = plan.waypoints["use_push"][:2], plan.waypoints["lift"][:2]
        self.assertLess(np.linalg.norm(push - target), np.linalg.norm(lift - target))
        self.assertGreater(np.dot(push - lift, [1.0, 0.0]), 0.0)   # along u = +x

    def test_skill_none_is_identical_to_stock_pick(self):
        stock = Planner(self.cfg, SC, self.post, self.home)
        ref = stock.plan(self._det(), self.home, self.post["open"])
        plan = self._plan(None)
        self.assertTrue(ref.ok, ref.reason)
        self.assertTrue(plan.ok, plan.reason)
        self.assertTrue(np.array_equal(plan.arm, ref.arm))
        self.assertNotIn("use_push", [n for n, _, _ in plan.phases])

    def test_brain_validates_use_skill_for_a_known_object(self):
        raw = [{"tool": "skill", "args": {"obj_id": "o0", "skill": "use"}}, {"tool": "done", "args": {}}]
        steps, errors = validate_plan(raw, dict(WORLD, instruction="drive the screw into the block with the power screwdriver"),
                                      classical_has_work=False)
        self.assertFalse(errors, errors)
        self.assertTrue(steps[0]["valid"])


if __name__ == "__main__":
    unittest.main()
