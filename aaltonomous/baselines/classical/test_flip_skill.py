"""planner.skill == 'flip' inserts a pan-toss phase after the lift; None leaves the pick byte-identical."""
from __future__ import annotations

import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .perception import Detection
from .planner import Planner


SC, CAL = Scene.load(), Calibration.load()


class TestFlipSkill(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(overlay="none")
        cls.post, cls.home = hand_postures(cls.cfg), home_arm(cls.cfg)
        cls.spec = JointSpec()
        L = SC.cfg["layout"]
        cls.top = SC.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]), SC.object_height()])

    def _plan(self, skill):
        planner = Planner(self.cfg, SC, self.post, self.home)
        planner.skill = skill
        det = Detection("yellow", self.top.copy(), 0, 0, "depth")
        return planner.plan(det, self.home, self.post["open"])

    def test_flip_phase_between_lift_and_carry_within_limits(self):
        plan = self._plan("flip")
        self.assertTrue(plan.ok, plan.reason)
        names = [n for n, _, _ in plan.phases]
        self.assertLess(names.index("lift"), names.index("flip"))
        self.assertEqual(names.index("flip") + 1, names.index("carry"))
        s, e = next((s, e) for n, s, e in plan.phases if n == "flip")
        ip = self.spec.arm_names.index(f"{plan.side}_wrist_pitch_joint")
        rng = float(np.ptp(plan.arm[s:e, ip]))
        self.assertGreaterEqual(rng, 0.4, f"wrist_pitch range {rng:.3f} rad")
        self.assertFalse(plan.safety_clips)

    def test_skill_none_is_identical_to_stock_pick(self):
        stock = Planner(self.cfg, SC, self.post, self.home)   # skill never set: the stock pick
        det = Detection("yellow", self.top.copy(), 0, 0, "depth")
        ref = stock.plan(det, self.home, self.post["open"])
        plan = self._plan(None)
        self.assertTrue(ref.ok, ref.reason)
        self.assertTrue(plan.ok, plan.reason)
        self.assertTrue(np.array_equal(plan.arm, ref.arm))
        self.assertNotIn("flip", [n for n, _, _ in plan.phases])


class TestHoldSkill(unittest.TestCase):
    """planner.skill == 'hold': lift, then the hand stays closed with the arm still for grasp.hold_s."""

    @classmethod
    def setUpClass(cls):
        TestFlipSkill.setUpClass.__func__(cls)

    def _plan(self, skill):
        planner = Planner(self.cfg, SC, self.post, self.home)
        planner.skill = skill
        det = Detection("yellow", self.top.copy(), 0, 0, "depth")
        return planner.plan(det, self.home, self.post["open"])

    def test_hold_phase_after_lift_is_still_and_hold_s_long(self):
        plan = self._plan("hold")
        self.assertTrue(plan.ok, plan.reason)
        names = [n for n, _, _ in plan.phases]
        self.assertEqual(names.index("lift") + 1, names.index("hold"))
        self.assertEqual(names.index("hold") + 1, names.index("carry"))
        s, e = next((s, e) for n, s, e in plan.phases if n == "hold")
        hold_s = float(self.cfg["grasp"].get("hold_s", 3.0))
        self.assertGreaterEqual(e - s, int(round(hold_s / (1.0 / 30.0))))
        self.assertEqual(float(np.ptp(plan.arm[s:e], axis=0).max()), 0.0)
        self.assertFalse(plan.safety_clips)


if __name__ == "__main__":
    unittest.main()
