"""planner.candidate_budget: a candidate whose IK keeps failing slowly is deferred within the budget and the next
candidate is planned. Run: PYTHONPATH=<repo> python -m pytest <repo>/aaltonomous/baselines/classical/test_candidate_budget.py -q"""
import time
import unittest

import numpy as np

from .config import load_config
from .kinematics import PlanBudget, PlanBudgetExceeded
from .planner import Plan
from .policy import ClassicalPolicy
from .test_retry_vary import FakePerception, FakePlanner, det, obs

HARD_X = 0.355
BUDGET = {"max_ik_calls": 30, "max_wall_s": 2.5}


class SlowFailPlanner(FakePlanner):
    """The hard candidate (x == HARD_X) calls the real kin.solve in a loop with a stub IK that always fails slowly."""

    def __init__(self, real, slow_s=0.02, hard_calls=200, **kw):
        super().__init__(real, **kw)
        self.slow_s, self.hard_calls, self.hard_seen, self.ik_calls = slow_s, hard_calls, 0, 0
        self.kin = real.kin
        self.kin._lsq = self._lsq

    def _lsq(self, side, target, q7, arm_q_full, w_o):
        time.sleep(self.slow_s)
        self.ik_calls += 1
        return np.asarray(q7, float), 1.0, 1.0

    def plan(self, det, arm_q, hand_q, **kw):
        if abs(det.top[0] - HARD_X) < 1e-6:
            self.hard_seen += 1
            for _ in range(self.hard_calls):
                self.kin.solve("left", det.top, arm_q, top_down=True)
            return Plan(False, color=det.color, reason="IK grasp left")
        return super().plan(det, arm_q, hand_q, **kw)


def make(dets, budget=BUDGET, **kw):
    ov = {"planner.candidate_budget": budget, "ik.random_restarts": 8, "policy.retry_vary_grasp": False}
    pol = ClassicalPolicy(log=None, overrides=ov)
    pol.perception = FakePerception(dets)
    pol.planner = SlowFailPlanner(pol.planner, **kw)
    pol.kin = pol.planner.kin
    pol.arm_cmd, pol.hand_cmd = pol.home.copy(), np.zeros(14)
    return pol


class TestCandidateBudget(unittest.TestCase):
    def test_budget_object_limits(self):
        b = PlanBudget(3, None)
        for _ in range(3):
            b.check(count=True)
        with self.assertRaises(PlanBudgetExceeded):
            b.check(count=True)
        now = [0.0]
        w = PlanBudget(None, 2.5, clock=lambda: now[0])
        w.check()
        now[0] = 2.6
        with self.assertRaises(PlanBudgetExceeded):
            w.check()

    def test_hard_candidate_deferred_then_next_planned(self):
        hard, easy = det("yellow", HARD_X, y=0.14), det("red", 0.30, y=-0.1)
        pol = make([hard, easy])
        t0 = time.monotonic()
        res = pol._choose([hard, easy], obs(), verify=False)
        dt = time.monotonic() - t0
        self.assertEqual(res[0], "plan")
        self.assertEqual(res[1].color, "red")                      # the next candidate is planned
        self.assertLess(dt, 3.0)                                   # budget (30 IK calls * 0.02 s) << unbudgeted 200 * 0.02 * ...
        self.assertLessEqual(pol.planner.ik_calls, 31 * 9 * 3)     # bounded by 30 solve calls
        self.assertEqual(pol.planner.hard_seen, 1)
        self.assertEqual(len(pol._deferred_cands), 1)
        self.assertEqual(pol.failed, [])                           # deferred is not failed

    def test_deferred_retried_once_after_the_others(self):
        hard = det("yellow", HARD_X, y=0.14)
        pol = make([hard])
        res = pol._choose([hard], obs(), verify=False)             # only candidate: deferred, then retried once, then failed
        self.assertEqual(res[0], "none")
        self.assertEqual(pol.planner.hard_seen, 2)
        self.assertEqual(len(pol.failed), 1)

    def test_wall_clock_budget(self):
        hard, easy = det("yellow", HARD_X, y=0.14), det("red", 0.30, y=-0.1)
        pol = make([hard, easy], budget={"max_ik_calls": 100000, "max_wall_s": 0.3}, slow_s=0.05)
        t0 = time.monotonic()
        p = pol._plan_pick(hard, {})
        self.assertTrue(p.deferred and not p.ok)
        self.assertLess(time.monotonic() - t0, 1.5)
        self.assertIsNone(pol.kin.budget)

    def test_flag_off_no_budget_no_defer(self):
        hard = det("yellow", HARD_X, y=0.14)
        pol = make([hard], budget=None, hard_calls=40, slow_s=0.0)
        p = pol._plan_pick(hard, {})
        self.assertFalse(getattr(p, "deferred", False))
        self.assertEqual(pol.planner.hard_seen, 1)
        self.assertEqual(pol._deferred_cands, [])

    def test_config_validation(self):
        load_config(overrides={"planner.candidate_budget": BUDGET})
        for bad in ({"max_ik_calls": 0, "max_wall_s": 1}, {"max_ik_calls": 5}, {"max_ik_calls": 5, "max_wall_s": -1}, 3):
            with self.assertRaises(ValueError):
                load_config(overrides={"planner.candidate_budget": bad})


if __name__ == "__main__":
    unittest.main()
