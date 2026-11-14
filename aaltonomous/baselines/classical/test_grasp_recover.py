"""policy.retry_cross_relocate / policy.late_sweep (both default off). Run from outside the repo:
    PYTHONPATH=<repo> python -m pytest <repo>/aaltonomous/baselines/classical/test_grasp_recover.py -q"""
import unittest

import numpy as np

from .config import FLAGS, load_config
from .planner import Plan
from .test_retry_vary import DEMO, V1, FakePlanner, det, make, miss, obs


class RelocPlanner(FakePlanner):
    """FakePlanner + plan_relocation: a right-arm plan to the counter (or a scripted failure)."""

    def __init__(self, *a, reloc_ok=True, **kw):
        super().__init__(*a, **kw)
        self.reloc_ok, self.reloc_calls = reloc_ok, 0

    def plan_relocation(self, d, bin_name, arm_q, hand_q, obstacles=(), drop_slot=None, clearance_obstacles=()):
        self.reloc_calls += 1
        if not self.reloc_ok:
            return Plan(False, color=d.color, reason="no relocation spot")
        p = self.plan(d, arm_q, hand_q)
        p.side, p.target_bin, p.reason = "right", "counter", "relocate for the left arm"
        return p


def make_x(dets, overrides=None, **kw):
    pol = make(dets, overrides=overrides)
    pol.planner = RelocPlanner(pol.planner.real, **kw)
    return pol


XREL = {"policy.retry_cross_relocate": True}


class TestFlags(unittest.TestCase):
    def test_default_off(self):
        for name in ("policy.retry_cross_relocate", "policy.late_sweep"):
            self.assertIn(name, FLAGS)
        for overlay in ("none", "finals_r23"):
            pol = load_config(overlay=overlay)["policy"]
            self.assertFalse(pol["retry_cross_relocate"])
            self.assertFalse(pol["late_sweep"])

    def test_ab_overlays_only_add_their_flag(self):
        base = load_config(overlay="finals_r23")
        for name, keys in (("ab_grasp_xrel", {"retry_cross_relocate"}), ("ab_grasp_sweep", {"late_sweep"}),
                           ("ab_grasp_both", {"retry_cross_relocate", "late_sweep"})):
            cfg = load_config(overlay=name)
            for k in keys:
                self.assertTrue(cfg["policy"][k], (name, k))
            cfg["policy"].update({k: False for k in keys})
            self.assertEqual(cfg["policy"], base["policy"], name)


class TestCrossRelocate(unittest.TestCase):
    def test_off_never_relocates(self):
        d = det("red", 0.30, y=-0.10, shape="cube")
        pol = make_x([d])
        miss(pol, pol._plan_pick(d, {}))
        p = pol._plan_pick(d, {})
        self.assertEqual(pol.planner.reloc_calls, 0)
        self.assertEqual(p.side, "left")

    def test_left_miss_cross_body_relocates_with_right_arm(self):
        d = det("red", 0.30, y=-0.10, shape="cube")
        pol = make_x([d], overrides=XREL)
        miss(pol, pol._plan_pick(d, {}))
        p = pol._plan_pick(d, {"bin_name": "left"})
        self.assertTrue(p.ok)
        self.assertEqual((p.side, p.target_bin), ("right", "counter"))
        self.assertEqual(pol._xrel_n, 1)

    def test_first_try_is_untouched(self):
        d = det("red", 0.30, y=-0.10, shape="cube")
        pol = make_x([d], overrides=XREL)
        p = pol._plan_pick(d, {"bin_name": "left"})
        self.assertEqual((p.side, pol.planner.reloc_calls), ("left", 0))

    def test_same_side_miss_keeps_stock_variants(self):
        d = det("red", 0.30, y=0.10, shape="cube")        # not cross-body
        pol = make_x([d], overrides=XREL)
        miss(pol, pol._plan_pick(d, {}))
        p = pol._plan_pick(d, {"bin_name": "left"})
        self.assertEqual((p.side, pol.planner.reloc_calls), ("left", 0))

    def test_right_arm_miss_never_relocates(self):
        d = det("red", 0.30, y=-0.10, shape="cube")
        pol = make_x([d], overrides=XREL)
        p = pol._plan_pick(d, {"bin_name": "left"})
        p.side = "right"
        miss(pol, p)
        self.assertNotEqual(pol._plan_pick(d, {"bin_name": "left"}).target_bin, "counter")

    def test_infeasible_relocation_falls_back_to_variants(self):
        d = det("red", 0.30, y=-0.10, shape="cube")
        pol = make_x([d], overrides=XREL, reloc_ok=False)
        miss(pol, pol._plan_pick(d, {}))
        p = pol._plan_pick(d, {"bin_name": "left"})
        self.assertTrue(p.ok)
        self.assertEqual(p.side, "left")
        self.assertEqual(pol.planner.reloc_calls, 1)

    def test_needs_time_and_is_capped(self):
        d = det("red", 0.30, y=-0.10, shape="cube")
        pol = make_x([d], overrides=XREL)
        miss(pol, pol._plan_pick(d, {}))
        pol._choose_t = 100.0                              # 6 s plan + 30 s needed > 20 s left
        self.assertEqual(pol._plan_pick(d, {"bin_name": "left"}).side, "left")
        pol._choose_t, pol._xrel_n = 10.0, 2
        self.assertEqual(pol._plan_pick(d, {"bin_name": "left"}).side, "left")


class TestLateSweep(unittest.TestCase):
    def _missed(self, overrides=None):
        d = det("red", 0.30, shape="cube")
        pol = make([d], overrides={"policy.max_attempts_vary": 3, **(overrides or {})})
        for _ in range(3):
            miss(pol, pol._plan_pick(d, {}))
        return pol, d

    def test_off_stops(self):
        pol, d = self._missed()
        self.assertEqual(pol._choose_retry([d], obs(40.0), False)[0], "none")

    def test_sweep_retries_with_time_left(self):
        pol, d = self._missed({"policy.late_sweep": True})
        res = pol._choose_retry([d], obs(40.0), False)
        self.assertEqual(res[0], "plan")
        self.assertEqual(pol._sweeps_used, 1)

    def test_no_sweep_without_time(self):
        pol, d = self._missed({"policy.late_sweep": True})
        self.assertEqual(pol._choose_retry([d], obs(100.0), False)[0], "none")
        self.assertEqual(pol._sweeps_used, 0)

    def test_sweep_is_bounded(self):
        pol, d = self._missed({"policy.late_sweep": True, "policy.late_sweep_max": 1})
        res = pol._choose_retry([d], obs(40.0), False)
        miss(pol, res[1])
        self.assertEqual(pol._choose_retry([d], obs(60.0), False)[0], "none")

    def test_failed_skip_is_reopened_and_repeat_grasp_allowed(self):
        d = det("red", 0.30, shape="cube")
        pol = make([d], overrides={"policy.late_sweep": True}, infeasible={(V1, 90), (V1, -90)})
        miss(pol, pol._plan_pick(d, {}))
        self.assertFalse(pol._plan_pick(d, {}).ok)          # every grasp repeats
        pol.failed.append(d.top[:2].copy())                 # what _choose does after that skip
        pol.attempts.append((d.top[:2].copy(), 1))
        res = pol._choose_retry([d], obs(40.0), False)
        self.assertEqual(res[0], "plan")
        self.assertEqual(pol.failed, [])

    def test_no_excluded_object_no_sweep(self):
        pol = make([], overrides={"policy.late_sweep": True})
        d = det("red", 0.30, shape="cube")
        self.assertEqual(pol._late_sweep(("none", None, 1), [d], obs(40.0), False)[0], "none")
        self.assertEqual(pol._sweeps_used, 0)


if __name__ == "__main__":
    unittest.main()
