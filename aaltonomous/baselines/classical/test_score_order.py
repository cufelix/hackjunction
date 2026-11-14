"""policy.score_order: EV-per-second pick order, EV <= 0 never, time budget, flags-off identity, act() latency."""
import time
import unittest

import numpy as np

from .perception import Detection
from .test_time_triage import FakePerception, FakePlanner, make, obs


def det(color, x, y, shape="cube", cc=1.0):
    d = Detection(color, np.array([x, y, 0.8]), 500, 400, "depth")
    d.shape, d.shape_conf, d.color_conf = shape, 0.95, cc
    return d


def scene5():
    return [det("red", 0.30, -0.15), det("blue", 0.32, 0.12), det("yellow", 0.34, 0.0, "triangle"),
            det("red", 0.52, 0.30, "cylinder"), det("blue", 0.35, -0.05, "cube", cc=0.35)]   # last: ambiguous


DUR = {"red": 10.0, "blue": 10.0, "yellow": 10.0}


def mk(dets, so=True, limit=120.0, **kw):
    pol = make(DUR, dets, time_limit=limit, order="nearest_bin", **kw, **({"score_order": True} if so else {}))
    return pol


def order(pol, t=0.0):
    cands, _ = pol._ordered_cands(pol.perception.detect(None, None), obs(t), False)
    return cands


class TestScoreOrder(unittest.TestCase):
    def test_certain_near_first_ambiguous_last_or_never(self):
        dets = scene5()
        c = order(mk(dets))
        ids = [id(d) for d in c]
        self.assertTrue(id(dets[4]) not in ids or ids[-1] == id(dets[4]))
        self.assertIn(id(c[0]), (id(dets[0]), id(dets[1])))          # certain near cubes lead
        self.assertLess(ids.index(id(dets[0])), ids.index(id(dets[3])))

    def test_ev_nonpositive_last_not_dropped(self):
        mid = det("red", 0.30, 0.0, cc=0.3)              # p_wrong 0.7 -> EV < 0, pw < 0.8: kept, but last
        good = det("blue", 0.3, 0.1)
        self.assertEqual([id(d) for d in order(mk([mid, good]))], [id(good), id(mid)])
        self.assertEqual([id(d) for d in order(mk([mid]))], [id(mid)])

    def test_pw_at_least_0_8_is_dropped(self):
        bad = det("red", 0.30, 0.0, cc=0.2)               # p_wrong 0.8 -> dropped
        pol = mk([bad, det("blue", 0.3, 0.1)])
        c = order(pol)
        self.assertNotIn(id(bad), [id(d) for d in c])
        pol = mk([bad])
        self.assertEqual(order(pol), [])
        a = pol.act(obs(0.0))
        self.assertTrue(a["done"])
        self.assertEqual(pol.planner.calls, [])

    def test_time_budget_stop(self):
        pol = mk([det("red", 0.30, 0.1)], limit=100.0)
        pol.planner.durations["red"] = 40.0
        pol.act(obs(70.0))                                # 70 + 40 + 3 > 100
        self.assertEqual([m for _, m in pol.events if m.startswith("execute")], [])
        pol = mk([det("red", 0.30, 0.1)], limit=100.0)
        pol.act(obs(10.0))
        self.assertEqual(len([m for _, m in pol.events if m.startswith("execute")]), 1)

    def test_endgame_keeps_all_certain_quickest_first(self):
        dets = [det("red", 0.30, 0.1), det("blue", 0.45, -0.2), det("yellow", 0.34, 0.0, "triangle", cc=0.8)]
        pol = mk(dets, limit=100.0)                       # 17 s usable < endgame 25 s; yellow is not certain
        c = order(pol, t=80.0)
        self.assertEqual([d.color for d in c], ["red", "blue"])
        tcs = [pol._so_estimate(d, 80.0)[2] for d in c]
        self.assertEqual(tcs, sorted(tcs))

    def test_flags_off_identical(self):
        dets = scene5()
        off = mk(dets, so=False)
        base = make(DUR, dets, time_limit=120.0, order="nearest_bin")
        self.assertEqual([id(d) for d in order(off)], [id(d) for d in order(base)])
        self.assertFalse(off.cfg["policy"]["score_order"])

    def test_act_latency_unchanged(self):
        def run(so):
            pol = mk(scene5(), so=so)
            t0 = time.perf_counter()
            for _ in range(20):
                pol.state = "look"
                pol.act(obs(0.0))
                pol.failed.clear()
            return (time.perf_counter() - t0) / 20
        run(True), run(False)
        self.assertLess(run(True), run(False) + 0.05)


if __name__ == "__main__":
    unittest.main()
