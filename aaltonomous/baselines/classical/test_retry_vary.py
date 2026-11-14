"""policy.retry_vary_grasp / policy.max_attempts_vary / grasp.border_posture. Run from outside the repo:
    PYTHONPATH=<repo> python -m pytest <repo>/aaltonomous/baselines/classical/test_retry_vary.py -q
Scripted planner tests (no IK) + mock-world tests with the real planner (IK, safety dry run)."""
import unittest

import numpy as np

from .config import load_config
from .perception import Detection
from .planner import Plan
from .policy import ClassicalPolicy

V1, DEMO = "closed_tight", "closed_demo"


class FakePerception:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, rgb, depth):
        return list(self.dets)

    def reset_calibration(self):
        pass


class FakePlanner:
    """plan(): posture = vary posture (else `stock`), finger heading = vary yaw (else `heading`); records each call."""

    def __init__(self, real, stock=V1, heading=0.0, infeasible=()):
        self.real, self.mapping, self.vary = real, real.mapping, None
        self.stock, self.heading, self.infeasible, self.calls = stock, heading, infeasible, []

    def bin_xy(self, name):
        return self.real.bin_xy(name)

    def drop_slots(self, name):
        return self.real.drop_slots(name)

    def finger_heading(self, plan):
        return plan._heading

    def plan(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True, drop_slot=None):
        v = dict(self.vary or {})
        self.calls.append(v)
        posture = v.get("posture") or self.stock
        yaw = self.heading if v.get("yaw") is None else v["yaw"]
        if (posture, None if v.get("yaw") is None else round(np.degrees(yaw))) in self.infeasible:
            return Plan(False, color=det.color, reason="scripted IK failure")
        n = 60
        p = Plan(True, side="left", color=det.color, target_bin="left", arm=np.zeros((n, 14)), hand=np.zeros((n, 14)),
                 waypoints={"grasp": det.top.copy(), "drop": det.top.copy()}, ik={"g": (True, 0.0, 0.0)},
                 safety_clips=0, posture=(posture, getattr(det, "shape", None), 1.0, "metric"),
                 phases=[("descend", 0, 10), ("close", 10, 20), ("lift", 20, 30), ("carry", 30, 60)])
        p._heading = yaw
        return p


def det(color, x, y=0.1, shape=None):
    d = Detection(color, np.array([x, y, 0.8]), 500, 400, "depth")
    d.shape = shape
    return d


def make(dets, overrides=None, **kw):
    ov = {"policy.retry_vary_grasp": True, "grasp.close_posture": V1, **(overrides or {})}   # V1 as in cand_r14
    pol = ClassicalPolicy(log=None, overrides=ov)
    pol.perception = FakePerception(dets)
    pol.planner = FakePlanner(pol.planner, **kw)
    pol.arm_cmd, pol.hand_cmd = pol.home.copy(), np.zeros(14)
    return pol


def miss(pol, plan):
    """Record `plan` as started and missed (held=no)."""
    pol._start(plan, 0.0)
    pol._abort = None
    pol.vary_hist[-1]["held"] = False
    pol.queue_a = pol.queue_h = None


def obs(t=10.0):
    return {"t": t, "arm_q": np.zeros(14), "hand_q": np.zeros(14), "head_rgb": None, "head_depth": None}


class TestVariants(unittest.TestCase):
    def test_retry_never_repeats_an_identical_grasp(self):
        for shape, stock in (("triangle", DEMO), ("triangle", V1), (None, V1), ("cube", V1), ("cylinder", V1)):
            d = det("red", 0.30, shape=shape)
            pol = make([d], stock=stock)
            tried = []
            for _ in range(3):
                p = pol._plan_pick(d, {})
                if not p.ok:
                    break
                key = pol._grasp_key(p)
                for h in tried:
                    self.assertFalse(pol._same_grasp(key, h), (shape, stock, key))
                miss(pol, p)
                tried.append(pol.vary_hist[-1])
            self.assertGreaterEqual(len(tried), 2, (shape, stock))
            if shape in ("cube", "cylinder"):   # cubes 7/25 with demo: a non-triangle never switches to it
                self.assertTrue(all(h["posture"] == V1 for h in tried))

    def test_first_retry_switches_posture_or_heading_as_specified(self):
        d = det("red", 0.30, shape=None)
        pol = make([d])
        miss(pol, pol._plan_pick(d, {}))
        p = pol._plan_pick(d, {})
        self.assertEqual(pol._posture_of(p), DEMO)          # label None: v1 -> demo
        d = det("red", 0.30, shape="triangle")
        pol = make([d], stock=DEMO)
        miss(pol, pol._plan_pick(d, {}))
        p = pol._plan_pick(d, {})
        self.assertEqual(pol._posture_of(p), DEMO)          # demo triangle keeps demo, heading +-60
        self.assertAlmostEqual(abs(np.degrees(p._heading)), 60.0, places=6)
        d = det("red", 0.30, shape="cube")
        pol = make([d])
        miss(pol, pol._plan_pick(d, {}))
        p = pol._plan_pick(d, {})
        self.assertAlmostEqual(abs(np.degrees(p._heading)), 90.0, places=6)

    def test_every_variant_identical_or_infeasible_is_refused(self):
        d = det("red", 0.30, shape="cube")
        pol = make([d], infeasible={(V1, 90), (V1, -90)})
        miss(pol, pol._plan_pick(d, {}))
        p = pol._plan_pick(d, {})
        self.assertFalse(p.ok)
        self.assertIn("repeats a tried", p.reason)

    def test_moved_object_is_not_identical(self):
        d = det("red", 0.30, shape="cube")
        pol = make([d], infeasible={(V1, 90), (V1, -90)})
        miss(pol, pol._plan_pick(d, {}))
        d2 = det("red", 0.33, shape="cube")                 # pushed 3 cm: same object (assoc 6 cm), other point
        self.assertTrue(pol._plan_pick(d2, {}).ok)

    def test_held_yes_or_other_colour_does_not_vary(self):
        d = det("red", 0.30, shape="cube")
        pol = make([d])
        pol._start(pol._plan_pick(d, {}), 0.0)
        pol.vary_hist[-1]["held"] = True
        self.assertEqual(pol._hist_for(d), [])
        pol.vary_hist[-1]["held"] = False
        self.assertEqual(pol._hist_for(det("blue", 0.30)), [])

    def test_max_attempts_vary_third_try_only_with_time(self):
        d = det("red", 0.30, shape="cube")
        for t, want in ((10.0, "plan"), (119.0, "none")):
            pol = make([d], overrides={"policy.max_attempts_vary": 3})
            for _ in range(2):
                miss(pol, pol._plan_pick(d, {}))
            res = pol._choose([d], obs(t), verify=False)
            self.assertEqual(res[0], want, t)
        pol = make([d])                                     # max_attempts_vary off: stock limit 2
        for _ in range(2):
            miss(pol, pol._plan_pick(d, {}))
        self.assertEqual(pol._choose([d], obs(), verify=False)[0], "none")

    def test_extra_try_comes_after_fresh_objects(self):
        a, b = det("red", 0.30, shape="cube"), det("blue", 0.32, y=-0.1, shape="cube")
        pol = make([a, b], overrides={"policy.max_attempts_vary": 3})
        for _ in range(2):
            miss(pol, pol._plan_pick(a, {}))
        res = pol._choose([a, b], obs(), verify=False)
        self.assertEqual(res[1].color, "blue")


class TestAbort(unittest.TestCase):
    def _plan(self, pol):
        p = pol._plan_pick(det("red", 0.30, shape="cube"), {})
        arm = np.zeros((25, 14)); arm[:, 0] = np.arange(25)     # lift ticks 20..29 + 15 ticks home
        p.abort = (20, arm, np.zeros((25, 14)))
        return p

    def _run_to_close_end(self, pol, plan, hand_q):
        pol._start(plan, 0.0)                               # pops ticks 0..9
        pol.act({**obs(), "hand_q": np.zeros(14)})          # ticks 10..19 (close)
        pol.hand_cmd = np.full(14, 0.5)
        return pol.act({**obs(), "hand_q": hand_q})         # done = 20 = close end -> held check

    def test_held_no_aborts_the_carry(self):
        pol = make([])
        plan = self._plan(pol)
        a = self._run_to_close_end(pol, plan, np.full(14, 0.5))   # fingers reached the command: closed on air
        self.assertFalse(pol.vary_hist[-1]["held"])
        self.assertEqual(list(a["arm_q_target"][:, 0]), list(range(10)))   # abort ticks 0..9 = the lift
        self.assertEqual(len(pol.queue_a), 15)                           # straight home, no carry/release
        self.assertTrue(any("abort" in m for _, m in pol.events))

    def test_held_yes_keeps_the_plan(self):
        pol = make([])
        plan = self._plan(pol)
        self._run_to_close_end(pol, plan, np.full(14, 0.3))       # fingers stalled on the object
        self.assertTrue(pol.vary_hist[-1]["held"])
        self.assertEqual(len(pol.queue_a), 30)                    # original 60 - 30 sent

    def test_no_abort_branch_keeps_the_plan(self):
        pol = make([])
        plan = self._plan(pol)
        plan.abort = None
        self._run_to_close_end(pol, plan, np.full(14, 0.5))
        self.assertFalse(pol.vary_hist[-1]["held"])
        self.assertEqual(len(pol.queue_a), 30)


class TestRealPlanner(unittest.TestCase):
    """Real IK + safety: the abort branch obeys the speed limits, the heading variant reaches its heading."""

    def test_abort_branch_and_heading_variant(self):
        from .config import hand_postures, home_arm
        from .planner import Planner
        from aaltonomous.scene.scene import Scene
        cfg = load_config(overlay="cand_r14", overrides={"policy.retry_vary_grasp": True})
        sc = Scene.load()
        pl = Planner(cfg, sc, hand_postures(cfg), home_arm(cfg))
        mo = cfg["motion"]
        L = sc.cfg["layout"]
        d = Detection("yellow", sc.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]),
                                                  sc.object_height()]), 0, 0, "depth")
        p = pl.plan(d, pl.home.copy(), pl.post["open"].copy())
        self.assertTrue(p.ok, p.reason)
        self.assertIsNotNone(p.abort)
        ls, arm, hand = p.abort
        full = np.vstack([p.arm[:ls], arm])
        self.assertLessEqual(np.abs(np.diff(full, axis=0)).max() * 30.0, mo["arm_max_vel"] + 1e-6)
        self.assertLess(len(arm), len(p.arm) - ls)               # shorter than carry + release + home
        np.testing.assert_allclose(arm[-1], pl.home, atol=1e-9)   # ends at home (started there)
        self.assertEqual(pl.dry_run_safety(pl.home, pl.post["open"], full, np.vstack([p.hand[:ls], hand])), 0)
        h0 = pl.finger_heading(p)
        pl.vary = {"posture": None, "yaw": h0 - np.radians(60.0)}
        q = pl.plan(d, pl.home.copy(), pl.post["open"].copy(), relocate=False)
        pl.vary = None
        if q.ok:
            dh = abs((pl.finger_heading(q) - (h0 - np.radians(60.0)) + np.pi) % (2 * np.pi) - np.pi)
            self.assertLessEqual(np.degrees(dh), cfg["grasp"]["yaw_tol_deg"] + 1e-6)

    def test_flag_off_plans_have_no_abort(self):
        from .config import hand_postures, home_arm
        from .planner import Planner
        from aaltonomous.scene.scene import Scene
        cfg = load_config(overlay="cand_r14")
        sc = Scene.load()
        pl = Planner(cfg, sc, hand_postures(cfg), home_arm(cfg))
        L = sc.cfg["layout"]
        d = Detection("yellow", sc.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]),
                                                  sc.object_height()]), 0, 0, "depth")
        p = pl.plan(d, pl.home.copy(), pl.post["open"].copy())
        self.assertTrue(p.ok, p.reason)
        self.assertIsNone(p.abort)
        self.assertIsNone(p.grasp_key)

    def test_mock_episode_with_flags_has_no_clips(self):
        from .mock_world import make_mock_env
        for ov in ("ab_retry_vary", "ab_retry_vary_slots"):
            cfg = load_config(overlay=ov)
            env = make_mock_env(seed=1)
            pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg, time_limit=120.0)
            o = env.reset()
            pol.reset({"instruction": o["instruction"], "time_limit": 120.0})
            for _ in range(20000):
                o, info = env.step(pol.act(o))
                if info["episode_over"]:
                    break
            self.assertEqual(info["safety_summary"]["clipped_steps"], 0, ov)


if __name__ == "__main__":
    unittest.main()
