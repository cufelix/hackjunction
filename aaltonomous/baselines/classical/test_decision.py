"""Decision layer D1-D4 (policy.confidence_order / object_time_cap / speed_tiers / confidence_tiers): synthetic object
lists and clocks, scripted perception + planner (no IK, no sim). Run:
    python -m unittest aaltonomous.baselines.classical.test_decision -v
Mapping (FIXED_MAPPING): red -> left bin, yellow -> centre bin, blue -> right bin.
"""
import unittest
from unittest import mock

import numpy as np

from . import decision as dec
from .config import effective_flags, load_config
from .perception import Detection
from .planner import Plan, lookup_bin
from .policy import ClassicalPolicy

FLAGS4 = ("confidence_order", "object_time_cap", "speed_tiers", "confidence_tiers")


# ----------------------------------------------------------------------------------------------- pure functions
class TestPure(unittest.TestCase):
    def test_hue_margin(self):
        yellow = [[[18, 70, 90], [35, 255, 255]]]
        self.assertAlmostEqual(dec.hue_margin_deg(yellow, 26), 2 * 8)          # 26 is 8 from 18 and 9 from 35
        self.assertEqual(dec.hue_margin_deg(yellow, 10), 0.0)                  # outside
        red = [[[0, 70, 70], [16, 255, 255]], [[170, 70, 70], [180, 255, 255]]]
        self.assertAlmostEqual(dec.hue_margin_deg(red, 178), 2 * 8)            # wraps: 170..196, 178 -> min(8, 18)
        self.assertAlmostEqual(dec.hue_margin_deg(red, 2), 2 * 12)             # 2 + 180 = 182: 12 from 170, 14 from 196
        self.assertEqual(dec.hue_margin_deg(red, 90), 0.0)

    def test_p_colour_monotone_and_bounded(self):
        self.assertAlmostEqual(dec.p_colour(500, 30.0, 1.0), 1.0)
        self.assertLess(dec.p_colour(500, 3.0, 1.0), dec.p_colour(500, 15.0, 1.0))          # hue margin
        self.assertLess(dec.p_colour(70, 30.0, 1.0), dec.p_colour(300, 30.0, 1.0))          # area
        self.assertEqual(dec.p_colour(40, None, None), 0.3)                                 # below 60 px
        self.assertLess(dec.p_colour(500, 30.0, 0.2), dec.p_colour(500, 30.0, 1.0))         # stability
        self.assertEqual(dec.p_colour(500, None, None), 1.0)                                # unknown = neutral

    def test_p_bin_centre_is_the_prior(self):
        self.assertEqual(dec.p_bin(True, 0.1), 1.0)
        self.assertEqual(dec.p_bin(False, None), 1.0)
        self.assertEqual(dec.p_bin(False, 0.6), 1.0)
        self.assertAlmostEqual(dec.p_bin(False, 0.3), 0.7)

    def test_p_grasp_shape_prior_and_ik_margin(self):
        xy, by = (0.30, 0.0), 0.26
        cube, cyl, tri = (dec.p_grasp(s, xy, by) for s in ("cube", "cylinder", "triangle"))
        self.assertGreater(cube, cyl)
        self.assertGreater(cyl, tri)
        self.assertAlmostEqual(cube, 0.9)
        self.assertAlmostEqual(cyl, 0.8)
        self.assertAlmostEqual(tri, 0.6)
        self.assertLess(dec.p_grasp("cube", (0.45, 0.0), by), cube)             # at the reach limit
        self.assertLess(dec.p_grasp("cube", (0.30, -0.2), by), cube)            # cross-body (object right, bin left)
        self.assertEqual(dec.p_grasp(None, xy, by), 0.75)                       # unknown shape

    def test_priority_is_two_p_over_time(self):
        self.assertAlmostEqual(dec.priority(0.9, 18.0), 0.1)

    def _item(self, p, t, shape, **kw):
        return dict(p=p, t=t, shape=shape, **kw)

    def test_rank_by_priority_then_shape(self):
        a, b, c = self._item(0.9, 20, "cube"), self._item(0.9, 10, "cube"), self._item(0.8, 10, "cylinder")
        out = dec.rank([a, b, c])
        self.assertEqual(out, [b, c, a])                                        # 0.18 > 0.16 > 0.09
        t1, t2 = self._item(0.6, 10, "triangle"), self._item(0.6, 10, "cube")   # equal priority: cube first
        self.assertEqual(dec.rank([t1, t2])[0], t2)

    def test_never_the_hardest_first(self):
        hard, easy = self._item(0.5, 5, "triangle"), self._item(0.9, 40, "cube")
        self.assertGreater(dec.priority(0.5, 5), dec.priority(0.9, 40))         # priority alone would lead with the hard one
        self.assertEqual(dec.rank([hard, easy]), [easy, hard])
        mid = self._item(0.7, 30, "cylinder")
        self.assertIsNot(dec.rank([hard, easy, mid])[0], hard)
        only = dec.rank([hard])
        self.assertEqual(only, [hard])                                          # a single object is still picked
        eq1, eq2 = self._item(0.8, 10, "cube"), self._item(0.8, 12, "cube")
        self.assertEqual(dec.rank([eq1, eq2])[0], eq1)                          # no strictly hardest: priority decides

    def test_rank_degraded_prefers_centre(self):
        side, centre = self._item(0.9, 10, "cube", centre_first=False), self._item(0.9, 30, "cube", centre_first=True)
        self.assertEqual(dec.rank([side, centre])[0], centre)

    def test_object_cap(self):
        self.assertAlmostEqual(dec.object_cap(120, 5), 21.0)                    # (120-15)/5
        self.assertEqual(dec.object_cap(120, 2), 35.0)                          # 52.5 -> clamp 35
        self.assertEqual(dec.object_cap(40, 3), 15.0)                           # 8.3 -> clamp 15
        self.assertAlmostEqual(dec.object_cap(75, 3), 20.0)
        self.assertEqual(dec.object_cap(100, 0), 35.0)                          # no division by zero

    def test_clocks(self):
        self.assertFalse(dec.no_new_object(99.9, 120))
        self.assertTrue(dec.no_new_object(100.0, 120))
        self.assertFalse(dec.endgame(111.9, 120))
        self.assertTrue(dec.endgame(112.0, 120))

    def test_endgame_action(self):
        self.assertEqual(dec.endgame_action("carry", True), "release")
        self.assertEqual(dec.endgame_action("carry", False), "retract")
        for ph in ("to_pregrasp", "descend", "close", "lift"):
            self.assertEqual(dec.endgame_action(ph, True), "retract")           # never release before the carry
        for ph in ("open", "retreat", "home", None):
            self.assertIsNone(dec.endgame_action(ph, False))
        ph = [("descend", 0, 10), ("carry", 10, 30), ("open", 30, 40)]
        self.assertEqual(dec.phase_at(ph, 10), "carry")
        self.assertEqual(dec.phase_at(ph, 9), "descend")
        self.assertIsNone(dec.phase_at(ph, 40))

    def test_speed_tiers_sequence(self):
        t = dec.SpeedTiers()
        self.assertEqual((t.scale(), t.clearance(), t.centre_only()), (0.7, 0.015, False))
        t.observe(True)
        self.assertEqual((t.scale(), t.clearance()), (0.85, 0.0))
        t.observe(True)
        self.assertEqual(t.scale(), 0.9)
        t.observe(False)
        self.assertEqual((t.scale(), t.centre_only(), t.clearance()), (0.6, True, 0.0))
        t.observe(True)                                                         # a clean success lifts the restriction
        self.assertEqual((t.scale(), t.centre_only()), (0.85, False))
        f = dec.SpeedTiers()
        f.observe(False)                                                        # failure on the very first object
        self.assertEqual((f.scale(), f.clearance(), f.centre_only()), (0.6, 0.0, True))
        for _ in range(6):
            t.observe(True)
        self.assertLessEqual(t.scale(), 1.0)

    def test_calib_confidence_and_tiers(self):
        self.assertIsNone(dec.calib_confidence())
        self.assertEqual(dec.calib_confidence(depth_valid_frac=0.6), 1.0)       # depth alone, renormalised
        self.assertEqual(dec.calib_confidence(depth_valid_frac=0.2), 0.0)
        self.assertAlmostEqual(dec.calib_confidence(pitch=1.0, bin_agree=0.0, depth_valid_frac=0.6), 0.65)
        self.assertEqual(dec.tier_of(0.6), dec.CONFIDENT)
        self.assertEqual(dec.tier_of(0.5999), dec.DEGRADED)
        self.assertEqual(dec.tier_of(0.3), dec.DEGRADED)
        self.assertEqual(dec.tier_of(0.2999), dec.PRIOR_ONLY)
        self.assertEqual(dec.tier_of(None), dec.PRIOR_ONLY)


# ----------------------------------------------------------------------------------------------- policy level
class FakePerception:
    hsv = {}

    def __init__(self, dets):
        self.dets = dets
        self.n_detect = 0

    def detect(self, rgb, depth):
        self.n_detect += 1
        return list(self.dets)

    def reset_calibration(self):
        pass


class FakePlanner:
    """plan() records the motion cfg it was called under and returns a scripted duration per colour."""

    def __init__(self, real, durations):
        self.real, self.durations, self.calls, self.seen = real, durations, [], []
        self.mapping = real.mapping
        self.cfg = real.cfg
        self.shared_seen = []     # the REAL (shared) cfg's arm_max_vel as seen by each plan() call

    def bin_xy(self, name):
        return self.real.bin_xy(name)

    def go_to(self, *a):
        return self.real.go_to(*a)

    def plan(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True):
        mo, g = self.cfg["motion"], self.cfg["grasp"]
        self.calls.append(det.color)
        self.shared_seen.append(self.real.cfg["motion"]["arm_max_vel"])
        self.seen.append((mo["arm_max_vel"], mo["arm_max_acc"], g.get("approach_extra_m", 0.0)))
        dur = self.durations[det.color]
        if dur is None:
            return Plan(False, color=det.color, reason="scripted failure")
        n = int(round(dur * 30))
        return Plan(True, side="left", color=det.color, target_bin=lookup_bin(self.cfg, self.mapping, det.color, getattr(det, "shape", None)), arm=np.zeros((n, 14)), hand=np.zeros((n, 14)),
                    waypoints={"grasp": det.top.copy(), "drop": det.top.copy()}, ik={"g": (True, 0.0, 0.0)},
                    safety_clips=0, phases=[("descend", 0, n // 4), ("carry", n // 4, n // 2), ("open", n // 2, n)])


def det(color, x=0.30, y=0.0, shape="cube", n_px=500):
    d = Detection(color, np.array([x, y, 0.8]), n_px, n_px, "depth")
    d.shape = shape
    return d


def obs(t, depth=None):
    return {"t": t, "arm_q": np.zeros(14), "hand_q": np.zeros(14), "head_rgb": None, "head_depth": depth}


def make(durations, dets, flags=(), time_limit=120.0, log=None, **extra):
    ov = {f"policy.{f}": True for f in flags}
    ov.update({k: v for k, v in extra.items()})
    pol = ClassicalPolicy(log=log, overrides=ov, time_limit=time_limit)
    pol.perception = FakePerception(dets)
    pol.planner = FakePlanner(pol.planner, durations)
    pol.state = "look"
    pol.arm_cmd, pol.hand_cmd = pol.home.copy(), np.zeros(14)
    return pol


DUR = {"red": 8.0, "yellow": 8.0, "blue": 8.0}


def first_color(pol, t=0.0):
    pol.act(obs(t))
    ex = [m for _, m in pol.events if m.startswith("execute")]
    return ex[-1].split()[1] if ex else None


class TestFlagsOff(unittest.TestCase):
    def test_four_flags_exist_and_default_off(self):
        cfg = load_config(overlay="none")
        for f in FLAGS4:
            self.assertFalse(cfg["policy"][f], f)
        self.assertEqual(cfg["grasp"]["approach_extra_m"], 0.0)
        self.assertFalse(any(on for _, _, on, _, _ in effective_flags(cfg)))

    def test_off_means_stock_order_and_untouched_motion(self):
        # stock `largest` order: red (900 px) first even though a cube would be the confidence pick
        r, y = det("red", shape="triangle", n_px=900), det("yellow", shape="cube", n_px=500)
        pol = make(DUR, [y, r], **{"policy.order": "largest"})
        base = (pol.cfg["motion"]["arm_max_vel"], pol.cfg["motion"]["arm_max_acc"])
        self.assertEqual(first_color(pol), "red")
        self.assertEqual(pol.planner.seen[0], base + (0.0,))
        self.assertIsNone(pol._exec)
        self.assertIsNone(pol.tier)

    def test_ab_decision_overlay_is_r16_nopp_plus_exactly_four_flags(self):
        a = {k for k, _, on, _, _ in effective_flags(load_config(overlay="ab_decision")) if on}
        b = {k for k, _, on, _, _ in effective_flags(load_config(overlay="cand_r17")) if on}
        self.assertEqual(a - b, {f"policy.{f}" for f in FLAGS4})
        self.assertEqual(b - a, set())
        ca, cb = load_config(overlay="ab_decision"), load_config(overlay="cand_r17")
        for f in FLAGS4:
            ca["policy"].pop(f)
            cb["policy"].pop(f)
        ca.pop("overlay"), cb.pop("overlay")
        self.assertEqual(ca, cb)


class TestConfidenceOrder(unittest.TestCase):
    def test_cube_before_prism_at_equal_geometry(self):
        tri, cube = det("red", shape="triangle", n_px=900), det("yellow", shape="cube", n_px=300)
        pol = make(DUR, [tri, cube], ("confidence_order",), **{"policy.order": "largest"})
        self.assertEqual(first_color(pol), "yellow")       # stock `largest` would lead with the 900 px prism

    def test_never_the_hardest_first(self):
        # the prism sits next to its bin (short estimate = highest 2p/t), the cube is far: the prism must not lead
        tri, cube = det("red", x=0.505, y=0.26, shape="triangle"), det("blue", x=0.20, y=0.30, shape="cube")
        pol = make(DUR, [tri, cube], ("confidence_order",))
        i_tri, i_cube = pol._conf_info(tri, obs(0)), pol._conf_info(cube, obs(0))
        self.assertGreater(dec.priority(i_tri["p"], i_tri["t"]), dec.priority(i_cube["p"], i_cube["t"]))
        self.assertLess(i_tri["p"], i_cube["p"])
        self.assertEqual(first_color(pol), "blue")

    def test_reranked_after_each_placement(self):
        a, b, c = det("yellow", shape="cube"), det("red", shape="cylinder"), det("blue", shape="triangle")
        pol = make(DUR, [a, b, c], ("confidence_order",))
        order1 = [d.color for d in pol._conf_rank([a, b, c], [a, b, c], obs(0), lambda d: ())]
        order2 = [d.color for d in pol._conf_rank([b, c], [b, c], obs(1), lambda d: ())]   # a was placed
        self.assertEqual(order1[0], "yellow")
        self.assertEqual(order2, [x for x in order1 if x != "yellow"])
        self.assertEqual(len(pol._stab), 2)                # one stability frame per look

    def test_stability_lowers_p_of_a_flickering_object(self):
        a = det("red")
        pol = make(DUR, [a], ("confidence_order",))
        pol._push_frame([a])
        pol._push_frame([])
        pol._push_frame([a])
        self.assertAlmostEqual(pol._stability(a), 2 / 3)
        stable = pol._conf_info(a, obs(0))["pc"]
        pol._stab = [[], [], []]
        self.assertLess(pol._conf_info(a, obs(0))["pc"], stable)

    def test_unmapped_object_gets_p_zero(self):
        pol = make(DUR, [], ("confidence_order",))
        pol.mapping = {"red": "left"}
        self.assertEqual(pol._conf_info(det("blue"), obs(0))["p"], 0.0)


class TestTimeCap(unittest.TestCase):
    def test_over_cap_plan_is_skipped_for_the_next_object(self):
        # 2 objects at t=0: cap = min(35, 52.5) = 35. red needs 40 s > 35, blue 10 s fits
        r, b = det("red", n_px=900), det("blue", n_px=500)
        pol = make({"red": 40.0, "blue": 10.0, "yellow": 8.0}, [r, b], ("object_time_cap",), **{"policy.order": "largest"})
        self.assertEqual(first_color(pol), "blue")
        self.assertEqual(pol.planner.calls, ["red", "blue"])
        self.assertEqual(pol.failed, [])                   # not marked failed: it comes back

    def test_cap_shrinks_with_the_clock(self):
        r, b = det("red", n_px=900), det("blue", n_px=500)
        dur = {"red": 20.0, "blue": 10.0, "yellow": 8.0}
        pol = make(dur, [r, b], ("object_time_cap",), **{"policy.order": "largest"})
        self.assertEqual(first_color(pol, 0.0), "red")     # cap 35 s: 20 s fits
        pol = make(dur, [r, b], ("object_time_cap",), **{"policy.order": "largest"})
        self.assertEqual(first_color(pol, 70.0), "blue")   # (120-70-15)/2 = 17.5 -> 15..35 clamp: 17.5 < 20 s

    def test_only_over_cap_object_is_still_taken(self):
        r = det("red")
        pol = make({"red": 40.0, "blue": 1.0, "yellow": 1.0}, [r], ("object_time_cap",))
        self.assertEqual(first_color(pol), "red")          # nothing else to do: the quickest over-cap plan, not 0 points

    def test_no_new_object_after_t_minus_20(self):
        pol = make(DUR, [det("red")], ("object_time_cap",))
        done = pol.act(obs(99.9))
        self.assertFalse(done["done"])
        self.assertEqual(pol.planner.calls, ["red"])       # 99.9: still allowed
        pol = make(DUR, [det("red")], ("object_time_cap",))
        done = pol.act(obs(100.0))
        self.assertTrue(done["done"])
        self.assertEqual(pol.planner.calls, [])            # 100.0: nothing planned, nothing started
        pol = make(DUR, [det("red")], time_limit=120.0)    # flag off: stock behaviour at t=100 still picks
        pol.act(obs(100.0))
        self.assertEqual(pol.planner.calls, ["red"])

    def _executing(self, phase_tick_frac, t_flag=("object_time_cap",)):
        pol = make({"red": 20.0, "blue": 20.0, "yellow": 20.0}, [det("red")], t_flag)
        plan = pol.planner.plan(det("red"), pol.arm_cmd, pol.hand_cmd)
        pol._start(plan, 0.0)
        pol.arm_cmd = np.full(14, 0.3)                     # where the arm is at T-8 s
        pol.hand_cmd = np.zeros(14)
        pol.queue_a, pol.queue_h = pol.queue_a[int(phase_tick_frac * len(plan.arm)):], pol.queue_h[int(phase_tick_frac * len(plan.arm)):]
        return pol, plan

    def test_t8_over_the_bin_releases_where_it_is(self):
        pol, plan = self._executing(0.35)                  # inside "carry"
        pol._hand_xyz = lambda p: np.r_[pol.planner.bin_xy("left"), 0.2]
        self.assertIsNone(pol._endgame(111.9))             # one tick before T-8: nothing happens
        a = pol._endgame(112.0)
        self.assertIsNotNone(a)
        A = np.vstack([a["arm_q_target"], pol.queue_a])
        H = np.vstack([a["hand_q_target"], pol.queue_h])
        rel = pol.post[pol.cfg["grasp"]["release_posture"]][:7]
        k = int(np.argmax(np.all(np.isclose(H[:, :7], rel, atol=1e-9), axis=1)))
        self.assertTrue(np.allclose(H[k, :7], rel))        # the hand reaches the release posture ...
        np.testing.assert_allclose(A[:k + 1], np.full((k + 1, 14), 0.3))   # ... while the arm has not moved at all
        np.testing.assert_allclose(A[-1][:7], pol.home[:7], atol=1e-9)     # then home
        np.testing.assert_allclose(H[-1], pol.post["open"], atol=1e-9)
        self.assertIsNone(pol._endgame(113.0))             # once per plan

    def test_t8_not_over_the_bin_retracts(self):
        pol, plan = self._executing(0.35)
        pol._hand_xyz = lambda p: np.array([0.30, 0.0, 0.2])      # over the counter
        a = pol._endgame(113.0)
        self.assertIsNotNone(a)
        np.testing.assert_allclose(pol.queue_a[-1][:7], pol.home[:7], atol=1e-9)
        np.testing.assert_allclose(pol.queue_h[-1], pol.post["open"], atol=1e-9)   # hand open at home

    def test_t8_before_the_carry_retracts_never_releases(self):
        pol, plan = self._executing(0.05)                  # still descending
        pol._hand_xyz = lambda p: np.r_[pol.planner.bin_xy("left"), 0.2]   # would be "over the bin"
        a = pol._endgame(115.0)
        self.assertIsNotNone(a)
        np.testing.assert_allclose(pol.queue_h[-1], pol.post["open"], atol=1e-9)
        np.testing.assert_allclose(pol.queue_a[-1][:7], pol.home[:7], atol=1e-9)

    def test_t8_already_releasing_is_left_alone(self):
        pol, plan = self._executing(0.8)                   # inside "open"
        n = len(pol.queue_a)
        self.assertIsNone(pol._endgame(118.0))
        self.assertEqual(len(pol.queue_a), n)

    def test_t8_through_act_and_flag_off(self):
        pol, plan = self._executing(0.35)
        pol._hand_xyz = lambda p: np.r_[pol.planner.bin_xy("left"), 0.2]
        pol.act(obs(112.5))
        self.assertTrue(pol._exec["t8"])
        np.testing.assert_allclose(pol.queue_a[-1][:7], pol.home[:7], atol=1e-9)   # the replaced motion, not the old plan
        off = make(DUR, [det("red")])
        plan = off.planner.plan(det("red"), off.arm_cmd, off.hand_cmd)
        off._start(plan, 0.0)
        self.assertIsNone(off._exec)                       # flag off: no bookkeeping, no T-8 handling
        n = len(off.queue_a)
        off.act(obs(119.0))
        self.assertEqual(len(off.queue_a), n - 10)         # just the next chunk of the stock plan


class TestSpeedTiers(unittest.TestCase):
    def _run_one(self, pol, dets_after, t):
        """Plan + start whatever is picked now, then look again with `dets_after` (the scene after the move)."""
        pol.act(obs(t))
        pol.queue_a = pol.queue_h = None
        pol.state = "look"
        pol.perception.dets = dets_after

    def test_first_object_is_slow_with_clearance_then_tiers_follow_outcomes(self):
        base_v, base_a = None, None
        yel, red, blue = det("yellow", y=0.0, n_px=1200), det("red", y=0.1, n_px=900), det("blue", y=-0.1, n_px=500)
        yel2 = det("yellow", x=0.35, y=0.02, n_px=400)
        pol = make(DUR, [yel, red, blue], ("speed_tiers",), **{"policy.order": "largest"})
        base_v, base_a = pol.cfg["motion"]["arm_max_vel"], pol.cfg["motion"]["arm_max_acc"]
        self._run_one(pol, [red, blue], 0.0)                # yellow picked and gone: clean
        self._run_one(pol, [blue], 10.0)                    # red picked and gone: clean (2nd)
        self._run_one(pol, [blue, yel2], 20.0)              # blue picked but still there: failure
        pol.act(obs(30.0))                                  # failure tier: only the centre-bin yellow is eligible
        self.assertEqual(pol.planner.calls, ["yellow", "red", "blue", "yellow"])
        seen = pol.planner.seen
        self.assertEqual(seen[0], (base_v * 0.7, base_a * 0.7, 0.015))        # first: 0.7 + 1.5 cm
        self.assertEqual(seen[1], (base_v * 0.85, base_a * 0.85, 0.0))        # after one clean
        self.assertEqual(seen[2], (base_v * 0.9, base_a * 0.9, 0.0))          # after two clean
        self.assertEqual(seen[3], (base_v * 0.6, base_a * 0.6, 0.0))          # after the failure
        self.assertTrue(all(v <= base_v and a <= base_a for v, a, _ in seen))  # never above the stock speed
        self.assertEqual((pol.cfg["motion"]["arm_max_vel"], pol.cfg["motion"]["arm_max_acc"],
                          pol.cfg["grasp"]["approach_extra_m"]), (base_v, base_a, 0.0))   # restored

    def test_failure_restricts_to_the_centre_bin(self):
        side, centre = det("red", y=0.1, n_px=900), det("yellow", y=0.0, n_px=500)
        pol = make(DUR, [side, centre], ("speed_tiers",), **{"policy.order": "largest"})
        pol.tiers.observe(False)                            # a failure happened
        self.assertEqual(first_color(pol), "yellow")        # the larger side-bin red is not eligible
        self.assertNotIn("red", pol.planner.calls)
        pol = make(DUR, [side], ("speed_tiers",))
        pol.tiers.observe(False)
        pol.act(obs(0.0))
        self.assertEqual(pol.planner.calls, ["red"])        # F1: no centre object left: the side bin is taken (no deadlock)

    def test_outcome_judged_by_the_object_still_being_there(self):
        a = det("red")
        pol = make(DUR, [a], ("speed_tiers",))
        pol._pend_out = {"color": "red", "xy": a.top[:2].copy()}
        pol._observe_outcome([det("red", x=0.31)])          # 1 cm away = same object = failure
        self.assertTrue(pol.tiers.failed)
        pol._pend_out = {"color": "red", "xy": a.top[:2].copy()}
        pol._observe_outcome([det("blue")])                 # other colour only = gone = clean
        self.assertFalse(pol.tiers.failed)
        self.assertEqual(pol.tiers.n_clean, 1)

    def test_corrective_picks_are_not_judged(self):
        pol = make(DUR, [det("red")], ("speed_tiers",))
        plan = pol.planner.plan(det("red"), pol.arm_cmd, pol.hand_cmd)
        plan.corrective = True
        pol._start(plan, 0.0)
        self.assertIsNone(pol._pend_out)


class TestConfidenceTiers(unittest.TestCase):
    def _hold_through(self, pol, depth, t0=0.0, step=0.2):
        """act() every 0.2 s until the first pick or done; returns (list of act times, time of the tier decision)."""
        t, decided = t0, None
        for _ in range(200):
            a = pol.act(obs(t, depth))
            if pol.tier is not None and decided is None:
                decided = t
            if a["done"] or any(m.startswith("execute") for _, m in pol.events):
                break
            t += step
        return t, decided

    def test_still_for_about_four_seconds_then_tier_confident(self):
        lines = []
        pol = make(DUR, [det("yellow")], ("confidence_tiers",), log=lines.append)
        t, decided = self._hold_through(pol, np.full((48, 64), 700.0))
        self.assertEqual(pol.tier, dec.CONFIDENT)
        self.assertGreaterEqual(decided, 4.0)               # hold_s
        self.assertLess(decided, 4.5)
        self.assertEqual(pol.planner.calls, ["yellow"])     # nothing planned during the hold
        self.assertEqual(pol.perception.n_detect >= 4, True)   # hold_frames detect frames + the look
        self.assertTrue(any(l.startswith("[tier] CONFIDENT C=1.00") for l in lines), lines)
        self.assertEqual(pol.planner.seen[0][0], pol.cfg["motion"]["arm_max_vel"])   # CONFIDENT: no slowdown

    def test_hold_does_not_move_the_arms(self):
        pol = make(DUR, [det("yellow")], ("confidence_tiers",))
        arm0 = pol.arm_cmd.copy()
        for k in range(10):
            a = pol.act(obs(k * 0.2, np.full((48, 64), 700.0)))
            np.testing.assert_array_equal(a["arm_q_target"], np.repeat(arm0[None], len(a["arm_q_target"]), 0))
        self.assertIsNone(pol.tier)

    def test_degraded_tier(self):
        lines = []
        depth = np.zeros((48, 64))
        depth[:, :26] = 700.0                               # ~0.41 valid -> depth component 0.52 -> C 0.52
        big_side, small_side, big_centre = det("red", n_px=500), det("blue", n_px=70), det("yellow", n_px=500)
        pol = make(DUR, [small_side, big_side, big_centre], ("confidence_tiers",), log=lines.append)
        self._hold_through(pol, depth)
        self.assertEqual(pol.tier, dec.DEGRADED, lines)
        self.assertTrue(any(l.startswith("[tier] DEGRADED") for l in lines))
        # p_colour*p_bin: big side bin 1.0 * (0.4+0.52) = 0.92 passes, the 70 px blob 0.62 * 0.92 fails (blue skipped)
        self.assertNotIn("blue", pol.planner.calls)
        self.assertEqual(pol.planner.calls[0], "yellow")    # centre bin preferred
        self.assertAlmostEqual(pol.planner.seen[0][0], pol.cfg["motion"]["arm_max_vel"] * 0.6)   # speed 0.6

    def test_degraded_gate_boundary(self):
        pol = make(DUR, [], ("confidence_tiers",))
        pol.tier, pol.C = dec.DEGRADED, 0.5
        ok, low = det("red", n_px=500), det("blue", n_px=70)
        self.assertEqual([d.color for d in pol._decision_filter([ok, low], obs(0))], ["red"])
        pol.C = 0.4                                         # side-bin trust 0.8 < 0.85: even a perfect blob fails
        self.assertEqual(pol._decision_filter([ok], obs(0)), [])
        self.assertEqual([d.color for d in pol._decision_filter([det("yellow")], obs(0))], ["yellow"])   # centre = prior

    def test_prior_only_one_centre_attempt_then_hold(self):
        lines = []
        pol = make(DUR, [det("red", y=0.12, n_px=900), det("yellow")], ("confidence_tiers",), log=lines.append, **{"policy.max_attempts": 1})
        self._hold_through(pol, np.zeros((48, 64)))         # no depth at all: C = 0
        self.assertEqual(pol.tier, dec.PRIOR_ONLY)
        self.assertTrue(any(l.startswith("[tier] PRIOR-ONLY") for l in lines))
        self.assertEqual(pol.planner.calls, ["yellow"])     # the side-bin red is never planned
        pol.queue_a = pol.queue_h = None                    # the attempt is over (object still there)
        pol.state = "look"
        pol.act(obs(30.0, np.zeros((48, 64))))
        self.assertEqual(pol.planner.calls, ["yellow", "red"])   # F1: centre exhausted -> one attempt at the side bin
        pol.queue_a = pol.queue_h = None
        pol.state = "look"
        a = pol.act(obs(40.0, np.zeros((48, 64))))
        self.assertTrue(a["done"])                          # at most one attempt per side bin, then hold
        self.assertEqual(pol.planner.calls, ["yellow", "red"])

    def test_prior_only_with_side_bin_objects_only_one_attempt_per_side_bin(self):
        pol = make(DUR, [det("red", n_px=900), det("blue")], ("confidence_tiers",), **{"policy.order": "largest"})
        z = np.zeros((48, 64))
        self._hold_through(pol, z)
        self.assertEqual(pol.planner.calls, ["red"])        # F1: no centre object -> side bins are allowed
        for t in (30.0, 40.0, 50.0):
            pol.queue_a = pol.queue_h = None
            pol.state = "look"
            pol.act(obs(t, z))
        self.assertEqual(pol.planner.calls, ["red", "blue"])   # one attempt per side bin, never a third

    def test_inputs_pitch_cal_and_bin_agreement_are_used_when_present(self):
        pol = make(DUR, [det("yellow")], ("confidence_tiers",))
        pol.perception.pitch_cal = 0.1                      # a poor pitch-cal result drags C down
        self._hold_through(pol, np.full((48, 64), 700.0))
        self.assertAlmostEqual(pol.C, (0.4 * 0.1 + 0.25 * 1.0) / 0.65)
        self.assertEqual(pol.tier, dec.DEGRADED)

    def test_tier_is_measured_once_per_episode_and_reset(self):
        pol = make(DUR, [det("yellow")], ("confidence_tiers",))
        self._hold_through(pol, np.full((48, 64), 700.0))
        n = pol.perception.n_detect
        pol.queue_a = pol.queue_h = None
        pol.state = "look"
        pol.act(obs(20.0, np.full((48, 64), 700.0)))
        self.assertEqual(pol.perception.n_detect, n + 1)    # a normal look, no second hold
        pol.reset({})
        self.assertIsNone(pol.tier)
        self.assertIsNone(pol.C)


if __name__ == "__main__":
    unittest.main()


class TestReviewFixes(unittest.TestCase):
    def test_f1_failed_centre_pick_falls_back_to_the_side_bins(self):
        yel = det("yellow", y=0.0, n_px=1200)
        red, red2, blue = det("red", y=0.1, n_px=900), det("red", x=0.35, y=0.12, n_px=800), det("blue", y=-0.1, n_px=500)
        pol = make(DUR, [yel, red, red2, blue], ("speed_tiers",), **{"policy.order": "largest", "policy.max_attempts": 1})
        pol.act(obs(0.0))
        self.assertEqual(pol.planner.calls, ["yellow"])     # the only centre object first
        pol.queue_a = pol.queue_h = None
        pol.state = "look"
        pol.act(obs(10.0))                                  # yellow is still there: failed; centre exhausted (attempt limit)
        self.assertTrue(pol.tiers.failed)
        self.assertEqual(pol.planner.calls[1:], ["red"])    # side-bin objects are picked, not "none"
        self.assertEqual(pol.planner.seen[1][0], pol.cfg["motion"]["arm_max_vel"] * 0.6)   # still at fail_scale
        pol.queue_a = pol.queue_h = None
        pol.state = "look"
        pol.act(obs(20.0))
        self.assertEqual(len(pol.planner.calls), 3)         # and the next one, no deadlock

    def test_f1_centre_only_still_holds_while_a_centre_candidate_exists(self):
        pol = make(DUR, [det("red", n_px=900), det("yellow")], ("speed_tiers",))
        pol.tiers.observe(False)
        self.assertEqual([d.color for d in pol._decision_filter(pol.perception.dets, obs(0))], ["yellow"])
        self.assertTrue(pol.tiers.centre_only())
        self.assertFalse(pol.tiers.centre_only(has_centre=False))

    def test_f2_endgame_tick_does_not_double_count_lift_on_stall(self):
        lines = []
        pol = make({"red": 20.0, "blue": 20.0, "yellow": 20.0}, [det("red")], ("object_time_cap",), log=lines.append)
        plan = pol.planner.plan(det("red"), pol.arm_cmd, pol.hand_cmd)      # 600 ticks: carry 150..300, open 300..600
        pol._start(plan, 0.0)
        pol.arm_cmd, pol.hand_cmd = np.full(14, 0.3), np.zeros(14)
        cut = 40                                            # 250 ticks played (10 by _start) ...
        pol.queue_a, pol.queue_h = pol.queue_a[240:], pol.queue_h[240:]
        pol.queue_a, pol.queue_h = pol.queue_a[cut:], pol.queue_h[cut:]     # ... then lift_on_stall skips 40 hold ticks
        pol._exec["cut"] += cut                             # (exactly what _lift_on_stall does)
        pol._hand_xyz = lambda p: np.r_[pol.planner.bin_xy("left"), 0.2]
        self.assertIsNotNone(pol._endgame(112.0))
        self.assertTrue(any("plan phase=carry" in l for l in lines), lines)   # tick 290 = carry, not 330 = open

    def test_f4_never_hardest_first_stays_inside_the_centre_group(self):
        mk = lambda p, t, cf: dict(p=p, t=t, shape="cube", centre_first=cf)
        c_hard, c_easy, side = mk(0.5, 5, True), mk(0.9, 40, True), mk(0.95, 5, False)
        out = dec.rank([side, c_hard, c_easy])
        self.assertEqual(out, [c_easy, c_hard, side])       # swap inside the centre group; the side object never leads
        self.assertIs(dec.rank([side, c_hard])[0], c_hard)  # a lone centre object is not swapped with a side one

    def test_f5_shared_cfg_is_never_mutated_while_planning(self):
        pol = make(DUR, [det("red")], ("speed_tiers",))
        base = (pol.cfg["motion"]["arm_max_vel"], pol.cfg["motion"]["arm_max_acc"], pol.cfg["grasp"].get("approach_extra_m"))
        pol.act(obs(0.0))
        self.assertEqual(pol.planner.shared_seen, [base[0]])   # the shared dict kept the stock value during the plan
        self.assertEqual(pol.planner.seen[0][0], base[0] * 0.7)   # while the plan itself was made at the scaled speed
        self.assertEqual(base[2], pol.cfg["grasp"].get("approach_extra_m"))
        self.assertEqual(pol.cfg["motion"]["arm_max_vel"], base[0])

    def test_f6_two_same_colour_objects_5cm_apart_clean_pick_is_not_failed(self):
        a = det("red", x=0.30)
        pol = make(DUR, [a], ("speed_tiers",))
        pol._pend_out = {"color": "red", "xy": a.top[:2].copy()}
        pol._observe_outcome([det("red", x=0.35)])          # the neighbour 5 cm away is another object
        self.assertFalse(pol.tiers.failed)
        self.assertEqual(pol.tiers.n_clean, 1)
        pol._pend_out = {"color": "red", "xy": a.top[:2].copy()}
        pol._observe_outcome([det("red", x=0.315)])         # 1.5 cm: the same object -> failure
        self.assertTrue(pol.tiers.failed)

    def test_f6_with_retry_tracks_the_survivor_must_belong_to_the_attempt_track(self):
        a = det("red", x=0.30)
        pol = make(DUR, [a], ("speed_tiers",))
        pol.attempt_tracks = [{"color": "red", "xy": np.array([0.30, 0.0]), "count": 1}]
        pol._pend_out = {"color": "red", "xy": a.top[:2].copy()}
        pol._observe_outcome([det("red", x=0.31)])
        self.assertTrue(pol.tiers.failed)
        pol.attempt_tracks = [{"color": "blue", "xy": np.array([0.30, 0.0]), "count": 1}]   # no red track here
        pol._pend_out = {"color": "red", "xy": a.top[:2].copy()}
        pol._observe_outcome([det("red", x=0.31)])
        self.assertFalse(pol.tiers.failed)

    def test_f7_a_raising_decision_helper_falls_back_to_stock_and_warns_once(self):
        lines = []
        pol = make(DUR, [det("red", n_px=900), det("yellow")], FLAGS4, log=lines.append)
        pol.tier = dec.CONFIDENT

        def boom(*a, **k):
            raise RuntimeError("kaput")
        pol._conf_info = boom
        for t in (0.0, 10.0):
            pol.queue_a = pol.queue_h = None
            pol.state = "look"
            pol.act(obs(t))                                 # must not raise
        warn = [l for l in lines if l.startswith("[decision] ")]
        self.assertEqual(len(warn), 2, warn)                # one per helper (_decision_filter, _conf_rank), once per episode
        self.assertTrue(all("stock behaviour" in l for l in warn))
        self.assertTrue(pol.planner.calls)                  # stock order still plans something
        pol2 = make(DUR, [det("red")], ("object_time_cap",), log=lines.append)
        plan = pol2.planner.plan(det("red"), pol2.arm_cmd, pol2.hand_cmd)
        pol2._start(plan, 0.0)
        pol2._over_target_bin = boom
        pol2._hand_xyz = lambda p: np.zeros(3)
        pol2.act(obs(115.0))                                # T-8 with a raising endgame helper: no exception


# ----------------------------------------------------------------------------------------------- decision.isolated_first
def _order(pol, dets):
    """Candidate order _choose hands to the planner: every plan fails, so each candidate is tried once, in order."""
    pol.planner.durations = {k: None for k in pol.planner.durations}
    pol._choose(dets, obs(0.0), False)
    return list(pol.planner.calls)


class TestIsolatedFirst(unittest.TestCase):
    ON = {"decision.isolated_first": True}
    LARGEST = {"policy.order": "largest"}

    def _touching_and_isolated(self):
        a = det("red", x=0.30, y=0.0, shape="cube", n_px=900)
        b = det("yellow", x=0.33, y=0.0, shape="cube", n_px=800)   # 3 cm from a: touching
        c = det("blue", x=0.30, y=0.20, shape="cube", n_px=300)    # 20 cm from both: isolated
        return [a, b, c]

    def test_isolated_object_first_touching_pair_after(self):
        pol = make(DUR, self._touching_and_isolated(), (), **self.LARGEST, **self.ON)
        self.assertEqual(_order(pol, self._touching_and_isolated()), ["blue", "red", "yellow"])

    def test_all_far_apart_order_unchanged_vs_flag_off(self):
        def far():
            return [det("red", x=0.30, y=0.0, n_px=900), det("yellow", x=0.30, y=0.20, n_px=800),
                    det("blue", x=0.30, y=-0.20, n_px=700)]
        off = _order(make(DUR, far(), (), **self.LARGEST), far())
        on = _order(make(DUR, far(), (), **self.LARGEST, **self.ON), far())
        self.assertEqual(off, ["red", "yellow", "blue"])
        self.assertEqual(on, off)

    def test_flag_off_touching_order_unchanged_and_code_path_not_run(self):
        dets = self._touching_and_isolated()
        with mock.patch.object(dec, "defer_touching", side_effect=AssertionError("must not run when off")):
            off = _order(make(DUR, dets, (), **self.LARGEST), self._touching_and_isolated())
        self.assertEqual(off, ["red", "yellow", "blue"])     # stock largest order, touching pair included

    def test_deferred_object_is_still_picked(self):
        dets = self._touching_and_isolated()
        durations = {"red": 8.0, "yellow": 8.0, "blue": None}   # the isolated blue plan fails
        pol = make(durations, dets, (), **self.LARGEST, **self.ON)
        self.assertEqual(first_color(pol), "red")                # blue tried first, then the deferred red
        self.assertEqual(pol.planner.calls, ["blue", "red"])
        self.assertEqual(len(pol.failed), 1)                     # blue skipped for its own reason, not dropped by the flag

    def test_only_touching_candidates_still_one_is_picked(self):
        a, b = det("red", x=0.30, y=0.0, n_px=900), det("yellow", x=0.33, y=0.0, n_px=800)
        pol = make(DUR, [a, b], (), **self.LARGEST, **self.ON)
        self.assertIn(first_color(pol), ("red", "yellow"))

    def test_degraded_tier_centre_first_wins_over_isolated_first(self):
        def dets():   # yellow = centre bin (fixed mapping), touching red; blue isolated
            return [det("red", x=0.30, y=0.0, n_px=900), det("yellow", x=0.33, y=0.0, n_px=800),
                    det("blue", x=0.30, y=0.20, n_px=300)]
        pol = make(DUR, dets(), (), **self.LARGEST)
        self.assertEqual(pol._centre_name(), "center")
        pol.tier = dec.DEGRADED
        self.assertEqual(_order(pol, dets()), ["yellow", "red", "blue"])           # flag off: centre first
        pol = make(DUR, dets(), (), **self.LARGEST, **self.ON)
        pol.tier = dec.DEGRADED
        self.assertEqual(_order(pol, dets()), ["yellow", "blue", "red"])           # centre first, then side: isolated first

    def test_defer_touching_within_keeps_group_positions(self):
        t1, t2 = det("red", x=0.0, y=0.0), det("yellow", x=0.03, y=0.0)   # touching pair
        i1 = det("blue", x=0.5, y=0.0)
        out = dec.defer_touching_within([t1, t2, i1], [t1, t2, i1], 0.06, key=lambda d: d.color == "yellow")
        self.assertEqual([id(d) for d in out], [id(i1), id(t2), id(t1)])   # red/blue group swap, yellow keeps slot 1

    def test_defer_touching_keeps_relative_order(self):
        t1, t2 = det("red", x=0.0, y=0.0), det("yellow", x=0.03, y=0.0)   # touching pair
        i1, i2 = det("blue", x=0.5, y=0.0), det("red", x=0.9, y=0.9)       # isolated
        cands = [t1, i1, t2, i2]
        out = dec.defer_touching(cands, cands, 0.06)
        self.assertEqual([id(d) for d in out], [id(i1), id(i2), id(t1), id(t2)])
