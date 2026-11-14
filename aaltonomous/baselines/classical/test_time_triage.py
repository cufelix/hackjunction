"""CPU tests of policy.time_triage with a scripted perception + planner (no IK, no sim). Run from outside the repo:
    PYTHONPATH=<repo> python -m pytest <repo>/aaltonomous/baselines/classical/test_time_triage.py -q
"""
import unittest

import numpy as np

from .perception import Detection
from .planner import Plan
from .policy import ClassicalPolicy


class FakePerception:
    def __init__(self, dets):
        self.dets = dets

    def detect(self, rgb, depth):
        return list(self.dets)


class FakePlanner:
    """plan() returns a scripted duration per colour (None = IK failure) and counts the calls."""

    def __init__(self, real, durations):
        self.real, self.durations, self.calls = real, durations, []
        self.mapping = real.mapping

    def bin_xy(self, name):
        return self.real.bin_xy(name)

    def plan(self, det, arm_q, hand_q, bin_name=None, obstacles=(), relocate=True):
        self.calls.append(det.color)
        dur = self.durations[det.color]
        if dur is None:
            return Plan(False, color=det.color, reason="scripted failure")
        n = int(round(dur * 30))
        return Plan(True, side="left", color=det.color, target_bin="x", arm=np.zeros((n, 14)), hand=np.zeros((n, 14)),
                    waypoints={"grasp": det.top.copy(), "drop": det.top.copy()}, ik={"g": (True, 0.0, 0.0)},
                    safety_clips=0)


def det(color, x, y=0.0, shape=None):
    d = Detection(color, np.array([x, y, 0.8]), 500, 400, "depth")
    if shape:
        d.shape = shape
    return d


def make(durations, dets, time_limit=100.0, **policy):
    ov = {f"policy.{k}": v for k, v in policy.items()}
    pol = ClassicalPolicy(log=None, overrides=ov, time_limit=time_limit)
    pol.perception = FakePerception(dets)
    pol.planner = FakePlanner(pol.planner, durations)
    pol.state = "look"
    pol.arm_cmd, pol.hand_cmd = pol.home.copy(), np.zeros(14)
    return pol


def obs(t):
    return {"t": t, "arm_q": np.zeros(14), "hand_q": np.zeros(14), "head_rgb": None, "head_depth": None}


def first_executed(pol, t):
    """act() once from the look state; returns (done, colour of the executed plan or None)."""
    a = pol.act(obs(t))
    ex = [m for _, m in pol.events if m.startswith("execute")]
    return a["done"], (ex[0].split()[1] if ex else None)


def far_near_dets():
    # nearest_bin order puts the colour nearest its bin first; positions chosen by the test through `order: largest`
    d1, d2 = det("red", 0.30), det("blue", 0.35)
    d1.n_px, d2.n_px = 900, 500        # order largest: red first, then blue
    return [d1, d2]


class TestTriage(unittest.TestCase):
    def test_flag_off_first_misfit_ends_episode(self):
        pol = make({"red": 60.0, "blue": 10.0}, far_near_dets(), time_limit=50.0, order="largest")
        done, ex = first_executed(pol, 10.0)
        self.assertTrue(done)
        self.assertIsNone(ex)
        self.assertEqual(pol.planner.calls, ["red"])          # blue was never even planned
        self.assertIn("not enough time", pol.events[-1][1])

    def test_long_first_short_second_runs_short(self):
        pol = make({"red": 60.0, "blue": 10.0}, far_near_dets(), time_limit=50.0, order="largest", time_triage=True)
        done, ex = first_executed(pol, 10.0)
        self.assertFalse(done)
        self.assertEqual(ex, "blue")
        self.assertEqual(pol.planner.calls, ["red", "blue"])
        self.assertEqual(pol.failed, [])                       # the slow object is not marked failed
        self.assertEqual(len(pol.too_long), 1)

    def test_nothing_fits_is_done(self):
        pol = make({"red": 60.0, "blue": 45.0}, far_near_dets(), time_limit=50.0, order="largest", time_triage=True)
        done, ex = first_executed(pol, 10.0)
        self.assertTrue(done)
        self.assertIsNone(ex)
        self.assertIn("no candidate fits", pol.events[-1][1])
        self.assertEqual(pol.failed, [])

    def test_too_long_cache_skips_replanning(self):
        pol = make({"red": 60.0, "blue": 10.0}, far_near_dets(), time_limit=50.0, order="largest", time_triage=True)
        first_executed(pol, 10.0)
        pol.planner.calls.clear()
        pol.state, pol.queue_a, pol.queue_h = "look", None, None
        pol.act(obs(25.0))
        self.assertNotIn("red", pol.planner.calls)             # known not to fit, no IK spent on it again

    def test_failed_ik_still_marks_failed_and_moves_on(self):
        pol = make({"red": None, "blue": 10.0}, far_near_dets(), time_limit=50.0, order="largest", time_triage=True)
        done, ex = first_executed(pol, 0.0)
        self.assertEqual(ex, "blue")
        self.assertEqual(len(pol.failed), 1)

    def test_tie_break_shape_then_confidence(self):
        # same cheap estimate (same x): cylinder is listed first by `largest`, the cube must win under `priority`
        cyl, cube = det("red", 0.30, shape="cylinder"), det("blue", 0.30, shape="cube")
        cyl.n_px, cube.n_px = 900, 500
        kw = dict(time_limit=100.0, order="nearest_bin", time_triage=True)
        pol = make({"red": 5.0, "blue": 5.0}, [cyl, cube], triage_order="priority", **kw)
        self.assertEqual(first_executed(pol, 0.0)[1], "blue")
        pol = make({"red": 5.0, "blue": 5.0}, [cyl, cube], triage_order="duration", **kw)   # off: estimate order only
        ref = sorted([cyl, cube], key=lambda d: np.linalg.norm(d.top[:2] - pol.planner.bin_xy(pol.mapping[d.color])))[0]
        self.assertEqual(first_executed(pol, 0.0)[1], ref.color)
        # equal shapes: higher mapping confidence first
        a, b = det("red", 0.30, shape="cube"), det("blue", 0.30, shape="cube")
        pol = make({"red": 5.0, "blue": 5.0}, [a, b], triage_order="priority", **kw)
        pol.mapping_conf = {"red": 0.4, "blue": 0.9}
        self.assertEqual(first_executed(pol, 0.0)[1], "blue")

    def test_priority_does_not_override_clearly_cheaper_estimate(self):
        near, far = det("red", 0.30, shape="triangle"), det("blue", 0.60, shape="cube")
        pol = make({"red": 5.0, "blue": 5.0}, [far, near], time_limit=100.0, order="largest", time_triage=True,
                   triage_order="priority", triage_tie_m=1e-6)
        pol.perception.dets[0].n_px = 900
        # `largest` keys are -n_px (huge gaps vs eps): the bigger blob stays first whatever its shape cost
        self.assertEqual(first_executed(pol, 0.0)[1], "blue")

    def test_margin(self):
        for triage in (False, True):
            # 10 + 40 = 50 <= 50 fits with margin 0; margin 0.5 makes it a misfit
            pol = make({"red": 40.0, "blue": 40.0}, far_near_dets(), time_limit=50.0, order="largest",
                       time_triage=triage)
            self.assertIsNotNone(first_executed(pol, 10.0)[1], f"triage={triage} margin 0")
            pol = make({"red": 40.0, "blue": 40.0}, far_near_dets(), time_limit=50.0, order="largest",
                       time_triage=triage, time_margin_s=0.5)
            done, ex = first_executed(pol, 10.0)
            self.assertTrue(done, f"triage={triage} margin 0.5")
            self.assertIsNone(ex)

    def test_margin_applies_in_triage_candidate_choice(self):
        pol = make({"red": 39.8, "blue": 30.0}, far_near_dets(), time_limit=50.0, order="largest", time_triage=True,
                   time_margin_s=0.5)
        self.assertEqual(first_executed(pol, 10.0)[1], "blue")

    def test_defaults_are_off(self):
        pol = make({"red": 1.0, "blue": 1.0}, far_near_dets())
        p = pol.cfg["policy"]
        self.assertFalse(p["time_triage"])
        self.assertEqual(p["triage_order"], "duration")
        self.assertEqual(p["time_margin_s"], 0.0)
        self.assertIsNone(p["max_total_attempts"])


def run_rolling(cap, n_looks=6, logs=None):
    """A rolling object knocked 5 cm (> same_object_radius) per attempt: every look it is a 'new' object.
    Returns (policy, number of looks that ended done)."""
    pol = make({"red": 5.0}, [det("red", 0.30)], time_limit=1000.0, max_total_attempts=cap)
    if logs is not None:
        pol.log = logs.append
    for i in range(n_looks):
        pol.perception.dets = [det("red", 0.30 + 0.05 * i)]
        pol.state, pol.queue_a, pol.queue_h = "look", None, None
        if pol.act(obs(10.0 * i))["done"]:
            return pol, i
    return pol, None


class TestAttemptCap(unittest.TestCase):
    def test_cap_reached_ends_episode(self):
        pol, done_at = run_rolling(3)
        self.assertEqual(pol.n_started, 3)
        self.assertEqual(done_at, 3)                           # 4th look: cap hit before perception/planning
        self.assertEqual(pol.planner.calls, ["red"] * 3)
        self.assertIn("attempt cap reached", pol.events[-1][1])

    def test_cap_null_keeps_going(self):
        pol, done_at = run_rolling(None)
        self.assertIsNone(done_at)
        self.assertEqual(pol.n_started, 6)                     # old behaviour: max_attempts never trips
        self.assertFalse(any("cap" in m for _, m in pol.events))

    def test_episode_end_summary_logged_once(self):
        logs = []
        pol, _ = run_rolling(2, logs=logs)
        pol.act(obs(99.0))                                     # further done ticks do not repeat the summary
        lines = [m for m in logs if "episode end" in m]
        self.assertEqual(lines, ["[classical] episode end: attempts=2 failed=0 too_long=0 started=2"])


if __name__ == "__main__":
    unittest.main()
