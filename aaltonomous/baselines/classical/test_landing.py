"""policy.verify_landing: post-release outcome classification, bounce re-queue, in-bin exclusion, time budget. Run:
    python -m unittest aaltonomous.baselines.classical.test_landing -v
Synthetic frames from the mock renderer; the policy's _look / _after_look / _start are called directly."""
import time
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from . import landing
from .config import load_config
from .mock_world import Renderer, make_mock_env
from .perception import Detection
from .policy import ClassicalPolicy

SC, CAL = Scene.load(), Calibration.load()
ON = {"policy.verify_landing": True}


def _obj(color, pos, shape="cube"):
    return dict(color=color, shape=shape, yaw=0.0, size=SC.size, height=SC.object_height(), pos=np.asarray(pos, float))


def _counter(color, x, y):
    return _obj(color, SC.mat_to_pelvis([x, y, SC.object_height() / 2]))


def _in_bin(pol, color, bin_name, slot=0):
    xy = pol.planner.drop_slots(bin_name)[slot]
    lo, _ = SC.bin_inner_box(bin_name)
    return _obj(color, np.r_[xy, SC.counter_z + lo[2] + SC.object_height() / 2])


class LandingBase(unittest.TestCase):
    overrides = ON

    def setUp(self):
        self.renderer = Renderer(SC, CAL)
        self.pol = ClassicalPolicy(config=load_config(overrides=self.overrides), log=None)
        self.pol.arm_cmd = self.pol.home.copy()
        self.pol.hand_cmd = self.pol.post["open"].copy()
        self.pol.reset()
        self.pol.arm_cmd = self.pol.home.copy()
        self.pol.hand_cmd = self.pol.post["open"].copy()
        self.logs = []
        self.pol.log = self.logs.append

    def obs(self, objects, t=10.0):
        rgb, depth = self.renderer.render(objects)
        return {"head_rgb": rgb, "head_depth": depth, "t": t}

    def pending(self, color, bin_name, src, pre=None):
        src = np.asarray(SC.mat_to_pelvis([*src, 0.0])[:2])
        self.pol._landing = {"color": color, "bin": bin_name, "src_xy": src, "pre_xy": [src] if pre is None else pre}


class TestClassification(LandingBase):
    def test_correct_landing_is_no_action(self):
        counter = _counter("yellow", 0.50, 0.15)
        self.pol._look(self.obs([_counter("red", 0.40, 0.15), counter]))      # the look that planned the pick
        self.pending("red", "left", (0.40, 0.15))
        res = self.pol._look(self.obs([_in_bin(self.pol, "red", "left"), counter]))
        self.assertEqual(self.pol.landing_events, [("ok", "red", "left")])
        self.assertIsNone(self.pol._landing)
        self.assertEqual(self.pol.attempts, [])                              # nothing refunded or re-queued
        self.assertEqual(res[0], "plan")
        self.assertEqual(res[1].color, "yellow")                              # the next normal pick, not the red

    def test_bounce_to_the_counter_is_repicked_with_jitter(self):
        self.pol._look(self.obs([_counter("red", 0.40, 0.15)]))
        self.pol.attempts.append((np.asarray(SC.mat_to_pelvis([0.40, 0.15, 0.0])[:2]), 1))
        self.pending("red", "left", (0.40, 0.15))
        bounced = SC.mat_to_pelvis([0.55, 0.20, 0.0])
        res = self.pol._look(self.obs([_counter("red", 0.55, 0.20)]))
        self.assertEqual(self.pol.landing_events, [("bounce", "red", "left")])
        self.assertEqual(res[0], "plan")
        self.assertEqual(res[1].color, "red")
        self.assertEqual(self.pol.attempts, [])                               # the attempt was refunded
        grasp = res[1].waypoints["grasp"][:2]
        off = grasp - bounced[:2]
        self.assertAlmostEqual(float(np.linalg.norm(off)), 0.01, delta=0.004)  # jitter of landing_jitter_m
        self.assertLess(grasp[0], bounced[0])                                  # inward, away from the bin rims
        self.assertTrue(any("bounced" in m for m in self.logs))

    def test_shadowed_purple_in_the_bin_counts_via_depth(self):
        self.pol._look(self.obs([_counter("yellow", 0.50, 0.15)]))
        self.pending("blue", "right", (0.40, 0.15))
        self.renderer.colors["blue"] = np.array([14.0, 14.0, 20.0])           # shadow: nearly black, hue gone
        frame = self.obs([_in_bin(self.pol, "blue", "right"), _counter("yellow", 0.50, 0.15)])
        _, rejected = self.pol.perception.detect(frame["head_rgb"], frame["head_depth"], debug=True, include_bins=True)
        self.assertFalse(any(d.color == "blue" for d, _ in rejected if d is not None))   # colour alone is blind
        self.pol._look(frame)
        self.assertEqual(self.pol.landing_events, [("ok", "blue", "right")])

    def test_wrong_bin_is_logged_not_corrected(self):
        self.pol._look(self.obs([]))
        self.pending("red", "left", (0.40, 0.15))
        res = self.pol._look(self.obs([_in_bin(self.pol, "red", "center")]))
        self.assertEqual(self.pol.landing_events, [("wrong_bin", "red", "left")])
        self.assertEqual(res[0], "none")                                       # no corrective pick, no stop-on-uncorrectable
        self.assertTrue(any("wrong bin" in m for m in self.logs))

    def test_lost_object_gets_one_relook_then_is_forgotten(self):
        self.pol._look(self.obs([]))
        self.pending("red", "left", (0.40, 0.15))
        res = self.pol._look(self.obs([]))
        self.assertEqual(res[0], "relook")
        self.assertIsNotNone(self.pol._landing)
        res = self.pol._look(self.obs([]))
        self.assertEqual(res[0], "none")
        self.assertEqual(self.pol.landing_events, [("lost", "red", "left")])

    def test_grasp_that_never_lifted_is_not_a_bounce(self):
        self.pol._look(self.obs([_counter("red", 0.40, 0.15)]))
        self.pending("red", "left", (0.40, 0.15))
        self.pol._look(self.obs([_counter("red", 0.40, 0.15)]))
        self.assertEqual(self.pol.landing_events, [("grasp_failed", "red", "left")])
        self.assertEqual(self.pol.bounce_count, {})

    def test_start_arms_the_verification(self):
        res = self.pol._look(self.obs([_counter("red", 0.40, 0.15)]))
        self.pol._start(res[1], 10.0)
        self.assertEqual((self.pol._landing["color"], self.pol._landing["bin"]), ("red", "left"))
        self.assertEqual(len(self.pol._landing["pre_xy"]), 1)

    def test_occupancy_is_blind_to_the_bin_walls_and_empty_floor(self):
        occ = landing.bin_occupancy(self.pol.perception, self.obs([])["head_depth"])
        for name in ("left", "center", "right"):
            self.assertIsNotNone(occ[name])
            self.assertLess(occ[name], 0.1, name)
        full = landing.bin_occupancy(self.pol.perception, self.obs([_in_bin(self.pol, "red", "left")])["head_depth"])
        self.assertGreater(full["left"] - occ["left"], 2.0 * float(self.pol.cfg["policy"]["landing_min_delta"]))


class TestInBinExclusion(LandingBase):
    overrides = {**ON, "bin.geometry_runtime": True}

    def _det(self, xy):
        top = np.r_[xy, SC.counter_z + SC.object_height()]
        return Detection("red", top, 500, 400, "depth")

    def test_in_bin_object_never_becomes_a_target(self):
        inside = self.pol.planner.drop_slots("left")[0]
        counter = SC.mat_to_pelvis([0.40, 0.15, 0.0])[:2]
        self.pol.perception.detect = lambda *a, **k: [self._det(inside), self._det(counter)]
        res = self.pol._look(self.obs([]))
        self.assertEqual(res[0], "plan")
        np.testing.assert_allclose(res[1].waypoints["grasp"][:2], counter, atol=1e-6)
        self.assertEqual([d for d in self.pol.last_detections if np.allclose(d.top[:2], inside)], [])

    def test_follows_the_runtime_bin_pose(self):
        """The bin moved 6 cm: its NEW footprint is excluded, the nominal spot is a counter spot again."""
        nominal = self.pol.planner.drop_slots("left")[0].copy()
        self.pol.planner.bin_shift = {"left": np.array([-0.06, 0.0])}
        shifted = nominal + np.array([-0.06, 0.0])
        self.pol.perception.detect = lambda *a, **k: [self._det(nominal)]
        self.assertEqual(landing.inside_bin(self.pol.perception, shifted, 0.0), "left")
        self.assertEqual(landing.inside_bin(self.pol.perception, np.array([0.30, 0.10]), 0.0), None)

    def test_flag_off_keeps_the_stock_detections(self):
        pol = ClassicalPolicy(config=load_config(overrides={"bin.geometry_runtime": True}), log=None)
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        inside = pol.planner.drop_slots("left")[0]
        pol.perception.detect = lambda *a, **k: [self._det(inside)]
        pol._look(self.obs([]))
        self.assertEqual(len(pol.last_detections), 1)


class TestTimeBudget(LandingBase):
    def test_no_requeued_pick_starts_when_it_cannot_finish(self):
        self.pol._look(self.obs([_counter("red", 0.40, 0.15)], t=100.0))
        self.pending("red", "left", (0.40, 0.15))
        res = self.pol._look(self.obs([_counter("red", 0.55, 0.20)], t=118.0))
        self.assertEqual(res[0], "plan")
        self.pol._after_look(res, 118.0)
        self.assertEqual(self.pol.state, "done")                               # duration > the 2 s left: no pick starts
        self.assertEqual(self.pol.n_started, 0)

    def test_a_requeued_pick_with_time_left_starts(self):
        self.pol._look(self.obs([_counter("red", 0.40, 0.15)], t=20.0))
        self.pending("red", "left", (0.40, 0.15))
        res = self.pol._look(self.obs([_counter("red", 0.55, 0.20)], t=30.0))
        self.pol._after_look(res, 30.0)
        self.assertEqual(self.pol.state, "executing")
        self.assertEqual(self.pol.n_started, 1)

    def test_regrant_needs_a_full_cycle_of_time(self):
        xy = SC.mat_to_pelvis([0.40, 0.15, 0.0])[:2]
        for t, expect in ((80.0, "plan"), (110.0, "none")):
            self.setUp()
            self.pol.attempts = [(xy.copy(), 1), (xy.copy(), 1)]               # max_attempts (2) used up
            res = self.pol._look(self.obs([_counter("red", 0.40, 0.15)], t=t))
            self.assertEqual(res[0], expect, t)
        self.setUp()
        self.pol.attempts = [(xy.copy(), 1), (xy.copy(), 1)]
        self.assertEqual(self.pol._look(self.obs([_counter("red", 0.40, 0.15)], t=80.0))[0], "plan")
        self.assertEqual(self.pol.landing_regrants, 1)

    def test_regrants_are_capped(self):
        xy = SC.mat_to_pelvis([0.40, 0.15, 0.0])[:2]
        self.pol.landing_regrants = int(self.pol.cfg["policy"]["landing_regrants_max"])
        self.pol.attempts = [(xy.copy(), 1), (xy.copy(), 1)]
        self.assertEqual(self.pol._look(self.obs([_counter("red", 0.40, 0.15)], t=20.0))[0], "none")


class TestFlagsOff(unittest.TestCase):
    def test_default_is_off_and_untouched(self):
        cfg = load_config(overlay="none")
        self.assertFalse(cfg["policy"]["verify_landing"])
        pol = ClassicalPolicy(config=cfg, log=None)
        self.assertFalse(pol._landing_on())
        self.assertFalse(ClassicalPolicy(config=load_config(overrides={**ON, "policy.verify_placement": True}),
                                         log=None)._landing_on())   # #6 keeps its own machinery

    def _episode(self, overrides, seed=1):
        env = make_mock_env(seed=seed)
        pol = ClassicalPolicy(config=load_config(overrides=overrides), log=None, safety=env.filter.cfg)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        stream = []
        while True:
            a = pol.act(obs)
            stream.append(np.concatenate([np.asarray(a["arm_q_target"]).ravel(), np.asarray(a["hand_q_target"]).ravel()]))
            obs, info = env.step(a)
            if info["episode_over"]:
                break
        return np.concatenate(stream), env.backend.outcomes(pol.mapping), pol

    def test_flag_off_is_stock_and_perfect_landings_make_the_flag_a_no_op(self):
        off, out_off, _ = self._episode({})
        off2, _, _ = self._episode({"policy.verify_landing": False})
        np.testing.assert_array_equal(off, off2)                               # flags-off byte-identical
        on, out_on, pol = self._episode(ON)
        self.assertEqual(out_off, out_on)
        np.testing.assert_array_equal(off, on)                                 # every release landed: nothing to do
        self.assertTrue(all(k == "ok" for k, _, _ in pol.landing_events), pol.landing_events)
        self.assertGreaterEqual(len(pol.landing_events), 1)


class TestActLatency(unittest.TestCase):
    def test_act_never_blocks_on_the_landing_review(self):
        env = make_mock_env(seed=2)
        pol = ClassicalPolicy(config=load_config(overrides=ON), log=None, safety=env.filter.cfg, threaded=True)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        worst, n = 0.0, 0
        while n < 3000:
            t0 = time.perf_counter()
            a = pol.act(obs)
            worst = max(worst, time.perf_counter() - t0)
            if pol._job is not None and pol._job["result"] is None:   # the look thread works while act() keeps being called
                time.sleep(0.02)
            obs, info = env.step(a)
            n += 1
            if info["episode_over"] or pol.state == "done":
                break
        self.assertGreaterEqual(len(pol.landing_events), 1)                    # the review really ran in the look thread
        self.assertLess(worst, 0.05)


if __name__ == "__main__":
    unittest.main()
