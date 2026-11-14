"""Six-object finals robustness: bin capacity, overflow, runtime geometry, and act() wall-time safety.

Run with the project environment (uses the canonical G1 + Dex3 URDF through ArmKinematics):
    python -m unittest aaltonomous.baselines.classical.test_bin_capacity -v
"""
from __future__ import annotations

import time
import unittest
from unittest import mock

import numpy as np

from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .mock_world import make_mock_env
from .perception import Detection
from .planner import Plan, Planner
from .policy import ClassicalPolicy


SC = Scene.load()
FINALS = ("finals_r19c", "finals_r20")
SHIFT_X = 0.07


def _detection(color="red", mat_xy=(0.41, 0.15), shape="cube"):
    top = SC.mat_to_pelvis([mat_xy[0], mat_xy[1], SC.object_height()])
    return Detection(color, top, 1200, 1000, "depth", shape=shape, shape_conf=1.0)


def _planner(overlay):
    cfg = load_config(overlay=overlay)
    return cfg, Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))


class TestSixObjectSlots(unittest.TestCase):
    def test_flag_off_keeps_the_legacy_slot_bytes(self):
        cfg0, pl0 = _planner("none")
        cfg1 = load_config(overlay="none", overrides={"grasp.drop_slot_grid": False})
        pl1 = Planner(cfg1, SC, hand_postures(cfg1), home_arm(cfg1))
        self.assertFalse(cfg0["grasp"]["drop_slot_grid"])
        for name in SC.bin_names:
            self.assertEqual([p.tobytes() for p in pl0.drop_slots(name)],
                             [p.tobytes() for p in pl1.drop_slots(name)])

    def test_finals_have_six_noncolliding_slots_inside_every_live_bin(self):
        for overlay in FINALS:
            cfg, pl = _planner(overlay)
            self.assertTrue(cfg["grasp"]["drop_slot_grid"], overlay)
            for name in SC.bin_names:
                slots = pl.drop_slots(name)
                self.assertEqual(len(slots), 6, (overlay, name, slots))
                for i, xy in enumerate(slots):
                    self.assertTrue(pl._in_bin_footprint(name, xy), (overlay, name, i, xy))
                    self.assertFalse(any(pl._in_bin_footprint(other, xy)
                                         for other in SC.bin_names if other != name), (overlay, name, i, xy))
                    for old in slots[:i]:
                        self.assertGreaterEqual(np.linalg.norm(xy - old), SC.size - 1e-9,
                                                (overlay, name, i, xy, old))

    def test_six_same_colour_choose_distinct_targets_then_overflow_stays_on_counter(self):
        """Exercise policy slot accounting separately from IK; the next test dry-runs all six through real IK."""
        cfg = load_config(overlay="finals_r19c")
        pol = ClassicalPolicy(config=cfg, log=None)
        det = _detection("red")
        target = pol.mapping["red"]

        def fake_plan(d, kw, plan_fn=None):
            slot = kw["drop_slot"]
            drop = pol.planner.drop_slots(kw["bin_name"])[slot]
            return Plan(True, side="right", color=d.color, target_bin=kw["bin_name"], drop_slot=slot,
                        waypoints={"grasp": d.top.copy(), "drop": np.r_[drop, SC.counter_z + 0.12]},
                        arm=np.empty((0, 14)), hand=np.empty((0, 14)), safety_clips=0)

        pol._plan_pick = fake_plan
        placed = []
        used = []
        for _ in range(6):
            result = pol._choose([det], {"t": 0.0}, verify=False)
            self.assertEqual(result[0], "plan", result)
            plan = result[1]
            xy = plan.waypoints["drop"][:2]
            self.assertNotIn(plan.drop_slot, used)
            self.assertTrue(pol.planner._in_bin_footprint(target, xy))
            self.assertFalse(any(pol.planner._in_bin_footprint(other, xy)
                                 for other in SC.bin_names if other != target))
            self.assertTrue(all(np.linalg.norm(xy - old) >= SC.size - 1e-9 for old in placed))
            used.append(plan.drop_slot)
            placed.append(xy.copy())
            pol.slot_used[target].add(plan.drop_slot)

        self.assertEqual(cfg["grasp"]["drop_slot_overflow"], "counter")
        overflow = pol._choose([det], {"t": 0.0}, verify=False)
        self.assertEqual(overflow[0], "none")       # no release command: the seventh object remains on the counter
        self.assertEqual(len(used), 6)

    def test_all_six_same_colour_slots_pass_real_ik_and_safety_dry_run(self):
        cfg, pl = _planner("finals_r19c")
        det = _detection("red")
        target = cfg["policy"]["fixed_mapping"]["red"]
        for slot, xy in enumerate(pl.drop_slots(target)):
            plan = pl.plan(det, pl.home, pl.post["open"], bin_name=target, drop_slot=slot, relocate=False)
            self.assertTrue(plan.ok, f"{target} slot {slot} {xy}: {plan.reason}")
            self.assertEqual(plan.safety_clips, 0, (slot, plan.reason))
            np.testing.assert_allclose(plan.waypoints["drop"][:2], xy, atol=1e-9)

    def test_plus_7cm_runtime_shift_moves_all_drop_targets_inside_shifted_bins(self):
        _, pl = _planner("finals_r20")
        nominal = {name: [p.copy() for p in pl.drop_slots(name)] for name in SC.bin_names}
        pl.bin_shift = {name: np.array([SHIFT_X, 0.0]) for name in SC.bin_names}
        for name in SC.bin_names:
            shifted = pl.drop_slots(name)
            self.assertEqual(len(shifted), 6)
            for xy, old in zip(shifted, nominal[name]):
                np.testing.assert_allclose(xy - old, [SHIFT_X, 0.0], atol=1e-12)
                self.assertTrue(pl._in_bin_footprint(name, xy), (name, xy, pl.bin_geometry(name)))
                self.assertFalse(any(pl._in_bin_footprint(other, xy)
                                     for other in SC.bin_names if other != name), (name, xy))


class TestFinalsSixMixed(unittest.TestCase):
    @staticmethod
    def _layout():
        # Two staggered rows, all pairwise >= 8 cm in the published spawn region.
        points = ((0.29, 0.13), (0.41, 0.13), (0.53, 0.13),
                  (0.35, 0.19), (0.47, 0.19), (0.59, 0.19))
        colors = ("red", "yellow", "blue", "yellow", "blue", "red")
        shapes = ("cube", "cylinder", "triangle", "triangle", "cube", "cylinder")
        return [dict(color=c, shape=s, x=x, y=y, yaw=0.0)
                for (x, y), c, s in zip(points, colors, shapes)]

    def test_finals_mixed_six_mock_has_bounded_act_no_exception_or_safety_stop(self):
        for overlay in FINALS:
            cfg = load_config(overlay=overlay)
            attempts_per_object = int(cfg["policy"].get("max_attempts_vary") or
                                      cfg["policy"]["max_attempts"])
            self.assertGreaterEqual(int(cfg["policy"]["max_total_attempts"]),
                                    6 * attempts_per_object, overlay)
            env = make_mock_env(seed=41, cfg=cfg, realtime=False)
            env.time_limit = 150.0
            pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg, time_limit=env.time_limit)
            try:
                obs = env.reset()
                env.layout = self._layout()
                env.backend.reset_scene(env.layout, obs["instruction"])
                obs = env._obs()
                pol.reset({"instruction": obs["instruction"], "time_limit": env.time_limit})
                max_act = 0.0
                modes = []
                info = None
                for _ in range(2500):
                    started = time.perf_counter()
                    action = pol.act(obs)
                    elapsed = time.perf_counter() - started
                    max_act = max(max_act, elapsed)
                    self.assertLess(elapsed, 0.050, f"{overlay} act() took {1000 * elapsed:.1f} ms")
                    obs, info = env.step(action)
                    modes.append(info["safety_mode"])
                    if info["episode_over"]:
                        break
                    time.sleep(0.001)     # real wall-time yield for the safety-shell planning workers
                self.assertIsNotNone(info, overlay)
                self.assertTrue(info["episode_over"], (overlay, env.t, max_act, pol.events[-5:]))
                self.assertNotIn("safe_stop", modes, overlay)
                self.assertNotIn("aborted", modes, overlay)
                self.assertEqual(pol._shell_stats["exception"], 0, (overlay, pol._shell_stats))
                self.assertFalse(any("look/plan error" in msg for _, msg in pol.events),
                                 (overlay, pol.events[-5:]))
                outcomes = env.backend.outcomes(pol.mapping)
                self.assertNotIn("wrong", outcomes, (overlay, outcomes))
                self.assertNotIn("off", outcomes, (overlay, outcomes))
            finally:
                env.close()

    def test_r20_bin_vote_runs_off_the_act_thread_in_real_wall_time(self):
        cfg = load_config(overlay="finals_r20")
        if not cfg["bin"]["vote"]["enabled"]:
            self.skipTest("bin.vote OFF in finals_r20 until the C3 rim-vote mirror fix lands")
        pol = ClassicalPolicy(config=cfg, log=None)
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        pol.state = "look"
        pol._pitch_calibration_done = True
        pol.pitch_calibration = object()
        n = int(cfg["bin"]["vote"]["n_frames"])
        frame = {"head_rgb": np.zeros((2, 2, 3), np.uint8), "head_depth": np.zeros((2, 2), np.uint16)}
        pol._bv_frames = [frame.copy() for _ in range(n - 1)]
        obs = {**frame, "arm_q": pol.home.copy(), "hand_q": pol.post["open"].copy(), "t": 1.0}

        def slow_vote(*args, **kwargs):
            time.sleep(0.10)
            return {}, {}, {}

        with mock.patch("aaltonomous.baselines.classical.policy.vote_bins", side_effect=slow_vote):
            deadline = time.perf_counter() + 1.0
            while not pol._bv_done and time.perf_counter() < deadline:
                started = time.perf_counter()
                pol.act(obs)
                self.assertLess(time.perf_counter() - started, 0.050)
                time.sleep(0.01)
        self.assertTrue(pol._bv_done)
        self.assertEqual(pol._shell_stats["exception"], 0)


if __name__ == "__main__":
    unittest.main()
