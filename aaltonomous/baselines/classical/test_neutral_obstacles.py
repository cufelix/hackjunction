"""perception.neutral_obstacles (FLAG, default off): dark / grey / colourless blocks are never detections, become
neutral obstacles with a 3D footprint (depth) and the planner keeps clear of them. Synthetic RGB-D, no GPU. Run:
    python -m unittest aaltonomous.baselines.classical.test_neutral_obstacles -v
"""
import time
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import FLAGS, hand_postures, home_arm, load_config
from .mock_world import Renderer
from .perception import Detection, NeutralObstacle, Perception
from .planner import Planner

SC, CAL = Scene.load(), Calibration.load()
H = SC.object_height()
BLACK, NAVY, GREY = (14.0, 14.0, 14.0), (8.0, 10.0, 52.0), (120.0, 120.0, 120.0)
RED_AT, BLACK_AT = (0.30, 0.155), (0.40, 0.155)     # mat xy (m): red target and a black block 10 cm away (region_y is 0.12-0.19)


def _cfg(on, **extra):
    return load_config(overlay="none", overrides={"perception.neutral_obstacles": on, **extra})


def _scene(extra=(), red=RED_AT, black=BLACK_AT, black_color="black"):
    objs = [dict(color="red", shape="cube", yaw=0.0, size=SC.size, height=H, pos=SC.mat_to_pelvis([*red, H / 2]))]
    if black is not None:
        objs.append(dict(color=black_color, shape="cube", yaw=0.0, size=SC.size, height=H,
                         pos=SC.mat_to_pelvis([*black, H / 2])))
    for xy, col in extra:
        objs.append(dict(color=col, shape="cube", yaw=0.0, size=SC.size, height=H, pos=SC.mat_to_pelvis([*xy, H / 2])))
    r = Renderer(SC, CAL)
    r.colors.update({"black": np.array(BLACK), "navy": np.array(NAVY), "grey": np.array(GREY)})
    return r.render(objs)


def _perc(on, **extra):
    return Perception(_cfg(on, **extra), SC, CAL, log=lambda *_: None)


def _box_obstacle(centre_xy, half=0.025, z_top=None):
    z_top = SC.counter_z + H if z_top is None else z_top
    c = np.asarray(centre_xy, float)
    return NeutralObstacle((0.0, 0.0), 1000, (0, 0, 10, 10), "dark", c - half, c + half, z_top)


class _Src:
    def __init__(self, obs):
        self.neutral_obstacles = obs


class TestFlag(unittest.TestCase):
    def test_registered_off_by_default_and_documented(self):
        self.assertIn("perception.neutral_obstacles", FLAGS)
        self.assertIn("v_dark", FLAGS["perception.neutral_obstacles"][2])
        cfg = load_config(overlay="none")["perception"]
        self.assertFalse(cfg["neutral_obstacles"])
        self.assertEqual(cfg["neutral_params"]["v_dark"], 55)
        self.assertFalse(load_config(overlay="submission")["perception"]["neutral_obstacles"])


class TestPerceptionNeutral(unittest.TestCase):
    def test_black_block_is_obstacle_not_detection_target_kept(self):
        rgb, depth = _scene()
        p = _perc(True)
        dets = p.detect(rgb, depth)
        self.assertEqual([d.color for d in dets], ["red"])
        self.assertEqual(len(p.neutral_obstacles), 1)
        ob = p.neutral_obstacles[0]
        self.assertTrue(ob.has_footprint)
        centre = SC.mat_to_pelvis([*BLACK_AT, 0.0])[:2]
        self.assertTrue(np.all(ob.xy_lo - 0.01 <= centre) and np.all(centre <= ob.xy_hi + 0.01))
        self.assertLess(float(np.max(ob.xy_hi - ob.xy_lo)), 0.08)
        self.assertAlmostEqual(ob.z_top, SC.counter_z + H, delta=0.01)

    def test_flag_off_black_block_changes_nothing(self):
        rgb, depth = _scene()
        p = _perc(False)
        ref = _perc(False).detect(*_scene(black=None))
        dets = p.detect(rgb, depth)
        self.assertEqual(p.neutral_obstacles, [])
        self.assertEqual([(d.color, d.n_px, d.method) for d in dets], [(d.color, d.n_px, d.method) for d in ref])
        np.testing.assert_array_equal(dets[0].top, ref[0].top)

    def test_flag_on_does_not_change_target_detections(self):
        rgb, depth = _scene()
        off, on = _perc(False).detect(rgb, depth), _perc(True).detect(rgb, depth)
        self.assertEqual(len(off), len(on))
        np.testing.assert_array_equal(off[0].top, on[0].top)

    def test_dark_blue_block_that_passes_the_blue_mask_is_not_a_detection(self):
        rgb, depth = _scene(black_color="navy")
        p = _perc(True)
        self.assertEqual([d.color for d in p.detect(rgb, depth)], ["red"])
        self.assertGreaterEqual(len(p.neutral_obstacles), 1)   # top face + dark side faces may be separate blobs

    def test_grey_block_is_obstacle(self):
        rgb, depth = _scene(black_color="grey")
        p = _perc(True)
        self.assertEqual([d.color for d in p.detect(rgb, depth)], ["red"])
        self.assertEqual(len(p.neutral_obstacles), 1)
        self.assertEqual(p.neutral_obstacles[0].reason, "grey")

    def test_two_black_blocks_and_three_targets(self):
        rgb, depth = _scene(extra=[((0.55, 0.155), "black"), ((0.62, 0.155), "yellow"), ((0.47, 0.155), "blue")])
        p = _perc(True)
        self.assertEqual(sorted(d.color for d in p.detect(rgb, depth)), ["blue", "red", "yellow"])
        self.assertEqual(len(p.neutral_obstacles), 2)

    def test_no_depth_is_image_only(self):
        rgb, _ = _scene()
        p = _perc(True)
        dets = p.detect(rgb, None)
        self.assertEqual([d.color for d in dets], ["red"])
        self.assertEqual(len(p.neutral_obstacles), 1)
        self.assertFalse(p.neutral_obstacles[0].has_footprint)
        cfg = _cfg(True)
        pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
        pl.neutral_src = p
        self.assertEqual(pl.neutral_boxes(), [])      # image-only: never used for collision

    def test_empty_counter_has_no_obstacles(self):
        rgb, depth = Renderer(SC, CAL).render([])
        p = _perc(True)
        self.assertEqual(p.detect(rgb, depth), [])
        self.assertEqual(p.neutral_obstacles, [])


class TestPlannerNeutral(unittest.TestCase):
    def setUp(self):
        self.cfg = _cfg(True)
        self.post, self.home = hand_postures(self.cfg), home_arm(self.cfg)
        self.top = SC.mat_to_pelvis([*RED_AT, H])
        self.det = Detection("red", self.top, 1000, 1000, "depth")

    def _planner(self, obs):
        pl = Planner(self.cfg, SC, self.post, self.home)
        pl.neutral_src = _Src(obs)
        return pl

    def _plan(self, pl):
        return pl.plan(self.det, self.home, self.post["open"], relocate=False)

    def test_baseline_plan_is_ok_without_obstacles(self):
        self.assertTrue(self._plan(self._planner([])).ok)

    def _timed(self, obs):
        pl, lines = self._planner(obs), []
        pl.log = lines.append
        t0 = time.monotonic()
        p = self._plan(pl)
        return pl, p, time.monotonic() - t0, lines

    def _assert_fail_open(self, obs, free, t_free):
        budget = self.cfg["perception"]["neutral_params"]["plan_timeout_s"]
        pl, p, dt, lines = self._timed(obs)
        self.assertTrue(p.ok, p.reason)
        np.testing.assert_array_equal(p.arm, free.arm)             # the stock plan, unchanged
        self.assertEqual(len([l for l in lines if l.startswith("[neutral]")]), 1, lines)
        # one complete stock plan + the budget: the old search took ~140 s here (every side / hint variant)
        self.assertLess(dt, 1.5 * t_free + budget + 1.0, (dt, t_free))
        print(f"\n[neutral timing] blocked plan {dt:.1f}s, stock plan {t_free:.1f}s, budget {budget}s")
        return pl, p

    def test_obstacle_at_the_target_fails_open_to_the_stock_plan_within_budget(self):
        t0 = time.monotonic()
        free = self._plan(self._planner([]))
        t_free = time.monotonic() - t0
        obs = [_box_obstacle(self.top[:2] + [0.0, 0.05])]          # 5 cm beside the target: inside the finger spread
        pl, p = self._assert_fail_open(obs, free, t_free)
        s = [e for n, _, e in free.phases if n == "descend"][0] - 1     # grasp pose
        self.assertLess(pl.neutral_clearance(free.side, free.arm[s], free.hand[s]), 0.02)   # it does violate the obstacle

    def test_obstacle_across_the_carry_path_fails_open_within_budget(self):
        t0 = time.monotonic()
        free = self._plan(self._planner([]))
        t_free = time.monotonic() - t0
        mid = 0.5 * (free.waypoints["lift"][:2] + free.waypoints["carry"][:2])
        obs = [_box_obstacle(mid, half=0.04, z_top=SC.counter_z + 0.20)]   # tall, mid carry leg
        pl, p = self._assert_fail_open(obs, free, t_free)
        s, e = [(s, e) for n, s, e in free.phases if n == "carry"][0]
        gaps = [pl.neutral_clearance(free.side, free.arm[i], free.hand[i]) for i in range(s, e, 3)]
        self.assertLess(min(gaps), 0.02)                                     # the straight carry crosses it

    def test_unblocked_plan_is_unchanged_and_silent(self):
        free = self._plan(self._planner([]))
        pl, p, dt, lines = self._timed([_box_obstacle(self.top[:2] + [0.0, -0.18])])
        self.assertTrue(p.ok)
        np.testing.assert_array_equal(p.arm, free.arm)
        self.assertEqual(lines, [])

    def test_plan_for_target_keeps_clearance_from_a_near_obstacle(self):
        margin = self.cfg["perception"]["neutral_params"]["margin_m"]
        pl = self._planner([_box_obstacle(self.top[:2] + [0.0, -0.18])])     # beside the target, outside the fingers
        p = self._plan(pl)
        self.assertTrue(p.ok, p.reason)
        boxes = pl.neutral_boxes()
        gaps = [pl.neutral_clearance(p.side, p.arm[k], p.hand[k], boxes) for k in range(0, len(p.arm), 3)]
        self.assertGreaterEqual(min(gaps), margin - 1e-9)

    def test_far_obstacle_does_not_change_the_plan(self):
        free = self._plan(self._planner([]))
        far = self._plan(self._planner([_box_obstacle(self.top[:2] + [-0.30, 0.0])]))
        self.assertTrue(far.ok)
        np.testing.assert_array_equal(free.arm, far.arm)

    def test_flag_off_planner_ignores_obstacles(self):
        cfg = _cfg(False)
        pl = Planner(cfg, SC, self.post, self.home)   # neutral_src stays None (policy only sets it when on)
        self.assertEqual(pl.neutral_boxes(), [])
        self.assertTrue(pl.plan(self.det, self.home, self.post["open"], relocate=False).ok)

    def test_policy_wires_the_source_only_when_on(self):
        from .policy import ClassicalPolicy
        on = ClassicalPolicy(log=None, overrides={"perception.neutral_obstacles": True})
        off = ClassicalPolicy(log=None)
        self.assertIs(on.planner.neutral_src, on.perception)
        self.assertIsNone(off.planner.neutral_src)


if __name__ == "__main__":
    unittest.main()
