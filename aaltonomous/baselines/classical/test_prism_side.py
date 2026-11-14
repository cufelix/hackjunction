"""perception.prism_side: a prism lying on its side (ridge up) is detected from depth and grasped across the ridge.
Run: python -m pytest aaltonomous/baselines/classical/test_prism_side.py -q"""
import unittest
from dataclasses import fields

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import home_arm, hand_postures, load_config
from .mock_world import Renderer, make_mock_env
from .perception import Perception
from .planner import Planner
from .policy import ClassicalPolicy

SC, CAL = Scene.load(), Calibration.load()
H = SC.object_height()
ON = {"perception.classify_shape": True, "perception.prism_side": True}
OFF = {"perception.classify_shape": True}


def _dets(ov, objs, seed=0):
    ob = [dict(color="red", shape="triangle", orient=None, yaw=0.0, size=SC.size, height=H, pos=None) | o
          for o in objs]
    for o, (x, y) in zip(ob, [(0.40, 0.15)] * len(ob)):
        o["pos"] = SC.mat_to_pelvis([*o.pop("xy", (x, y)), H / 2])
    rgb, depth = Renderer(SC, CAL).render(ob, np.random.default_rng(seed), 1.0, 0.02)
    return Perception(load_config(overrides=ov), SC, CAL).detect(rgb, depth)


def _mod_pi(a):
    return abs((a + np.pi / 2) % np.pi - np.pi / 2)


class TestPrismSide(unittest.TestCase):
    def test_upright_prism_is_still_triangle(self):
        for ov in (ON, OFF):
            (d,) = _dets(ov, [dict(orient="up", yaw=0.3)])
            self.assertEqual(d.shape, "triangle", ov)
            self.assertIsNone(d.ridge_yaw)

    def test_lying_prism_triangle_ridge_and_grasp_perpendicular(self):
        cfg = load_config(overrides=ON)
        pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
        for yaw_deg in (0, 45, 90):
            for xy in ((0.40, 0.15), (0.30, 0.12)):
                (d,) = _dets(ON, [dict(orient="side", yaw=np.radians(yaw_deg), xy=xy)])
                self.assertEqual(d.shape, "triangle", (yaw_deg, xy))
                self.assertLess(np.degrees(_mod_pi(d.ridge_yaw - np.radians(yaw_deg))), 10.0, (yaw_deg, xy, d.ridge_yaw))
                self.assertAlmostEqual(d.top[2] - SC.counter_z, SC.size, delta=0.006)   # ridge height ~ triangle height
                p = pl.plan(d, home_arm(cfg), hand_postures(cfg)["open"], bin_name="right", relocate=False)
                self.assertTrue(p.ok, p.reason)
                hd = pl.finger_heading(p)                    # fingers along the ridge -> closing across it
                self.assertLess(np.degrees(_mod_pi(hd - d.ridge_yaw)), 13.0, (yaw_deg, xy))
                self.assertLess(np.degrees(_mod_pi(hd + np.pi / 2 - (d.ridge_yaw + np.pi / 2))), 13.0)
                self.assertAlmostEqual(p.waypoints["grasp"][2] - SC.counter_z, 0.5 * SC.size, delta=0.006)

    def test_cube_and_cylinder_are_never_prisms(self):
        for shape in ("cube", "cylinder"):
            for yaw in (0.0, 0.4, 0.785, 1.2):
                (d,) = _dets(ON, [dict(shape=shape, yaw=yaw)])
                self.assertIsNone(d.ridge_yaw, (shape, yaw))
                self.assertEqual(d.shape, shape)

    def test_flag_off_identical_detections(self):
        objs = [[dict(orient="side", yaw=0.8)], [dict(orient="up", yaw=0.3)], [dict(shape="cube", yaw=0.3)]]
        for o in objs:
            a, b = _dets(OFF, o), _dets({**OFF, "perception.prism_side": False}, o)
            self.assertEqual(len(a), len(b))
            for x, y in zip(a, b):
                for f in fields(x):
                    np.testing.assert_array_equal(getattr(x, f.name), getattr(y, f.name), f.name)
                self.assertIsNone(x.ridge_yaw)
        (d,) = _dets(OFF, [dict(orient="side", yaw=0.8)])     # off: the lying prism is mislabelled as today
        self.assertNotEqual(d.shape, "triangle")

    def test_mock_episode_lying_prism_picked_and_placed(self):
        n_ok = 0
        for seed, yaw in ((0, 0.0), (1, 0.8), (2, 1.57)):
            cfg = load_config(overrides=ON)
            env = make_mock_env(seed=seed, cfg=cfg)
            pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
            env.reset()
            env.layout = [dict(color="red", shape="triangle", orient="side", x=0.40, y=0.15, yaw=yaw)]
            env.backend.reset_scene(env.layout, env.instruction)
            obs = env._obs()
            pol.reset({"instruction": obs["instruction"]})
            while True:
                obs, info = env.step(pol.act(obs))
                if info["episode_over"]:
                    break
            out = env.backend.outcomes(pol.mapping)
            n_ok += out == ["correct"]
            print(f"[prism_side e2e] seed={seed} yaw={yaw} outcome={out} end={info['end_reason']} t={info['t']:.0f}")
        self.assertEqual(n_ok, 3)


if __name__ == "__main__":
    unittest.main()
