"""perception.counter_plane: per-episode counter z from the head depth (fail-safe). Run:
    python -m pytest aaltonomous/baselines/classical/test_counter_plane.py -q
    python -m aaltonomous.baselines.classical.test_counter_plane      # prints the offline accuracy table
"""
import copy
import time
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .calibration_sweep import jittered_camera
from .config import hand_postures, home_arm, load_config
from .mock_world import Renderer, make_mock_env
from .perception import Detection, Perception, scene_with_counter_z
from .planner import Planner
from .policy import ClassicalPolicy

SC, CAL = Scene.load(), Calibration.load()
CFG = load_config()
CFG_ON = load_config(overrides={"perception.counter_plane.enabled": True})


def _objects(scene):
    h = scene.object_height()
    return [dict(color=c, shape="cube", yaw=0.3 * i, size=scene.size, height=h,
                 pos=scene.mat_to_pelvis([x, y, h / 2]))
            for i, (c, x, y) in enumerate((("red", 0.33, 0.14), ("yellow", 0.45, 0.17), ("blue", 0.58, 0.13)))]


def _rotvec(axis, deg):
    return np.asarray(axis, float) * np.radians(deg)


def render(dz=0.0, rotvec=(0, 0, 0), seed=0, noise=1.5, holes=0.05):
    """Counter (and everything on it) dz above nominal; actual head camera rotated by rotvec (= a tilted counter as
    seen from the nominal camera). Returns rgb, depth, true scene, true perceived counter plane height function."""
    true = scene_with_counter_z(SC, SC.counter_z + dz)
    T0 = CAL._T("head")
    T = jittered_camera(T0, np.zeros(3), rotvec)
    rgb, depth = Renderer(true, CAL, T_pelvis_optical=T).render(
        _objects(true), np.random.default_rng(seed), depth_noise_mm=noise, hole_frac=holes)
    # nominal-camera deprojection maps a true point p to M p + k with M = R0 R^T, k = o - M o (o unchanged)
    M = T0[:3, :3] @ T[:3, :3].T
    n = M @ np.array([0, 0, 1.0])
    p0 = (M @ np.array([0, 0, true.counter_z])) + (T[:3, 3] - M @ T[:3, 3])

    def z_at(xy):
        return float(p0[2] - (n[0] * (xy[0] - p0[0]) + n[1] * (xy[1] - p0[1])) / n[2])
    return rgb, depth, true, z_at


def ref_xy(scene=SC):
    L = scene.cfg["layout"]
    return scene.mat_to_pelvis([np.mean(L["region_x"]), np.mean(L["region_y"]), 0.0])[:2]


def grasp_z(scene, det):
    cfg = load_config()
    post, home = hand_postures(cfg), home_arm(cfg)
    p = Planner(cfg, scene, post, home).plan(det, home, post["open"], relocate=False)
    return p.waypoints["grasp"][2] if p.ok else np.nan


class TestCounterPlane(unittest.TestCase):
    def test_flag_off_is_stock(self):
        rgb, depth, _, _ = render(dz=0.03)
        p, stock = Perception(CFG, SC, CAL), Perception(CFG, SC, CAL)
        cp = p.estimate_counter_plane(depth, rgb)
        self.assertFalse(cp.success)
        self.assertEqual(cp.reason, "disabled")
        self.assertIs(p.scene, SC)
        self.assertEqual(repr(p.detect(rgb, depth)), repr(stock.detect(rgb, depth)))

    def test_flag_off_policy_actions_identical(self):
        def rollout(cfg, steps=60):
            env = make_mock_env(seed=5)
            pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            out = []
            for _ in range(steps):
                a = pol.act(obs)
                out.append(repr(sorted((k, np.asarray(v).tolist()) for k, v in a.items())))
                obs, _ = env.step(a)
            self.assertIs(pol.planner.scene, pol.scene)
            return out
        bare = copy.deepcopy(CFG)
        del bare["perception"]["counter_plane"]
        self.assertEqual(rollout(CFG), rollout(bare))

    def test_shifted_counter_recovered_and_grasp_z_follows(self):
        det_z0 = None
        for dz in (-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05):
            rgb, depth, true, z_at = render(dz=dz, seed=int(1000 * dz) + 50)
            p = Perception(CFG_ON, SC, CAL)
            cp = p.estimate_counter_plane(depth, rgb)
            self.assertTrue(cp.success, cp)
            self.assertLess(abs(cp.z - true.counter_z), 0.003, f"dz {dz}: {cp}")
            self.assertLess(cp.elapsed_ms, 60.0)   # CI margin; ~5 ms typical
            dets = {d.color: d for d in p.detect(rgb, depth)}
            self.assertEqual(set(dets), {"red", "yellow", "blue"}, f"dz {dz}")
            gz = grasp_z(p.scene, dets["yellow"]) - p.scene.counter_z
            if det_z0 is None:
                det_z0 = gz
            self.assertLess(abs(gz - det_z0), 0.004, f"grasp z above counter changes at dz {dz}")

    def test_tilt_recovered(self):
        for axis in ((1, 0, 0), (0, 1, 0)):
            for deg in (-2.0, 2.0):
                rgb, depth, _, z_at = render(rotvec=_rotvec(axis, deg), seed=3)
                cp = Perception(CFG_ON, SC, CAL).estimate_counter_plane(depth, rgb)
                self.assertTrue(cp.success, cp)
                self.assertLess(abs(cp.z - z_at(ref_xy())), 0.003, cp)
                self.assertAlmostEqual(cp.tilt_deg, abs(deg), delta=0.3)

    def test_bad_fit_falls_back(self):
        p = Perception(CFG_ON, SC, CAL)
        # nothing at counter height (wall at 0.5 m)
        cp = p.estimate_counter_plane(np.full((480, 640), 500, np.uint16))
        self.assertFalse(cp.success)
        self.assertIs(p.scene, SC)
        # counter 10 cm off nominal: outside the max_dz gate
        rgb, depth, _, _ = render(dz=0.10)
        cp = p.estimate_counter_plane(depth, rgb)
        self.assertFalse(cp.success, cp)
        self.assertIs(p.scene, SC)
        # very noisy depth: residual / inlier gates
        rgb, depth, _, _ = render(noise=25.0)
        cp = p.estimate_counter_plane(depth, rgb)
        self.assertFalse(cp.success, cp)
        self.assertIs(p.scene, SC)
        # no depth frame: not a valid frame (policy retries next tick)
        self.assertFalse(p.estimate_counter_plane(None).valid_frame)

    def test_policy_applies_once_and_resets(self):
        env = make_mock_env(seed=3)
        logs = []
        pol = ClassicalPolicy(config=CFG_ON, log=logs.append, safety=env.filter.cfg)
        obs = env.reset()
        info = {"instruction": obs["instruction"]}
        pol.reset(info)
        pol.act(obs)
        self.assertTrue(pol._counter_done)
        self.assertIs(pol.planner.scene, pol.perception.scene)
        self.assertTrue(any("counter plane accepted" in m for m in logs), logs)
        pol.reset(info)
        self.assertIs(pol.planner.scene, pol.scene)
        self.assertIs(pol.perception.scene, pol.scene)


def table():
    print("shift_cm tilt_deg | est_dz_mm err_mm tilt_est | grasp_z-true_counter mm | ms")
    for dz in (-0.05, -0.03, -0.01, 0.0, 0.01, 0.03, 0.05):
        for rv, tag in (((0, 0, 0), 0.0), (_rotvec((0, 1, 0), 2.0), 2.0), (_rotvec((1, 0, 0), -2.0), -2.0)):
            rgb, depth, true, z_at = render(dz=dz, rotvec=rv, seed=7)
            p = Perception(CFG_ON, SC, CAL)
            t = time.perf_counter()
            cp = p.estimate_counter_plane(depth, rgb)
            ms = (time.perf_counter() - t) * 1000
            dets = {d.color: d for d in p.detect(rgb, depth)}
            gz = grasp_z(p.scene, dets["yellow"]) - z_at(dets["yellow"].top[:2]) if "yellow" in dets else np.nan
            gs = grasp_z(SC, dets["yellow"]) - z_at(dets["yellow"].top[:2]) if "yellow" in dets else np.nan
            print(f"{100 * dz:+5.0f} {tag:+5.1f} | {cp.dz * 1000:+6.1f} {(cp.z - z_at(ref_xy())) * 1000:+5.1f} "
                  f"{cp.tilt_deg:4.2f} {'ok ' if cp.success else 'REJ'} | on {gz * 1000:5.1f} stock {gs * 1000:5.1f} | "
                  f"{ms:4.1f}")


if __name__ == "__main__":
    table()
