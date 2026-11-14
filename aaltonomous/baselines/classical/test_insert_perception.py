"""insert_perception: lid + hole pose from synthetic head depth (flat counter, plate on the bin rim, holes to the bin
floor, hand blobs, noise). Run: python -m unittest aaltonomous.baselines.classical.test_insert_perception -v"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from . import insert_perception as ip
from .config import load_config
from .mock_world import Renderer
from .perception import Perception

CAL = Calibration.load()
SC = Scene.load()
MODEL = ip.LidModel.from_scene(SC)
PERC = Perception(load_config(), SC, CAL, log=lambda *a: None)


def render(dx=0.0, dy=0.0, yaw=0.0, noise_mm=1.0, blobs=(), seed=0, cam_dz=0.0):
    """uint16 mm depth for a lid at the nominal pose + (dx, dy, yaw)."""
    r = Renderer(SC, CAL)
    c0, _, z_top = ip._prior(SC, None, MODEL)
    t_c = r._plane(SC.counter_z)
    t_p = r._plane(z_top)
    xy = r.o[:2] + t_p[..., None] * r.rays[..., :2]
    q = (xy - (c0 + [dx, dy])) @ ip._rot(yaw)          # = R(-yaw) (xy - c)
    solid = (np.abs(q) <= MODEL.half).all(-1) & np.isfinite(t_p)
    for _, poly in MODEL.holes.values():
        solid &= ~ip._in_convex(q[..., 0], q[..., 1], poly)
    t_floor = r._plane(SC.counter_z + 0.003)
    d = np.where(solid, t_p, np.where(np.isfinite(t_p) & (np.abs(q) <= MODEL.half).all(-1), t_floor, t_c))
    for (cx, cy, rad) in blobs:                       # hands: nearer than the plate, in pelvis xy
        near = np.hypot(xy[..., 0] - cx, xy[..., 1] - cy) < rad
        d = np.where(near, t_p - 0.06, d)
    d = d * 1000.0 + cam_dz * 0 + np.random.default_rng(seed).normal(0, noise_mm, d.shape)
    return np.clip(np.nan_to_num(d, posinf=0), 0, 65535).astype(np.uint16)


def fit(depth, **kw):
    return ip.estimate_lid(np.zeros((480, 640, 3), np.uint8), depth, PERC, SC, **kw)


class TestLidFit(unittest.TestCase):
    def test_nominal_pose_recovered(self):
        e = fit(render())
        self.assertIsNotNone(e)
        nom = ip.nominal_lid(SC)
        self.assertLess(np.abs(e.centre_xy - nom.centre_xy).max(), 0.002)
        self.assertLess(abs(e.z_top - nom.z_top), 0.001)

    def test_perturbed_poses(self):
        rng = np.random.default_rng(1)
        nom = ip.nominal_lid(SC)
        errs = []
        for i in range(10):
            dx, dy = rng.uniform(-0.02, 0.02, 2)
            yaw = np.radians(rng.uniform(-6, 6))
            e = fit(render(dx, dy, yaw, seed=i))
            self.assertIsNotNone(e, (dx, dy, yaw))
            for s, h in e.holes.items():
                tru = ip._make(MODEL, nom.centre_xy + [dx, dy], yaw, nom.z_top, 1, (0, 0, 0), {}).holes[s]
                errs.append((np.linalg.norm(h.xy - tru.xy) * 1000, abs(np.degrees(h.yaw - tru.yaw))))
        errs = np.array(errs)
        print(f"\nhole err over {len(errs)}: xy max {errs[:,0].max():.2f} mm mean {errs[:,0].mean():.2f}; yaw max {errs[:,1].max():.2f} deg")
        self.assertLess(errs[:, 0].max(), 4.0)
        self.assertLess(errs[:, 1].max(), 2.0)

    def test_hand_occlusion(self):
        e = fit(render(0.01, -0.01, 0.05, blobs=[(0.45, 0.18, 0.04), (0.56, 0.33, 0.035)]))
        self.assertIsNotNone(e)
        self.assertLess(np.abs(e.centre_xy - (ip.nominal_lid(SC).centre_xy + [0.01, -0.01])).max(), 0.004)

    def test_no_lid_returns_none(self):
        r = Renderer(SC, CAL)
        d = np.nan_to_num(r._plane(SC.counter_z) * 1000.0, posinf=0).astype(np.uint16)
        self.assertIsNone(fit(d))
        self.assertIsNone(ip.estimate_lid(None, None, PERC, SC))

    def test_targets_and_nominal(self):
        e = fit(render(0.005, 0.0, 0.0))
        t = ip.to_targets(e, SC)
        self.assertEqual(set(t), {"cube", "cylinder", "triangle"})
        base = ip.hole_targets(SC) if hasattr(ip, "hole_targets") else __import__("aaltonomous.baselines.classical.insert_shapes", fromlist=["x"]).hole_targets(SC)
        for s in t:
            self.assertLess(np.linalg.norm(t[s]["xy"] - base[s]["xy"] - [0.005, 0.0]), 0.003)
            self.assertAlmostEqual(t[s]["yaw"], base[s]["yaw"], delta=np.radians(2))
        self.assertEqual(ip.to_targets(ip.nominal_lid(SC, conf=0.5), SC, min_conf=0.7), {})


if __name__ == "__main__":
    unittest.main()
