"""perception.rim_z_from_depth (FLAG, default off): per-bin rim z from head depth reaches the planner's rim clearance,
drop / carry height and wall collision. Synthetic depth, no GPU. Run:
    python -m unittest aaltonomous.baselines.classical.test_rim_z -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import FLAGS, hand_postures, home_arm, load_config
from .mock_world import Renderer
from .perception import Detection, Perception
from .planner import Planner

CAL = Calibration.load()
ON = {"perception.rim_z_from_depth": True}


def _depth(raise_by, noise_mm=0.0, bins=None, seed=0):
    """uint16 mm depth: flat counter + each bin's 3 mm wall ring at (nominal rim + raise_by[bin])."""
    sc = Scene.load()
    r = Renderer(sc, CAL)
    d = np.nan_to_num(r._plane(sc.counter_z) * 1000.0, posinf=0.0)
    b = sc.cfg["bins"]
    for name, dz in (raise_by if isinstance(raise_by, dict) else {n: raise_by for n in sc.bin_names}).items():
        t = r._plane(sc.counter_z + float(b["wall_height"]) + dz)
        xy = sc.pelvis_to_mat(r.o + t[..., None] * r.rays)[..., :2]
        lo, hi = sc.bin_inner_box(name)
        c, half = np.asarray(b["centers"][name], float), np.asarray(b["outer_size"], float) / 2
        in_out = np.all(np.abs(xy - c) <= half, axis=-1)
        in_in = np.all((xy >= lo[:2]) & (xy <= hi[:2]), axis=-1)
        ring = in_out & ~in_in & np.isfinite(t)
        d = np.where(ring, t * 1000.0, d)
    if noise_mm:
        d = d + np.random.default_rng(seed).normal(0, noise_mm, d.shape) * (d > 0)
    return np.clip(d, 0, 65535).astype(np.uint16)


def _perc(on=True):
    sc, logs = Scene.load(), []
    p = Perception(load_config(overrides=ON if on else None), sc, CAL, log=logs.append)
    return p, sc, logs


def _looks(p, depth, n=3):
    rgb = np.zeros((480, 640, 3), np.uint8)
    for _ in range(n):
        p.detect(rgb, depth)


def _plan(sc, bin_name="left"):
    cfg = load_config()
    post, home = hand_postures(cfg), home_arm(cfg)
    pl = Planner(cfg, sc, post, home)
    L = sc.cfg["layout"]
    top = sc.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]), sc.object_height()])
    top[2] = sc.counter_z + sc.object_height()
    plan = pl.plan(Detection("red", top, 1000, 1000, "depth"), home, post["open"], bin_name=bin_name, relocate=False)
    return pl, plan


class TestRimZ(unittest.TestCase):
    def test_flag_default_off_and_registered(self):
        self.assertIn("perception.rim_z_from_depth", FLAGS)
        self.assertFalse(load_config()["perception"]["rim_z_from_depth"])

    def test_measured_within_5mm_at_plus_2_and_3cm(self):
        for dz in (0.02, 0.03):
            p, sc, logs = _perc()
            _looks(p, _depth(dz, noise_mm=1.5))
            nom = sc.counter_z + float(sc.cfg["bins"]["wall_height"])
            self.assertEqual(set(sc.rim_z_runtime), set(sc.bin_names), logs[-1])
            for n in sc.bin_names:
                self.assertAlmostEqual(sc.rim_z(n), nom + dz, delta=0.005, msg=f"{n} +{dz}")

    def test_plan_follows_raised_rim(self):
        sc0 = Scene.load()
        _, base = _plan(sc0)
        self.assertTrue(base.ok, base.reason)
        for dz in (0.02, 0.03):
            p, sc, _ = _perc()
            _looks(p, _depth(dz))
            pl, plan = _plan(sc)
            self.assertTrue(plan.ok, plan.reason)
            self.assertAlmostEqual(plan.waypoints["drop"][2] - base.waypoints["drop"][2], dz, delta=0.005)
            self.assertGreaterEqual(plan.waypoints["lift"][2], base.waypoints["lift"][2] - 1e-9)
            # descend / rim clearance: a hand point nominal-rim + 1 cm high over the bin is now below the raised rim
            self.assertAlmostEqual(sc.rim_z("left") - sc0.rim_z("left"), dz, delta=0.005)

    def test_clearance_and_wall_collision_use_measured_rim(self):
        p, sc, _ = _perc()
        _looks(p, _depth(0.03))
        pl0, _ = _plan(Scene.load())
        pl1 = Planner(load_config(), sc, hand_postures(load_config()), home_arm(load_config()))
        arm, hand = home_arm(load_config()), hand_postures(load_config())["open"]
        # same pose, same bins: clearance over the bin is exactly the rim shift smaller (inf if the hand is not over a bin)
        c0, c1 = pl0.bin_clearance("left", arm, hand), pl1.bin_clearance("left", arm, hand)
        if np.isfinite(c0):
            self.assertAlmostEqual(c0 - c1, sc.rim_z("left") - Scene.load().rim_z("left"), delta=0.006)
        self.assertEqual(pl1.hits_bin_wall("left", arm, hand), pl0.hits_bin_wall("left", arm, hand))

    def test_per_bin_independent(self):
        p, sc, _ = _perc()
        _looks(p, _depth({"left": 0.03, "center": 0.0, "right": 0.0} if "center" in Scene.load().bin_names
                         else {n: (0.03 if i == 0 else 0.0) for i, n in enumerate(Scene.load().bin_names)}))
        nom = sc.counter_z + float(sc.cfg["bins"]["wall_height"])
        first = sc.bin_names[0]
        self.assertAlmostEqual(sc.rim_z(first), nom + 0.03, delta=0.005)
        self.assertAlmostEqual(sc.rim_z(sc.bin_names[-1]), nom, delta=0.005)

    def test_clamped(self):
        p, sc, _ = _perc()
        _looks(p, _depth(0.06))   # beyond max_delta_m = 0.04 window: no wall points in window -> nominal
        nom = sc.counter_z + float(sc.cfg["bins"]["wall_height"])
        for n in sc.bin_names:
            self.assertLessEqual(sc.rim_z(n), nom + 0.04 + 1e-9)

    def test_bad_depth_is_nominal(self):
        for bad in (np.full((480, 640), np.nan), np.zeros((0, 0)), None, np.zeros((480, 640), np.uint16),
                    np.random.default_rng(1).integers(0, 65535, (480, 640)).astype(np.uint16),
                    np.full((480, 640), 600, np.uint16), np.zeros(5)):
            p, sc, logs = _perc()
            p.update_rim_z(bad)
            nom = Scene.load()
            for n in sc.bin_names:
                self.assertAlmostEqual(sc.rim_z(n), nom.rim_z(n), delta=0.0, msg=repr(np.shape(bad)))
            self.assertEqual(sc.rim_wall(), float(sc.cfg["bins"]["wall_height"]))

    def test_flag_off_inert_and_identical_plans(self):
        p, sc, logs = _perc(on=False)
        _looks(p, _depth(0.03))
        self.assertIsNone(sc.rim_z_runtime)
        self.assertEqual([l for l in logs if "[rim_z]" in l], [])
        _, a = _plan(sc)
        _, b = _plan(Scene.load())
        self.assertEqual(a.ok, b.ok)
        for k in b.waypoints:
            np.testing.assert_array_equal(a.waypoints[k], b.waypoints[k])
        np.testing.assert_array_equal(a.arm, b.arm)

    def test_reset_restores_nominal(self):
        p, sc, _ = _perc()
        _looks(p, _depth(0.03))
        self.assertTrue(sc.rim_z_runtime)
        p.reset_calibration()
        self.assertIsNone(sc.rim_z_runtime)


if __name__ == "__main__":
    unittest.main()
