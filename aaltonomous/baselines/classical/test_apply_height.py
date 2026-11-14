"""calib.plane.apply_height: one runtime counter_z (scene.counter_z) from head depth reaches every consumer."""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .mock_world import Renderer
from .perception import Detection, Perception
from .planner import Planner

CAL = Calibration.load()
NOM = Scene.load().counter_z
ON = {"calib.plane.apply_height": True}


def _depth(z, tilt_deg=0.0, patch=None):
    """uint16 mm depth of a plane z = z + tan(tilt) * x (pelvis), optionally with a 4 cm square at height z+patch."""
    sc = Scene.load()
    r = Renderer(sc, CAL)
    plane = lambda zz: np.nan_to_num(r._plane(zz) * 1000.0, posinf=0.0)
    zmap = z + np.tan(np.radians(tilt_deg)) * (r.o[0] + r.rays[..., 0] * 0.8)   # ~x of the hit point (tilt test only)
    d = plane(z) if tilt_deg == 0 else plane(float(zmap.mean()))
    if tilt_deg:   # true tilted plane: solve per ray  z0 + k*x = o_z + t*rz,  x = o_x + t*rx
        k = np.tan(np.radians(tilt_deg))
        t = (z + k * r.o[0] - r.o[2]) / (r.rays[..., 2] - k * r.rays[..., 0])
        d = np.where(t > 0, t * 1000.0, 0.0)
    if patch:
        u, v = np.rint(r._project(np.array([0.30, 0.0, z + patch]))[:2]).astype(int)
        d[v - 12:v + 12, u - 12:u + 12] = plane(z + patch)[v - 12:v + 12, u - 12:u + 12]
    return d.astype(np.uint16)


def _perc(on=True, **kw):
    sc, logs = Scene.load(), []
    p = Perception(load_config(overrides=ON if on else None), sc, CAL, log=logs.append)
    return p, sc, logs


def _looks(p, depth, n=3):
    rgb = np.zeros((480, 640, 3), np.uint8)
    for _ in range(n):
        p.detect(rgb, depth)


class TestApplyHeight(unittest.TestCase):
    def test_flag_default_off_and_inert(self):
        self.assertFalse(load_config()["calib"]["plane"]["apply_height"])
        p, sc, logs = _perc(on=False)
        _looks(p, _depth(NOM + 0.10))
        self.assertEqual((sc.counter_z, logs), (NOM, []))

    def test_estimate_plus_10cm(self):
        p, sc, logs = _perc()
        _looks(p, _depth(NOM + 0.10))
        self.assertAlmostEqual(sc.counter_z, NOM + 0.10, delta=0.005)
        self.assertEqual(len(logs), 3)
        self.assertRegex(logs[-1], r"\[apply_height\] est=.* applied=.* n=3 tilt_deg=.* fallback=none")
        p.reset_calibration()
        self.assertEqual(sc.counter_z, NOM)

    def test_clamped_to_20cm(self):
        p, sc, logs = _perc()
        _looks(p, _depth(NOM + 0.35))
        self.assertAlmostEqual(sc.counter_z, NOM + 0.20, delta=1e-6)
        self.assertIn("fallback=clamped", logs[-1])

    def test_bad_depth_is_nominal(self):
        for bad in (np.full((480, 640), np.nan), np.zeros((0, 0)), None, np.zeros((480, 640), np.uint16)):
            p, sc, logs = _perc()
            p.update_counter_height(bad)
            self.assertEqual(sc.counter_z, NOM)
            self.assertRegex(logs[-1], r"n=0 .*fallback=\S+")
            self.assertNotIn("fallback=none", logs[-1])

    def test_tilted_plane_is_nominal_and_logs_tilt(self):
        p, sc, logs = _perc()
        _looks(p, _depth(NOM + 0.10, tilt_deg=12.0))
        self.assertEqual(sc.counter_z, NOM)
        self.assertRegex(logs[-1], r"tilt_deg=1[12]\.\d+ fallback=tilt")

    def test_every_consumer_shifts(self):
        dz = 0.10
        # perception height gate: an object top at table+0.045 is kept only with the runtime height
        kept = {}
        for on in (False, True):
            p, sc, _ = _perc(on)
            d = _depth(NOM + dz, patch=p.height)
            p.update_counter_height(d), p.update_counter_height(d), p.update_counter_height(d)
            u, v = np.rint(p._project(np.array([0.30, 0.0, NOM + dz + p.height]))[:2]).astype(int)
            kept[on] = int(p._keep_mask(d)[v - 8:v + 8, u - 8:u + 8].astype(bool).sum())
        self.assertEqual(kept[False], 0)
        self.assertGreater(kept[True], 100)
        # ray ∩ plane: z of the plane detection and its xy against the analytic ray hit
        tops = {}
        for on in (False, True):
            p, sc, _ = _perc(on)
            _looks(p, _depth(NOM + dz))
            tops[on] = p._from_plane("red", (320.0, 300.0), 100, 0).top
        self.assertAlmostEqual(tops[True][2] - tops[False][2], dz, delta=0.005)
        T, K = p._T(), p.cal.cams[p.cam].K
        ray = T[:3, :3] @ np.array([(320 - K[0, 2]) / K[0, 0], (300 - K[1, 2]) / K[1, 1], 1.0])
        hit = T[:3, 3] + ray * ((tops[True][2] - 0.25 * p.height - T[2, 3]) / ray[2])
        np.testing.assert_allclose(tops[True][:2], hit[:2] + p.xy_offset, atol=1e-9)
        # planner: grasp, carry, drop shift with the same scene
        cfg = load_config()
        post, home = hand_postures(cfg), home_arm(cfg)
        wp = {}
        for z in (NOM, NOM + dz):
            sc = Scene.load()
            sc.counter_z_runtime = None if z == NOM else z
            pl = Planner(cfg, sc, post, home)
            L = sc.cfg["layout"]
            top = sc.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]), sc.object_height()])
            top[2] = z + sc.object_height()
            plan = pl.plan(Detection("red", top, 1000, 1000, "depth"), home, post["open"], bin_name="left", relocate=False)
            self.assertTrue(plan.ok, plan.reason)
            wp[z] = plan.waypoints
            rim = pl.scene.counter_z + float(sc.cfg["bins"]["wall_height"])
            self.assertEqual(rim, z + float(sc.cfg["bins"]["wall_height"]))
        for k in ("grasp", "lift", "drop"):
            self.assertAlmostEqual(wp[NOM + dz][k][2] - wp[NOM][k][2], dz, delta=0.005, msg=k)


if __name__ == "__main__":
    unittest.main()
