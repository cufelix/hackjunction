"""calib.wide_counter_window + planner.bin_wall_extra_m: counter height at nominal -5 .. +20 cm, synthetic depth."""
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
WIDE = {"calib.wide_counter_window": True}


def _world(dz, objects=(("red", 0.30, 0.0),), tilt_deg=0.0):
    """(rgb, depth, truth_tops): a counter at NOM + dz (bins ride on it), objects resting on it."""
    sc = Scene.load()
    sc.counter_z_runtime, sc.mat_z_follows_counter = NOM + dz, True      # renderer scene only
    r = Renderer(sc, CAL)
    h = sc.object_height()
    objs = [{"color": c, "shape": "cube", "yaw": 0.0, "size": sc.size, "height": h,
             "pos": np.array([x, y, NOM + dz + h / 2])} for c, x, y in objects]
    rgb, depth = r.render(objs)
    if tilt_deg:   # true tilted plane z = NOM + dz + tan(t) * x, no objects / bins
        k = np.tan(np.radians(tilt_deg))
        t = (NOM + dz + k * r.o[0] - r.o[2]) / (r.rays[..., 2] - k * r.rays[..., 0])
        depth = np.where(t > 0, t * 1000.0, 0.0).astype(np.uint16)
    return rgb, depth, [np.array([x, y, NOM + dz + h]) for _, x, y in objects]


def _perc(overrides=None):
    sc, logs = Scene.load(), []
    p = Perception(load_config(overrides=overrides), sc, CAL, log=logs.append)
    return p, sc, logs


def _top(p, rgb, depth, n=3):
    for _ in range(n):
        dets = p.detect(rgb, depth)
    return dets


class TestWideCounterWindow(unittest.TestCase):
    def test_flags_default_off(self):
        cfg = load_config()
        self.assertFalse(cfg["calib"]["wide_counter_window"])
        self.assertEqual(cfg["planner"]["bin_wall_extra_m"], 0.0)
        self.assertTrue(load_config(overlay="finals_r22_counter")["calib"]["wide_counter_window"])
        self.assertEqual(load_config(overlay="finals_r22_counter")["planner"]["bin_wall_extra_m"], 0.01)

    def test_height_and_object_position_within_1cm(self):
        for dz in (-0.05, 0.05, 0.15, 0.20):
            p, sc, logs = _perc(WIDE)
            rgb, depth, truth = _world(dz)
            dets = _top(p, rgb, depth)
            self.assertAlmostEqual(sc.counter_z, NOM + dz, delta=0.01, msg=f"dz={dz} {logs[-1:]}")
            self.assertEqual(len(dets), 1, (dz, dets))
            np.testing.assert_allclose(dets[0].top, truth[0], atol=0.01, err_msg=f"dz={dz}")

    def test_flag_off_plus_15cm_fails_as_documented(self):
        """Current behaviour (no runtime height): the 15 cm high counter is not seen. Documented, expected."""
        p, sc, logs = _perc()
        rgb, depth, truth = _world(0.15)
        dets = _top(p, rgb, depth)
        self.assertEqual(sc.counter_z, NOM)                       # nominal kept: wrong by 15 cm
        self.assertFalse(any(np.abs(d.top[2] - truth[0][2]) <= 0.01 and np.allclose(d.top[:2], truth[0][:2], atol=0.01)
                             for d in dets), dets)                # no correct detection

    def test_flag_off_is_inert(self):
        p, sc, logs = _perc()
        rgb, depth, _ = _world(0.10)
        _top(p, rgb, depth)
        self.assertEqual((sc.counter_z, sc.mat_z_follows_counter, logs), (NOM, False, []))

    def test_tilted_plane_beyond_gate_is_rejected_and_falls_back(self):
        p, sc, logs = _perc(WIDE)
        rgb, depth, _ = _world(0.10, objects=(), tilt_deg=12.0)
        _top(p, rgb, depth)
        self.assertEqual(sc.counter_z, NOM)
        self.assertNotIn("fallback=none", logs[-1])

    def test_estimate_outside_the_window_is_rejected(self):
        p, sc, logs = _perc(WIDE)
        rgb, depth, _ = _world(0.30, objects=())
        _top(p, rgb, depth)
        self.assertEqual(sc.counter_z, NOM)

    def test_bad_depth_falls_back(self):
        for bad in (None, np.zeros((480, 640), np.uint16), np.full((480, 640), np.nan)):
            p, sc, logs = _perc(WIDE)
            p.update_counter_height(bad)
            self.assertEqual(sc.counter_z, NOM)

    def test_mat_z_follows_runtime_counter(self):
        sc = Scene.load()
        sc.mat_z_follows_counter, sc.counter_z_runtime = True, NOM + 0.15
        z = sc.mat_to_pelvis([0.3, 0.2, 0.06])[2]
        self.assertAlmostEqual(z, NOM + 0.15 + 0.06, places=9)
        self.assertAlmostEqual(sc.pelvis_to_mat(np.array([0.0, 0.0, z]))[2], 0.06, places=9)
        sc.mat_z_follows_counter = False
        self.assertAlmostEqual(sc.mat_to_pelvis([0.3, 0.2, 0.06])[2], NOM + 0.06, places=9)

    def test_bin_hull_follows_runtime_counter(self):
        """keep-mask bin hulls are projected from mat coords: they must move with the runtime counter."""
        hulls = {}
        for flag in (False, True):
            p, sc, _ = _perc(WIDE if flag else None)
            sc.counter_z_runtime = NOM + 0.15
            hulls[flag] = int((p._bin_hulls_mask((480, 640)) == 0).sum())
            sc.counter_z_runtime = None
            hulls[(flag, "nom")] = int((p._bin_hulls_mask((480, 640)) == 0).sum())
        self.assertEqual(hulls[False], hulls[(False, "nom")])      # off: hull ignores the runtime counter
        self.assertNotEqual(hulls[True], hulls[(True, "nom")])     # on: it follows


class TestUnmeasuredHeight(unittest.TestCase):
    """depth=None: use the configured prior, else fail OPEN on the workspace rule (flag on); flag off unchanged."""
    GATE = {"perception.color_gate.enabled": True}

    def _det_far(self, p):
        # a point whose x is beyond the workspace box at the nominal height but valid
        g = p.color_gate_cfg
        return Detection("red", np.array([float(g["ws_x"][1]) + 0.05, 0.0, NOM + 0.15 + p.height]), 1000, 1000, "plane"), g

    def test_prior_is_used_when_unmeasured(self):
        p, sc, _ = _perc({**WIDE, "calib.plane.counter_z_prior_m": NOM + 0.15})
        p.update_counter_height(None)
        self.assertAlmostEqual(sc.counter_z, NOM + 0.15, places=9)
        p.reset_calibration()
        self.assertEqual(sc.counter_z, NOM)

    def test_scene_prior_attribute_wins(self):
        p, sc, _ = _perc(WIDE)
        sc.counter_z_prior = NOM + 0.12
        p.update_counter_height(None)
        self.assertAlmostEqual(sc.counter_z, NOM + 0.12, places=9)

    def test_no_prior_workspace_fails_open_and_off_still_rejects(self):
        for flag in (True, False):
            p, sc, _ = _perc({**(WIDE if flag else {}), **self.GATE})
            p.update_counter_height(None)
            self.assertIsNone(sc.counter_z_runtime)
            det, g = self._det_far(p)
            self.assertEqual(p._color_gate_workspace(det, g), None if flag else "color_gate:workspace")

    def test_measured_height_keeps_the_rule(self):
        p, sc, _ = _perc({**WIDE, **self.GATE})
        sc.counter_z_runtime = NOM + 0.15
        det, g = self._det_far(p)
        self.assertEqual(p._color_gate_workspace(det, g), "color_gate:workspace")


class TestBinWallExtra(unittest.TestCase):
    def _plan(self, extra, dz=0.0, bin_name="left"):
        cfg = load_config(overrides={"planner.bin_wall_extra_m": extra})
        post, home = hand_postures(cfg), home_arm(cfg)
        sc = Scene.load()
        sc.counter_z_runtime = None if dz == 0 else NOM + dz
        pl = Planner(cfg, sc, post, home)
        L = sc.cfg["layout"]
        top = sc.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]), sc.object_height()])
        top[2] = sc.counter_z + sc.object_height()
        plan = pl.plan(Detection("red", top, 1000, 1000, "depth"), home, post["open"], bin_name=bin_name, relocate=False)
        self.assertTrue(plan.ok, plan.reason)
        return pl, plan

    def test_drop_and_carry_follow_runtime_counter_plus_wall_extra(self):
        wall = float(Scene.load().cfg["bins"]["wall_height"])
        base = {}
        for dz in (0.0, 0.15):
            for extra in (0.0, 0.01):
                pl, plan = self._plan(extra, dz)
                base[(dz, extra)] = plan.waypoints
        for dz in (0.0, 0.15):
            self.assertAlmostEqual(base[(dz, 0.01)]["drop"][2] - base[(dz, 0.0)]["drop"][2], 0.01, delta=1e-9)
            # carry = counter + max(carry_height, wall + H + 0.03): carry_height dominates at stock, never lower
            self.assertGreaterEqual(base[(dz, 0.01)]["lift"][2], base[(dz, 0.0)]["lift"][2])
        for extra in (0.0, 0.01):
            self.assertAlmostEqual(base[(0.15, extra)]["drop"][2] - base[(0.0, extra)]["drop"][2], 0.15, delta=0.005)
        h, ab = Scene.load().object_height(), load_config()["grasp"]["drop_above_rim"]
        self.assertAlmostEqual(base[(0.15, 0.01)]["drop"][2], NOM + 0.15 + wall + 0.01 + h / 2 + ab, delta=1e-9)

    def test_rim_keepout_and_descend_clearance_follow_the_extra(self):
        pl0, _ = self._plan(0.0)
        pl1, _ = self._plan(0.01)
        side = "left"
        q = pl0.home.copy()
        hand = hand_postures(pl0.cfg)["open"]
        b = pl0.scene.cfg["bins"]
        c0 = pl0.scene.mat_to_pelvis(np.r_[b["centers"]["left"], 0.0])
        pts = pl0.kin.hand_points(side, q, hand)
        # rim keepout: the clearance reported over a bin drops by exactly the extra (same pose, same bin footprint)
        probe = pl0.kin.hand_points
        shift = np.zeros(3)
        shift[:2] = c0[:2] - pts[0, :2]
        pl0.kin.hand_points = pl1.kin.hand_points = lambda s, a, h, _p=pts + shift: _p
        try:
            cl0, cl1 = pl0.bin_clearance(side, q, hand), pl1.bin_clearance(side, q, hand)
        finally:
            pl0.kin.hand_points = pl1.kin.hand_points = probe
        self.assertTrue(np.isfinite(cl0))
        self.assertAlmostEqual(cl0 - cl1, 0.01, delta=1e-9)

    def test_default_is_stock(self):
        a = self._plan(0.0)[1]
        b = Planner(load_config(), Scene.load(), hand_postures(load_config()), home_arm(load_config()))
        self.assertEqual(b._wall(0.06), 0.06)
        self.assertTrue(a.ok)


if __name__ == "__main__":
    unittest.main()
