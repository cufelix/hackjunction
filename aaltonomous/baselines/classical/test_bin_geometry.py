"""bin.geometry_runtime: ONE accessor (bin_vote.bin_geometry) feeds planner (bin_xy/drop_xy/slots/clearance/wall hits) and
perception (reject/near_bin/keep mask/hulls/colour gate). Synthetic +7 cm pelvis-x shift (the real drop_2 offset).
Flag off = the old numbers exactly. Run: python -m unittest aaltonomous.baselines.classical.test_bin_geometry -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .bin_vote import bin_geometry
from .config import load_config
from .perception import Detection, Perception
from .policy import ClassicalPolicy

SC, CAL = Scene.load(), Calibration.load()
DX = 0.07                                   # pelvis x = mat y
SHIFT = {n: np.array([DX, 0.0]) for n in SC.bin_names}
H = SC.object_height()


class _FakeKin:
    def __init__(self, pts):
        self.pts = np.asarray(pts, float)

    def hand_points(self, side, arm_q, hand_q):
        return self.pts


class _Src:
    def __init__(self, shift):
        self.bin_shift = shift


def _planner(overlay="none", shift=None):
    pl = ClassicalPolicy(config=load_config(overlay=overlay)).planner
    pl.bin_shift = dict(shift or {})
    return pl


def _perc(geo, vote=False, live=False, shift=SHIFT):
    p = Perception(load_config(overlay="none"), SC, CAL, log=lambda *_: None)
    p.geometry_runtime, p.bin_vote_on, p.live_bins = geo, vote, live
    p.bin_shift_src = _Src(shift)
    return p


def _det(mat_xy):
    return Detection("red", SC.mat_to_pelvis([mat_xy[0], mat_xy[1], H / 2]), 5000, 500, "depth")


class TestAccessor(unittest.TestCase):
    def test_nominal(self):
        for n, g in bin_geometry(SC).items():
            c = np.asarray(SC.cfg["bins"]["centers"][n], float)
            np.testing.assert_array_equal(g["centre_mat"], c)
            np.testing.assert_array_equal(g["centre_pelvis"], SC.mat_to_pelvis(np.r_[c, 0.0])[:2])
            lo, hi = SC.bin_inner_box(n)
            np.testing.assert_array_equal(g["inner_lo"], lo)
            self.assertIsNone(g["shift"])

    def test_shift_moves_everything_in_both_frames(self):
        nom, sh = bin_geometry(SC), bin_geometry(SC, SHIFT)
        for n in SC.bin_names:
            np.testing.assert_allclose(sh[n]["centre_pelvis"] - nom[n]["centre_pelvis"], [DX, 0])
            np.testing.assert_allclose(sh[n]["centre_mat"] - nom[n]["centre_mat"], [0, DX])   # pelvis x = mat y
            np.testing.assert_allclose(sh[n]["outer_hi"] - nom[n]["outer_hi"], [0, DX])
            np.testing.assert_allclose(sh[n]["inner_lo"] - nom[n]["inner_lo"], [0, DX, 0], atol=1e-12)
            np.testing.assert_allclose(np.subtract(sh[n]["inner_pelvis"], nom[n]["inner_pelvis"]), [DX, DX, 0, 0])


class TestPlanner(unittest.TestCase):
    def test_flag_off_is_the_old_numbers_exactly(self):
        pl = _planner()
        for n in SC.bin_names:
            np.testing.assert_array_equal(pl.bin_xy(n), SC.mat_to_pelvis(np.r_[SC.cfg["bins"]["centers"][n], 0.0])[:2])
            np.testing.assert_array_equal(pl.drop_xy(n), pl._drop_xy0(n))
            for a, b in zip(pl.drop_slots(n), pl._drop_slots0(n)):
                np.testing.assert_array_equal(a, b)
            self.assertTrue(pl._in_bin_footprint(n, pl.drop_xy(n)))

    def test_drop_point_is_the_centre_of_the_shifted_bin(self):
        pl = _planner("ab_bingeo", SHIFT)
        pl.cfg["grasp"]["drop_inset"] = None              # release at the centre
        for n in SC.bin_names:
            nom = SC.mat_to_pelvis(np.r_[SC.cfg["bins"]["centers"][n], 0.0])[:2]
            np.testing.assert_allclose(pl.drop_xy(n), nom + [DX, 0])
            np.testing.assert_allclose(pl.drop_xy(n), pl.bin_geometry(n)["centre_pelvis"])

    def test_drop_and_slots_inside_the_shifted_footprint(self):
        pl = _planner("ab_bingeo", SHIFT)
        pl0 = _planner()
        for n in SC.bin_names:
            self.assertTrue(pl._in_bin_footprint(n, pl.drop_xy(n)))
            np.testing.assert_allclose(pl.drop_xy(n) - pl0.drop_xy(n), [DX, 0])
            for s, s0 in zip(pl.drop_slots(n), pl0.drop_slots(n)):
                self.assertTrue(pl._in_bin_footprint(n, s))
                np.testing.assert_allclose(s - s0, [DX, 0])
            xlo, xhi, ylo, yhi = pl.bin_geometry(n)["inner_pelvis"]
            edge = [xhi - 1e-3, (ylo + yhi) / 2]       # inside the shifted footprint, 6.9 cm past the nominal one
            self.assertTrue(pl._in_bin_footprint(n, edge))
            self.assertFalse(pl0._in_bin_footprint(n, edge))

    def test_collision_uses_the_shifted_walls(self):
        pl = _planner("ab_bingeo")
        b = SC.cfg["bins"]
        c = np.asarray(b["centers"]["left"], float)
        half = np.asarray(b["outer_size"], float) / 2
        m = np.array([c[0], c[1] + half[1] + 0.02, float(b["wall_height"]) - 0.02])   # below rim, 2 cm outside nominal wall
        pl.kin = _FakeKin([SC.mat_to_pelvis(m)])
        self.assertFalse(pl.hits_bin_wall("left", None, None))
        pl.bin_shift = {"left": np.array([DX, 0.0])}
        self.assertTrue(pl.hits_bin_wall("left", None, None))
        # the shifted cavity is allowed for a corrective pick into that bin, the nominal one is not
        inner = bin_geometry(SC, pl.bin_shift, "left")
        cav = (inner["inner_lo"] + inner["inner_hi"]) / 2
        pl.kin = _FakeKin([SC.mat_to_pelvis(np.r_[cav[:2], float(b["wall_height"]) - 0.02])])
        self.assertFalse(pl.hits_bin_wall("left", None, None, allowed_inside="left"))
        # bin_clearance: a point above the shifted rim, outside the nominal box
        pl.kin = _FakeKin([SC.mat_to_pelvis(np.r_[m[:2], float(b["wall_height"]) + 0.02])])
        pl.bin_shift = {}
        self.assertEqual(pl.bin_clearance("left", None, None), np.inf)
        pl.bin_shift = {"left": np.array([DX, 0.0])}
        self.assertAlmostEqual(pl.bin_clearance("left", None, None), 0.02, places=9)


class TestPerception(unittest.TestCase):
    C = np.asarray(SC.cfg["bins"]["centers"]["center"], float)
    HALF = np.asarray(SC.cfg["bins"]["outer_size"], float) / 2 + load_config(overlay="none")["perception"]["bin_exclusion_margin"]

    def test_flag_off_is_nominal(self):
        p = _perc(False)
        for n, c in SC.cfg["bins"]["centers"].items():
            np.testing.assert_array_equal(p._bin_centres_mat()[n], np.asarray(c, float))

    def test_geometry_flag_alone_makes_reject_and_near_bin_follow(self):
        c, hy = self.C, self.HALF[1]
        out_nom = (c[0], c[1] + hy + 0.02)          # 2 cm past the nominal +y edge: inside once shifted +7 cm in mat y
        self.assertIsNone(_perc(False)._reject(_det(out_nom)))
        self.assertEqual(_perc(True)._reject(_det(out_nom)), "in_bin:center")
        in_nom = (c[0], c[1] - hy + 0.005)          # inside nominal, outside the shifted box
        self.assertEqual(_perc(False)._reject(_det(in_nom)), "in_bin:center")
        self.assertIsNone(_perc(True)._reject(_det(in_nom)))
        cfg = load_config(overlay="none")["perception"]
        near = SC.cfg["bins"]["outer_size"][1] / 2 + cfg["bin_pixel_margin"] + _perc(False).size
        top = np.atleast_2d(SC.mat_to_pelvis([c[0], c[1] - near + 0.01, H]))   # inside the nominal band, 6 cm outside shifted
        self.assertTrue(_perc(False)._near_bin(top))
        self.assertFalse(_perc(True)._near_bin(top))

    def test_exclusion_mask_and_hulls_follow(self):
        depth = np.full((480, 640), 700, np.uint16)
        a, b = _perc(False)._bin_hulls_mask(depth.shape), _perc(True)._bin_hulls_mask(depth.shape)
        self.assertFalse(np.array_equal(a, b))
        np.testing.assert_array_equal(a, _perc(False, vote=True, shift={})._bin_hulls_mask(depth.shape))
        np.testing.assert_array_equal(b, _perc(False, vote=True)._bin_hulls_mask(depth.shape))   # = the old bin.vote path

    def test_colour_gate_boxes_follow_without_live_bins(self):
        g = {"near_margin_m": 0.10}
        off, on = _perc(False)._color_gate_boxes(g), _perc(True)._color_gate_boxes(g)
        for (lo0, hi0), (lo1, hi1) in zip(off, on):
            np.testing.assert_allclose(lo1 - lo0, [0, DX])
            np.testing.assert_allclose(hi1 - hi0, [0, DX])


class TestFlag(unittest.TestCase):
    def test_default_off_and_ab_overlay_on(self):
        self.assertFalse(load_config(overlay="none")["bin"]["geometry_runtime"])
        cfg = load_config(overlay="ab_bingeo")
        self.assertTrue(cfg["bin"]["geometry_runtime"] and cfg["bin"]["vote"]["enabled"])
        self.assertTrue(ClassicalPolicy(config=cfg).perception.geometry_runtime)
        self.assertFalse(ClassicalPolicy(config=load_config(overlay="none")).perception.geometry_runtime)


if __name__ == "__main__":
    unittest.main()
