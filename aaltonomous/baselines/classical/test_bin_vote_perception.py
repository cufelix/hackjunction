"""bin.vote in perception: the bin footprints (_reject in_bin, _keep_mask, _bin_hulls_mask, _near_bin) follow the
planner's bin_shift when bin.vote is on; flag off = nominal scene.yaml centres. Synthetic only. Run:
    python -m unittest aaltonomous.baselines.classical.test_bin_vote_perception -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import Renderer
from .perception import Detection, Perception

SC, CAL = Scene.load(), Calibration.load()
CFG = load_config(overlay="none")
H_TOP = SC.object_height()
BIN = np.asarray(SC.cfg["bins"]["centers"]["center"], float)                       # mat (0.45, 0.365)
HALF = np.asarray(SC.cfg["bins"]["outer_size"], float) / 2 + CFG["perception"]["bin_exclusion_margin"]
SHIFT_PELVIS = np.array([-0.07, 0.0])                                             # pelvis dx -7 cm = mat y -7 cm
# mat points (x, y) at object-top height, checked against the footprints in the tests below
P_SHIFT_ONLY = (BIN[0], BIN[1] - HALF[1] - 0.02)     # inside shifted box (y 0.18..0.41), 2 cm outside nominal (0.25..0.48)
P_NOM_ONLY = (BIN[0], BIN[1] + HALF[1] - 0.005)      # inside nominal, 6.5 cm outside shifted


class _Planner:
    def __init__(self, shift):
        self.bin_shift = shift


def _perception(vote_on):
    p = Perception(CFG, SC, CAL, log=lambda *_: None)
    p.bin_vote_on = vote_on
    p.bin_shift_src = _Planner({"center": SHIFT_PELVIS})
    return p


def _top(mat_xy, h=H_TOP):
    return SC.mat_to_pelvis([float(mat_xy[0]), float(mat_xy[1]), h])


def _det(xy, h=H_TOP / 2):
    return Detection("red", _top(xy, h), 5000, 500, "depth")


class TestRejectFootprint(unittest.TestCase):
    def test_flag_off_is_nominal_and_ignores_the_shift(self):
        p = _perception(False)
        self.assertEqual(p._reject(_det(BIN)), "in_bin:center")
        self.assertIsNone(p._reject(_det(P_SHIFT_ONLY)))
        self.assertEqual(p._reject(_det(P_NOM_ONLY)), "in_bin:center")

    def test_flag_on_follows_the_shifted_footprint(self):
        p = _perception(True)
        self.assertEqual(p._reject(_det(BIN)), "in_bin:center")
        self.assertEqual(p._reject(_det(P_SHIFT_ONLY)), "in_bin:center")
        self.assertIsNone(p._reject(_det(P_NOM_ONLY)))

    def test_near_bin_follows_the_shift(self):
        top = np.atleast_2d(_top(P_NOM_ONLY))   # 6.5 cm outside the shifted box, but within its near-bin band (nominal)
        self.assertTrue(_perception(False)._near_bin(top))
        self.assertFalse(_perception(True)._near_bin(top))


class TestFootprintMasks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rgb, cls.depth = Renderer(SC, CAL).render(
            [dict(color="blue", shape="cube", yaw=0.0, size=SC.size, height=H_TOP,
                  pos=SC.mat_to_pelvis([P_SHIFT_ONLY[0], P_SHIFT_ONLY[1], H_TOP / 2]))])

    def _px(self, p, xy, h=H_TOP):
        uvz = p._project(np.atleast_2d(_top(xy, h)))[0]
        return int(round(uvz[1])), int(round(uvz[0]))   # (row, col)

    def test_keep_mask_with_depth_honours_the_shift(self):
        off, on = _perception(False), _perception(True)
        self.assertEqual(off._keep_mask(self.depth)[self._px(off, P_SHIFT_ONLY)], 255)
        self.assertEqual(on._keep_mask(self.depth)[self._px(on, P_SHIFT_ONLY)], 0)

    def test_bin_hulls_mask_honours_the_shift(self):
        shape = self.depth.shape
        off, on = _perception(False), _perception(True)
        far = (BIN[0], 0.19)   # 5 cm outside the nominal box, 2 cm inside the shifted one (the hull walls make y 0.23 ambiguous)
        self.assertEqual(off._bin_hulls_mask(shape)[self._px(off, far, H_TOP / 2)], 255)
        self.assertEqual(on._bin_hulls_mask(shape)[self._px(on, far, H_TOP / 2)], 0)


if __name__ == "__main__":
    unittest.main()
