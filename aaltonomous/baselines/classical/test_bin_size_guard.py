"""perception.bin_size_guard (real-only hotfix): a bin-sized / bin-tall colour blob is never a pick candidate. Run:
    python -m unittest aaltonomous.baselines.classical.test_bin_size_guard -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
H, W = 480, 640
MM_PER_PX = 1.0   # synthetic camera: 1 px = 1 mm on the counter


def _perc(overlay="finals_r25_gen", overrides=None):
    p = Perception(load_config(overlay=overlay, overrides=overrides), SC, CAL, log=lambda *_: None)
    return p


def _rig(p, heights, expected_px=3000.0):
    """Stub the camera: pixel (u, v) -> pelvis (0.3 + u mm, v mm, counter_z + heights[v, u])."""
    zc = float(p.scene.counter_z)

    def pix(u, v, d, frame=None, transform_snapshot=None):
        u, v = np.asarray(u), np.asarray(v)
        return np.c_[0.3 + u * MM_PER_PX / 1000.0, v * MM_PER_PX / 1000.0, zc + heights[v, u]]
    p._pixel_to_pelvis = pix
    p.expected_area_px = lambda blob, centroid, depth: expected_px
    return np.full((H, W), 600, np.uint16)


def _blob(x0, y0, w, h):
    m = np.zeros((H, W), np.uint8)
    m[y0:y0 + h, x0:x0 + w] = 1
    return m


class TestBinSizeGuard(unittest.TestCase):
    def test_flag_real_on_sim_off_stock_off(self):
        self.assertTrue(_perc().bin_size_guard)
        real = load_config(overlay="finals_r25_gen")
        sim = load_config(overlay="finals_r25_gen", overrides=dict(real["profile_auto"]["sim_overrides"]))
        self.assertFalse(Perception(sim, SC, CAL, log=lambda *_: None).bin_size_guard)
        self.assertFalse(_perc(overlay="none").bin_size_guard)

    def test_cube_accepted(self):
        p = _perc()
        hts = np.zeros((H, W))
        cube = _blob(300, 200, 45, 45)            # 4.5 cm footprint, 4.5 cm tall
        hts[cube > 0] = 0.045
        depth = _rig(p, hts)
        self.assertIsNone(p._bin_sized_blob(cube, int(cube.sum()), (322, 222), depth))

    def test_two_touching_cubes_accepted(self):
        p = _perc()
        hts = np.zeros((H, W))
        pair = _blob(300, 200, 90, 45)
        hts[pair > 0] = 0.045
        depth = _rig(p, hts)
        self.assertIsNone(p._bin_sized_blob(pair, int(pair.sum()), (345, 222), depth))

    def test_cube_without_depth_accepted(self):
        p = _perc()
        _rig(p, np.zeros((H, W)))
        cube = _blob(300, 200, 45, 45)
        self.assertIsNone(p._bin_sized_blob(cube, int(cube.sum()), (322, 222), None))

    def test_bin_sized_orange_blob_rejected_by_area(self):
        p = _perc()
        hts = np.zeros((H, W))
        wall = _blob(200, 100, 210, 90)            # 21 cm bin wall + rim, 6 cm rim
        hts[wall > 0] = 0.06
        depth = _rig(p, hts)
        self.assertEqual(p._bin_sized_blob(wall, int(wall.sum()), (305, 145), depth), "bin_size:area")

    def test_bin_rejected_by_extent_when_area_looks_small(self):
        p = _perc()
        hts = np.zeros((H, W))
        wall = _blob(200, 100, 210, 30)            # thin 21 cm wall sliver at a 6 cm rim
        hts[wall > 0] = 0.06
        depth = _rig(p, hts, expected_px=1e9)
        self.assertEqual(p._bin_sized_blob(wall, int(wall.sum()), (305, 115), depth), "bin_size:extent")

    def test_bin_rejected_by_top_height(self):
        p = _perc()
        hts = np.zeros((H, W))
        rim = _blob(300, 200, 60, 40)              # small rim fragment, 9 cm above the counter
        hts[rim > 0] = 0.09
        depth = _rig(p, hts, expected_px=1e9)
        self.assertEqual(p._bin_sized_blob(rim, int(rim.sum()), (330, 220), depth), "bin_size:top")


if __name__ == "__main__":
    unittest.main()
