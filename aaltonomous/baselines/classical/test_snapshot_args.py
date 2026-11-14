"""Regression: fixvote's explicit transform_snapshot and fixcal's FrameTransform share _pixel_to_pelvis.
A raw 4x4 passed positionally used to land in `frame` and be transposed (r22 bin.vote snapshot path)."""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from .config import load_config
from .perception import Perception
from aaltonomous.scene.scene import Scene


class TestSnapshotArgs(unittest.TestCase):
    def setUp(self):
        self.p = Perception(load_config(overlay="none"), Scene.load(), Calibration.load(), log=lambda *_: None)
        self.u, self.v = np.array([200.0, 320.0, 450.0]), np.array([300.0, 260.0, 380.0])
        self.d = np.array([900.0, 1100.0, 1300.0])

    def test_explicit_snapshot_equals_current_pose(self):
        T = self.p.projection_snapshot()
        np.testing.assert_allclose(self.p._pixel_to_pelvis(self.u, self.v, self.d, transform_snapshot=T),
                                   self.p._pixel_to_pelvis(self.u, self.v, self.d), atol=1e-9)
        np.testing.assert_allclose(self.p._pixel_to_pelvis(self.u, self.v, self.d, frame=self.p.frame_transform()),
                                   self.p._pixel_to_pelvis(self.u, self.v, self.d), atol=1e-9)

    def test_raw_matrix_in_frame_slot_is_rejected(self):
        with self.assertRaises(TypeError):
            self.p._pixel_to_pelvis(self.u, self.v, self.d, self.p.projection_snapshot())
        with self.assertRaises(TypeError):
            self.p._project(np.zeros((1, 3)), self.p.projection_snapshot())

    def test_counter_plane_snapshot_matches_live_pose(self):
        depth = np.full((480, 640), 1000, np.uint16)
        c = load_config(overlay="none")["calib"]["plane"]
        live = self.p._counter_plane(depth, c)
        snap = self.p._counter_plane(depth, c, transform_snapshot=self.p.projection_snapshot())
        self.assertEqual(live[2], snap[2])
        if live[0] is not None:
            np.testing.assert_allclose(np.asarray(snap[0]), np.asarray(live[0]), atol=1e-9)

    def test_r22_bin_vote_snapshot_path_matches_live_path(self):
        import copy
        from .bin_vote import vote_bins
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(config=load_config(overlay="finals_r22"), log=lambda *_: None)
        rng = np.random.default_rng(3)
        depth = (1000 + rng.integers(-3, 4, (480, 640))).astype(np.uint16)
        rgb = np.zeros((480, 640, 3), np.uint8)
        T = pol.perception.projection_snapshot()
        frames = [{"head_rgb": rgb, "head_depth": depth, "transform": pol.perception.frame_transform(),
                   "transform_snapshot": T} for _ in range(3)]
        out = {}
        for snap in (False, True):
            cfg = copy.deepcopy(pol.cfg)
            cfg["bin"]["vote"]["frame_transform_snapshot"] = snap
            logs = []
            out[snap] = (vote_bins(pol.perception, pol.planner, frames, cfg, log=logs.append), logs)
        self.assertFalse(any("failed" in m for m in out[True][1]), out[True][1])
        for a, b in zip(out[False][0], out[True][0]):
            self.assertEqual(set(a), set(b))
            for k in a:
                np.testing.assert_allclose(np.asarray(a[k], float), np.asarray(b[k], float), atol=1e-9)

    def test_tape_votes_snapshot_equals_live_with_and_without_depth(self):
        import cv2
        from .bin_vote import tape_votes
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(config=load_config(overlay="finals_r22"), log=lambda *_: None)
        per, v = pol.perception, pol.cfg["bin"]["vote"]
        names = list(pol.planner.scene.bin_names)
        centers = {n: np.asarray(pol.planner.bin_xy(n), float) for n in names}
        lo, hi = np.asarray(v["tape_hsv"][0], int), np.asarray(v["tape_hsv"][1], int)
        rgb_px = cv2.cvtColor(((lo + hi) // 2).astype(np.uint8)[None, None], cv2.COLOR_HSV2RGB)[0, 0]
        rgb = np.zeros((480, 640, 3), np.uint8)
        depth = np.zeros((480, 640), np.uint16)
        sc = per.scene
        rim_z = sc.counter_z + float(sc.cfg["bins"]["wall_height"])
        drawn = 0
        for n in names:
            u, w, z = per._project(np.r_[centers[n], rim_z])
            u, w = int(round(u)), int(round(w))
            if 10 <= u < 630 and 10 <= w < 200:
                rgb[w - 4:w + 4, u - 12:u + 12] = rgb_px
                depth[w - 4:w + 4, u - 12:u] = int(z * 1000.0)   # half with depth, half on the rim plane
                drawn += 1
        self.assertGreater(drawn, 0)
        T = per.projection_snapshot()
        live = tape_votes(per, rgb, depth, centers, v)
        snap = tape_votes(per, rgb, depth, centers, v, transform_snapshot=T)
        self.assertTrue(any(x is not None for x in live.values()), live)
        for n in names:
            if live[n] is None:
                self.assertIsNone(snap[n])
            else:
                np.testing.assert_allclose(np.asarray(snap[n], float), np.asarray(live[n], float), atol=1e-9)


if __name__ == "__main__":
    unittest.main()
