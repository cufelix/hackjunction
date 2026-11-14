"""FK arm/hand masking, including a dense rendered-arm regression for New H2."""
import unittest
from unittest import mock

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import FLAGS, hand_postures, home_arm, load_config
from .hand_calib import ArmSurface
from .mock_world import Renderer
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
H_TOP = SC.object_height()


def _perception(fk=True, overlay="none", overrides=None):
    values = {"perception.color_gate.fk_mask": fk}
    values.update(overrides or {})
    cfg = load_config(overlay=overlay, overrides=values)
    return Perception(cfg, SC, CAL, log=lambda *_: None)


def _scene():
    objs = [dict(color="blue", shape="cube", yaw=0.0, size=SC.size, height=H_TOP,
                 pos=SC.mat_to_pelvis([0.45, 0.16, H_TOP / 2]))]
    return Renderer(SC, CAL).render(objs)


def _render_arm_links(p, rgb, depth, arm_q, hand_q, preserve=None, fringe_px=2):
    """Paint dense coloured G1/Dex3 link silhouettes into an RGB-D frame.

    This deliberately does not repeat the production sparse-point raster: each link's 5 mm FK surface samples are
    projected through the head-camera model and filled as a convex hull.  A two-pixel fringe represents RGB/FK edge
    residual.  ``preserve`` keeps separately rendered spawn-object pixels visible while placing arms next to them.
    """
    import cv2

    if p._arm_surface is None:
        p._arm_surface = ArmSurface()
    rendered = np.zeros(rgb.shape[:2], np.uint8)
    colours = (np.asarray(SC.colors["red"], np.uint8), np.asarray(SC.colors["yellow"], np.uint8),
               np.asarray(SC.colors["blue"], np.uint8))
    for i, (points, _) in enumerate(p._arm_surface.posed_links(arm_q, hand_q)):
        uvz = p._project(points)
        ok = np.all(np.isfinite(uvz), axis=1) & (uvz[:, 2] > 0)
        uv = np.round(uvz[ok, :2]).astype(np.int32)
        h, w = rendered.shape
        uv = uv[(uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)]
        if len(uv) < 3:
            continue
        link = np.zeros_like(rendered)
        cv2.fillConvexPoly(link, cv2.convexHull(uv), 1)
        if fringe_px:
            link = cv2.dilate(link, cv2.getStructuringElement(cv2.MORPH_ELLIPSE,
                                                               (2 * fringe_px + 1, 2 * fringe_px + 1)))
        if preserve is not None:
            link[np.asarray(preserve, bool)] = 0
        rgb[link > 0] = colours[i % len(colours)]
        depth[link > 0] = 0                 # unknown self depth is the hard colour-only false-positive case
        rendered |= link
    return rendered.astype(bool)


class TestFkMask(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.home = home_arm(load_config(profile="sim", overlay="none"))

    def test_sim_home_hands_are_in_view_and_masked(self):
        p = _perception()
        m = p._fk_arm_mask((480, 640), self.home)
        self.assertIsNotNone(m)
        self.assertGreater(int(m.sum()), 500)        # SIM home: hands in the head camera image (Timi / Oliver)
        self.assertLess(m.mean(), 0.6)               # a silhouette, not the whole image

    def test_dilation_is_a_default_off_registered_flag(self):
        self.assertIn("perception.color_gate.fk_dilate_px", FLAGS)
        self.assertEqual(load_config(overlay="none")["perception"]["color_gate"]["fk_dilate_px"], 0)
        self.assertEqual(load_config(overlay="finals_r21")["perception"]["color_gate"]["fk_dilate_px"], 3)

    def test_zero_dilation_preserves_the_stock_sparse_point_mask(self):
        p = _perception(overrides={"perception.color_gate.fk_dilate_px": 0})
        actual = p._fk_arm_mask((480, 640), self.home)
        points, _ = p._arm_surface.posed(self.home)
        uvz = p._project(points)
        ok = np.all(np.isfinite(uvz), axis=1) & (uvz[:, 2] > 0)
        u, v = np.round(uvz[ok, 0]).astype(int), np.round(uvz[ok, 1]).astype(int)
        inb = (u >= 0) & (u < 640) & (v >= 0) & (v < 480)
        stock = np.zeros((480, 640), bool)
        stock[v[inb], u[inb]] = True
        np.testing.assert_array_equal(actual, stock)

    def test_finals_rendered_arm_silhouette_has_zero_self_candidates_and_keeps_nearby_objects(self):
        import cv2

        cfg = load_config(profile="sim", overlay="finals_r21")
        arm = home_arm(cfg)
        hand = hand_postures(cfg)["open"]
        p = Perception(cfg, SC, CAL, log=lambda *_: None)

        # Arms only: the test render contains filled per-link silhouettes plus a 2 px colour fringe, not sparse dots.
        rgb, depth = Renderer(SC, CAL).render([])
        rendered = _render_arm_links(p, rgb, depth, arm, hand)
        self.assertGreater(int(rendered.sum()), 1000)
        unmasked = _perception(False, "finals_r21").detect(rgb, depth, debug=True)
        self.assertGreater(len(unmasked[0]) + len(unmasked[1]), 0)   # meaningful: coloured self pixels form candidates
        mask = p._fk_arm_mask(rendered.shape, arm, hand)
        self.assertEqual(int(np.count_nonzero(rendered & ~mask)), 0)
        dets, rejected = p.detect(rgb, depth, debug=True, robot_q=(arm, hand))
        self.assertEqual((dets, rejected), ([], []))

        # One spawn object per frame avoids blob merging.  These are near the visible hands in the head view; every
        # object remains detectable even though the rendered arm silhouette is within 40 px of its colour mask.
        spots = ((0.23, -0.20, "red"), (0.27, 0.15, "yellow"), (0.31, -0.15, "blue"))
        for x, y, colour in spots:
            pos = SC.mat_to_pelvis([y + 0.45, x - 0.14, H_TOP / 2])
            obj = [dict(color=colour, shape="cube", yaw=0.0, size=SC.size, height=H_TOP, pos=pos)]
            rgb, depth = Renderer(SC, CAL).render(obj)
            object_pixels = np.zeros(rgb.shape[:2], bool)
            for blob in p.masks(rgb).values():
                object_pixels |= blob > 0
            rendered = _render_arm_links(p, rgb, depth, arm, hand, preserve=object_pixels)
            near = cv2.dilate(object_pixels.astype(np.uint8), np.ones((81, 81), np.uint8)) > 0
            self.assertTrue(np.any(rendered & near), (x, y, colour))
            dets = p.detect(rgb, depth, robot_q=(arm, hand))
            self.assertEqual([d.color for d in dets], [colour], (x, y, colour))

    def test_object_under_the_arm_mask_is_dropped(self):
        rgb, depth = _scene()
        p = _perception()
        base = p.detect(rgb, depth)
        self.assertEqual([d.color for d in base], ["blue"])           # no robot_q: unmasked
        with mock.patch.object(p, "_fk_arm_mask", return_value=np.ones((480, 640), bool)):
            self.assertEqual(p.detect(rgb, depth, robot_q=(self.home, None)), [])

    def test_flag_off_ignores_robot_q_and_never_builds_the_mask(self):
        rgb, depth = _scene()
        p = _perception(fk=False)
        with mock.patch.object(p, "_fk_arm_mask", side_effect=AssertionError("mask built with the flag off")):
            self.assertEqual([d.color for d in p.detect(rgb, depth, robot_q=(self.home, None))], ["blue"])

    def test_mask_failure_is_unmasked_not_a_crash(self):
        p = _perception()
        self.assertIsNone(p._fk_arm_mask((480, 640), np.zeros(3)))    # wrong arm_q size -> logged, None


if __name__ == "__main__":
    unittest.main()
