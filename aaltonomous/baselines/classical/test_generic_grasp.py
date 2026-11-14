"""Synthetic pinhole tests for the unknown-object grasp proposal."""
import unittest

import numpy as np

from .generic_grasp import propose_grasp


def _camera():
    K = np.array([[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]])
    T = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, -1.0, 0.0, 0.0],
                  [0.0, 0.0, -1.0, 0.5], [0.0, 0.0, 0.0, 1.0]])
    return {"height": 480, "width": 640, "K": K, "T": T}


def _render_plane(cam, z_plane, inside):
    height, width, K, T = cam["height"], cam["width"], cam["K"], cam["T"]
    v, u = np.meshgrid(np.arange(height), np.arange(width), indexing="ij")
    rays = np.stack(((u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u)), -1)
    rays = rays @ T[:3, :3].T
    origin = T[:3, 3]
    scale = (z_plane - origin[2]) / rays[..., 2]
    points = origin + scale[..., None] * rays
    mask = (scale > 0.0) & inside(points[..., 0], points[..., 1])
    return mask, np.where(mask, scale, 0.0)


def _rectangle(centre, length, width, yaw):
    c, s = np.cos(yaw), np.sin(yaw)

    def inside(x, y):
        dx, dy = x - centre[0], y - centre[1]
        return ((np.abs(c * dx + s * dy) <= length / 2.0) &
                (np.abs(-s * dx + c * dy) <= width / 2.0))
    return inside


def _disk(centre, radius):
    return lambda x, y: (x - centre[0]) ** 2 + (y - centre[1]) ** 2 <= radius ** 2


def _yaw_error(a, b):
    return abs(float((a - b + np.pi / 2.0) % np.pi - np.pi / 2.0))


class TestGenericGrasp(unittest.TestCase):
    def test_rotated_rectangle_centre_yaw_and_width(self):
        cam, yaw = _camera(), np.deg2rad(30.0)
        mask, depth = _render_plane(cam, 0.03, _rectangle((0.02, -0.015), 0.09, 0.045, yaw))
        proposal = propose_grasp(mask, depth, cam["K"], cam["T"])
        self.assertIsNotNone(proposal)
        np.testing.assert_allclose(proposal.centre, [0.02, -0.015, 0.03], atol=0.003)
        self.assertLess(_yaw_error(proposal.yaw, yaw), np.deg2rad(3.0))
        self.assertAlmostEqual(proposal.width, 0.045, delta=0.004)
        self.assertGreaterEqual(proposal.confidence, 0.9)

    def test_circle_has_finite_proposal(self):
        cam = _camera()
        mask, depth = _render_plane(cam, 0.04, _disk((-0.01, 0.02), 0.0225))
        proposal = propose_grasp(mask, depth, cam["K"], cam["T"])
        self.assertIsNotNone(proposal)
        np.testing.assert_allclose(proposal.centre, [-0.01, 0.02, 0.04], atol=0.003)
        self.assertTrue(np.isfinite(proposal.yaw))
        self.assertAlmostEqual(proposal.width, 0.045, delta=0.004)

    def test_concave_object_centre_is_on_observed_support(self):
        cam = _camera()

        def ring(x, y):
            r2 = (x - 0.01) ** 2 + (y + 0.01) ** 2
            return (r2 <= 0.030 ** 2) & (r2 >= 0.012 ** 2)

        mask, depth = _render_plane(cam, 0.035, ring)
        proposal = propose_grasp(mask, depth, cam["K"], cam["T"])
        self.assertIsNotNone(proposal)
        radius = np.linalg.norm(proposal.centre[:2] - [0.01, -0.01])
        self.assertGreaterEqual(radius, 0.010)
        self.assertLessEqual(radius, 0.032)

    def test_holes_and_depth_spikes_are_robust(self):
        cam, yaw = _camera(), np.deg2rad(-25.0)
        mask, depth = _render_plane(cam, 0.03, _rectangle((0.02, -0.015), 0.09, 0.045, yaw))
        rng = np.random.default_rng(7)
        noisy = depth.copy()
        noisy[rng.random(noisy.shape) < 0.30] = 0.0
        pixels = np.argwhere(noisy > 0.0)
        spikes = pixels[rng.random(len(pixels)) < 0.05]
        noisy[spikes[:, 0], spikes[:, 1]] *= 0.5
        proposal = propose_grasp(mask, noisy, cam["K"], cam["T"])
        self.assertIsNotNone(proposal)
        np.testing.assert_allclose(proposal.centre, [0.02, -0.015, 0.03], atol=0.005)
        self.assertLess(_yaw_error(proposal.yaw, yaw), np.deg2rad(3.0))

    def test_sparse_or_invalid_inputs(self):
        cam = _camera()
        mask = np.zeros((480, 640), bool)
        depth = np.zeros(mask.shape)
        self.assertIsNone(propose_grasp(mask, depth, cam["K"], cam["T"]))
        mask[10:13, 10:15], depth[10:13, 10:15] = True, 0.4
        self.assertIsNone(propose_grasp(mask, depth, cam["K"], cam["T"]))
        with self.assertRaises(ValueError):
            propose_grasp(mask, depth, np.eye(4), cam["T"])
        bad_K = cam["K"].copy(); bad_K[0, 0] = 0.0
        with self.assertRaises(ValueError):
            propose_grasp(mask, depth, bad_K, cam["T"])

    def test_confidence_penalises_border_and_excess_width(self):
        cam = _camera()
        mask, depth = _render_plane(cam, 0.03, _disk((0.0, 0.0), 0.0225))
        baseline = propose_grasp(mask, depth, cam["K"], cam["T"])
        bordered = mask.copy(); bordered[0, 0] = True
        bordered_depth = depth.copy(); bordered_depth[0, 0] = 0.47
        border = propose_grasp(bordered, bordered_depth, cam["K"], cam["T"])
        missing_border_depth = propose_grasp(bordered, depth, cam["K"], cam["T"])
        wide_mask, wide_depth = _render_plane(cam, 0.03, _disk((0.0, 0.0), 0.05))
        wide = propose_grasp(wide_mask, wide_depth, cam["K"], cam["T"])
        self.assertLess(border.confidence, baseline.confidence)
        self.assertLess(border.confidence, 0.5)
        self.assertLess(missing_border_depth.confidence, 0.5)
        self.assertGreater(wide.width, 0.06)
        self.assertLess(wide.confidence, baseline.confidence)


if __name__ == "__main__":
    unittest.main()
