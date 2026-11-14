"""perception.inject_cam_error: the head extrinsics perturbation (test-only flag). Run:
    python -m pytest -q aaltonomous/baselines/classical/test_inject_cam_error.py
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration

from .config import load_config
from .perception import perturb_extrinsics
from .policy import ClassicalPolicy


class TestInjection(unittest.TestCase):
    def setUp(self):
        self.cal = Calibration.load()
        self.T = self.cal.cams["head"].T_pelvis_optical

    def test_zero_error_is_identity(self):
        np.testing.assert_allclose(perturb_extrinsics(self.T, [0, 0, 0, 0, 0]), self.T)

    def test_translation_shifts_deprojected_points(self):
        T2 = perturb_extrinsics(self.T, [0, 0, 0.03, -0.02, 0.0])
        p = np.array([0.1, 0.2, 0.7]) @ self.T[:3, :3].T + self.T[:3, 3]
        q = np.array([0.1, 0.2, 0.7]) @ T2[:3, :3].T + T2[:3, 3]
        np.testing.assert_allclose(q - p, [0.03, -0.02, 0.0], atol=1e-12)

    def test_pitch_minus_7_gives_a_few_cm_object_error(self):
        T2 = perturb_extrinsics(self.T, [-7, 0, 0, 0, 0])
        R = T2[:3, :3]
        self.assertAlmostEqual(abs(np.linalg.det(R) - 1), 0, places=4)
        c = self.cal.cams["head"]
        errs = []
        for X in ([0.35, 0.0, -0.05], [0.4, 0.2, -0.05], [0.3, -0.2, -0.05]):    # table points
            uvz = self.cal.project("head", np.array(X))
            d = uvz[2] * 1000.0
            P0 = self.cal.pixel_to_pelvis("head", uvz[0], uvz[1], d)
            pt = np.array([(uvz[0] - c.K[0, 2]) / c.K[0, 0], (uvz[1] - c.K[1, 2]) / c.K[1, 1], 1.0]) * uvz[2]
            P1 = T2[:3, :3] @ pt + T2[:3, 3]
            errs.append(np.linalg.norm(P1[:2] - P0[:2]))
        self.assertTrue(all(0.02 < e < 0.12 for e in errs), errs)

    def test_perception_uses_the_injected_extrinsics_and_the_shared_calibration_is_untouched(self):
        cfg = load_config(overrides={"perception.inject_cam_error": [-7, 0, 0, 0, 0]})
        pol = ClassicalPolicy(config=cfg, log=None)
        T = pol.cal.cams["head"].T_pelvis_optical.copy()
        self.assertFalse(np.allclose(pol.perception.cal.cams["head"].T_pelvis_optical, T))
        np.testing.assert_allclose(pol.cal.cams["head"].T_pelvis_optical, T)

    def test_camera_derived_bin_geometry_moves_under_injection(self):
        """Bin image hulls (keep mask) and bin points deprojected from depth both go through the perturbed pose."""
        base = ClassicalPolicy(config=load_config(), log=None)
        inj = ClassicalPolicy(config=load_config(overrides={"perception.inject_cam_error": [-7, 0, 0, 0, 0]}), log=None)
        m0, m1 = (p.perception._bin_hulls_mask((480, 640)) for p in (base, inj))
        self.assertGreater(int((m0 != m1).sum()), 2000)
        sc = base.scene
        c = list(sc.cfg["bins"]["centers"].values())[0]
        P = sc.mat_to_pelvis([c[0], c[1], float(sc.cfg["bins"]["wall_height"])])
        u, v, z = base.cal.project("head", P)
        q0 = base.perception.cal.pixel_to_pelvis("head", u, v, z * 1000.0)
        q1 = inj.perception.cal.pixel_to_pelvis("head", u, v, z * 1000.0)
        np.testing.assert_allclose(q0, P, atol=1e-6)
        self.assertGreater(np.linalg.norm((q1 - q0)[:2]), 0.02)

    def test_bad_inject_value_fails_at_load(self):
        with self.assertRaises(ValueError):
            load_config(overrides={"perception.inject_cam_error": [1, 2]})


if __name__ == "__main__":
    unittest.main()
