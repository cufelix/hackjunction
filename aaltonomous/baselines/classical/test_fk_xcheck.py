"""calib.fk_xcheck (B2, log-only): synthetic image tests on the real head camera, plus the policy hook."""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .fk_xcheck import FkHandXCheck, FkXCheckResult, axis_rotation, project_residual
from .mock_world import make_mock_env
from .perception import Perception
from .policy import ClassicalPolicy

SC, CAL = Scene.load(), Calibration.load()
# Synthetic dots are small: relax the area / opening gates only (the real gates are in config.yaml).
CFG = dict(load_config(overlay="none")["calib"]["fk_xcheck_config"], open_px=1, min_area_px=5, max_area_px=500,
           min_valid_depth_px=5)


def _perception(pitch_deg=None):
    p = Perception(load_config(overlay="none"), SC, CAL, log=None)
    if pitch_deg is not None:   # what perception.online_pitch_cal commits: pitch_T = corrected pose
        T = p._T().copy()
        T[:3, :3] = axis_rotation(pitch_deg, 0.0) @ T[:3, :3]
        p.pitch_T = T
    return p


def _hand_points(T, K):
    """Hand-like points 0.35-0.5 m in front of the camera, spread over the image (camera frame -> pelvis)."""
    rng = np.random.default_rng(0)
    u, v = rng.uniform(200, 440, 12), rng.uniform(160, 320, 12)
    z = rng.uniform(0.35, 0.5, 12)
    cam = np.c_[(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z]
    return cam @ T[:3, :3].T + T[:3, 3]


class TestProjection(unittest.TestCase):
    def test_zero_residual_is_perceptions_own_projection(self):
        """The cross-check projects exactly like Perception._project, nominal and pitch-corrected."""
        for pitch in (None, 4.0):
            p = _perception(pitch)
            pts = _hand_points(p._T(), p.cal.cams[p.cam].K)
            np.testing.assert_allclose(project_residual(pts, p._T(), p.cal.cams[p.cam].K),
                                       p._project(pts), rtol=1e-10, atol=1e-9)


class TestFkXCheck(unittest.TestCase):
    def _render(self, p, pitch, yaw=0.0):
        import cv2
        cam = p.cal.cams[p.cam]
        rgb = np.full((480, 640, 3), 235, np.uint8)
        for u, v, z in project_residual(self.points, p._T(), cam.K, pitch, yaw):
            if z > 0 and 0 <= u < 640 and 0 <= v < 480:
                cv2.circle(rgb, (round(u), round(v)), 2, (35, 35, 35), -1)
        return rgb

    def _setup(self, pitch_deg):
        self.p = _perception(pitch_deg)
        self.points = _hand_points(self.p._T(), self.p.cal.cams[self.p.cam].K)
        self.check = FkHandXCheck(dict(CFG), self.p)

    def test_recovers_residual_on_the_pose_in_use(self):
        for pose in (None, 4.0):      # nominal and online-pitch-corrected
            self._setup(pose)
            for pitch in (-9.0, -4.0, 0.0, 3.0):
                with self.subTest(pose=pose, pitch=pitch):
                    got = self.check.estimate(self._render(self.p, pitch), self.points)
                    self.assertTrue(got.visible, got)
                    self.assertLessEqual(abs(got.pitch_deg - pitch), 1.0, got)
                    self.assertLessEqual(abs(got.yaw_deg), 1.0, got)

    def test_no_hand_is_noop_with_low_confidence(self):
        self._setup(None)
        got = self.check.estimate(np.full((480, 640, 3), 235, np.uint8), self.points)
        self.assertFalse(got.visible)
        self.assertEqual((got.pitch_deg, got.yaw_deg), (0.0, 0.0))
        self.assertLess(got.confidence, 0.1)

    def test_online_disagreement_over_two_degrees_lowers_confidence(self):
        self._setup(None)
        rgb = self._render(self.p, 3.0)
        standalone = self.check.estimate(rgb, self.points)
        disagreed = self.check.estimate(rgb, self.points, online_pitch_deg=-3.0)
        self.assertGreater(disagreed.disagreement_deg, 2.0)
        self.assertLess(disagreed.confidence, standalone.confidence)


class TestPolicyHook(unittest.TestCase):
    def test_off_by_default_creates_nothing(self):
        pol = ClassicalPolicy(config=load_config(overlay="none"), log=None)
        self.assertFalse(pol.fk_xcheck_enabled)
        self.assertFalse(hasattr(pol, "_fk_xcheck_thread"))

    def test_on_launches_a_worker_and_publishes_a_log_only_result(self):
        env = make_mock_env(seed=0)
        obs = env.reset()
        cfg = load_config(overlay="none", overrides={"calib.fk_xcheck": True})
        pol = ClassicalPolicy(config=cfg, log=None)
        pol.reset({"instruction": obs["instruction"], "time_limit": 120.0})
        pol._poll_fk_xcheck(obs, 0.0)                 # launches the worker
        pol._fk_xcheck_thread.join(timeout=30)
        pol._poll_fk_xcheck(obs, 0.0)                 # publishes it
        self.assertIsInstance(pol.fk_xcheck_result, FkXCheckResult)
        self.assertIsNone(pol._fk_xcheck_thread)      # period not reached: no second worker yet


if __name__ == "__main__":
    unittest.main()
