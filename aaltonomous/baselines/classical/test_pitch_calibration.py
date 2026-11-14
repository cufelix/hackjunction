"""Synthetic tests for the opt-in per-episode head-camera pitch calibration."""
import threading
import time
import unittest
from unittest import mock

import numpy as np
from scipy.spatial.transform import Rotation

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config, pitch_cal_conflicts
from .mock_world import Renderer, make_mock_env
from .perception import Perception
from .policy import ClassicalPolicy


SC = Scene.load()
CAL = Calibration.load()
PITCH_ON = {"perception.online_pitch_cal": True}
TABLE_ON = {"perception.online_pitch_cal": "table"}


def _pose(deg, roll=0.0, height_delta=0.0):
    nominal = CAL._T("head")
    T = nominal.copy()
    T[:3, :3] = nominal[:3, :3] @ Perception._optical_rz(roll) @ Perception._optical_rx(deg)
    T[2, 3] += float(height_delta)
    return T


def _project(T, points):
    q = (np.asarray(points, float) - T[:3, 3]) @ T[:3, :3]
    K = CAL.cams["head"].K
    return np.stack([K[0, 0] * q[..., 0] / q[..., 2] + K[0, 2],
                     K[1, 1] * q[..., 1] / q[..., 2] + K[1, 2], q[..., 2]], axis=-1)


def _synthetic_rims(deg, keep_bins=None):
    """Orange vertical walls whose top pixels are the projected nominal far rims."""
    rgb = np.zeros((480, 640, 3), np.uint8)
    depth = np.full((480, 640), 800, np.uint16)
    bins = SC.cfg["bins"]
    half = np.asarray(bins["outer_size"], float) / 2.0
    keep_bins = set(bins["centers"]) if keep_bins is None else set(keep_bins)
    for name, center in bins["centers"].items():
        if name not in keep_bins:
            continue
        center = np.asarray(center, float)
        rim = SC.mat_to_pelvis(
            [[center[0] - half[0], center[1] + half[1], bins["wall_height"]],
             [center[0] + half[0], center[1] + half[1], bins["wall_height"]]])
        uvz = _project(_pose(deg), rim)
        u0, u1 = sorted(np.rint(uvz[:, 0]).astype(int))
        v = int(round(float(np.mean(uvz[:, 1]))))
        u0, u1 = max(0, u0), min(rgb.shape[1] - 1, u1)
        if 0 <= v < rgb.shape[0] and u0 <= u1:
            rgb[v:min(v + 45, rgb.shape[0]), u0:u1 + 1] = [230, 110, 30]
    return rgb, depth


def _synthetic_table(deg, roll=0.0, height_delta=0.0, seed=123):
    """Noisy counter depth rendered with an erroneous camera tilt, plus holes and non-plane clutter."""
    h, w = 480, 640
    vv, uu = np.indices((h, w))
    K = CAL.cams["head"].K
    rays = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones((h, w))], -1)
    T = _pose(deg, roll, height_delta)
    direction = rays @ T[:3, :3].T
    with np.errstate(divide="ignore", invalid="ignore"):
        optical_z = (SC.counter_z - T[2, 3]) / direction[..., 2]
    rng = np.random.default_rng(seed)
    depth = np.rint(optical_z / CAL.cams["head"].depth_scale + rng.normal(0.0, 1.2, (h, w)))
    good = np.isfinite(depth) & (optical_z > 0.0) & (depth >= 150) & (depth <= 2000)
    depth = np.where(good, depth, 0).astype(np.uint16)
    depth[rng.random((h, w)) < 0.12] = 0

    rgb = np.zeros((h, w, 3), np.uint8)
    # A coloured object is explicitly masked; the grey patch acts like unsegmented hand clutter for RANSAC.
    rgb[290:350, 270:350] = [200, 30, 30]
    depth[290:350, 270:350] = np.maximum(depth[290:350, 270:350].astype(int) - 70, 151)
    rgb[355:430, 410:485] = [90, 90, 90]
    depth[355:430, 410:485] = np.maximum(depth[355:430, 410:485].astype(int) - 110, 151)
    return rgb, depth


class TestOnlinePitchCalibration(unittest.TestCase):
    def test_recovers_known_optical_x_offsets(self):
        cfg = load_config(overrides=PITCH_ON)
        for expected in (-9.0, -4.0, 0.0, 3.0):
            with self.subTest(expected=expected):
                rgb, depth = _synthetic_rims(expected)
                perception = Perception(cfg, SC, CAL)
                result = perception.calibrate_pitch(rgb, depth)
                self.assertTrue(result.success, result)
                self.assertAlmostEqual(result.pitch_deg, expected, delta=0.5)
                np.testing.assert_allclose(perception._T(), _pose(result.pitch_deg), atol=1e-12)

    def test_empty_image_keeps_nominal_extrinsic(self):
        cfg = load_config(overrides=PITCH_ON)
        perception = Perception(cfg, SC, CAL)
        nominal = CAL._T("head").copy()
        result = perception.calibrate_pitch(np.zeros((480, 640, 3), np.uint8),
                                            np.full((480, 640), 800, np.uint16))
        self.assertTrue(result.valid_frame)
        self.assertFalse(result.success)
        self.assertIn("too few", result.reason)
        self.assertIsNone(perception.pitch_T)
        np.testing.assert_array_equal(perception._T(), nominal)

    def test_missing_one_bin_keeps_nominal_extrinsic(self):
        cfg = load_config(overrides=PITCH_ON)
        perception = Perception(cfg, SC, CAL)
        rgb, depth = _synthetic_rims(-9.0, keep_bins=("left", "center"))
        result = perception.calibrate_pitch(rgb, depth)
        self.assertTrue(result.valid_frame)
        self.assertFalse(result.success)
        self.assertIn("not all bins found", result.reason)
        self.assertIsNone(perception.pitch_T)
        np.testing.assert_array_equal(perception._T(), CAL._T("head"))

    def test_inconsistent_rims_fail_residual_gate(self):
        rgb = np.zeros((480, 640, 3), np.uint8)
        for name, pitch in zip(SC.cfg["bins"]["centers"], (-9.0, 0.0, 9.0)):
            one_bin, depth = _synthetic_rims(pitch, keep_bins=(name,))
            rgb = np.maximum(rgb, one_bin)
        cfg = load_config(overrides={**PITCH_ON, "perception.pitch_calibration.edge_band_px": 150})
        perception = Perception(cfg, SC, CAL)
        result = perception.calibrate_pitch(rgb, depth)
        self.assertFalse(result.success)
        self.assertIn("high residual", result.reason)
        self.assertIsNone(perception.pitch_T)
        np.testing.assert_array_equal(perception._T(), CAL._T("head"))

    def test_invalid_depth_retries_and_reset_discards_cached_pose(self):
        cfg = load_config(overrides=PITCH_ON)
        perception = Perception(cfg, SC, CAL)
        rgb, depth = _synthetic_rims(-4.0)
        invalid = perception.calibrate_pitch(rgb, np.zeros_like(depth))
        self.assertFalse(invalid.valid_frame)
        accepted = perception.calibrate_pitch(rgb, depth)
        self.assertTrue(accepted.success, accepted)
        self.assertIsNotNone(perception.pitch_T)
        perception.reset_calibration()
        self.assertIsNone(perception.pitch_calibration)
        self.assertIsNone(perception.pitch_T)
        np.testing.assert_array_equal(perception._T(), CAL._T("head"))

    def test_flag_off_is_nominal(self):
        perception = Perception(load_config(overlay="none"), SC, CAL)
        self.assertFalse(perception.online_pitch_cal)
        self.assertIsNone(perception.pitch_T)
        np.testing.assert_array_equal(perception._T(), CAL._T("head"))
        u = np.array([10, 320, 630])
        v = np.array([20, 240, 470])
        depth = np.array([450, 700, 1100])
        np.testing.assert_array_equal(perception._pixel_to_pelvis(u, v, depth),
                                      CAL.pixel_to_pelvis("head", u, v, depth))
        points = SC.mat_to_pelvis([[0.2, 0.2, 0.0], [0.45, 0.35, 0.06], [0.7, 0.5, 0.1]])
        np.testing.assert_array_equal(perception._project(points), CAL.project("head", points))

    def test_policy_retries_once_logs_once_and_reset_discards_fit(self):
        cfg = load_config(overrides=PITCH_ON)
        env = make_mock_env(seed=2, cfg=cfg)
        lines = []
        policy = ClassicalPolicy(config=cfg, log=lines.append, safety=env.filter.cfg)
        obs = env.reset()
        policy.reset({"instruction": obs["instruction"]})
        invalid = dict(obs, head_depth=np.zeros_like(obs["head_depth"]))

        policy.act(invalid)
        self.assertFalse(policy._pitch_calibration_done)
        self.assertFalse([line for line in lines if line.startswith("[pitch_cal]")])
        policy.act(obs)
        self.assertTrue(policy._pitch_calibration_done)
        policy._pitch_thread.join(30)
        cached = policy.pitch_calibration
        self.assertIsNotNone(cached)
        policy.act(obs)
        self.assertIs(policy.pitch_calibration, cached)
        self.assertEqual(len([line for line in lines if line.startswith("[pitch_cal]")]), 1)

        policy.reset({"instruction": obs["instruction"]})
        self.assertFalse(policy._pitch_calibration_done)
        self.assertIsNone(policy.pitch_calibration)
        self.assertIsNone(policy.perception.pitch_T)


class TestTablePitchCalibration(unittest.TestCase):
    def test_recovers_known_pitch_from_noisy_plane_with_holes(self):
        cfg = load_config(overrides=TABLE_ON)
        self.assertEqual(cfg["perception"]["table_pitch_calibration"]["min_inlier_ratio"], 0.60)
        for expected in (-9.0, -4.0, 0.0, 3.0):
            with self.subTest(expected=expected):
                rgb, depth = _synthetic_table(expected)
                perception = Perception(cfg, SC, CAL)
                result = perception.calibrate_pitch(rgb, depth)
                self.assertTrue(result.success, result)
                self.assertEqual(result.mode, "table")
                self.assertAlmostEqual(result.pitch_deg, expected, delta=0.5)
                self.assertAlmostEqual(result.roll_deg, 0.0, delta=0.5)
                self.assertAlmostEqual(result.plane_height_m, SC.counter_z, delta=0.005)
                self.assertAlmostEqual(result.camera_height_m, CAL._T("head")[2, 3] - SC.counter_z,
                                       delta=0.005)
                self.assertGreaterEqual(result.n_inliers,
                                        cfg["perception"]["table_pitch_calibration"]["min_inliers"])
                self.assertLess(result.residual_m, 0.005)

    def test_recovers_roll_and_corrects_plane_normal(self):
        cfg = load_config(overrides=TABLE_ON)
        rgb, depth = _synthetic_table(-4.0, roll=3.0)
        perception = Perception(cfg, SC, CAL)
        result = perception.calibrate_pitch(rgb, depth)
        self.assertTrue(result.success, result)
        self.assertAlmostEqual(result.pitch_deg, -4.0, delta=0.5)
        self.assertAlmostEqual(result.roll_deg, 3.0, delta=0.5)
        np.testing.assert_allclose(perception._T()[:3, :3], _pose(result.pitch_deg, result.roll_deg)[:3, :3],
                                   atol=2e-3)

    def test_reports_camera_height_but_never_writes_z(self):
        cfg = load_config(overrides=TABLE_ON)
        rgb, depth = _synthetic_table(3.0, roll=-2.0, height_delta=0.027)
        perception = Perception(cfg, SC, CAL)
        result = perception.calibrate_pitch(rgb, depth)
        expected = _pose(3.0, -2.0, 0.027)
        self.assertTrue(result.success, result)
        self.assertAlmostEqual(result.camera_height_m, expected[2, 3] - SC.counter_z, delta=0.005)
        self.assertIn("not applied", result.reason)
        self.assertEqual(perception._T()[2, 3], CAL._T("head")[2, 3])
        np.testing.assert_array_equal(perception._T()[:3, 3], CAL._T("head")[:3, 3])
        np.testing.assert_allclose(perception._T()[:3, :3], expected[:3, :3], atol=2e-3)

    def test_quality_and_angle_gates_keep_nominal_extrinsic(self):
        cases = [
            ({"perception.table_pitch_calibration.min_inlier_ratio": 0.95}, -4.0, 0.0,
             "low plane inlier ratio"),
            ({"perception.table_pitch_calibration.max_rms_m": 0.00001}, -4.0, 0.0, "high plane RMS"),
            ({}, 16.0, 0.0, "pitch outside guard"),
            ({}, -4.0, 6.0, "roll outside guard"),
        ]
        for extra, pitch, roll, reason in cases:
            with self.subTest(reason=reason):
                cfg = load_config(overrides={**TABLE_ON, **extra})
                rgb, depth = _synthetic_table(pitch, roll=roll)
                perception = Perception(cfg, SC, CAL)
                result = perception.calibrate_pitch(rgb, depth)
                self.assertFalse(result.success, result)
                self.assertIn(reason, result.reason)
                self.assertIsNone(perception.pitch_T)
                np.testing.assert_array_equal(perception._T(), CAL._T("head"))

    def test_rejects_too_little_plane_and_keeps_nominal(self):
        cfg = load_config(overrides=TABLE_ON)
        rgb, depth = _synthetic_table(-4.0)
        # Keep the overall frame live, but leave only a tiny number of candidates in the table ROI.
        depth[264:441, 76:563] = 0
        depth[300:315, 300:315] = 600
        perception = Perception(cfg, SC, CAL)
        result = perception.calibrate_pitch(rgb, depth)
        self.assertTrue(result.valid_frame)
        self.assertFalse(result.success)
        self.assertIn("too few table points", result.reason)
        self.assertIsNone(perception.pitch_T)
        np.testing.assert_array_equal(perception._T(), CAL._T("head"))

    def test_policy_logs_mode_tilt_height_inliers_and_decides_once(self):
        cfg = load_config(overrides=TABLE_ON)
        env = make_mock_env(seed=2, cfg=cfg)
        lines = []
        policy = ClassicalPolicy(config=cfg, log=lines.append, safety=env.filter.cfg)
        obs = env.reset()
        rgb, depth = _synthetic_table(3.0)
        obs = dict(obs, head_rgb=rgb, head_depth=depth)
        policy.reset({"instruction": obs["instruction"]})
        policy.act(obs)
        policy._pitch_thread.join(30)
        pitch_lines = [line for line in lines if line.startswith("[pitch_cal]")]
        self.assertEqual(len(pitch_lines), 1)
        for field in ("mode=table", "dpitch=", "droll=", "height=", "inliers=", "rms=", "accepted", "not applied"):
            self.assertIn(field, pitch_lines[0])
        policy.act(obs)
        self.assertEqual(len([line for line in lines if line.startswith("[pitch_cal]")]), 1)

    def test_policy_act_never_blocks_and_uses_nominal_until_swap(self):
        cfg = load_config(overrides=TABLE_ON)
        env = make_mock_env(seed=2, cfg=cfg)
        policy = ClassicalPolicy(config=cfg, log=lambda *a: None, safety=env.filter.cfg)
        obs = env.reset()
        policy.reset({"instruction": obs["instruction"]})
        gate, calls = threading.Event(), []
        real = policy.perception.compute_pitch

        def slow(rgb, depth):
            calls.append(1)
            gate.wait(30)
            return real(rgb, depth)

        with mock.patch.object(policy.perception, "compute_pitch", slow):
            policy.act(obs)
            policy.act(obs)           # worker is stuck; act() must return anyway
            self.assertIsNone(policy.perception.pitch_T)
            np.testing.assert_array_equal(policy.perception._T(), CAL._T("head"))
            gate.set()
            policy._pitch_thread.join(30)
        self.assertEqual(len(calls), 1)    # once per episode
        self.assertIsNotNone(policy.pitch_calibration)

    def test_stale_worker_cannot_write_into_next_episode(self):
        cfg = load_config(overrides=TABLE_ON)
        env = make_mock_env(seed=2, cfg=cfg)
        policy = ClassicalPolicy(config=cfg, log=lambda *a: None, safety=env.filter.cfg)
        obs = env.reset()
        rgb, depth = _synthetic_table(3.0)
        obs = dict(obs, head_rgb=rgb, head_depth=depth)
        policy.reset({"instruction": obs["instruction"]})
        gate = threading.Event()
        real = policy.perception.compute_pitch
        with mock.patch.object(policy.perception, "compute_pitch", lambda r, d: (gate.wait(30), real(r, d))[1]):
            policy.act(obs)
            old = policy._pitch_thread
            policy.reset({"instruction": obs["instruction"]})   # new episode while the old fit is still running
            gate.set()
            old.join(30)
        self.assertIsNone(policy.pitch_calibration)
        self.assertIsNone(policy.perception.pitch_T)
        self.assertFalse(policy._pitch_calibration_done)

    def test_worker_exception_keeps_nominal(self):
        cfg = load_config(overrides=TABLE_ON)
        env = make_mock_env(seed=2, cfg=cfg)
        policy = ClassicalPolicy(config=cfg, log=lambda *a: None, safety=env.filter.cfg)
        obs = env.reset()
        policy.reset({"instruction": obs["instruction"]})
        with mock.patch.object(policy.perception, "compute_pitch", side_effect=RuntimeError("boom")):
            policy.act(obs)
            policy._pitch_thread.join(30)
        self.assertFalse(policy.pitch_calibration.success)
        self.assertIn("boom", policy.pitch_calibration.reason)
        np.testing.assert_array_equal(policy.perception._T(), CAL._T("head"))


def _render_counter(deg, objects=(), seed=5, hole_frac=0.0):
    """Counter + bins rendered by the mock world's ray caster with an independently built camera pose
    (scipy rotation about the optical x axis, not Perception._optical_rx)."""
    nominal = CAL._T("head").copy()
    nominal[:3, :3] = nominal[:3, :3] @ Rotation.from_euler("x", deg, degrees=True).as_matrix()
    return Renderer(SC, CAL, T_pelvis_optical=nominal).render(
        list(objects), rng=np.random.default_rng(seed), depth_noise_mm=1.2, hole_frac=hole_frac)


class TestTableAgainstMockRender(unittest.TestCase):
    def test_bins_hand_and_holes_are_accepted(self):
        cfg = load_config(overrides=TABLE_ON)
        for deg in (0.0, 3.0):
            with self.subTest(deg=deg):
                rgb, depth = _render_counter(deg, hole_frac=0.40)
                depth[355:440, 410:495] = np.maximum(depth[355:440, 410:495].astype(int) - 110, 151)  # the hand
                rgb[355:440, 410:495] = [90, 90, 90]
                result = Perception(cfg, SC, CAL).calibrate_pitch(rgb, depth)
                self.assertTrue(result.success, result)
                self.assertAlmostEqual(result.pitch_deg, deg, delta=0.5)

    def test_sign_and_magnitude_against_independent_render(self):
        cfg = load_config(overrides=TABLE_ON)
        for deg in (-2.0, 2.0):
            with self.subTest(deg=deg):
                rgb, depth = _render_counter(deg)
                result = Perception(cfg, SC, CAL).calibrate_pitch(rgb, depth)
                self.assertTrue(result.success, result)
                self.assertEqual(np.sign(result.pitch_deg), np.sign(deg), result)
                self.assertAlmostEqual(result.pitch_deg, deg, delta=0.5)


class TestPitchCalReleaseGate(unittest.TestCase):
    def test_conflicts(self):
        self.assertEqual(pitch_cal_conflicts(load_config(overlay="none")), [])
        on = {"perception.online_pitch_cal": True}
        self.assertEqual(pitch_cal_conflicts(load_config(overrides=on)), [])
        self.assertEqual(pitch_cal_conflicts(load_config(overrides={**on, "perception.y_affine": [0.0, 0.0]})), [])
        self.assertEqual(len(pitch_cal_conflicts(load_config(overrides={**on, "perception.y_affine": [0.02, 0.0]}))), 1)
        self.assertEqual(len(pitch_cal_conflicts(load_config(overrides={**on, "perception.xy_offset": [0.0, 0.01]}))), 1)
        off = {"perception.y_affine": [0.02, 0.0]}
        self.assertEqual(pitch_cal_conflicts(load_config(overrides=off)), [])


if __name__ == "__main__":
    unittest.main()
