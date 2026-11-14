"""Synthetic FK/depth tests for the finals pre-motion calibration stack."""
import time
import unittest
from dataclasses import replace

import numpy as np

from aaltonomous.interface.calibration import Calibration

from .calibration_fit import CounterHeightTracker, FrameGuard, PrimaryCalibrationFitter
from .config import hand_postures, load_config
from .hand_calib import ArmSurface, _rot_deg, perturb, render_depth, table_depth


CFG = load_config(overlay="none")
CAL = Calibration.load()
CAM = "head"
T0 = CAL.cams[CAM].T_pelvis_optical.copy()
K = CAL.cams[CAM].K
ARM = np.asarray(CFG["home_arm_q"], float)
HAND = hand_postures(CFG)["open"]
SURFACE = None
FITTER = None


def fitter(crosscheck=True):
    global SURFACE, FITTER
    if SURFACE is None:
        SURFACE = ArmSurface()
    if FITTER is None or FITTER.crosscheck != crosscheck:
        FITTER = PrimaryCalibrationFitter(CFG["calib"]["primary_fit"], SURFACE, crosscheck=crosscheck)
    return FITTER


def synthetic(T, seed=0, noise_mm=0.8, dropout=0.0, arms=True, table_z=0.01):
    bg = table_depth(T, K, z_table=table_z)
    bg = np.where(np.isfinite(bg), bg, 0).astype(np.uint16)
    if arms:
        points, _ = fitter().surface.posed(ARM, HAND)
        arm = render_depth(points, T, K, spacing=fitter().surface.spacing)
        depth = render_depth(points, T, K, spacing=fitter().surface.spacing, background=table_depth(T, K, z_table=table_z))
    else:
        arm, depth = np.zeros_like(bg), bg
    rng = np.random.default_rng(seed)
    if dropout:
        depth[(arm > 0) & (rng.random(depth.shape) < dropout)] = 0
    if noise_mm:
        valid = depth > 0
        noisy = depth.astype(float)
        noisy[valid] += rng.normal(0.0, noise_mm, int(valid.sum()))
        depth = np.clip(np.rint(noisy), 0, 65535).astype(np.uint16)
    return depth


class TestPrimaryFit(unittest.TestCase):
    def test_recovers_3cm_4deg_envelope(self):
        cases = [
            ([0.008, -0.006, 0.004], [0.7, -0.5, 0.4]),
            ([0.024, -0.015, 0.009], [3.2, -1.8, 1.4]),
            ([-0.018, 0.021, -0.009], [-2.6, 2.4, -1.6]),
        ]
        for seed, (dt, dr) in enumerate(cases):
            true = perturb(T0, dt, dr)
            result = fitter().fit(CAL, CAM, synthetic(true, seed), ARM, HAND, 0.01)
            self.assertTrue(result.hand_accepted, result.summary())
            self.assertLess(np.linalg.norm(result.T[:3, 3] - true[:3, 3]), 0.005, result.summary())
            self.assertLess(_rot_deg(true[:3, :3].T @ result.T[:3, :3]), 0.5, result.summary())

    def test_black_link_40pct_dropout_recovers_or_cleanly_falls_back(self):
        true = perturb(T0, [0.022, -0.012, 0.008], [2.5, -1.5, 1.0])
        result = fitter().fit(CAL, CAM, synthetic(true, 9, dropout=0.40), ARM, HAND, 0.01)
        self.assertIn(result.source, {"hand", "plane", "nominal"})
        self.assertTrue(result.reason)
        if result.source == "hand":
            self.assertLess(np.linalg.norm(result.T[:3, 3] - true[:3, 3]), 0.005, result.summary())
            self.assertLess(_rot_deg(true[:3, :3].T @ result.T[:3, :3]), 0.5, result.summary())
        elif result.source == "nominal":
            np.testing.assert_array_equal(result.T, T0)

    def test_armless_negative_control_is_always_rejected(self):
        for seed in range(8):
            true = perturb(T0, [0.01, -0.008, 0.005], [1.0, -0.8, 0.5])
            result = fitter().fit(CAL, CAM, synthetic(true, seed, arms=False), ARM, HAND, 0.01)
            self.assertFalse(result.hand_accepted, (seed, result.summary()))
            self.assertNotEqual(result.source, "hand")

    def test_fallback_chain_ends_at_nominal_plus_15cm_counter_prior(self):
        depth = np.zeros((480, 640), np.uint16)
        result = fitter().fit(CAL, CAM, depth, ARM, HAND, 0.01)
        self.assertEqual(result.source, "nominal")
        np.testing.assert_array_equal(result.T, T0)
        self.assertAlmostEqual(result.counter_z, 0.16)
        self.assertTrue(any(name == "fallback selection" and passed for name, passed, _ in result.checks))

    def test_plane_tilt_disagreement_rejects_hand_commit(self):
        local = PrimaryCalibrationFitter(CFG["calib"]["primary_fit"], fitter().surface, crosscheck=True)
        original, calls = local.fit_plane, [0]

        def disagree(*args, **kwargs):
            plane = original(*args, **kwargs)
            calls[0] += 1
            return replace(plane, tilt_deg=3.0) if calls[0] == 2 else plane

        local.fit_plane = disagree
        true = perturb(T0, [0.008, -0.006, 0.004], [0.6, -0.5, 0.4])
        result = local.fit(CAL, CAM, synthetic(true, 5), ARM, HAND, 0.01)
        self.assertFalse(result.hand_accepted, result.summary())
        self.assertEqual(result.source, "plane")
        self.assertTrue(any(name == "plane tilt cross-check" and not passed for name, passed, _ in result.checks))

    def test_warm_fit_is_under_150ms_cpu_budget(self):
        true = perturb(T0, [0.018, -0.012, 0.007], [2.0, -1.4, 0.8])
        depth = synthetic(true, 4)
        fitter().fit(CAL, CAM, depth, ARM, HAND, 0.01)  # import/allocation warm-up is outside an episode
        samples = []
        for _ in range(3):
            started = time.perf_counter()
            result = fitter().fit(CAL, CAM, depth, ARM, HAND, 0.01)
            samples.append((time.perf_counter() - started) * 1000.0)
            self.assertTrue(result.hand_accepted, result.summary())
        self.assertLess(max(samples), 150.0, samples)


class TestFreshnessAndCounter(unittest.TestCase):
    def test_stale_frame_and_joint_motion_are_rejected(self):
        guard = FrameGuard(pair_s=0.1, joint_rad=0.02)
        q = np.zeros(28)
        self.assertFalse(guard.check(1.0, q, last_motion_time=1.1)[0])
        self.assertFalse(guard.check(1.2, q, require_pair=True)[0])  # primer
        ok, reason = guard.check(1.31, q + 0.021, require_pair=True)
        self.assertFalse(ok)
        self.assertIn("joints moved", reason)

    def test_counter_rejects_frame_older_than_last_motion(self):
        tracker = CounterHeightTracker(fitter(), CFG["calib"]["counter_fit"], 0.01)
        update = tracker.observe(CAL, CAM, synthetic(T0, 2), T0, 1.0, np.r_[ARM, HAND], last_motion_time=1.1)
        self.assertFalse(update.accepted)
        self.assertIn("stale timestamp", update.reason)

    def test_counter_needs_three_distinct_mad_consistent_samples(self):
        tracker = CounterHeightTracker(fitter(), CFG["calib"]["counter_fit"], 0.01)
        true = perturb(T0, [0.01, -0.008, 0.006], [1.0, -0.7, 0.4])
        depth = synthetic(true, 3, table_z=0.16)
        q = np.r_[ARM, HAND]
        updates = [tracker.observe(CAL, CAM, depth, true, t, q) for t in (0.0, 0.11, 0.22, 0.33)]
        self.assertFalse(updates[0].accepted)  # primer does not count
        self.assertFalse(updates[2].committed)
        self.assertTrue(updates[3].committed, updates[3])
        self.assertAlmostEqual(updates[3].counter_z, 0.16, delta=0.004)

    def test_refit_smoothing_caps_a_single_update(self):
        params = dict(CFG["calib"]["counter_fit"], max_samples=3, mad_floor_m=0.05,
                      smooth_alpha=1.0, max_update_m=0.01)
        tracker = CounterHeightTracker(fitter(), params, 0.01)
        q = np.r_[ARM, HAND]
        low = synthetic(T0, 1, table_z=0.16)
        for t in (0.0, 0.11, 0.22, 0.33):
            first = tracker.observe(CAL, CAM, low, T0, t, q)
        self.assertTrue(first.committed)
        before = first.counter_z
        high = synthetic(T0, 2, table_z=0.19)
        tracker.observe(CAL, CAM, high, T0, 0.44, q)
        moved = tracker.observe(CAL, CAM, high, T0, 0.55, q)
        self.assertTrue(moved.committed)
        self.assertLessEqual(abs(moved.counter_z - before), 0.0100001)


class TestPolicyAsyncPhase(unittest.TestCase):
    def test_new_flags_default_off_and_do_not_construct_the_phase(self):
        from .policy import ClassicalPolicy
        policy = ClassicalPolicy(log=None)
        self.assertIsNone(policy.primary_calibration)
        self.assertTrue(policy._primary_done)
        self.assertFalse(policy.perception.counter_after_hand)
        self.assertEqual(policy._counter_kw({}), {})

    def test_act_holds_and_never_waits_for_fit(self):
        from .policy import ClassicalPolicy
        policy = ClassicalPolicy(log=None, overrides={"calib.hand_icp_primary": True})
        real_fit = policy.primary_calibration.fitter.fit

        def slow(*args):
            time.sleep(0.35)
            return real_fit(*args)

        policy.primary_calibration.fitter.fit = slow
        true = perturb(T0, [0.008, -0.006, 0.004], [0.6, -0.5, 0.4])
        obs = {"head_rgb": np.zeros((480, 640, 3), np.uint8), "head_depth": synthetic(true, 7),
               "arm_q": ARM.copy(), "hand_q": HAND.copy(), "t": 0.0}
        first = policy.act(obs)
        obs["t"] = 0.11
        started = time.perf_counter()
        second = policy.act(obs)
        elapsed = time.perf_counter() - started
        self.assertLess(elapsed, 0.10)
        self.assertEqual(policy.state, "calibrate")   # rc7: the fit runs inside fixcal's pre-motion calibrate state
        np.testing.assert_allclose(first["arm_q_target"], np.repeat(ARM[None], len(first["arm_q_target"]), axis=0))
        np.testing.assert_allclose(second["arm_q_target"], np.repeat(ARM[None], len(second["arm_q_target"]), axis=0))


if __name__ == "__main__":
    unittest.main()
