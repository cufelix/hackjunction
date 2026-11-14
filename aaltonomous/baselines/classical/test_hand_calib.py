"""perception.hand_calibration (hand_calib.py): head-camera pose from the arms in depth. Run:
    python -m unittest aaltonomous.baselines.classical.test_hand_calib -v
Synthetic depth = the shipped arm surface model posed by FK, rendered from a KNOWN perturbed camera + the counter plane.
"""
import time
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration

from . import hand_calib as hc
from .config import hand_postures, load_config

CFG = load_config(overlay="none")
HOME = np.asarray(CFG["home_arm_q"], float)          # sim profile: arms down, hands in the head view
HAND = hand_postures(CFG)["open"]
CAL = Calibration.load()
K = CAL.cams["head"].K
T0 = CAL.cams["head"].T_pelvis_optical.copy()
_SOLVER = None
# object top centres on the counter (pelvis frame)
OBJ = np.c_[np.stack(np.meshgrid(np.linspace(0.30, 0.55, 4), np.linspace(-0.3, 0.3, 5)), -1).reshape(-1, 2),
            np.full(20, 0.055)]


def solver():
    global _SOLVER
    if _SOLVER is None:
        _SOLVER = hc.HandCalibrator(CFG["perception"]["hand_calibration"])
    return _SOLVER


def scene_depth(T_true, arm_q=HOME):
    P, _ = solver().surface.posed(arm_q, HAND)
    return hc.render_depth(P, T_true, K, spacing=solver().surface.spacing, background=hc.table_depth(T_true, K))


def obj_xy_err_mm(T_true, T_est):
    pc = (OBJ - T_true[:3, 3]) @ T_true[:3, :3]
    p = pc @ T_est[:3, :3].T + T_est[:3, 3]
    return np.linalg.norm(p[:, :2] - OBJ[:, :2], axis=1) * 1000


class TestSolver(unittest.TestCase):
    def test_nominal_stays_nominal(self):
        r = solver().solve(CAL, "head", scene_depth(T0), HOME, HAND)
        self.assertTrue(r.success, r.summary())
        self.assertLess(np.linalg.norm(r.dt_m), 0.002, r.summary())
        self.assertLess(r.drot_deg, 0.2, r.summary())

    def test_recovers_known_perturbation(self):
        rng = np.random.default_rng(0)
        for _ in range(3):
            T_true = hc.perturb(T0, rng.uniform(-0.01, 0.01, 3), rng.uniform(-1, 1, 3))
            r = solver().solve(CAL, "head", scene_depth(T_true), HOME, HAND)
            self.assertTrue(r.success, r.summary())
            self.assertLess(np.linalg.norm(r.T[:3, 3] - T_true[:3, 3]), 0.003, r.summary())
            self.assertLess(hc._rot_deg(T_true[:3, :3].T @ r.T[:3, :3]), 0.3, r.summary())
            self.assertLess(obj_xy_err_mm(T_true, r.T).max(), 3.0)
            self.assertLess(obj_xy_err_mm(T_true, r.T).max(), obj_xy_err_mm(T_true, T0).max())
        self.assertTrue(np.array_equal(CAL.cams["head"].T_pelvis_optical, T0))     # shared object untouched

    def test_pm1cm_1deg_object_xy_median(self):
        """+-1 cm / +-1 deg camera error: median object xy error after correction <= 1.5 mm."""
        rng = np.random.default_rng(7)
        errs = []
        for _ in range(8):
            T_true = hc.perturb(T0, rng.uniform(-0.01, 0.01, 3), rng.uniform(-1, 1, 3))
            r = solver().solve(CAL, "head", scene_depth(T_true), HOME, HAND)
            self.assertTrue(r.success, r.summary())
            errs.append(obj_xy_err_mm(T_true, r.T))
        self.assertLessEqual(float(np.median(np.concatenate(errs))), 1.5)

    def test_depth_holes_on_the_arms_fall_back(self):
        """60..100 % of the arm pixels have no depth: never crash, nominal pose, a logged reason."""
        T_true = hc.perturb(T0, [0.006, -0.004, 0.003], [0.5, -0.4, 0.3])
        d = scene_depth(T_true)
        arm = np.abs(d.astype(float) - hc.table_depth(T_true, K)) > 20       # arm pixels (depth units differ from table)
        self.assertGreater(arm.sum(), 1000)
        rng = np.random.default_rng(3)
        for frac in (0.6, 0.8, 0.9, 1.0):
            dd = d.copy()
            dd[arm & (rng.random(d.shape) < frac)] = 0
            r = solver().solve(CAL, "head", dd, HOME, HAND)
            self.assertFalse(r.success, f"{frac}: {r.summary()}")
            self.assertTrue(np.array_equal(r.T, T0), frac)
            self.assertTrue(r.reason, frac)

    def test_bad_data_falls_back_to_nominal(self):
        h, w = 480, 640
        bad = {"no depth": None,
               "all zero": np.zeros((h, w), np.uint16),
               "counter only": hc.table_depth(T0, K).clip(0, 60000).astype(np.uint16),
               "noise": np.random.default_rng(1).integers(300, 900, (h, w)).astype(np.uint16)}
        for name, d in bad.items():
            r = solver().solve(CAL, "head", d, HOME, HAND)
            self.assertFalse(r.success, name)
            self.assertTrue(np.array_equal(r.T, T0), name)
            self.assertTrue(r.reason, name)

    def test_arms_out_of_view(self):
        robot_home = np.asarray(load_config(overlay="none", profile="none")["home_arm_q"], float)
        r = solver().solve(CAL, "head", scene_depth(T0, robot_home), robot_home, HAND)
        self.assertFalse(r.success)
        self.assertTrue(np.array_equal(r.T, T0))

    def test_correction_cap(self):
        T_true = hc.perturb(T0, [0.04, -0.03, 0.02], [0, 0, 0])      # 5.4 cm: beyond the 3 cm cap
        r = solver().solve(CAL, "head", scene_depth(T_true), HOME, HAND)
        self.assertFalse(r.success, r.summary())
        self.assertTrue(np.array_equal(r.T, T0))


def settle(pol, o, n=400):
    """act() until the async hand calibration is decided (worker thread), like successive policy ticks."""
    for _ in range(n):
        pol.act(o)
        if pol._hc_done:
            return
        time.sleep(0.005)


class TestPrincipalPointReset(unittest.TestCase):
    def test_reset_keeps_the_principal_point_copy_with_the_flag_off(self):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None, overrides={"perception.principal_point": "center"})
        w = pol.cal.cams["head"].width
        self.assertEqual(pol.perception.cal.cams["head"].K[0, 2], w / 2.0)
        pol.reset()
        self.assertEqual(pol.perception.cal.cams["head"].K[0, 2], w / 2.0)
        self.assertEqual(pol.perception.cal.cams["head"].K[0, 2], 320.0)


class TestPolicyFlag(unittest.TestCase):
    def obs(self, T_true):
        return {"head_rgb": np.zeros((480, 640, 3), np.uint8), "head_depth": scene_depth(T_true), "arm_q": HOME.copy(),
                "hand_q": HAND.copy(), "t": 0.0}

    def test_flag_off_is_stock(self):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None)
        self.assertIsNone(pol.hand_calib)
        pol.act(self.obs(T0))
        pol.act(self.obs(T0))
        self.assertIs(pol.perception.cal, pol.cal)
        self.assertIsNone(pol.hand_calibration)

    def test_flag_on_corrects_a_copy_once_per_episode(self):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None, overrides={"perception.hand_calibration.enabled": True})
        T_true = hc.perturb(T0, [0.008, -0.006, 0.004], [0.6, -0.5, 0.4])
        settle(pol, self.obs(T_true))
        self.assertTrue(pol.hand_calibration.success, pol.hand_calibration.summary())
        self.assertIsNot(pol.perception.cal, pol.cal)
        self.assertTrue(np.array_equal(pol.cal.cams["head"].T_pelvis_optical, T0))       # shared object nominal
        T = pol.perception.cal.cams["head"].T_pelvis_optical
        self.assertLess(np.linalg.norm(T[:3, 3] - T_true[:3, 3]), 0.003)
        self.assertTrue(any("hand calibration accepted" in m for _, m in pol.events))
        pol.reset()
        self.assertIs(pol.perception.cal, pol.cal)                                          # next episode: nominal

    def test_chains_after_an_earlier_pose_correction(self):
        """perception.pitch_T (online_pitch_cal) is the start pose: the cap and the result are relative to it."""
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None, overrides={"perception.hand_calibration.enabled": True})
        T_a = hc.perturb(T0, [0.0, 0.0, 0.0], [-6.0, 0.0, 0.0])           # "A" corrected pitch (6 deg > the 3 deg cap)
        pol.perception.pitch_T = T_a.copy()
        T_true = hc.perturb(T_a, [0.005, -0.004, 0.003], [0.5, -0.4, 0.3])
        settle(pol, self.obs(T_true))
        self.assertTrue(pol.hand_calibration.success, pol.hand_calibration.summary())
        self.assertLess(np.linalg.norm(pol.perception.pitch_T[:3, 3] - T_true[:3, 3]), 0.003)
        self.assertLess(hc._rot_deg(T_true[:3, :3].T @ pol.perception.current_T()[:3, :3]), 0.3)

    def test_flag_on_gives_up_on_bad_frames(self):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None, overrides={"perception.hand_calibration.enabled": True,
                                                   "perception.hand_calibration.max_tries": 2})
        o = self.obs(T0)
        o["head_depth"] = np.zeros((480, 640), np.uint16)
        settle(pol, o)
        self.assertIs(pol.perception.cal, pol.cal)
        self.assertFalse(pol.hand_calibration.success)
        self.assertTrue(any("hand calibration fallback" in m for _, m in pol.events))

    def test_act_does_not_block_on_a_slow_solve(self):
        """A 0.6 s solve runs on a worker: the per-tick hook stays fast, the result lands afterwards."""
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None, overrides={"perception.hand_calibration.enabled": True})
        real = pol.hand_calib.solve

        def slow(*a, **k):
            time.sleep(0.6)
            return real(*a, **k)

        pol.hand_calib.solve = slow
        T_true = hc.perturb(T0, [0.005, -0.004, 0.003], [0.5, -0.4, 0.3])
        o = self.obs(T_true)
        pol._hand_calibrate(o, 0.0)                                  # first tick only records arm_q
        worst, t0 = 0.0, time.perf_counter()
        while not pol._hc_done and time.perf_counter() - t0 < 5:
            a = time.perf_counter()
            pol._hand_calibrate(o, 0.0)       # exactly what act() adds per tick (the policy's own first look is ~190 ms)
            worst = max(worst, time.perf_counter() - a)
            time.sleep(0.005)
        self.assertGreater(time.perf_counter() - t0, 0.4)          # the solve really was in flight across ticks
        self.assertLess(worst, 0.200, f"_hand_calibrate took {worst * 1000:.1f} ms while the solve ran")
        self.assertTrue(pol.hand_calibration.success, pol.hand_calibration.summary())

    def test_needs_consecutive_still_ticks(self):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None, overrides={"perception.hand_calibration.enabled": True})
        o = self.obs(T0)
        for _ in range(4):                                  # 1 recording tick + 3 still ticks < still_ticks 5
            pol._hand_calibrate(o, 0.0)
        self.assertIsNone(pol._hc_job)
        o2 = dict(o, hand_q=o["hand_q"] + 0.1)             # the hands move: the count restarts
        pol._hand_calibrate(o2, 0.0)
        self.assertEqual(pol._hc_still, 0)

    def test_warns_on_double_correction(self):
        from .policy import ClassicalPolicy
        msgs = []
        ClassicalPolicy(log=msgs.append, overrides={"perception.hand_calibration.enabled": True,
                                                    "perception.y_affine": [0.01, 0.0]})
        self.assertTrue(any("double-correction" in str(m) for m in msgs), msgs)

    def test_stuck_solve_times_out_to_nominal(self):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(log=None, overrides={"perception.hand_calibration.enabled": True,
                                                   "perception.hand_calibration.job_timeout_s": 0.2})
        pol.hand_calib.solve = lambda *a, **k: time.sleep(1.0)
        o = self.obs(T0)
        settle(pol, o, 200)
        self.assertTrue(pol._hc_done)
        self.assertIs(pol.perception.cal, pol.cal)
        self.assertTrue(any("timed out" in m for _, m in pol.events))


if __name__ == "__main__":
    unittest.main()
