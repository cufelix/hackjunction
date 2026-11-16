"""Venue 2026-11-15 causes 3+4: bin.vote tape_body_reject + voter logging, grasp.wrong_object_recovery. Stubs only (no URDF). Run:
    python -m unittest aaltonomous.baselines.classical.test_wrong_object_recovery -v
"""
import copy
import unittest

import numpy as np

from aaltonomous.scene.scene import Scene

from .bin_vote import tape_votes, vote_bins
from .config import load_config
from .held import HeldDetector
from .policy import ClassicalPolicy
from .test_held_detector import trace

SC = Scene.load()
TAPE_RGB = [150, 52, 32]   # OpenCV HSV ~[5, 201, 150], inside bin.vote.tape_hsv (and the venue's paper-wrapped bin colour)


class _Perc:
    """Pixel (u, v) -> pelvis point on the centre bin's rim plane; z linear in depth so the depth-less rim-height solve works."""
    scene = SC
    cfg = {"depth_range_mm": [100, 3000]}

    def _pixel_to_pelvis(self, u, v, d, transform_snapshot=None):
        u, d = np.asarray(u, float), np.asarray(d, float)
        rim = SC.counter_z + float(SC.cfg["bins"]["wall_height"])
        return np.c_[np.full(len(u), 0.505), (u - 12) * 0.001, rim + (d - 1500.0) * 1e-3]


def _vote(rgb, **over):
    cfg = load_config(overlay="finals_r25_gen", overrides=over)
    v = dict(cfg["bin"]["vote"], min_tape_px=20)
    return tape_votes(_Perc(), rgb, None, {"center": np.array([0.505, 0.0])}, v)["center"]


def _rgb(side):
    rgb = np.zeros((120, 120, 3), np.uint8)
    rgb[10:10 + side, 10:10 + side] = TAPE_RGB
    return rgb


class TestTapeBodyReject(unittest.TestCase):
    def test_flag_registered_default_off_enabled_in_finals_r25_gen(self):
        self.assertFalse(load_config(overlay="none")["bin"]["vote"]["tape_body_reject"])
        self.assertTrue(load_config(overlay="finals_r25_gen")["bin"]["vote"]["tape_body_reject"])
        with self.assertRaises(ValueError):
            load_config(overrides={"bin.vote.tape_body_reject": "yes"})

    def test_bin_body_sized_blob_votes_only_with_the_flag_off(self):
        body = _rgb(80)   # 6400 px > tape_max_px 3000
        self.assertIsNotNone(_vote(body, **{"bin.vote.tape_body_reject": False}))
        self.assertIsNone(_vote(body, **{"bin.vote.tape_body_reject": True}))

    def test_real_tape_sized_blob_still_votes_with_the_flag_on(self):
        self.assertIsNotNone(_vote(_rgb(30), **{"bin.vote.tape_body_reject": True}))

    def test_body_blob_does_not_hide_a_small_tape_blob(self):
        rgb = _rgb(60)
        rgb[20:35, 100:115] = TAPE_RGB
        diag = {}
        cfg = load_config(overlay="finals_r25_gen")
        v = dict(cfg["bin"]["vote"], min_tape_px=20)
        tape_votes(_Perc(), rgb, None, {"center": np.array([0.505, 0.0])}, v, diag=diag)
        self.assertEqual(diag["rejected_body_px"], [3600])
        self.assertEqual(diag["blob_px"], [3600, 225])


class _P:   # minimal planner/perception for vote_bins
    scene = SC

    def __init__(self):
        self.bin_names = SC.bin_names

    def bin_xy(self, n):
        return np.asarray(SC.mat_to_pelvis(np.r_[SC.cfg["bins"]["centers"][n], 0.0])[:2], float)

    def frame_transform(self):
        return None

    def use_frame_transform(self, f):
        import contextlib
        return contextlib.nullcontext()


class _Perc2(_Perc):
    def frame_transform(self):
        return None

    def use_frame_transform(self, f):
        import contextlib
        return contextlib.nullcontext()


class TestVoterLogging(unittest.TestCase):
    def test_vote_bins_logs_every_voter_and_why_one_is_missing(self):
        planner = _P()
        cfg = load_config(overlay="finals_r25_gen", overrides={"bin.vote.frame_transform_snapshot": False})
        logs = []
        vote_bins(_Perc2(), planner, [{"head_rgb": _rgb(30), "head_depth": None}], cfg, log=logs.append)
        txt = "\n".join(logs)
        for n in planner.bin_names:
            self.assertIn(f"[bin.vote.voters] {n}: prior=(", txt)
        self.assertIn("rim=none", txt)
        self.assertIn("frame0 rim: {'why': 'no depth'}", txt)
        self.assertIn("blob_px", txt)
        self.assertTrue(any(m.startswith("[bin.vote] ") for m in logs))   # the old summary line is unchanged


def _pol(**params):
    pol = ClassicalPolicy.__new__(ClassicalPolicy)
    cfg = load_config(overlay="finals_r25_gen")
    cfg["grasp"]["wrong_object_recovery_params"].update(params)
    pol.cfg = cfg
    pol.logs = []
    pol.log = pol.logs.append
    pol._bad_grasp, pol._wor_recoveries, pol._rel_fail = [], 0, False
    det = HeldDetector("finger_sum", 0.045)
    det._reference_count = 1            # bias reference present (e0 = 0)
    pol.held_detector = det
    return pol


def _g(side="right", close=None):
    plan = type("Plan", (), {"side": side, "color": "red", "waypoints": {"grasp": np.array([0.42, -0.21, 0.04])}})()
    return {"plan": plan, "t": 11.9, "samples": {"close": close} if close else {}}


class TestLiftMargin(unittest.TestCase):
    def test_flag_default_off_enabled_in_gen_overlay(self):
        self.assertFalse(load_config(overlay="none")["grasp"]["wrong_object_recovery"])
        self.assertTrue(load_config(overlay="finals_r25_gen")["grasp"]["wrong_object_recovery"])
        with self.assertRaises(ValueError):
            load_config(overrides={"grasp.wrong_object_recovery": 1})

    def test_venue_bin_wall_reads_are_rejected_and_blacklisted(self):
        for lift in (0.046, 0.088):    # run 1 / run 2 lift reads, threshold 0.045
            pol = _pol()
            cmd, q = trace("right", 0.0, lift)
            self.assertTrue(pol.held_detector.held("right", cmd, q)[0])   # the old detector accepted both
            self.assertTrue(pol._lift_margin_fail(_g(), q, cmd))
            self.assertTrue(pol._bad_grasp_near((0.43, -0.20)))
            self.assertFalse(pol._bad_grasp_near((0.30, -0.05)))
            self.assertTrue(any("[wrong_object]" in m for m in pol.logs))

    def test_clear_grip_passes_and_is_not_blacklisted(self):
        pol = _pol()
        cmd, q = trace("right", 0.0, 0.15)
        self.assertFalse(pol._lift_margin_fail(_g(), q, cmd))
        self.assertEqual(pol._bad_grasp, [])

    def test_ratio_rule_is_optional_and_uses_the_close_read(self):
        close = trace("right", 0.0, 0.30)[::-1]            # samples hold (tick, hq, hc)
        cmd, q = trace("right", 0.0, 0.12)
        g = _g(close=(300, close[0], close[1]))
        self.assertFalse(_pol()._lift_margin_fail(g, q, cmd))                       # off by default
        self.assertTrue(_pol(lift_min_ratio=0.5)._lift_margin_fail(g, q, cmd))      # 0.12 < 0.5 x 0.30

    def test_flag_off_unsure_and_soft_never_fail(self):
        cmd, q = trace("right", 0.0, 0.046)
        pol = _pol()
        pol.cfg["grasp"]["wrong_object_recovery"] = False
        self.assertFalse(pol._lift_margin_fail(_g(), q, cmd))
        self.assertFalse(pol._bad_grasp_near((0.42, -0.21)))
        pol = _pol()
        pol.held_detector = HeldDetector("finger_sum", 0.045)        # no bias reference
        self.assertFalse(pol._lift_margin_fail(_g(), q, cmd))
        pol = _pol()
        self.assertFalse(pol._lift_margin_fail(_g(), q[:7], cmd))   # missing data
        self.assertFalse(pol._lift_margin_fail(_g(), np.full(14, np.nan), cmd))
        pol.held_detector.soft = {"rest": np.zeros(14)}
        self.assertFalse(pol._lift_margin_fail(_g(), q, cmd))


class _Planner:
    def go_to(self, arm, hand, home, post):
        return np.zeros((4, 14)), np.zeros((4, 14))


class TestReleaseRecovery(unittest.TestCase):
    def _stuck(self, **params):
        pol = _pol(**params)
        pol.planner = _Planner()
        pol.home = np.zeros(14)
        pol.post = {"open": np.zeros(14)}
        pol.arm_cmd, pol.hand_cmd = np.ones(14), np.zeros(14)
        pol._held_hand_lo, pol._held_hand_hi = np.full(14, -2.0), np.full(14, 2.0)
        pol._pop = lambda: "popped"
        plan = _g()["plan"]
        pol._rel = {"plan": plan, "phase": "check", "t0": 0.0, "reopened": True, "rest": (None, None), "n": 10, "open_end": 1}
        return pol

    def _hand_stuck(self):
        cmd, q = trace("right", 0.0, 0.2)
        return {"hand_q": 2 * cmd - q}, cmd   # fingers still closed past the "open" command: not empty

    def _step(self, pol, t=5.0):
        obs, cmd = self._hand_stuck()
        pol.hand_cmd = cmd
        return pol._release_confirm(obs, t)

    def test_failed_release_continues_up_to_max_recoveries_then_stops(self):
        pol = self._stuck()
        for i in (1, 2):
            self.assertEqual(self._step(pol), "popped")
            self.assertFalse(pol._rel_fail, i)
            self.assertEqual(pol._wor_recoveries, i)
            self.assertIsNone(pol._rel)
            pol._rel = self._stuck()._rel
        self.assertEqual(self._step(pol), "popped")
        self.assertTrue(pol._rel_fail)             # third failure: stop as before
        self.assertEqual(len(pol._bad_grasp), 2)   # the two recovered grasps are blacklisted

    def test_flag_off_stops_like_before(self):
        pol = self._stuck()
        pol.cfg["grasp"]["wrong_object_recovery"] = False
        self.assertEqual(self._step(pol), "popped")
        self.assertTrue(pol._rel_fail)
        self.assertEqual(pol._wor_recoveries, 0)
        self.assertEqual(pol._bad_grasp, [])

    def test_recovery_uses_the_existing_open_home_path_only(self):
        pol = self._stuck()
        calls = []
        pol.planner.go_to = lambda *a: calls.append(a) or (np.zeros((4, 14)), np.zeros((4, 14)))
        self._step(pol)
        self.assertEqual(len(calls), 1)
        np.testing.assert_array_equal(calls[0][3], pol.post["open"])   # fully open hand
        self.assertTrue(any("recovery 1/2" in m for m in pol.logs))


if __name__ == "__main__":
    unittest.main()
