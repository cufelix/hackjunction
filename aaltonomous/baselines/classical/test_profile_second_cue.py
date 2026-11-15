"""profile_auto.second_cue (cue B = temporal depth noise): real needs ONE cue (holes OR flicker), sim needs BOTH.
NOTE: organiser drop_2/drop_3 are RGB-only (no depth), so the 'real' depth here is a synthetic D435i model
(0.7 mm RMS temporal noise on a mm grid + a 3 % shadow band), not a recording."""
import os
import unittest
from unittest import mock

import numpy as np

from . import policy_server
from .config import load_config
from .policy_server import ClassicalServerPolicy
from .profile_detect import TEMP_CHANGED_FRAC_SIM_MAX, detect_profile, temporal_noise
from .test_profile_detect import H, W, SIM_OBJS, REAL_OBJS, _frame, _plane_mm, real_frame, sim_frame
from .test_profile_harden import OverridePolicy


def flicker_frames(n=5, sigma=0.7, holes=0.0, seed=0):
    """Same scene, per-frame independent depth noise on a 1 mm grid (D435i-like), optional hole fraction."""
    out = []
    for i in range(n):
        r = np.random.default_rng(seed + i)
        f = _frame(SIM_OBJS)
        z = np.round(f["head_depth"] + r.normal(0, sigma, f["head_depth"].shape))
        if holes:
            z[np.random.default_rng(99).random(z.shape) < holes] = 0.0   # same holes each frame (static shadow)
        f["head_depth"] = z
        out.append(f)
    return out


def det(frames, **kw):
    return detect_profile(frames, second_cue=True, **kw)


class TestCueB(unittest.TestCase):
    def test_sim_frames_sim(self):
        p, info = det([sim_frame() for _ in range(5)])
        self.assertEqual(p, "sim")
        self.assertEqual(info["cue_b_changed_frac"], 0.0)
        self.assertTrue(info["cue_a_sim"] and info["cue_b_sim"])

    def test_synthetic_real_frames_real(self):
        self.assertEqual(det([real_frame(i) for i in range(5)])[0], "none")

    def test_hole_free_but_flickering_depth_is_real(self):
        # the exact failure we guard: a real camera that reports 0 holes (hole-filled / post-processed depth)
        frames = flicker_frames(holes=0.0)
        self.assertEqual(detect_profile(frames)[0], "sim")            # old behaviour = the false sim
        p, info = det(frames)
        self.assertEqual(p, "none")
        self.assertGreater(info["cue_b_changed_frac"], 2 * TEMP_CHANGED_FRAC_SIM_MAX)

    def test_holes_but_frozen_frame_is_real(self):
        z = flicker_frames(1, holes=0.03)[0]
        self.assertEqual(det([z] * 5)[0], "none")           # cue B is quiet (identical), holes say real

    def test_hole_fraction_edges(self):
        for holes, expect in ((0.0, "sim"), (0.004, "sim"), (0.0051, "none"), (0.01, "none"), (0.02, "none")):
            f = sim_frame()
            f["head_depth"] = f["head_depth"].copy()
            f["head_depth"].reshape(-1)[: int(round(holes * H * W))] = 0.0
            self.assertEqual(det([f] * 5)[0], expect, holes)

    def test_exactly_half_percent_is_sim_boundary_of_cue_a(self):
        f = sim_frame()
        f["head_depth"] = f["head_depth"].copy()
        f["head_depth"].reshape(-1)[: int(0.005 * H * W)] = 0.0     # == 0.005: <= threshold, cue B quiet
        self.assertEqual(det([f] * 5)[0], "sim")

    def test_all_zero_depth_real(self):
        f = sim_frame()
        f["head_depth"] = np.zeros((H, W))
        self.assertEqual(det([f] * 5)[0], "none")

    def test_failure_modes_real(self):
        s = sim_frame()
        z = s["head_depth"]
        nan = z.copy()
        nan[::2] = np.nan
        for depth in (None, np.zeros_like(z), np.full_like(z, np.nan), z[..., None], nan, "x"):
            d = dict(s, head_depth=depth)
            try:
                self.assertEqual(det([d] * 5)[0], "none", repr(type(depth)))
            except Exception as e:   # the server wraps this; the detector must not be the thing that crashes silently
                self.fail(repr(e))
        self.assertEqual(det([])[0], "none")

    def test_single_frame_cue_b_unmeasurable_is_real(self):
        self.assertEqual(det([sim_frame()])[0], "none")
        self.assertEqual(detect_profile([sim_frame()])[0], "sim")     # flag off: unchanged

    def test_motion_pixels_do_not_hide_flicker_nor_fake_it(self):
        a = sim_frame()
        b = sim_frame()
        b["head_depth"] = b["head_depth"].copy()
        b["head_depth"][100:200, 100:300] -= 120.0     # an arm sweeping through: > static cap, excluded
        self.assertEqual(det([a, b, b, b, b])[0], "sim")

    def test_flag_off_identical(self):
        for fr in ([sim_frame()] * 5, [real_frame(i) for i in range(5)], flicker_frames(), []):
            a, ia = detect_profile(fr)
            b, ib = detect_profile(fr, second_cue=False)
            self.assertEqual((a, ia["votes"]), (b, ib["votes"]))

    def test_temporal_noise_unit(self):
        self.assertEqual(temporal_noise([sim_frame()["head_depth"]]), (None, None, 0))
        fr, ma, n = temporal_noise([f["head_depth"] for f in flicker_frames()])
        self.assertEqual(n, 4)
        self.assertGreater(fr, 0.2)


class TestServer(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("AALTO_PROFILE", None)
        self._p = mock.patch.object(policy_server, "ClassicalPolicy", OverridePolicy)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._env.stop()

    def run_server(self, frames, second=True, n=9, overlay="finals_r25", force=None):
        srv = ClassicalServerPolicy(overlay=overlay)
        if srv.probe_cfg is not None:
            srv.probe_cfg["profile_auto"]["second_cue"] = second
        srv.reset("i", task="sort_fixed", seed=0)
        with self.assertLogs("aaltonomous.baseline", "INFO") as cm:
            for i in range(n):
                srv.act({"t": i, **frames[min(i, len(frames) - 1)]})
        return srv, cm.output

    def test_flag_default_and_overlays(self):
        self.assertIs(load_config(profile="none")["profile_auto"]["second_cue"], False)
        for ov in ("finals_r25", "finals_r25_xy"):
            self.assertIs(load_config(profile="none", overlay=ov)["profile_auto"]["second_cue"], True, ov)
        self.assertIs(load_config(profile="none", overlay="finals_r23")["profile_auto"]["second_cue"], False)

    def test_server_sim_and_real_and_logs(self):
        srv, out = self.run_server([sim_frame()] * 5)
        self.assertEqual(srv.pol.cfg["profile"], "sim")
        self.assertTrue(any("second_cue" in m and "changed_frac=0.0" in m for m in out), out)
        srv, out = self.run_server(flicker_frames(holes=0.0))
        self.assertEqual(srv.pol.cfg["profile"], "none")

    def test_profile_never_flips_after_decision(self):
        srv, _ = self.run_server([sim_frame()] * 5, n=5)
        self.assertEqual(srv.pol.cfg["profile"], "sim")
        pol = srv.pol
        with mock.patch.object(policy_server, "detect_profile", side_effect=AssertionError("re-detected")):
            for i in range(20):
                srv.act({"t": 100 + i, **real_frame(i)})
        self.assertIs(srv.pol, pol)
        self.assertEqual(srv.pol.cfg["profile"], "sim")

    def test_detector_exception_defaults_to_real(self):
        with mock.patch.object(policy_server, "detect_profile", side_effect=RuntimeError("boom")):
            srv = ClassicalServerPolicy(overlay="finals_r25")
            srv.reset("i", task="sort_fixed", seed=0)
            with self.assertLogs("aaltonomous.baseline", "ERROR"):
                for i in range(5):
                    srv.act({"t": i, **sim_frame()})
        self.assertEqual(srv.pol.cfg["profile"], "none")

    def test_forced_profile_ignores_detection(self):
        for force, expect in (("sim", "sim"), ("none", "none")):
            with mock.patch.object(policy_server, "detect_profile", side_effect=AssertionError("detect ran")):
                import aaltonomous.baselines.classical.config as C
                real_load = C.load_config

                def lc(*a, **k):
                    c = real_load(*a, **k)
                    if k.get("profile") == "none" and not k.get("overrides"):
                        c["profile_auto"]["force"] = force
                    return c
                with mock.patch.object(C, "load_config", lc):
                    srv = ClassicalServerPolicy(overlay="finals_r25")
                    srv.reset("i", task="sort_fixed", seed=0)
                    for i in range(3):
                        srv.act({"t": i, **(real_frame(0) if force == "sim" else sim_frame())})
                self.assertEqual(srv.pol.cfg["profile"], expect)


if __name__ == "__main__":
    unittest.main()
