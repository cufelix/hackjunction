"""profile_detect: sim vs real from the first frames (profile_auto.auto_detect), and the ClassicalServerPolicy wiring."""
import os
import unittest
from unittest import mock

import numpy as np

from . import policy_server
from .config import load_config
from .policy_server import ClassicalServerPolicy
from .profile_detect import detect_profile, rgb_noise

H, W = 480, 640
# scene.yaml colours: red (200,30,30), blue (30,70,200) = hue ~113 (sim band), yellow (235,200,20).
# real-like: orange (255,140,0) hue ~11 (neutral), purple (93,60,200) hue ~127 (real), yellow.
SIM_OBJS = [((400, 460), (100, 160), (200, 30, 30)), ((380, 440), (300, 360), (30, 70, 200)),
            ((350, 410), (500, 560), (235, 200, 20))]
REAL_OBJS = [((400, 460), (100, 160), (255, 140, 0)), ((380, 440), (300, 360), (93, 60, 200)),
             ((350, 410), (500, 560), (235, 200, 20))]


def _plane_mm(noise_rng=None, holes=0.0):
    """Counter as an exact plane: 1/z affine in (u, v), z ~ 600-700 mm."""
    vv, uu = np.mgrid[0:H, 0:W].astype(float)
    inv = 1 / 640.0 + 2e-7 * (uu - 320) + 1e-7 * (vv - 240)
    z = 1.0 / inv
    if noise_rng is not None:
        z = z + noise_rng.normal(0, 3.0, z.shape)
    if holes:
        z[noise_rng.random(z.shape) < holes] = 0.0
    return z


def _frame(objs, depth_fn=None, rgb_bg=(60, 60, 60)):
    rgb = np.zeros((H, W, 3), np.uint8)
    rgb[:] = rgb_bg
    z = _plane_mm() if depth_fn is None else depth_fn()
    for (r0, r1), (c0, c1), col in objs:
        rgb[r0:r1, c0:c1] = col
        z[r0:r1, c0:c1] = z[r0:r1, c0:c1] - 45.0          # 45 mm cube on the counter
    return {"head_rgb": rgb, "head_depth": z}


def sim_frame():
    return _frame(SIM_OBJS)


def real_frame(seed=0):
    rng = np.random.default_rng(seed)
    return _frame(REAL_OBJS, depth_fn=lambda: _plane_mm(rng, holes=0.05))


class TestDetectProfile(unittest.TestCase):
    def test_synthetic_sim_frames_are_sim(self):
        profile, info = detect_profile([sim_frame() for _ in range(5)])
        self.assertEqual(profile, "sim")
        self.assertEqual((info["votes"], info["frames"], info["palette"]), (5, 5, "sim"))
        self.assertEqual(info["depth_holes"], 0.0)
        self.assertLess(info["plane_std_mm"], 1.0)

    def test_sim_depth_without_objects_in_view_is_sim(self):
        # palette inconclusive (no saturated object blobs) must not veto: depth says sim
        empty = [{"head_rgb": np.full((H, W, 3), 60, np.uint8), "head_depth": _plane_mm()} for _ in range(5)]
        profile, info = detect_profile(empty)
        self.assertEqual(profile, "sim")
        self.assertEqual(info["palette"], "inconclusive")

    def test_one_bad_frame_of_five_is_none(self):
        # unanimity: a single moving-arm / odd-depth frame (counter std 3 mm) sends the whole decision to none
        bad = _frame(SIM_OBJS, depth_fn=lambda: _plane_mm(np.random.default_rng(3)))
        profile, info = detect_profile([sim_frame(), sim_frame(), bad, sim_frame(), sim_frame()])
        self.assertEqual(profile, "none")
        self.assertEqual(info["votes"], 4)

    def test_four_sim_plus_one_noisy_frame_is_none(self):
        bad = _frame(SIM_OBJS, depth_fn=lambda: _plane_mm(np.random.default_rng(3)))
        self.assertEqual(detect_profile([sim_frame()] * 4 + [bad])[0], "none")

    def test_real_like_frames_are_none(self):
        profile, info = detect_profile([real_frame(i) for i in range(5)])
        self.assertEqual(profile, "none")
        self.assertEqual(info["votes"], 0)
        self.assertGreater(info["depth_holes"], 0.03)
        self.assertGreater(info["plane_std_mm"], 2.0)

    def test_real_depth_with_sim_colours_is_none(self):
        noisy = _frame(SIM_OBJS, depth_fn=lambda: _plane_mm(np.random.default_rng(4), holes=0.05))
        self.assertEqual(detect_profile([noisy] * 5)[0], "none")

    def test_purple_palette_is_log_only(self):
        # purple (real) measures inside the sim blue band; the palette must not change the vote
        profile, info = detect_profile([_frame(REAL_OBJS)] * 5)
        self.assertEqual(info["palette"], "real")
        self.assertEqual(profile, "sim")
        self.assertEqual(info["votes"], 5)

    def test_missing_depth_on_all_frames_is_none(self):
        frames = []
        for _ in range(5):
            f = sim_frame()
            f.pop("head_depth")
            frames.append(f)
        profile, info = detect_profile(frames)
        self.assertEqual(profile, "none")
        self.assertEqual(info["votes"], 0)

    def test_no_frames_is_none(self):
        self.assertEqual(detect_profile([])[0], "none")

    def test_rgb_noise_separates_flat_from_noisy_and_does_not_vote(self):
        flat = np.full((H, W, 3), 60, np.uint8)
        noisy = np.clip(flat + np.random.default_rng(0).normal(0, 3, (H, W, 3)), 0, 255).astype(np.uint8)
        self.assertLess(rgb_noise(flat), 0.5)
        self.assertGreater(rgb_noise(noisy), 1.0)
        # log-only: a noisy RGB with a clean sim depth still votes sim
        profile, info = detect_profile([{"head_rgb": noisy, "head_depth": _plane_mm()} for _ in range(5)])
        self.assertEqual(profile, "sim")
        self.assertGreater(info["rgb_noise"][0], 1.0)


class FakePolicy:
    """Stands in for ClassicalPolicy: real config per profile/overlay, no planner."""
    instances = []

    def __init__(self, threaded, require_confirm, time_limit, overrides, profile, overlay, log, **kw):
        self.cfg = load_config(profile=profile, overlay=overlay)
        self.fixed_mapping, self._job = {}, None
        self.resets, self.calls = [], []
        FakePolicy.instances.append(self)

    def reset(self, info):
        self.resets.append(info)

    def act(self, obs):
        self.calls.append(obs)
        return {"arm_q_target": np.zeros((1, 14)), "hand_q_target": np.zeros((1, 14)), "done": False}


def _info(profile, depth_holes=0.0, std=0.4, palette="sim", frames=5, votes=None):
    votes = (frames if profile == "sim" else 0) if votes is None else votes
    return profile, {"frames": frames, "votes": votes, "depth_holes": depth_holes, "plane_std_mm": std,
                     "palette": palette, "per_frame": []}


class TestServerAutoProfile(unittest.TestCase):
    def setUp(self):
        FakePolicy.instances = []
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("AALTO_PROFILE", None)
        self._p = mock.patch.object(policy_server, "ClassicalPolicy", FakePolicy)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._env.stop()

    def _server(self, overlay="finals_r22_auto", **kw):
        return ClassicalServerPolicy(overlay=overlay, **kw)

    def test_flag_on_sim_detection_builds_sim_and_replays_reset(self):
        calls = []
        obs = [{"t": 0.1 * i} for i in range(7)]
        with mock.patch.object(policy_server, "detect_profile",
                               side_effect=lambda frames: (calls.append(frames), _info("sim"))[1]):
            srv = self._server()
            self.assertEqual(srv.probe_cfg["profile_auto"]["auto_frames"], 5)
            srv.reset("pick the red cube", "sort_fixed", 7)
            self.assertEqual(FakePolicy.instances, [])
            for o in obs[:4]:
                out = srv.act(o)
                self.assertEqual(np.asarray(out["arm_q_target"]).shape, (0, 14))   # zero-row holds
            with self.assertLogs("aaltonomous.baseline", "INFO") as cm:
                out = srv.act(obs[4])
            self.assertEqual(np.asarray(out["arm_q_target"]).shape, (1, 14))
            self.assertTrue(any("[profile] auto=sim votes=5/5 (depth_holes=" in m for m in cm.output))
            srv.reset("again", "sort_fixed", 8)            # sticky: no re-detection
            srv.act(obs[5])
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 5)
        pol = FakePolicy.instances[-1]
        self.assertEqual(pol.cfg["profile"], "sim")
        self.assertEqual(pol.cfg["overlay"], "finals_r22_auto")
        self.assertEqual(pol.resets[0]["instruction"], "pick the red cube")
        self.assertEqual(pol.resets[0]["seed"], 7)
        self.assertEqual(pol.calls, [obs[4], obs[5]])
        self.assertEqual(pol.resets[-1]["instruction"], "again")

    def test_flag_on_none_detection_builds_real_profile(self):
        with mock.patch.object(policy_server, "detect_profile",
                               return_value=_info("none", 0.05, 3.0, "inconclusive", votes=2)):
            srv = self._server()
            for i in range(5):
                srv.act({"t": 0.1 * i})
        self.assertEqual(FakePolicy.instances[-1].cfg["profile"], "none")

    def test_flag_off_does_not_buffer_or_detect(self):
        with mock.patch.object(policy_server, "detect_profile", side_effect=AssertionError("no detection")):
            srv = self._server(overlay="finals_r22")
            self.assertIsNotNone(srv.pol)
            out = srv.act({"t": 0.0})
        self.assertEqual(np.asarray(out["arm_q_target"]).shape, (1, 14))
        self.assertEqual(srv.pol.cfg["profile"], "none")

    def test_explicit_env_wins_over_auto(self):
        os.environ["AALTO_PROFILE"] = "none"
        with mock.patch.object(policy_server, "detect_profile", side_effect=AssertionError("no detection")):
            srv = self._server()
        self.assertIsNotNone(srv.pol)
        self.assertEqual(srv.pol.cfg["profile"], "none")

    def test_explicit_arg_wins_over_auto(self):
        with mock.patch.object(policy_server, "detect_profile", side_effect=AssertionError("no detection")):
            srv = self._server(profile="none")
        self.assertEqual(srv.pol.cfg["profile"], "none")

    def test_explicit_sim_with_finals_still_refused(self):
        with self.assertRaises(RuntimeError):
            self._server(profile="sim", overlay="finals_r22")

    def test_auto_sim_on_finals_overlay_is_allowed(self):
        with mock.patch.object(policy_server, "detect_profile", return_value=_info("sim")):
            srv = self._server()
            for i in range(5):
                srv.act({"t": 0.1 * i})
        self.assertEqual(srv.pol.cfg["profile"], "sim")


if __name__ == "__main__":
    unittest.main()
