"""profile_auto.require_rgb_cue (default OFF): second sim cue, failure modes of the sim/real auto-detect."""
import os
import unittest
from unittest import mock

import numpy as np

from . import policy_server
from .config import effective_flags, load_config
from .policy_server import ClassicalServerPolicy
from .profile_detect import RGB_SIM_MAX, detect_profile, rgb_noise
from .test_profile_detect import H, W, real_frame, sim_frame
from .test_profile_sim_overrides import OverridePolicy as _OP


class OverridePolicy(_OP):
    _job = None


def noisy_rgb(f, sigma=2.5, seed=0):
    rng = np.random.default_rng(seed)
    g = dict(f)
    g["head_rgb"] = np.clip(f["head_rgb"].astype(float) + rng.normal(0, sigma, f["head_rgb"].shape), 0, 255).astype(np.uint8)
    return g


def with_(f, **kw):
    g = dict(f)
    g.update(kw)
    return g


def decide(frames, cue=True):
    return detect_profile(frames, require_rgb_cue=cue)


def five(f):
    return [f] * 5


class TestRgbCue(unittest.TestCase):
    def test_flag_default_off(self):
        self.assertIs(load_config(profile="none", overlay="finals_r23")["profile_auto"]["require_rgb_cue"], False)
        self.assertTrue(any(k == "profile_auto.require_rgb_cue" and not on for k, v, on, *_ in effective_flags(
            load_config(profile="none", overlay="finals_r23"))))

    def test_sim_still_sim(self):
        p, info = decide(five(sim_frame()))
        self.assertEqual((p, info["votes"], info["disagree"]), ("sim", 5, 0))
        self.assertLessEqual(rgb_noise(sim_frame()["head_rgb"]), RGB_SIM_MAX)

    def test_clean_depth_noisy_rgb_is_real_only_with_flag(self):
        f = noisy_rgb(sim_frame())               # hole-filled real depth + sensor-noisy rgb
        self.assertEqual(decide(five(f), cue=False)[0], "sim")   # old behaviour: the false sim
        p, info = decide(five(f))
        self.assertEqual((p, info["votes"], info["disagree"]), ("none", 0, 5))

    def test_real_frames_none(self):
        self.assertEqual(decide([real_frame(i) for i in range(5)])[0], "none")

    def test_black_and_flat_rgb_not_sim(self):
        for rgb in (np.zeros((H, W, 3), np.uint8), np.full((H, W, 3), 90, np.uint8)):
            self.assertEqual(decide(five(with_(sim_frame(), head_rgb=rgb)))[0], "none")

    def test_missing_or_malformed_rgb(self):
        s = sim_frame()
        bad = [None, np.zeros((H, W), np.uint8), np.zeros((4, 4, 3), np.uint8), "x",
               np.full((H, W, 3), np.nan, np.float32)]
        for rgb in bad:
            self.assertEqual(decide(five(with_(s, head_rgb=rgb)))[0], "none", repr(type(rgb)))
        d = dict(s)
        del d["head_rgb"]
        self.assertEqual(decide(five(d))[0], "none")

    def test_depth_failure_modes_stay_none(self):
        s = sim_frame()
        z = s["head_depth"]
        nan = z.copy()
        nan[::2] = np.nan
        for depth in (None, np.zeros_like(z), np.full_like(z, np.nan), z[..., None], nan):
            self.assertEqual(decide(five(with_(s, head_depth=depth)))[0], "none")

    def test_depth_dtype_and_resolution(self):
        s = sim_frame()
        u16 = with_(s, head_depth=np.round(s["head_depth"]).astype(np.uint16))
        self.assertEqual(decide(five(u16))[0], "sim")       # exact uint16 mm depth, quiet rgb: still sim
        f32 = with_(s, head_depth=s["head_depth"].astype(np.float32) / 1000.0)
        self.assertEqual(decide(five(f32))[0], "sim")
        small = with_(s, head_depth=s["head_depth"][::2, ::2])   # other resolution: must not raise
        self.assertEqual(decide(five(small))[0], "sim")

    def test_black_first_frames_majority(self):
        s, black = sim_frame(), with_(sim_frame(), head_rgb=np.zeros((H, W, 3), np.uint8))
        self.assertEqual(decide([black, black, s, s, s])[0], "sim")    # 3/5
        self.assertEqual(decide([black, black, black, s, s])[0], "none")  # 2/5

    def test_frozen_real_frame_with_noise_is_not_sim(self):
        f = noisy_rgb(sim_frame())
        self.assertEqual(decide([f] * 5)[0], "none")

    def test_empty(self):
        self.assertEqual(decide([])[0], "none")


class TestServerWiring(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("AALTO_PROFILE", None)
        self._p = mock.patch.object(policy_server, "ClassicalPolicy", OverridePolicy)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._env.stop()

    def run_server(self, frames, cue):
        srv = ClassicalServerPolicy(overlay="finals_r23")
        srv.probe_cfg["profile_auto"]["require_rgb_cue"] = cue
        srv.reset("i", task="sort_fixed", seed=0)
        outs = []
        with self.assertLogs("aaltonomous.baseline", "INFO") as cm:
            for i, f in enumerate(frames):
                outs.append(srv.act({"t": i, **f}))
        return srv, outs, cm.output

    def test_no_motion_before_decision(self):
        srv, outs, _ = self.run_server(five(noisy_rgb(sim_frame())), cue=True)
        for o in outs[:4]:
            self.assertEqual(np.asarray(o["arm_q_target"]).shape, (0, 14))
            self.assertEqual(np.asarray(o["hand_q_target"]).shape, (0, 14))
            self.assertFalse(o["done"])

    def test_disagreement_builds_real_and_warns(self):
        srv, _, out = self.run_server(five(noisy_rgb(sim_frame())), cue=True)
        self.assertEqual(srv.pol.cfg["profile"], "none")
        self.assertTrue(any("rgb disagrees" in m for m in out), out)
        self.assertEqual(srv.pol.cfg["policy"]["fixed_mapping"], {"yellow": "left", "blue": "center", "red": "right"})

    def test_flag_off_unchanged(self):
        srv, _, out = self.run_server(five(noisy_rgb(sim_frame())), cue=False)
        self.assertEqual(srv.pol.cfg["profile"], "sim")

    def test_clean_sim_builds_sim_and_is_not_redecided_on_reset(self):
        srv, _, _ = self.run_server(five(sim_frame()), cue=True)
        self.assertEqual(srv.pol.cfg["profile"], "sim")
        srv.reset("i", task="sort_fixed", seed=1)           # decided once per process: noisy frames later change nothing
        for i in range(10):
            srv.act({"t": i, **noisy_rgb(sim_frame())})
        self.assertEqual(srv.pol.cfg["profile"], "sim")


if __name__ == "__main__":
    unittest.main()
