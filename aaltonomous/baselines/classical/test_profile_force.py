"""profile_auto.force (Dockerfile ARG PROFILE_MODE): force=sim == auto on a sim decision, force=none == auto on a real
decision (effective config and every action byte-identical); force beats detection; auto (null) is unchanged."""
import hashlib
import os
import unittest
from unittest import mock

import numpy as np

from . import policy_server
from .config import load_config
from .policy_server import ClassicalServerPolicy


class CfgPolicy:
    """ClassicalPolicy stand-in: real config per profile/overlay/overrides; actions are a function of cfg + obs."""
    def __init__(self, threaded, require_confirm, time_limit, overrides, profile, overlay, log, **kw):
        self.cfg = load_config(profile=profile, overlay=overlay, overrides=overrides)
        self.fixed_mapping = {}
        self._job = None
        self._d = hashlib.sha256(repr(sorted(self.cfg["policy"].items(), key=str)).encode()).digest()

    def reset(self, info):
        pass

    def act(self, obs):
        r = np.random.default_rng(int.from_bytes(self._d[:4], "little") + int(obs["t"] * 100))
        return {"arm_q_target": r.normal(size=(3, 14)), "hand_q_target": r.normal(size=(3, 14)), "done": False}


def _info(profile):
    return profile, {"frames": 5, "votes": 5 if profile == "sim" else 0, "depth_holes": 0.0, "plane_std_mm": 0.4,
                     "palette": profile, "per_frame": []}


OBS = [{"t": 0.1 * i} for i in range(8)]


def _run(srv, n_hold):
    outs = [srv.act(o) for o in OBS]
    return outs[n_hold:]


class TestProfileForce(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("AALTO_PROFILE", None)
        self._p = mock.patch.object(policy_server, "ClassicalPolicy", CfgPolicy)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._env.stop()

    def _auto(self, decided, overlay="finals_r24"):
        with mock.patch.object(policy_server, "detect_profile", return_value=_info(decided)):
            srv = ClassicalServerPolicy(overlay=overlay)
            srv.reset("x", "sort_fixed", 1)
            outs = _run(srv, 4)    # frames 1-4 are zero-row holds; the decision frame (5th) is the first real action
        return srv, outs

    def _forced(self, mode, overlay="finals_r24"):
        with mock.patch.object(policy_server, "detect_profile", side_effect=AssertionError("detection must not run")):
            srv = self._forced_server(mode, overlay)
            srv.reset("x", "sort_fixed", 1)
            outs = _run(srv, 4)
        return srv, outs

    def _forced_server(self, mode, overlay):
        from . import config
        orig = config.load_config

        def patched(*a, **k):
            cfg = orig(*a, **k)
            if k.get("profile") == "none" and "overrides" not in k:   # the server's flags-only probe
                cfg["profile_auto"]["force"] = mode
            return cfg
        with mock.patch.object(config, "load_config", patched):
            return ClassicalServerPolicy(overlay=overlay)

    def _same(self, a, b):
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            for k in ("arm_q_target", "hand_q_target"):
                self.assertEqual(np.asarray(x[k]).tobytes(), np.asarray(y[k]).tobytes())
            self.assertEqual(x["done"], y["done"])

    def test_force_sim_identical_to_auto_sim(self):
        sa, oa = self._auto("sim")
        sf, of = self._forced("sim")
        self.assertEqual(sf.pol.cfg, sa.pol.cfg)
        self.assertEqual(sf.pol.cfg["profile"], "sim")
        self.assertFalse(sf.pol.cfg["policy"]["safety_shell"])        # a sim_overrides key really applied
        self._same(oa, of)

    def test_force_none_identical_to_auto_real(self):
        sa, oa = self._auto("none")
        sf, of = self._forced("none")
        self.assertEqual(sf.pol.cfg, sa.pol.cfg)
        self.assertEqual(sf.pol.cfg["profile"], "none")
        self.assertTrue(sf.pol.cfg["policy"]["safety_shell"])
        self._same(oa, of)

    def test_force_ignores_a_contradicting_detection(self):
        # forced real while the frames look like sim: detect_profile is never consulted
        sf, _ = self._forced("none")
        self.assertEqual(sf.pol.cfg["profile"], "none")

    def test_null_force_keeps_auto_detect(self):
        with mock.patch.object(policy_server, "detect_profile", return_value=_info("sim")):
            srv = ClassicalServerPolicy(overlay="finals_r24")
            self.assertIsNone(srv.pol)       # pending: detection after 5 frames
            self.assertIsNone(srv.probe_cfg["profile_auto"]["force"])

    def test_bad_value_rejected(self):
        with self.assertRaises(Exception):
            load_config(profile="none", overlay="finals_r24", overrides={"profile_auto.force": "real"})

    def test_overlays_have_force_null_and_addons_off(self):
        for ov in ("finals_r24", "finals_r24_classic"):
            c = load_config(profile="none", overlay=ov)
            self.assertIsNone(c["profile_auto"]["force"])
            self.assertFalse(c["profile_auto"]["require_rgb_cue"])
            self.assertEqual(c["grasp"]["left_hand_fix"]["shift_z"], 0.01)   # A102 decision
            self.assertEqual(c["scene"]["counter_z_prior"], 0.03)   # real FK, drop_3
            self.assertEqual(c["grasp"]["arm_xy_offset"], {"left": [0.0, 0.0], "right": [0.0, 0.0]})
            for k in ("retry_cross_relocate", "late_sweep", "retry_repeat_ok", "late_regrant", "frozen_rgb_guard"):
                self.assertFalse(c["policy"][k], k)
            self.assertFalse(c["planner"]["carry_corridor"])
            so = c["profile_auto"]["sim_overrides"]
            for k in ("policy.frozen_rgb_guard", "planner.carry_corridor", "perception.real_merge_split",
                      "perception.real_border_partial", "perception.split_by_solidity",
                      "perception.prism_speckle_floor", "perception.color_gate.fk_silhouette"):
                self.assertIs(so[k], False, k)
        c = load_config(profile="none", overlay="finals_r24_classic")
        self.assertFalse(c["brain"]["enabled"])
        self.assertFalse(c["perception"]["novel_objects"])


class TestRealDemoValues(unittest.TestCase):
    """drop_3 real-demo values: ON in the real profile only; sim (via profile_auto.sim_overrides) keeps r23 values."""
    OLD = {"grasp.point_in_palm": [0.0862, -0.0864, 0.0143], "hand.closed_tight": [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6],
           "grasp.close_posture_by_shape": True, "grasp.drop_height_extra_m": 0.0, "grasp.approach_slow_mps": 0.0}

    def _g(self, cfg, k):
        for kk in k.split("."):
            cfg = cfg[kk]
        return cfg

    def test_real_gets_demo_values_sim_keeps_old(self):
        for ov in ("finals_r24", "finals_r24_classic"):
            real = load_config(profile="none", overlay=ov)
            so = real["profile_auto"]["sim_overrides"]
            sim = load_config(profile="sim", overlay=ov, overrides=so)
            self.assertEqual(self._g(real, "grasp.point_in_palm"), [0.1094, -0.0694, 0.0142])
            self.assertEqual(self._g(real, "hand.closed_tight"), [0.0, 0.84, 0.21, -0.79, -1.24, -0.62, -1.23])
            self.assertFalse(self._g(real, "grasp.close_posture_by_shape"))
            self.assertEqual(self._g(real, "grasp.drop_height_extra_m"), 0.06)
            self.assertEqual(self._g(real, "grasp.approach_slow_mps"), 0.05)
            self.assertEqual(self._g(real, "perception.online_pitch_cal"), "table")
            self.assertEqual(self._g(real, "calib.plane.apply_height"), True)
            self.assertEqual(self._g(real, "bin.vote.max_shift_m"), 0.18)
            self.assertEqual(self._g(real, "scene.counter_z_prior"), 0.03)
            self.assertIsNone(self._g(sim, "scene.counter_z_prior"))
            self.assertIs(self._g(sim, "perception.online_pitch_cal"), False)
            for k, v in self.OLD.items():
                self.assertEqual(self._g(sim, k), v, k)

    def test_thumb0_is_zero_in_every_mirrored_posture(self):
        # the right-hand mirror flips th0's sign, which is wrong per FK: th0 must stay 0 (TODO: fix the mirror, then lift)
        for ov in ("finals_r24", "finals_r24_classic"):
            cfg = load_config(profile="none", overlay=ov)
            self.assertEqual(cfg["hand"]["closed_tight"][0], 0.0)
            self.assertEqual(cfg["hand"]["closed_demo"][0], 0.0)


if __name__ == "__main__":
    unittest.main()
