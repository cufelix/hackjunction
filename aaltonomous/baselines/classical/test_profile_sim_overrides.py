"""profile_auto.sim_overrides: applied as load_config overrides only when auto-detect picks sim."""
import os
import unittest
from unittest import mock

from . import config, policy_server
from .config import load_config
from .policy_server import ClassicalServerPolicy
from .test_profile_detect import sim_frame


class OverridePolicy:
    """Stands in for ClassicalPolicy: real config per profile/overlay/overrides, no planner."""
    def __init__(self, threaded, require_confirm, time_limit, overrides, profile, overlay, log, **kw):
        self.cfg = load_config(profile=profile, overlay=overlay, overrides=overrides)
        self.fixed_mapping = dict(self.cfg["policy"].get("fixed_mapping") or {})

    def reset(self, info):
        pass

    def act(self, obs):
        return {"arm_q_target": [[0.0] * 14], "hand_q_target": [[0.0] * 14], "done": False}


def _info(profile):
    votes = 5 if profile == "sim" else 0
    return profile, {"frames": 5, "votes": votes, "depth_holes": 0.0, "plane_std_mm": 0.4, "palette": profile,
                     "per_frame": []}


class TestSimOverrides(unittest.TestCase):
    def setUp(self):
        self._env = mock.patch.dict(os.environ, {}, clear=False)
        self._env.start()
        os.environ.pop("AALTO_PROFILE", None)
        self._p = mock.patch.object(policy_server, "ClassicalPolicy", OverridePolicy)
        self._p.start()

    def tearDown(self):
        self._p.stop()
        self._env.stop()

    def _decide(self, profile):
        with mock.patch.object(policy_server, "detect_profile", return_value=_info(profile)):
            srv = ClassicalServerPolicy(overlay="finals_r23")
            for i in range(5):
                srv.act({"t": i})
        return srv.pol.cfg

    def test_auto_sim_applies_rc1_values(self):
        cfg = self._decide("sim")
        self.assertIsNone(cfg["policy"]["fixed_mapping"])
        self.assertIsNone(cfg["scene"]["counter_z_prior"])
        self.assertFalse(cfg["grasp"]["release_confirm"])
        self.assertTrue(cfg["perception"]["gate_below_bins"])   # not listed: overlay value kept

    def test_auto_none_keeps_finals_values(self):
        cfg = self._decide("none")
        self.assertEqual(cfg["policy"]["fixed_mapping"], {"yellow": "left", "blue": "center", "red": "right"})
        self.assertEqual(cfg["scene"]["counter_z_prior"], 0.15)
        self.assertTrue(cfg["grasp"]["release_confirm"])

    def test_image_cmd_path_runs_auto_detect_and_sim_overrides(self):
        # mirrors policy_server __main__ in the image: no AALTO_PROFILE, profile=None, baked-style overlay
        os.environ.pop("AALTO_PROFILE", None)
        pol = ClassicalServerPolicy(profile=os.environ.get("AALTO_PROFILE") or None, overlay="finals_r23")
        self.assertIsNone(pol.pol)   # auto-detect pending, not skipped
        with self.assertLogs("aaltonomous.baseline", "INFO") as cm:
            for i in range(5):
                pol.act({"t": i, **sim_frame()})
        self.assertIsNotNone(pol.pol)
        self.assertTrue(any("[profile] auto=sim" in m for m in cm.output), cm.output)
        self.assertTrue(any("sim_overrides applied" in m for m in cm.output), cm.output)
        self.assertIs(pol.pol.cfg["calib"]["plane"]["apply_height"], False)

    def test_bad_key_fails_at_construction(self):
        orig = config.load_config

        def with_typo(**kw):
            c = orig(**kw)
            c["profile_auto"]["sim_overrides"]["policy.no_such_key"] = 1
            return c

        with mock.patch.object(config, "load_config", side_effect=with_typo):
            with self.assertRaises(KeyError):
                ClassicalServerPolicy(overlay="finals_r23")


if __name__ == "__main__":
    unittest.main()
