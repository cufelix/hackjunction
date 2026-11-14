"""C2/L1: the server is real-profile by default and finals configs are audited under both profiles."""
import copy
import os
import unittest
from unittest import mock

from .config import OVERLAY_DIR, effective_flags, load_config
from . import policy_server


class _ConfigOnlyPolicy:
    def __init__(self, config=None, overrides=None, profile=None, overlay=None, **_kw):
        self.cfg = (config if isinstance(config, dict)
                    else load_config(config, overrides=overrides, profile=profile, overlay=overlay))


class TestPolicyServerProfile(unittest.TestCase):
    def server(self, **kw):
        with mock.patch.object(policy_server, "ClassicalPolicy", _ConfigOnlyPolicy):
            return policy_server.ClassicalServerPolicy(**kw)

    def test_server_defaults_to_real_profile_and_snapshots_effective_config(self):
        with mock.patch.dict(os.environ, {}, clear=True), self.assertLogs("aaltonomous.baseline", "INFO") as logs:
            server = self.server(overlay="none")
        self.assertEqual(server.config_snapshot, load_config(profile="none", overlay="none"))
        self.assertIsNot(server.config_snapshot, server.pol.cfg)
        self.assertIn("profile=none", "\n".join(logs.output))
        self.assertIn("arm_vel=0.500", "\n".join(logs.output))

    def test_sim_remains_an_explicit_nonfinals_option(self):
        server = self.server(profile="sim", overlay="none")
        self.assertEqual(server.config_snapshot["profile"], "sim")
        self.assertEqual(server.config_snapshot["motion"]["arm_max_acc"], 8.0)
        with mock.patch.dict(os.environ, {"AALTO_PROFILE": "sim"}, clear=True):
            from_env = self.server(overlay="none")
        self.assertEqual(from_env.config_snapshot["profile"], "sim")

    def test_finals_sim_profile_fails_loudly(self):
        with self.assertRaisesRegex(RuntimeError, "refusing finals overlay .* profile 'sim'"):
            self.server(profile="sim", overlay="finals_r21")
        with mock.patch.dict(os.environ, {"AALTO_PROFILE": "sim"}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "refusing finals overlay .* profile 'sim'"):
                self.server(overlay="finals_r21")
        cfg = load_config(profile="sim", overlay="finals_r21")
        with self.assertRaisesRegex(RuntimeError, "refusing finals overlay"):
            self.server(config=cfg)

    def test_every_finals_overlay_loads_under_sim_and_none(self):
        finals = sorted(path.stem for path in OVERLAY_DIR.glob("finals*.yaml"))
        self.assertIn("finals_r21", finals)
        for name in finals:
            configs = {}
            for profile in ("sim", "none"):
                with self.subTest(overlay=name, profile=profile):
                    cfg = configs[profile] = load_config(profile=profile, overlay=name)
                    self.assertEqual(cfg["profile"], profile)
                    self.assertEqual(cfg["overlay"], name)
            with self.subTest(overlay=name, server_profile="none"):
                server = self.server(profile="none", overlay=name)
                self.assertEqual(server.config_snapshot, configs["none"])
            with self.subTest(overlay=name, server_profile="sim"):
                with self.assertRaisesRegex(RuntimeError, "refusing finals overlay"):
                    self.server(profile="sim", overlay=name)
            sim_on = {key for key, _, on, _, _ in effective_flags(configs["sim"]) if on}
            real_on = {key for key, _, on, _, _ in effective_flags(configs["none"]) if on}
            self.assertEqual(sim_on, real_on, name)

    def test_server_snapshot_is_the_exact_finals_real_config(self):
        server = self.server(profile="none", overlay="finals_r21")
        expected = load_config(profile="none", overlay="finals_r21")
        self.assertEqual(server.config_snapshot, expected)
        mutated = copy.deepcopy(server.config_snapshot)
        mutated["motion"]["arm_max_vel"] = 99
        self.assertNotEqual(mutated, server.pol.cfg)


if __name__ == "__main__":
    unittest.main()
