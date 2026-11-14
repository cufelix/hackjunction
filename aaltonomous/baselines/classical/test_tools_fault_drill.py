"""Fast checks for the classical fault drill: one short episode per fault, over the real websocket path."""
import contextlib
import io
import unittest

import numpy as np

from .tools_fault_drill import ActionAudit, main, run_episode
from .config import load_config


class TestFaultDrill(unittest.TestCase):
    def test_two_faults_one_short_episode_each(self):
        for fault in ("instruction", "depth_blackout"):
            with self.subTest(fault=fault):
                r = run_episode("submission", fault, episode=0, time_limit=3.0)
                self.assertEqual(r.crashes, 0, r.crash_heads[:1] or r.detail)
                self.assertTrue(r.completed, r.detail)
                self.assertTrue(r.sane, r.detail)
                self.assertTrue(r.act_ms)

    def test_audit_flags_nan_and_cli_exit_code(self):
        obs = {"arm_q": np.zeros(14), "hand_q": np.zeros(14)}
        audit = ActionAudit(load_config(overlay="submission"), obs)
        audit.check({"arm_q_target": np.full((1, 14), np.nan), "hand_q_target": np.zeros((1, 14))})
        self.assertFalse(audit.sane)
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(["--overlay", "submission", "--fault", "instruction", "--episodes", "1",
                                   "--time-limit", "3"]), 0)


if __name__ == "__main__":
    unittest.main()
