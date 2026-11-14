"""Tests for the offline FAULT-DRILL harness."""
import contextlib
import io
import unittest
from unittest import mock

import numpy as np

from .tools_fault_drill import ActionAudit, FAULTS, Result, main, run_episode
from .config import load_config


class TestFaultDrill(unittest.TestCase):
    def test_all_faults_hold_and_meet_contract(self):
        rows = [run_episode(fault, overlay="none", seed=i) for i, fault in enumerate(FAULTS)]
        failed = [row for row in rows if not row.passed]
        self.assertEqual(failed, [], [(r.fault, r.__dict__) for r in failed])
        self.assertTrue(all(r.p99_ms < 33.0 for r in rows))

    def test_audit_rejects_nan_and_cli_fails(self):
        obs = {"arm_q": np.zeros(14), "hand_q": np.zeros(14)}
        audit = ActionAudit(load_config(overlay="none"), obs)
        audit.check({"arm_q_target": np.full((1, 14), np.nan), "hand_q_target": np.zeros((1, 14))})
        self.assertFalse(audit.limits)
        with mock.patch("aaltonomous.baselines.classical.tools_fault_drill.run_suite",
                        return_value=[Result("x", "x", no_exception=False)]):
            with contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(main(["--overlay", "x"]), 1)


if __name__ == "__main__":
    unittest.main()
