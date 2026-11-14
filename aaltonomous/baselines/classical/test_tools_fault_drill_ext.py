"""Tests for the RC drills (offline cold start, ws drop, release gate)."""
import unittest

from .tools_fault_drill_ext import Drill, drill_offline, drill_ws_drop, release_gate


class TestDrillExt(unittest.TestCase):
    def test_pass_logic(self):
        good = dict(no_exception=True, p50=1.0, p99=5.0, max=9.0, wrong=0, ok=True)
        self.assertTrue(Drill("x", "o", **good).passed)
        for bad in (dict(wrong=1), dict(p99=40.0), dict(ok=False), dict(no_exception=False)):
            self.assertFalse(Drill("x", "o", **{**good, **bad}).passed, bad)

    def test_release_gate(self):
        self.assertEqual(release_gate("submission"), [])
        diff = {k for k, *_ in release_gate("rc1")}
        self.assertIn("instruction.normalize", diff)

    def test_offline_cold_start_makes_no_socket_attempt(self):
        d = drill_offline("rc1", ticks=30)
        self.assertTrue(d.no_exception and d.ok and d.wrong == 0, d.detail)

    def test_ws_drop_reconnect(self):
        d = drill_ws_drop("none", drop_at=20)
        # p99 is not asserted: client, server and planner share one GIL in-process (see module notes)
        self.assertTrue(d.no_exception and d.ok and d.wrong == 0, d.detail)
        self.assertIn("reconnect", d.detail)


if __name__ == "__main__":
    unittest.main()
