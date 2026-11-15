"""planner.lift_from_contact: when the staging route fails and the hand is still in contact with the table, the policy
holds and never takes the joint-space rc6 go_to through the table. Flag off, or a safe start, keeps the rc6 fallback.
Mock world, virtual clock.   python -m unittest aaltonomous.baselines.classical.test_lift_hold_policy -v
"""
import time
import unittest
from unittest import mock

from .config import load_config
from .mock_world import make_mock_env
from .policy import ClassicalPolicy


def _drive(pol, env, until, max_ticks=300):
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"], "time_limit": 60.0})
    n = 0
    while n < max_ticks and not until(pol):
        pt, ch = getattr(pol, "_pitch_thread", None), pol._jobs.get("counter_height")
        if (pt is not None and pt.is_alive()) or (ch is not None and ch["result"] is None):
            time.sleep(0.002)
            continue
        obs, info = env.step(pol.act(obs))
        n += 1
        if info["episode_over"]:
            break
    return n


def _held(pol):
    return any("lift_from_contact: hold" in m for _, m in pol.events)


def _moving(pol):
    return any("to home" in m for _, m in pol.events)


class TestLiftHoldPolicy(unittest.TestCase):
    def _run(self, lift, clearance):
        env = make_mock_env(seed=0)
        ev = []
        cfg = load_config(overlay="finals_r22", overrides={"planner.lift_from_contact": lift})
        pol = ClassicalPolicy(config=cfg, safety=env.filter.cfg, time_limit=60.0, log=lambda m: ev.append(str(m)))
        pol._stage_route = lambda *a, **k: (None, "forced fail")
        go = mock.Mock(wraps=pol.planner.go_to)
        hold = mock.Mock(wraps=pol._hold)
        with mock.patch.object(pol.planner, "clearance", return_value=clearance), \
                mock.patch.object(pol.planner, "go_to", go), mock.patch.object(pol, "_hold", hold):
            _drive(pol, env, (lambda p: _held(p)) if lift else _moving)
        return pol, go, hold

    def test_flag_on_contact_holds_and_never_calls_go_to(self):
        pol, go, hold = self._run(lift=True, clearance=0.0)
        self.assertTrue(_held(pol), pol.events)
        self.assertEqual(go.call_count, 0)
        self.assertGreater(hold.call_count, 0)
        self.assertFalse(any("fallback rc6" in m for _, m in pol.events))

    def test_flag_on_safe_start_keeps_rc6_fallback(self):
        pol, go, _ = self._run(lift=True, clearance=0.05)
        self.assertGreater(go.call_count, 0)
        self.assertTrue(any("fallback rc6 home route" in m for _, m in pol.events), pol.events)

    def test_flag_off_contact_still_takes_go_to(self):
        pol, go, _ = self._run(lift=False, clearance=0.0)
        self.assertGreater(go.call_count, 0)
        self.assertFalse(_held(pol))
        self.assertTrue(any("fallback rc6 home route" in m for _, m in pol.events), pol.events)


if __name__ == "__main__":
    unittest.main()
