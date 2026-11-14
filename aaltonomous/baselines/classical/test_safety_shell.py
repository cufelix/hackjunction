"""policy.safety_shell: hold-last watchdog, 20 % reserve vs the SafetyFilter velocity limit, never raises."""
import hashlib
import unittest

import numpy as np

from .config import load_config
from .mock_world import make_mock_env
from .policy import ClassicalPolicy
from .safety_shell import reserve_pair

DT, AV, HV, RES = 1 / 30, 1.0, 3.0, 0.2


def ramp_to(peak_vel, n=40):
    """(n, 14) arm path moving joint 0 at peak_vel rad/s, hand parked."""
    a = np.zeros((n, 14))
    a[:, 0] = peak_vel * DT * np.arange(1, n + 1)
    return a, np.zeros((n, 14))


def max_ratio(a, h, a0, h0):
    da = np.abs(np.diff(np.vstack([a0, a]), axis=0)).max() / DT / AV
    dh = np.abs(np.diff(np.vstack([h0, h]), axis=0)).max() / DT / HV
    return max(da, dh)


class TestReserve(unittest.TestCase):
    def test_over_reserve_is_slowed_to_08_and_ends_on_target(self):
        a, h = ramp_to(0.95 * AV)
        a2, h2 = reserve_pair(a, h, np.zeros(14), np.zeros(14), DT, AV, HV, RES)
        self.assertLessEqual(max_ratio(a2, h2, np.zeros(14), np.zeros(14)), 0.8 + 1e-9)
        self.assertGreater(len(a2), len(a))                        # slower, never faster
        np.testing.assert_allclose(a2[-1], a[-1])

    def test_hand_over_reserve_is_slowed_too(self):
        a, h = np.zeros((30, 14)), np.zeros((30, 14))
        h[:, 3] = 0.95 * HV * DT * np.arange(1, 31)
        a2, h2 = reserve_pair(a, h, np.zeros(14), np.zeros(14), DT, AV, HV, RES)
        self.assertLessEqual(max_ratio(a2, h2, np.zeros(14), np.zeros(14)), 0.8 + 1e-9)

    def test_within_reserve_is_untouched(self):
        a, h = ramp_to(0.5 * AV)
        a2, h2 = reserve_pair(a, h, np.zeros(14), np.zeros(14), DT, AV, HV, RES)
        self.assertIs(a2, a)
        self.assertIs(h2, h)

    def test_planner_output_obeys_reserve_when_set(self):
        pol = ClassicalPolicy(config=load_config(overrides={"policy.safety_shell": True,
                                                            "policy.safety_shell_vel_reserve": 0.2}), log=None)
        a, h = pol.planner.go_to(np.zeros(14), np.zeros(14), pol.home, pol.post["open"])
        self.assertLessEqual(max_ratio(a, h, np.zeros(14), np.zeros(14)), 0.8 + 1e-9)


def drive(pol, env, steps):
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    out = []
    for _ in range(steps):
        a = pol.act(obs)
        out.append(a)
        obs, info = env.step(a)
        if info["episode_over"]:
            break
    return obs, out


class TestShell(unittest.TestCase):
    def setUp(self):
        self.env = make_mock_env(seed=0)
        self.pol = ClassicalPolicy(config=load_config(overrides={"policy.safety_shell": True}), log=None,
                                   safety=self.env.filter.cfg)

    def test_deadline_miss_returns_identical_hold_and_keeps_the_chunk(self):
        obs, out = drive(self.pol, self.env, 3)
        last = out[-1]["arm_q_target"][-1].copy(), out[-1]["hand_q_target"][-1].copy()
        self.pol._shell_budget = -1.0                              # every act() is now late
        m1, m2 = self.pol.act(obs), self.pol.act(obs)
        for m in (m1, m2):
            self.assertFalse(m["done"])
            self.assertTrue((m["arm_q_target"] == last[0]).all() and (m["hand_q_target"] == last[1]).all())
        np.testing.assert_array_equal(m1["arm_q_target"], m2["arm_q_target"])
        self.pol._shell_budget = 1e9                                # recovered: the discarded chunk is served next
        nxt = self.pol.act(obs)
        env2 = make_mock_env(seed=0)                                # twin without misses: same next action
        pol2 = ClassicalPolicy(config=load_config(overrides={"policy.safety_shell": True}), log=None,
                               safety=env2.filter.cfg)
        obs2, _ = drive(pol2, env2, 3)
        want = pol2.act(obs2)
        np.testing.assert_array_equal(nxt["arm_q_target"], want["arm_q_target"])
        self.assertEqual(self.pol._shell_stats["miss"], 2)

    def test_exception_in_planning_holds_and_does_not_raise(self):
        logs = []
        self.pol.log = logs.append
        obs, out = drive(self.pol, self.env, 2)
        last = out[-1]["arm_q_target"][-1].copy()
        self.pol.planner.plan = self.pol.planner.plan_to_counter = self.pol.planner.go_to = self._boom
        self.pol._act = self._boom                                  # also anything else inside act()
        res = self.pol.act(obs)
        np.testing.assert_array_equal(res["arm_q_target"][-1], last)
        self.assertEqual(self.pol._shell_stats["exception"], 1)
        self.assertEqual(sum("[safety_shell]" in str(m) for m in logs), 1)

    @staticmethod
    def _boom(*a, **k):
        raise RuntimeError("planner exploded")

    def test_exception_on_the_very_first_call_still_holds(self):
        self.pol._act = self._boom
        obs = self.env.reset()
        res = self.pol.act(obs)
        np.testing.assert_allclose(res["arm_q_target"][0], obs["arm_q"])

    def test_shell_episode_runs_without_clips(self):
        obs = self.env.reset()
        self.pol.reset({"instruction": obs["instruction"]})
        n = 0
        while n < 6000:
            obs, info = self.env.step(self.pol.act(obs))
            n += 1
            if info["episode_over"]:
                break
        self.assertEqual(info["safety_summary"]["clipped_steps"], 0)
        self.assertEqual(self.pol._shell_stats["exception"], 0)


class TestFlagOff(unittest.TestCase):
    @staticmethod
    def stream(cfg, steps=400):
        env = make_mock_env(seed=1)
        pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
        h = hashlib.sha256()
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        for _ in range(steps):
            a = pol.act(obs)
            h.update(np.ascontiguousarray(a["arm_q_target"], np.float64).tobytes())
            h.update(np.ascontiguousarray(a["hand_q_target"], np.float64).tobytes())
            obs, info = env.step(a)
            if info["episode_over"]:
                break
        return h.hexdigest()

    def test_default_off_and_stream_identical_to_explicit_off(self):
        self.assertFalse(load_config()["policy"]["safety_shell"])
        self.assertEqual(self.stream(load_config()), self.stream(load_config(overrides={"policy.safety_shell": False})))

    def test_flag_on_default_reserve_leaves_plans_unchanged(self):
        on = ClassicalPolicy(config=load_config(overrides={"policy.safety_shell": True}), log=None)
        off = ClassicalPolicy(log=None)
        self.assertEqual(on.cfg["policy"]["safety_shell_vel_reserve"], 0.0)
        self.assertNotIn("plan", vars(on.planner))
        self.assertNotIn("go_to", vars(on.planner))
        a, h = on.planner.go_to(np.zeros(14), np.zeros(14), on.home, on.post["open"])
        a0, h0 = off.planner.go_to(np.zeros(14), np.zeros(14), off.home, off.post["open"])
        np.testing.assert_array_equal(a, a0)
        np.testing.assert_array_equal(h, h0)

    def test_shell_off_policy_has_no_wrappers(self):
        pol = ClassicalPolicy(log=None)
        self.assertFalse(pol._shell)
        self.assertNotIn("plan", vars(pol.planner))


if __name__ == "__main__":
    unittest.main()
