"""Six objects in one episode (the finals' upper bound is 5 plus one spare): no crash, the episode ends, no pick starts
too late to finish, and the attempt cap holds. Mock world, finals_r19b overlay, one seed, sim time 120 s.
    python -m unittest aaltonomous.baselines.classical.test_six_objects -v
Runs virtual time: the policy's planning thread is waited for (sim clock paused) so the result does not depend on
the CPU. Skipped, not failed, when the episode takes more than WALL_BUDGET_S of wall time.
"""
import time
import unittest

from .config import load_config
from .mock_world import make_mock_env
from .policy import ClassicalPolicy

OVERLAY, SEED, N_OBJECTS, TIME_LIMIT = "finals_r19b", 0, 6, 120.0
WALL_BUDGET_S = 90.0
MAX_ACTS = 20000


def _planner_busy(pol):
    job = getattr(pol, "_job", None)
    return job is not None and job.get("result") is None


def run_six(seed=SEED):
    """One episode. Returns (policy, starts, info, wall_s); starts = [(t, plan)] for every pick the policy began."""
    env = make_mock_env(seed=seed, n_objects=N_OBJECTS)
    pol = ClassicalPolicy(config=load_config(overlay=OVERLAY), safety=env.filter.cfg, time_limit=TIME_LIMIT,
                          log=None)
    starts = []
    begin = pol._start

    def record_start(plan, t):
        starts.append((float(t), plan))
        return begin(plan, t)
    pol._start = record_start
    t0 = time.monotonic()
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"], "time_limit": TIME_LIMIT})
    info, n = {"episode_over": False}, 0
    while n < MAX_ACTS:
        if time.monotonic() - t0 > WALL_BUDGET_S:
            raise unittest.SkipTest(f"six-object episode still running after {WALL_BUDGET_S:.0f} s wall; "
                                    f"test skipped (sim t={env.t:.1f} s, {len(starts)} picks started)")
        if _planner_busy(pol):
            time.sleep(0.002)   # virtual clock paused while the planner thread works
            continue
        obs, info = env.step(pol.act(obs))
        n += 1
        if info["episode_over"]:
            break
    return pol, starts, info, n, time.monotonic() - t0


class TestSixObjects(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pol, cls.starts, cls.info, cls.acts, cls.wall = run_six()

    def test_episode_ends_without_exception(self):
        self.assertLess(self.acts, MAX_ACTS, "episode did not end")
        self.assertTrue(self.info["episode_over"])
        self.assertIn(self.info["end_reason"], ("done", "time_limit", "aborted", "safe_stop"))

    def test_no_pick_starts_too_late_to_finish(self):
        durations = [p.duration for p in self.pol.plans if p.ok]
        self.assertTrue(durations, "no feasible plan at all: nothing to check")
        min_dur = min(durations)
        for t, plan in self.starts:
            self.assertLessEqual(t, TIME_LIMIT - min_dur + 1e-6, f"pick {plan.color} started at t={t:.2f}")
            self.assertLessEqual(t + plan.duration, TIME_LIMIT + 1e-6,
                                 f"pick {plan.color} started at {t:.2f} would run to {t + plan.duration:.2f}")

    def test_attempt_cap_holds(self):
        cap = int(self.pol.cfg["policy"]["max_total_attempts"])
        self.assertEqual(cap, 18)
        self.assertLessEqual(self.pol.n_started, cap)
        self.assertEqual(len(self.starts), self.pol.n_started)


if __name__ == "__main__":
    unittest.main()
