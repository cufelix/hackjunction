"""Review-fix batch B: over-estimated time kept, cooldown commit, verify_landing bounce."""
import copy
import types
import unittest

import numpy as np

from .test_score_order import det, make, mk, order
from .test_time_triage import obs


class TestOverEstimateTime(unittest.TestCase):
    def test_overestimated_candidate_kept_when_real_plan_fits(self):
        dets = [det("red", 0.30, 0.1)]                    # estimate ~15.4 s, real plan 10 s
        pol = mk(dets, limit=95.0)                        # left at t=80 is 12 s < estimate
        self.assertEqual([d.color for d in order(pol, t=80.0)], ["red"])
        pol.act(obs(80.0))
        self.assertEqual(len([m for _, m in pol.events if m.startswith("execute")]), 1)


class TestCooldownCommit(unittest.TestCase):
    def test_cooldown_cooled_in_worker_is_committed_and_stamped_from_live_clock(self):
        d = det("red", 0.30, 0.1)
        pol = make({"red": 10.0}, [d], time_limit=100.0)
        pol.cfg["planner"]["per_target_guard"] = True
        pol.cfg["planner"]["per_target_cooldown_s"] = 30.0
        pol._latest_t = 10.0
        worker = copy.copy(pol)                           # the look worker: a copy of the policy, lists copied
        worker._target_cooldowns = list(pol._target_cooldowns)
        worker._cool_target(d, 10.0, "test")
        result = types.SimpleNamespace(
            state=tuple((n, getattr(worker, n)) for n in pol._LOOK_COMMIT_FIELDS if hasattr(worker, n)),
            perception_state=(), counter_z_runtime=worker.scene.counter_z_runtime, logs=())
        pol._latest_t = 12.0                              # the live clock moved on while the worker ran
        pol._commit_look_result(result)
        self.assertTrue(pol._target_guarded(d, 12.0))
        self.assertAlmostEqual(pol._target_cooldowns[0]["until"], 42.0)


class TestLandingBounce(unittest.TestCase):
    def test_two_hop_bounce_moves_track_and_releases_slot(self):
        pol = make({"red": 10.0}, [det("red", 0.30, 0.26)], time_limit=100.0)
        pol._vary_on = lambda: True                       # retry-vary tracks are the path that carries the count
        bin_name = pol.scene.bin_names[0]
        pol.attempt_tracks = [{"color": "red", "xy": np.array([0.30, 0.10]), "count": 2}]
        pol.attempts = [(np.array([0.30, 0.10]), 1)]
        pol.slot_used[bin_name] = {3}
        pol.reserved_drop_slots[bin_name] = {3}
        pend = {"color": "red", "bin": bin_name, "src_xy": np.array([0.30, 0.10]), "slot": 3}
        bounced = det("red", 0.30, 0.26)                  # bounced 16 cm onto the counter
        pol._landing_requeue(pend, bounced, [bounced])
        tr = pol.attempt_tracks[0]
        self.assertTrue(np.allclose(tr["xy"], [0.30, 0.26], atol=0.02))   # track follows the object
        self.assertEqual(tr["count"], 1)                                  # one attempt refunded
        self.assertEqual(pol._attempt_count(np.array([0.30, 0.26]), "red"), 1)
        self.assertNotIn(3, pol.slot_used[bin_name])
        self.assertNotIn(3, pol.reserved_drop_slots[bin_name])


if __name__ == "__main__":
    unittest.main()
