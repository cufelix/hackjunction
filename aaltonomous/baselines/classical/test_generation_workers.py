"""H7 generation-safe look workers and non-blocking server reset."""
import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

from .config import load_config
from .policy import ClassicalPolicy
from .policy_server import ClassicalServerPolicy


class TestGenerationWorkers(unittest.TestCase):
    def test_reset_invalidates_before_stale_worker_finishes_and_stale_never_publishes(self):
        pol = ClassicalPolicy(config=load_config(overrides={"policy.generation_workers": True}), log=None)
        old_generation = pol._look_generation
        entered, release = threading.Event(), threading.Event()

        class Worker:
            perception = SimpleNamespace()
            scene = SimpleNamespace(counter_z_runtime=None)

            def _look(self, _obs):
                entered.set()
                release.wait(2.0)
                return ("none", None, 0)

        job = {"generation": old_generation, "obs": {}, "result": None}
        thread = threading.Thread(target=pol._look_job, args=(job, Worker(), []), daemon=True)
        pol._look_thread = thread
        thread.start()
        self.assertTrue(entered.wait(1.0))
        t0 = time.perf_counter()
        pol.reset({"instruction": ""})
        self.assertLess(time.perf_counter() - t0, 0.2)
        self.assertGreater(pol._look_generation, old_generation)
        self.assertIsNone(pol._look_thread, "the new generation must not wait for the stale worker")
        release.set()
        thread.join(1.0)
        self.assertIsNone(job["result"])

    def test_worker_snapshot_does_not_mutate_live_episode_objects(self):
        pol = ClassicalPolicy(config=load_config(overrides={"policy.generation_workers": True}), log=None)
        worker, _ = pol._look_snapshot()
        worker.failed.append("stale")
        worker.planner.bin_shift["left"] = [1, 2]
        worker.kin.offset["left"][0] = 99.0
        worker.perception._ah_est.append(9.0)
        worker.scene.counter_z_runtime = 99.0
        self.assertNotIn("stale", pol.failed)
        self.assertNotIn("left", pol.planner.bin_shift)
        self.assertNotEqual(pol.kin.offset["left"][0], 99.0)
        self.assertIsNot(worker.kin.data, pol.kin.data)
        self.assertNotIn(9.0, pol.perception._ah_est)
        self.assertNotEqual(pol.scene.counter_z_runtime, 99.0)

    def test_server_reset_with_flag_never_sleeps_for_pending_job(self):
        server = ClassicalServerPolicy.__new__(ClassicalServerPolicy)
        server.pol = SimpleNamespace(
            cfg={"policy": {"generation_workers": True}},
            _job={"result": None},
            reset=mock.Mock(),
        )
        with mock.patch("aaltonomous.baselines.classical.policy_server.time.sleep",
                        side_effect=AssertionError("reset slept")):
            server.reset("sort", task="sort_fixed", seed=7)
        server.pol.reset.assert_called_once_with({"instruction": "sort", "task": "sort_fixed", "seed": 7})


if __name__ == "__main__":
    unittest.main()
