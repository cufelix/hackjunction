"""act() must never run perception on its own thread in threaded mode (production: policy_server.py).
With perception.detect slowed to 0.2 s, every act() call stays under 20 ms and detect never runs on the act thread.
Run: python -m unittest aaltonomous.baselines.classical.test_act_never_blocks -v
"""
import threading
import time
import unittest

from .config import load_config
from .mock_world import make_mock_env
from .policy import ClassicalPolicy


class TestActNeverBlocks(unittest.TestCase):
    def test_slow_detect_stays_off_the_act_thread(self):
        cfg = load_config(overlay="finals_r19", overrides={"brain.backend": "stub", "brain.find": False})
        env = make_mock_env(seed=1, cfg=cfg)
        pol = ClassicalPolicy(config=cfg, threaded=True, require_confirm=False, time_limit=120.0, log=None)
        self.assertTrue(pol.threaded)
        self.assertFalse(pol._shell)
        act_thread = threading.get_ident()
        detect_threads = []
        orig_detect = pol.perception.detect

        def slow_detect(*a, **kw):
            detect_threads.append(threading.get_ident())
            time.sleep(0.2)
            return orig_detect(*a, **kw)

        pol.perception.detect = slow_detect
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        worst = 0.0
        for _ in range(200):
            s = time.perf_counter()
            pol.act(obs)
            worst = max(worst, (time.perf_counter() - s) * 1000.0)
        self.assertLess(worst, 20.0, f"worst act() {worst:.1f} ms")
        self.assertTrue(detect_threads, "look never reached perception.detect")
        self.assertNotIn(act_thread, detect_threads, "perception.detect ran on the act() thread")
        pol.reset({"instruction": obs["instruction"]})  # let any look thread finish before exit


if __name__ == "__main__":
    unittest.main()
