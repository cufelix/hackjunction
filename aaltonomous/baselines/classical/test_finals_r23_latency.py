"""finals_r23 under profile 'none' (real) with policy.safety_shell on: act() p95 stays under the 37 ms budget."""
import time
import unittest

import numpy as np

from .config import load_config
from .mock_world import make_mock_env
from .policy import ClassicalPolicy


class TestFinalsR23Latency(unittest.TestCase):
    def test_act_p95_under_37ms_with_safety_shell(self):
        env = make_mock_env(seed=0)
        cfg = load_config(profile="none", overlay="finals_r23")
        self.assertTrue(cfg["policy"]["safety_shell"])
        pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        pol.act(obs)                                   # warm-up (first chunk builds plans)
        ts = []
        for _ in range(30):
            t0 = time.perf_counter()
            pol.act(obs)
            ts.append(time.perf_counter() - t0)
        p95 = float(np.percentile(ts, 95)) * 1e3
        print(f"\nact() p95 = {p95:.1f} ms over {len(ts)} calls")
        self.assertLess(p95, 37.0)


if __name__ == "__main__":
    unittest.main()
