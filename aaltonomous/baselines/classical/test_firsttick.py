"""First tick before any encoder sample: a zero-row hold must survive the M4 limiter and the R42a mirror."""
import unittest

import numpy as np

from .config import load_config
from .planner import MotionOutputLimiter
from .policy import ClassicalPolicy

DT = 1.0 / 30.0


class TestFirstTick(unittest.TestCase):
    def test_nan_first_tick_on_r22_returns_zero_rows_without_error(self):
        pol = ClassicalPolicy(config=load_config(overlay="finals_r22"))
        self.assertTrue(pol._mirror_im)
        self.assertIsNotNone(pol.output_limiter)
        obs = {"arm_q": np.full(14, np.nan), "hand_q": np.full(14, np.nan), "t": 0.0}
        out = pol.act(obs)
        self.assertEqual(np.asarray(out["arm_q_target"]).shape, (0, 14))
        self.assertEqual(np.asarray(out["hand_q_target"]).shape, (0, 14))

    def test_limiter_clips_overshoot_to_urdf_limits(self):
        lo, hi = -np.ones(14) * 0.5, np.ones(14) * 0.5
        lim = MotionOutputLimiter(0.85, 4.0, 1.5, 8.0, DT, arm_lim=(lo, hi), hand_lim=(lo, hi))
        lim.limit(np.zeros((1, 14)), np.zeros((1, 14)), np.zeros(14), np.zeros(14), 0.0)
        lim.arm_v[:] = 0.85   # fast start: carried velocity toward the joint limit
        lim.hand_v[:] = 1.5
        target = np.full((30, 14), 0.5)   # the URDF limit itself
        a, h = lim.limit(target, target, lim.arm_q, lim.hand_q, DT * 1.0)
        self.assertTrue(np.all(a <= hi + 1e-12) and np.all(a >= lo - 1e-12))
        self.assertTrue(np.all(h <= hi + 1e-12) and np.all(h >= lo - 1e-12))
        self.assertGreater(float(a.max()), 0.49)   # still reaches the limit, not stuck short of it


if __name__ == "__main__":
    unittest.main()
