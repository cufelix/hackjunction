"""motion.arm_tau_ff = gravity: shape / finite / clamp / flag-off identity (needs the G1 URDF, like the other policy tests)."""
import unittest

import numpy as np

from .gravity_ff import GravityFF, TAU_MAX


class TestGravityFF(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.g = GravityFF()

    def test_matches_rnea_and_runner_row_order(self):
        pin = self.g.pin
        rng = np.random.default_rng(0)
        for _ in range(5):
            q = np.clip(rng.uniform(-1, 1, 14) * 0.6, self.g.spec.arm_lower, self.g.spec.arm_upper)
            full = self.g.q0.copy()
            full[self.g.arm_iq] = q
            ref = pin.rnea(self.g.model, self.g.data, full, np.zeros(self.g.model.nv), np.zeros(self.g.model.nv))
            want = np.clip(ref[self.g.arm_iv], -self.g.tau_max, self.g.tau_max)
            np.testing.assert_allclose(self.g.row(q), want, atol=1e-9)

    def test_shapes_finite_clamp(self):
        q = np.zeros((6, 14)); q[:, 0] = -1.5; q[:, 7] = -1.5     # arms stretched forward
        t = self.g.rows(q, np.zeros((6, 14)))
        self.assertEqual(t.shape, (6, 14))
        self.assertTrue(np.isfinite(t).all() and (np.abs(t) <= TAU_MAX + 1e-12).all())
        self.assertTrue((np.abs(t) <= self.g.tau_max + 1e-12).all())
        self.assertEqual(self.g.rows(q[0]).shape, (14,))
        self.assertIsNone(self.g.rows(np.zeros((3, 13))))
        self.assertGreater(abs(t[0, 0]), 3.0)                        # a real shoulder load
        z = self.g.row(np.full(14, np.nan))                          # NaN pose -> 0 torque, never NaN
        self.assertTrue(np.isfinite(z).all())

    def test_policy_flag_off_has_no_key_and_on_keeps_q(self):
        from .mock_world import make_mock_env
        from .policy import ClassicalPolicy
        outs = {}
        for name, ov in (("off", None), ("on", {"motion.arm_tau_ff": "gravity"})):
            env = make_mock_env(seed=1)
            pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=20.0, overrides=ov)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"], "time_limit": 20.0})
            acts = []
            for _ in range(25):
                a = pol.act(obs)
                acts.append(a)
                obs, info = env.step(a)
                if info["episode_over"]:
                    break
            outs[name] = acts
        for a, b in zip(outs["off"], outs["on"]):
            self.assertNotIn("arm_tau_ff", a)
            np.testing.assert_array_equal(a["arm_q_target"], b["arm_q_target"])
            np.testing.assert_array_equal(a["hand_q_target"], b["hand_q_target"])
            if len(b["arm_q_target"]):
                self.assertEqual(np.asarray(b["arm_tau_ff"]).shape, np.asarray(b["arm_q_target"]).shape)
                self.assertTrue(np.isfinite(b["arm_tau_ff"]).all())
        self.assertTrue(any("arm_tau_ff" in b for b in outs["on"]))


if __name__ == "__main__":
    unittest.main()
