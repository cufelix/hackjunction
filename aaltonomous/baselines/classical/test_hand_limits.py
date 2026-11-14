"""R42a/R36: real Dex3 hand safety. For every finals_* overlay: postures + emitted hand targets inside the joint limits
and the real 1.746 rad idx1/mid1 saturation, index == middle per hand on every tick (hand.mirror_index_middle),
left/right closing signs consistent, flag off -> unchanged targets."""
import unittest
from pathlib import Path

import numpy as np

from aaltonomous.interface.safety import JointSpec

from .config import hand_postures, load_config
from .mock_world import make_mock_env
from .policy import ClassicalPolicy

FINALS = sorted(p.stem for p in (Path(__file__).parent / "overlays").glob("finals_*.yaml"))
SAT = 1.746                       # real Dex3 idx1/mid1 saturate here (R36); |q| beyond it is never reachable
SPEC = JointSpec()
STEPS = 500


def stream(cfg, seed, steps=STEPS):
    env = make_mock_env(seed=seed)
    pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    out = []
    for _ in range(steps):
        a = pol.act(obs)
        out.append(np.asarray(a["hand_q_target"], float).reshape(-1, 14))
        obs, info = env.step(a)
        if info["episode_over"]:
            break
    return np.vstack(out)


def pairs_equal(h):               # index == middle per hand, whichever slot order the robot uses
    return (np.array_equal(h[:, 3:5], h[:, 5:7]) and np.array_equal(h[:, 10:12], h[:, 12:14]))


class TestHandLimits(unittest.TestCase):
    def test_flag_default_off(self):
        self.assertFalse(load_config(overlay="none")["hand"]["mirror_index_middle"])

    def test_postures(self):
        for name in FINALS:
            cfg = load_config(overlay=name)
            self.assertTrue(cfg["hand"]["mirror_index_middle"], name)
            for pn, v in hand_postures(cfg).items():
                self.assertTrue(((v >= SPEC.hand_lower) & (v <= SPEC.hand_upper)).all(), (name, pn))   # (1)
                self.assertLessEqual(abs(v[[4, 6, 11, 13]]).max(), SAT, (name, pn))
                self.assertTrue(pairs_equal(v[None]), (name, pn))                                       # (2)
                # (3) closing sign: left idx/mid joints <= 0, right >= 0 (joint_spec.yaml; no real-Dex3 note in the repo)
                self.assertTrue((v[3:7] <= 0).all() and (v[10:14] >= 0).all(), (name, pn))
                self.assertTrue(np.allclose(v[3:7], -v[10:14]) or pn == "closed_demo", (name, pn))

    def test_pair_choice(self):
        post = hand_postures(load_config(overlay="finals_real"))
        d = post["closed_demo"]                       # raw left: mid -0.50,-1.41 / idx -0.49,-1.21 -> the more closed
        np.testing.assert_allclose(d[3:7], [-0.50, -1.41, -0.50, -1.41])
        np.testing.assert_allclose(post["release"][3:7], -0.05)

    def test_episodes(self):
        for name in FINALS:
            for seed in (1, 2):
                h = stream(load_config(overlay=name), seed)
                self.assertTrue(((h >= SPEC.hand_lower - 1e-9) & (h <= SPEC.hand_upper + 1e-9)).all(), (name, seed))  # (1)
                self.assertLessEqual(abs(h[:, [4, 6, 11, 13]]).max(), SAT, (name, seed))
                self.assertTrue(pairs_equal(h), (name, seed))                                           # (2)
                self.assertTrue((h[:, 3:7] <= 1e-9).all() and (h[:, 10:14] >= -1e-9).all(), (name, seed))   # (3)

    def test_flag_off_identical(self):
        cfg = load_config(overlay="finals_real")
        off = load_config(overlay="finals_real", overrides={"hand.mirror_index_middle": False})
        self.assertFalse(off["hand"]["mirror_index_middle"])
        plain = load_config(overlay="finals_real", overrides={"hand.mirror_index_middle": False})
        a, b = stream(off, 1), stream(plain, 1)
        self.assertEqual(a.tobytes(), b.tobytes())                                                     # (4) deterministic off
        on = stream(cfg, 1)
        self.assertEqual(on.shape, a.shape)
        # closed_demo is the only posture with index != middle; flag-on may only differ on those joints
        diff = np.flatnonzero(np.abs(on - a).max(axis=0) > 1e-12)
        self.assertTrue(set(diff) <= {3, 4, 5, 6, 10, 11, 12, 13}, diff)


if __name__ == "__main__":
    unittest.main()


class TestPostureAwareEmission(unittest.TestCase):
    def test_closed_demo_keeps_full_flexion_and_open_stays_open(self):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(config=load_config(overlay="finals_r21"), log=None)
        d = pol.post["closed_demo"].copy()
        d[5:7] = [-0.49, -1.21]                       # an un-equalized ramp target: one pair less closed
        np.testing.assert_allclose(pol._im_emit(d)[3:7], [-0.50, -1.41, -0.50, -1.41])
        o = pol.post["open"].copy()
        o[6] = o[4] - 0.2 * np.sign(o[4] or -1.0)     # one open finger slightly more curled
        e = pol._im_emit(o)
        self.assertLessEqual(abs(e[6]), abs(o[6]))    # open side takes the more open value
        self.assertEqual(e[4], e[6])
