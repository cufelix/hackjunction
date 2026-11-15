"""motion.sag_comp: mock PD + gravity plant (runner kp/kd, +-0.12 rad target clip, G1 URDF gravity).

(a) no comp, (b) comp with runner gravity ff OFF, (c) comp with runner ff ON (kappa -> 0, output bit-identical, no overshoot).
Run with -s to print the end-effector error table.
"""
import unittest

import numpy as np

from .config import load_config, OVERLAY_DIR
from .gravity_ff import GravityFF
from .kinematics import ArmKinematics
from .planner import MotionOutputLimiter
from .sag_comp import KP, SagComp, SLEW

KD = np.array([3.0, 3, 3, 3, 1.5, 1.5, 1.5] * 2)
INERTIA = np.array([0.9, 0.7, 0.4, 0.3, 0.06, 0.06, 0.06] * 2)
DT, HZ = 0.002, 30.0
CLIP = 0.12


class Plant:
    """Runner PD: tau = kp*(clip(cmd, q+-0.12) - q) - kd*qd + tau_ff, gravity from the URDF at the true q."""

    def __init__(self, gff, q0, ff_on):
        self.g, self.q, self.qd, self.ff_on = gff, np.array(q0, float), np.zeros(14), ff_on
        self.cmd = self.q.copy()

    def step(self, cmd, n):
        for _ in range(n):
            tgt = np.clip(cmd, self.q - CLIP, self.q + CLIP)
            tau_ff = float(self.ff_on) * self.g.row(cmd) if self.ff_on else 0.0     # runner: gravity(target), clipped like ArmGravity
            tau = KP * (tgt - self.q) - KD * self.qd + tau_ff - self.g.raw(self.q)
            self.qd = self.qd + tau / INERTIA * DT
            self.q = self.q + self.qd * DT


def ramp(a, b, vel=0.6):
    n = max(2, int(np.ceil(np.abs(b - a).max() / vel * HZ)))
    return a + (b - a) * (np.arange(1, n + 1) / n)[:, None]


class TestSagComp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(overlay=str(OVERLAY_DIR / "finals_r25_gen.yaml"))
        cls.g = GravityFF()
        cls.kin = ArmKinematics(cls.cfg, cls.g.spec)
        from .config import home_arm
        cls.home = home_arm(cls.cfg, cls.g.spec)
        cls.poses = {"home": cls.home}
        seed = cls.home.copy()
        for name, tgt in (("hover", (0.42, -0.18, 0.26)), ("grasp", (0.42, -0.18, 0.08)), ("release", (0.52, -0.05, 0.25))):
            q, ok, err, _ = cls.kin.solve("right", np.array(tgt), seed)
            cls.assertTrue(ok or err < 0.02, f"IK {name}")
            seed = q
            cls.poses[name] = q
        cls.mo = cls.cfg["motion"]

    def ee(self, q):
        return self.kin.grasp_pose("right", q)[0]

    def run_case(self, comp, ff_on, stages=("home", "hover", "grasp", "release"), hold=3.0, home_hold=4.0):
        plant = Plant(self.g, self.home, ff_on)
        sag = SagComp(self.g) if comp else None
        lim = MotionOutputLimiter(self.mo["arm_max_vel"], self.mo["arm_max_acc"], self.mo["hand_max_vel"], self.mo["hand_max_acc"],
                                  1 / HZ, arm_lim=(self.g.spec.arm_lower, self.g.spec.arm_upper),
                                  hand_lim=(self.g.spec.hand_lower, self.g.spec.hand_upper))
        hand = np.zeros((1, 14))
        t, cur, errs, kap, emitted = 0.0, self.home.copy(), {}, [], []
        prev = None
        for si, name in enumerate(stages):
            tgt = self.poses[name]
            plan = np.vstack([ramp(cur, tgt) if si else np.empty((0, 14))] + [np.repeat(tgt[None], int((home_hold if si == 0 else hold) * HZ), 0)])
            rows_to_hold = int((home_hold if si == 0 else hold) * HZ)
            samples = []
            for k, row in enumerate(plan):
                a = row[None]
                if sag is not None:
                    sag.observe(t, plant.q, None)
                    a = sag.process(a)
                a, _ = lim.limit(a, hand, plant.q, hand[0], t)
                if sag is not None:
                    sag.note_emitted(a[-1])
                emitted.append(a[-1].copy())
                plant.step(a[-1], int(1 / HZ / DT))
                t += 1 / HZ
                kap.append(sag.kappa if sag else 0.0)
                if k >= len(plan) - 9:
                    samples.append(self.ee(plant.q) - self.ee(tgt))
            cur = tgt
            errs[name] = float(np.linalg.norm(np.mean(samples, 0)) * 1e3)
        return errs, np.array(kap), np.array(emitted), plant

    def test_error_table(self):
        res = {}
        for label, comp, ff in (("a no comp, ff off", False, False), ("b comp, ff off", True, False),
                                ("c comp, ff on", True, True), ("(ref) no comp, ff on", False, True)):
            res[label] = self.run_case(comp, ff)
        print("\nEE error (mm) at the end of each hold, right hand, kappa at the end")
        print(f"{'case':24s}" + "".join(f"{s:>9s}" for s in ("home", "hover", "grasp", "release")) + "   kappa")
        for label, (e, k, _, _) in res.items():
            print(f"{label:24s}" + "".join(f"{e[s]:9.1f}" for s in ("home", "hover", "grasp", "release")) + f"   {k[-1]:.2f}")
        ea, eb, ec, er = (res[k][0] for k in res)
        for s in ("home", "hover", "grasp", "release"):
            self.assertGreater(ea[s], 10.0)                          # sag is real without comp
            self.assertLess(eb[s], 0.4 * ea[s] + 3.0, s)             # comp removes most of it
        self.assertGreater(res["b comp, ff off"][1][-1], 0.8)
        # runner ff ON: inert, kappa 0, no worse than the plain ff-on runner
        self.assertTrue((res["c comp, ff on"][1] == 0.0).all())
        for s in ("home", "hover", "grasp", "release"):
            self.assertLessEqual(ec[s], er[s] + 0.5, s)

    def test_inert_when_runner_has_gravity_ff(self):
        """Comp ON with ff ON must emit exactly the stream comp OFF emits (same plant, same plan)."""
        _, k, em_on, _ = self.run_case(True, True)
        _, _, em_off, _ = self.run_case(False, True)
        self.assertTrue((k == 0.0).all())
        np.testing.assert_array_equal(em_on, em_off)

    def test_caps_and_clip_respected(self):
        _, k, em, plant = self.run_case(True, False)
        self.assertGreater(k[-1], 0.8)
        d = np.diff(em, axis=0) * HZ
        self.assertLessEqual(np.abs(d).max(), self.mo["arm_max_vel"] + 1e-9)
        self.assertLessEqual(np.abs(np.diff(d, axis=0) * HZ).max(), self.mo["arm_max_acc"] + 1e-6)
        # offset <= 0.09 rad and slew-limited
        sag = SagComp(self.g); sag.kappa = 1.0
        q = self.poses["grasp"]
        rows = sag.process(np.repeat(q[None], 40, 0))
        self.assertLessEqual(np.abs(rows - q).max(), 0.09 + 1e-12)
        self.assertLessEqual(np.abs(np.diff(rows, axis=0)).max(), SLEW / HZ + 1e-12)
        self.assertLess(np.abs(rows - q).max(), CLIP)

    def test_partial_runner_ff_gives_partial_kappa(self):
        """Runner carries half the gravity: the estimate settles near 0.5 (never above the 1.2 clamp)."""
        e, k, _, _ = self.run_case(True, 0.5)
        self.assertLessEqual(k.max(), 1.2 + 1e-12)
        self.assertAlmostEqual(k[-1], 0.5, delta=0.25)
        e0, _, _, _ = self.run_case(False, 0.5)
        for s in ("hover", "grasp"):
            self.assertLess(e[s], 0.5 * e0[s] + 2.0)

    def test_policy_on_mock_without_sag_is_identical(self):
        """Mock env has no sag (measured == command): sag_comp ON learns kappa 0 and the action stream is bit-identical."""
        from .mock_world import make_mock_env
        from .policy import ClassicalPolicy
        outs = {}
        for name, flag in (("off", False), ("on", True)):
            env = make_mock_env(seed=1)
            pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=20.0,
                                  overrides={"motion.output_caps": {"arm_vel": 0.85, "arm_acc": 4.0, "hand_vel": 1.5, "hand_acc": 8.0},
                                             "motion.sag_comp": flag})
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"], "time_limit": 20.0})
            acts = []
            for _ in range(120):
                a = pol.act(obs)
                acts.append(a)
                obs, info = env.step(a)
                if info["episode_over"]:
                    break
            outs[name] = (acts, pol)
        for a, b in zip(outs["off"][0], outs["on"][0]):
            np.testing.assert_array_equal(a["arm_q_target"], b["arm_q_target"])
        self.assertEqual(getattr(outs["on"][1]._sag, "kappa", 0.0), 0.0)
        self.assertIsNone(outs["off"][1]._sag)


if __name__ == "__main__":
    unittest.main()
