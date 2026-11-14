"""grasp.prism_posture (demo | narrow) and grasp.prism_offset (off | table, port of Oliver e344273), both default off.
Run: python -m unittest aaltonomous.baselines.classical.test_prism"""
import unittest

import numpy as np

from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .perception import Detection
from .planner import Planner

SC = Scene.load()
OV = {"grasp.close_posture": "closed_tight", "hand.closed_tight": [0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6],
      "grasp.point_in_palm": [0.0862, -0.0864, 0.0143], "grasp.close_posture_by_shape": True}
TAB_L = [-1.2, 0.2, 1.8, 3.2, 4.5, 2.8, 1.2, -0.2, -1.8, -3.5, -4.5, -2.8]
TAB_R = [-1.2, -2.8, -4.5, -3.5, -1.8, -0.2, 1.2, 2.8, 4.5, 3.2, 1.8, 0.2]
TRI = dict(shape="triangle", shape_conf=0.9, shape_path="metric")


def _det(y=None, **kw):
    L = SC.cfg["layout"]
    top = SC.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]), SC.object_height()])
    if y is not None:
        top[1] = y
    d = Detection("red", top, 0, 0, kw.pop("method", "depth"))
    for k, v in kw.items():
        setattr(d, k, v)
    return d


def _plan(ov, **det_kw):
    cfg = load_config(overrides={**OV, **ov})
    post, home = hand_postures(cfg), home_arm(cfg)
    pl = Planner(cfg, SC, post, home)
    pl.log_lines = []
    pl.log = pl.log_lines.append
    p = pl.plan(_det(**det_kw), home, post["open"])
    assert p.ok, p.reason
    return p, post, pl


def _jaw(kin, side, q14):
    t = kin.tips_in_palm(side, q14)
    return t, np.linalg.norm(t[2] - t[0])


class TestNarrowPosture(unittest.TestCase):
    def test_defaults_off(self):
        g = load_config()["grasp"]
        self.assertEqual((g["prism_posture"], g["prism_offset"]), ("demo", "off"))
        self.assertEqual((g["prism_offset_table_l"], g["prism_offset_table_r"]), (TAB_L, TAB_R))

    def test_bad_values_fail_at_load(self):
        for k, v in (("grasp.prism_posture", "wide"), ("grasp.prism_offset", "on"), ("grasp.prism_offset", False)):
            with self.assertRaises(ValueError):
                load_config(overrides={k: v})

    def test_fk_jaw_and_exact_right_mirror(self):
        cfg = load_config()
        post, kin = hand_postures(cfg), ArmKinematics(cfg)
        tl, jl = _jaw(kin, "left", post["closed_narrow"])
        tr, jr = _jaw(kin, "right", post["closed_narrow"])
        self.assertAlmostEqual(jl * 1000, 21.5, delta=0.3)                  # < 28 mm the prism needs; demo = 29.1
        self.assertAlmostEqual(_jaw(kin, "left", post["closed_demo"])[1] * 1000, 29.1, delta=0.3)
        np.testing.assert_allclose(tr, tl * np.array([1, -1, 1]), atol=1e-6)   # right = exact mirror (thumb0 kept)
        np.testing.assert_allclose((tl[0] + tl[2]) / 2, cfg["grasp"]["point_in_palm_narrow"], atol=1e-4)
        self.assertGreater(np.linalg.norm(tl[1] - (tl[0] + tl[2]) / 2), 0.03)   # middle tip stays clear of the jaw

    def test_choice(self):
        for mode, want in (("demo", "closed_demo"), ("narrow", "closed_narrow")):
            p, post, pl = _plan({"grasp.prism_posture": mode}, **TRI)
            self.assertEqual(p.posture[0], want)
            sl = ArmKinematics.sl(p.side)
            end = p.phases[[n for n, _, _ in p.phases].index("close")][2] - 1
            np.testing.assert_allclose(p.hand[end][sl], post[want][sl], atol=1e-9)
        p, _, _ = _plan({"grasp.prism_posture": "narrow"}, shape="cube", shape_conf=0.9, shape_path="metric")
        self.assertEqual(p.posture[0], "closed_tight")                        # only gated prism picks change

    def test_demo_is_unchanged(self):
        a, _, _ = _plan({}, **TRI)
        b, _, _ = _plan({"grasp.prism_posture": "demo"}, **TRI)
        np.testing.assert_array_equal(a.arm, b.arm)
        np.testing.assert_array_equal(a.hand, b.hand)


class TestPrismOffset(unittest.TestCase):
    def test_table_interp_wraps(self):
        f = Planner.prism_table_d
        for rel, d in ((30, 3.2), (45, 3.65), (115, -2.0), (-5, -2.0), (270, 3.2)):
            self.assertAlmostEqual(f(TAB_L, rel), d)
        for k in range(12):
            self.assertEqual(TAB_R[k], TAB_L[(-k) % 12])

    def test_gate(self):
        on = {"grasp.prism_offset": "table"}
        p, _, _ = _plan(on, apex_yaw=0.3, **TRI)
        self.assertIsNotNone(p.prism_offset)
        for kw in (dict(apex_yaw=None), dict(apex_yaw=float("nan")), dict(method="plane", apex_yaw=0.3)):
            p, _, _ = _plan(on, **{**TRI, **kw})
            self.assertIsNone(p.prism_offset, kw)
        p, _, _ = _plan(on, shape="cube", shape_conf=0.9, shape_path="metric", apex_yaw=0.3)
        self.assertIsNone(p.prism_offset)                                     # pinch_v1 pick: table not valid
        p, _, _ = _plan({**on, "grasp.prism_posture": "narrow"}, apex_yaw=0.3, **TRI)
        self.assertIsNone(p.prism_offset)                                     # table is for pinch_demo only
        p, _, _ = _plan({}, apex_yaw=0.3, **TRI)
        self.assertIsNone(p.prism_offset)                                     # flag off

    def test_shift_follows_table_and_clamp_axis(self):
        on = {"grasp.prism_offset": "table", "policy.grasp_diag_log": True}
        for y in (0.10, -0.10):
            p0, _, pl = _plan(on, y=y, apex_yaw=0.0, **TRI)
            h = p0.prism_offset["heading"]
            for rel in (30.0, 45.0, 100.0):
                p, _, pl = _plan(on, y=y, apex_yaw=h + np.radians(rel), **TRI)
                r = p.prism_offset
                tab = TAB_L if p.side == "left" else TAB_R
                self.assertAlmostEqual(r["rel_deg"], rel, places=3)
                self.assertAlmostEqual(r["d_mm"], Planner.prism_table_d(tab, rel), places=6)
                self.assertAlmostEqual(np.linalg.norm(r["u"]), 1.0, places=9)
                np.testing.assert_allclose(r["shift"], -r["d_mm"] / 1000 * r["u"], atol=1e-12)
                a = (np.degrees(np.arctan2(r["u"][1], r["u"][0]) - h) + 180) % 360 - 180
                self.assertTrue(80 < (1 if p.side == "left" else -1) * a < 150, (p.side, a))   # demo clamp ~98 deg
                self.assertTrue(any(m.startswith("[prism_offset]") for m in pl.log_lines))

    def test_off_is_byte_identical(self):
        a, _, _ = _plan({}, apex_yaw=0.7, **TRI)
        b, _, pl = _plan({"grasp.prism_offset": "off"}, apex_yaw=0.7, **TRI)
        np.testing.assert_array_equal(a.arm, b.arm)
        self.assertEqual(pl.log_lines, [])


class TestCloseToContact(unittest.TestCase):
    def _pol(self, **ov):
        from .policy import ClassicalPolicy
        cfg = load_config(overrides={**OV, "grasp.close_to_contact": True, **ov})
        return ClassicalPolicy(config=cfg)

    def _setup(self, pol, n=60, s0=5, e0=40, closed=1.0, hold=None, lim=(-9.0, 9.0)):
        """A fake left pick: hand joint 1 closes 0.03/tick from 0 to `closed` (hv ~ 1.2 rad/s = 0.04/tick), holds to
        row e0 (= start of the release), then opens to 0."""
        sl = slice(0, 7)
        h = np.zeros((n, 14))
        r = min(int(np.ceil(closed / 0.03)), e0 - s0)
        h[s0:s0 + r, 1] = np.minimum(np.arange(1, r + 1) * 0.03, closed)
        h[s0 + r:e0, 1] = closed
        h[e0:, 1] = np.linspace(closed, 0.0, n - e0 + 1)[1:]
        pol.queue_a, pol.queue_h = np.zeros((n, 14)), h.copy()
        pol.hand_cmd = np.zeros(14)
        d = np.zeros(7); d[1] = 1.0
        lo, hi = np.full(7, lim[0]), np.full(7, lim[1])
        pol._contact = {"n": n, "win": (s0, e0), "sl": sl, "dir": d, "lock": {}, "prev": None, "hold": s0 + r,
                        "open": e0, "lo": lo, "hi": hi, "orig": h.copy(), "still": np.zeros(7, int), "t": 0.0, "plan": None}
        return h

    def _run(self, pol, stall_at):
        """Measured joint 1 follows the command until it stalls at stall_at."""
        meas = np.zeros(14)
        sent = []
        while pol.queue_a is not None and len(pol.queue_a):
            if pol._contact is not None:
                pol._close_to_contact({"hand_q": meas.copy()})
            a = pol._pop()
            for row in a["hand_q_target"]:
                sent.append(row[1])
                pol.hand_cmd[1] = row[1]
                meas[1] = min(row[1], stall_at)
        return np.array(sent)

    def test_cap_after_stall(self):          # deep posture: contact mid-ramp, closing travel removed
        pol = self._pol()
        h = self._setup(pol)
        sent = self._run(pol, stall_at=0.4)
        self.assertGreaterEqual(sent[:40].max(), 0.4 + 0.12 - 1e-9)
        self.assertLess(sent[:40].max(), 0.7)
        self.assertLessEqual(np.abs(np.diff(sent)).max(), float(pol.cfg["motion"]["hand_max_vel"]) / 30 + 1e-9)
        np.testing.assert_allclose(sent[-1], h[-1, 1], atol=1e-9)          # release ends at the planned posture

    def test_shallow_posture_is_deepened(self):   # closed 0.43 barely past contact 0.4 (lag < contact_err)
        pol = self._pol()
        self._setup(pol, closed=0.43)
        sent = self._run(pol, stall_at=0.4)
        self.assertGreaterEqual(sent[:40].max(), 0.4 + 0.12 - 1e-9)       # beyond the planned 0.43
        self.assertLessEqual(np.abs(np.diff(sent)).max(), float(pol.cfg["motion"]["hand_max_vel"]) / 30 + 1e-9)
        np.testing.assert_allclose(sent[-1], 0.0, atol=1e-9)

    def test_squeeze_is_a_config_value(self):
        for sq in (0.25, 0.40):
            pol = self._pol(**{"grasp.contact_squeeze_rad": sq})
            self._setup(pol, closed=0.43)
            self.assertAlmostEqual(self._run(pol, stall_at=0.4)[:40].max(), 0.4 + sq, places=6)

    def test_joint_limit_clamps(self):
        pol = self._pol(**{"grasp.contact_squeeze_rad": 0.40})
        self._setup(pol, closed=0.43, lim=(-9.0, 0.6))
        self.assertAlmostEqual(self._run(pol, stall_at=0.4).max(), 0.6, places=6)

    def test_real_dex3_limits_are_used(self):
        pol = self._pol()
        sp = pol.spec
        self.assertTrue(np.all(sp.hand_lower[:7] <= sp.hand_upper[:7]))
        self.assertAlmostEqual(float(sp.hand_upper[2]), 1.7453, places=3)    # left thumb_2 URDF upper

    def test_no_contact_still_squeezes_nothing_before_hold(self):
        pol = self._pol()
        h = self._setup(pol)
        pol.queue_h[:, 1] = h[:, 1]
        sent = self._run(pol, stall_at=10.0)          # never stalls mid-ramp; at the hold the joint follows the command
        np.testing.assert_allclose(sent[:10], h[:10, 1])

    def test_bad_params_fail(self):
        with self.assertRaises(ValueError):
            load_config(overrides={"grasp.close_to_contact": True, "grasp.contact_squeeze_rad": 0.01})

    def test_mock_episode_runs(self):
        from .mock_world import make_mock_env
        from .policy import ClassicalPolicy
        for on in (False, True):
            cfg = load_config(overrides={**OV, "grasp.close_to_contact": on, "grasp.contact_scope": "all", "policy.grasp_diag_log": True})
            env = make_mock_env(seed=1, cfg=cfg)
            logs = []
            pol = ClassicalPolicy(config=cfg, log=logs.append, safety=env.filter.cfg)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            while True:
                obs, info = env.step(pol.act(obs))
                if info["episode_over"]:
                    break
            if on:
                self.assertEqual(len([m for m in logs if m.startswith("[contact]")]), pol.n_started)
            self.assertEqual(env.filter.counts if hasattr(env.filter, "counts") else {}, {} if not hasattr(env.filter, "counts") else env.filter.counts)


if __name__ == "__main__":
    unittest.main()
