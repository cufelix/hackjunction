"""planner.use_perceived_bins (drops / rim clearance / wall tests follow Perception.bin_layout). Run:
    python -m pytest aaltonomous/baselines/classical/test_bin_drop.py -q
"""
import unittest
from unittest import mock

import numpy as np

from aaltonomous.interface.safety import JointSpec, SafetyConfig
from aaltonomous.scene.scene import Scene

from . import bin_detect
from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .planner import Planner

SC = Scene.load()
SPEC = JointSpec()
CFG_OFF = load_config(overlay="none")
CFG_ON = load_config(overlay="none", overrides={"planner.use_perceived_bins": True})
KIN = ArmKinematics(CFG_OFF, SPEC)
NOMINAL = bin_detect.nominal_layout(SC)


def make(cfg, safety=None):
    logs = []
    p = Planner(cfg, SC, hand_postures(cfg, SPEC), home_arm(cfg, SPEC), KIN, safety or SafetyConfig(dt=1 / 30), SPEC)
    p.log = logs.append
    return p, logs


def shifted(dx, dy, deg, names=None):
    return {n: {"xy": v["xy"] + np.array([dx, dy]), "yaw": float(np.radians(deg)),
                "source": "perceived" if names is None or n in names else "nominal"} for n, v in NOMINAL.items()}


def local(v, xy):
    """pelvis xy -> frame of the (rotated) bin v."""
    c, s = np.cos(v["yaw"]), np.sin(v["yaw"])
    d = np.asarray(xy, float) - v["xy"]
    return np.array([c * d[0] + s * d[1], -s * d[0] + c * d[1]])


INNER = np.asarray(SC.cfg["bins"]["outer_size"], float)[::-1] / 2 - SC.cfg["bins"]["wall_thickness"]
OUTER = np.asarray(SC.cfg["bins"]["outer_size"], float)[::-1] / 2
RIM = SC.counter_z + SC.cfg["bins"]["wall_height"]


def geometry(p, slots=True):
    out = []
    for n in SC.bin_names:
        out += [p.bin_xy(n), p.drop_xy(n)] + (list(p.drop_slots(n)) if slots else [])
    return np.array(out)


class TestBinDrop(unittest.TestCase):
    def test_shifted_rotated_bins_drop_inside_the_perceived_bin(self):
        """+0.10 m x, -0.06 m y, 5 deg: drop point + every slot inside the SHIFTED inner box (reach check stubbed:
        +0.10 m is past the arm's top-down reach, see test_unreachable_falls_back)."""
        cfg = load_config(overlay="none", overrides={"planner.use_perceived_bins": True, "grasp.drop_slots": True})
        p, logs = make(cfg)
        lay = shifted(0.10, -0.06, 5)
        with mock.patch.object(Planner, "_drop_unreachable", return_value=None):
            acc = p.set_bin_layout(lay)
        self.assertEqual(sorted(acc), sorted(SC.bin_names))
        inset = cfg["grasp"]["drop_inset"]
        margin = cfg["grasp"]["drop_slot_wall_margin"]
        for n, v in lay.items():
            np.testing.assert_allclose(p.bin_xy(n), v["xy"])
            q = local(v, p.drop_xy(n))
            self.assertTrue(np.all(np.abs(q) <= INNER - inset + 1e-9), (n, q))
            self.assertAlmostEqual(q[0], -(INNER[0] - inset), places=9)       # near-wall band (reach-biased)
            for s in p.drop_slots(n):
                q = local(v, s)
                self.assertTrue(np.all(np.abs(q) <= INNER + 1e-9), (n, q))
                self.assertLessEqual(abs(q[1]), INNER[1] - margin + 1e-9)
            # scoring test (#6 _score_bin) follows the perceived box
            self.assertEqual(p.bin_of(np.r_[v["xy"], SC.counter_z + 0.02]), n)
        # the stock planner would drop the centre object 10 cm in front of the moved bin's inner box
        nom, _ = make(CFG_OFF)
        self.assertLess(local(lay["center"], nom.drop_xy("center"))[0], -INNER[0])
        self.assertTrue(any("[bin_drop] center: perceived d=(+0.100,-0.060) +5.0deg" in s for s in logs), logs)

    def test_rim_clearance_and_wall_follow_the_perceived_footprint(self):
        p, _ = make(CFG_ON)
        nom, _ = make(CFG_OFF)
        lay = shifted(0.10, -0.06, 5)
        with mock.patch.object(Planner, "_drop_unreachable", return_value=None):
            p.set_bin_layout(lay)
        v = lay["center"]
        c, s = np.cos(v["yaw"]), np.sin(v["yaw"])
        R = np.array([[c, -s], [s, c]])
        far_wall = v["xy"] + R @ np.array([OUTER[0] - 0.001, 0.0])     # in the moved bin's far wall, out of nominal
        cavity = v["xy"]
        self.assertFalse(SC.bin_of(SC.pelvis_to_mat(np.r_[far_wall, RIM - 0.02])))
        pts = lambda xy, z: np.array([[xy[0], xy[1], z]] * 3)
        with mock.patch.object(KIN, "hand_points", return_value=pts(far_wall, RIM - 0.02)):
            self.assertLess(p.bin_clearance("left", None, None), 0)
            self.assertEqual(nom.bin_clearance("left", None, None), np.inf)
            self.assertTrue(p.hits_bin_wall("left", None, None))
            self.assertTrue(p.hits_bin_wall("left", None, None, allowed_inside="center"))   # wall, not cavity
            self.assertFalse(nom.hits_bin_wall("left", None, None))
        with mock.patch.object(KIN, "hand_points", return_value=pts(cavity, RIM - 0.02)):
            self.assertTrue(p.hits_bin_wall("left", None, None))
            self.assertFalse(p.hits_bin_wall("left", None, None, allowed_inside="center"))
        with mock.patch.object(KIN, "hand_points", return_value=pts(cavity, RIM + 0.05)):
            self.assertAlmostEqual(p.bin_clearance("left", None, None), 0.05)

    def test_reachable_lateral_shift_with_real_ik(self):
        """-0.06 m y + 5 deg at the nominal depth: every bin passes the real IK + workspace check."""
        p, logs = make(CFG_ON)
        lay = shifted(0.0, -0.06, 5)
        acc = p.set_bin_layout(lay)
        self.assertEqual(sorted(acc), sorted(SC.bin_names), logs)
        for n, v in lay.items():
            self.assertTrue(np.all(np.abs(local(v, p.drop_xy(n))) <= INNER - CFG_ON["grasp"]["drop_inset"] + 1e-9))

    def test_unreachable_falls_back_to_nominal_per_bin(self):
        p, logs = make(CFG_ON)
        nom, _ = make(CFG_OFF)
        acc = p.set_bin_layout(shifted(0.15, 0.0, 0))           # 15 cm farther: past the top-down reach (~0.50 m)
        self.assertEqual(acc, {})
        np.testing.assert_array_equal(geometry(p), geometry(nom))
        self.assertEqual(sum("[bin_drop]" in s and "unreachable" in s and "-> nominal" in s for s in logs), 3, logs)
        # workspace box: a box that ends before the drop point rejects without IK
        p2, logs2 = make(CFG_ON, SafetyConfig(dt=1 / 30, workspace_max=(0.30, 0.70, 0.60)))
        with mock.patch.object(KIN, "solve", side_effect=AssertionError("IK must not run")):
            self.assertEqual(p2.set_bin_layout(shifted(0.0, -0.02, 0)), {})
        self.assertTrue(all("outside the workspace box" in s for s in logs2), logs2)
        # one bin unreachable (IK stub fails only for it): the others keep their perceived pose
        real = KIN.solve

        def solve(side, target, *a, **kw):
            q, ok, err, tilt = real(side, target, *a, **kw)
            return q, ok and target[1] > -0.1, err, tilt
        p3, logs3 = make(CFG_ON)
        with mock.patch.object(KIN, "solve", side_effect=solve):
            acc = p3.set_bin_layout(shifted(0.0, 0.03, 3))
        self.assertEqual(sorted(acc), ["center", "left"], logs3)
        np.testing.assert_array_equal(p3.drop_xy("right"), nom.drop_xy("right"))

    def test_deadband_keeps_nominal(self):
        nom, _ = make(CFG_OFF)
        p, logs = make(CFG_ON)
        with mock.patch.object(KIN, "solve", side_effect=AssertionError("no IK inside the deadband")):
            self.assertEqual(p.set_bin_layout(shifted(-0.002, 0.001, -0.1)), {})   # sim detector noise
        np.testing.assert_array_equal(geometry(p), geometry(nom))
        with mock.patch.object(Planner, "_drop_unreachable", return_value=None):
            self.assertEqual(sorted(p.set_bin_layout(shifted(0.0, 0.0, 3))), sorted(SC.bin_names))   # yaw alone

    def test_clamp_offset_and_yaw(self):
        p, logs = make(CFG_ON)
        lay = shifted(0.0, -0.30, 40)
        with mock.patch.object(Planner, "_drop_unreachable", return_value=None):
            acc = p.set_bin_layout(lay)
        for n, (xy, yaw) in acc.items():
            self.assertAlmostEqual(float(np.linalg.norm(xy - NOMINAL[n]["xy"])), 0.15)
            self.assertAlmostEqual(np.degrees(yaw), 15.0)
        self.assertTrue(all("(clamped)" in s for s in logs), logs)

    def test_nominal_source_and_flag_off_are_identical(self):
        nom, _ = make(CFG_OFF)
        ref = geometry(nom)
        on, _ = make(CFG_ON)
        with mock.patch.object(KIN, "solve", side_effect=AssertionError("no IK for nominal bins")):
            self.assertEqual(on.set_bin_layout(NOMINAL), {})                       # every bin 'nominal'
        np.testing.assert_array_equal(geometry(on), ref)
        off, logs = make(CFG_OFF)
        self.assertEqual(off.set_bin_layout(shifted(0.10, -0.06, 5)), {})           # flag off: ignored
        np.testing.assert_array_equal(geometry(off), ref)
        self.assertEqual(logs, [])
        on.set_bin_layout(shifted(0.0, -0.06, 5))
        on.reset_bins()                                                              # new episode
        np.testing.assert_array_equal(geometry(on), ref)

    def test_policy_hands_the_perceived_layout_to_the_planner_once(self):
        from .policy import ClassicalPolicy
        from .test_bin_detect import render, true_layout
        cfg = load_config(overlay="none", overrides={"perception.detect_bins": True,
                                                     "planner.use_perceived_bins": True})
        poses = true_layout((0.0, -0.05), np.radians(3))
        rgb, depth = render(poses)
        pol = ClassicalPolicy(config=cfg, log=None)
        pol.perception.detect(rgb, depth)
        pol._sync_bins()
        self.assertEqual(sorted(pol.planner.perceived), sorted(SC.bin_names))
        for n, (xy, yaw) in poses.items():
            self.assertLess(np.linalg.norm(pol.planner.bin_xy(n) - xy), 0.01)
            v = {"xy": xy, "yaw": yaw}
            self.assertTrue(np.all(np.abs(local(v, pol.planner.drop_xy(n))) <= INNER - 0.04), n)
        pol.reset()
        self.assertEqual(pol.planner.perceived, {})
        # detect_bins on, use_perceived_bins off: the planner never moves
        pol2 = ClassicalPolicy(config=load_config(overlay="none", overrides={"perception.detect_bins": True}), log=None)
        pol2.perception.detect(rgb, depth)
        pol2._sync_bins()
        self.assertEqual(pol2.planner.perceived, {})
        self.assertEqual(pol2.perception.bin_layout["left"]["source"], "perceived")


if __name__ == "__main__":
    unittest.main()
