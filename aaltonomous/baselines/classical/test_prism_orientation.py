"""perception.prism_orientation + grasp.lying_prism: synthetic ray-cast RGB-D of standing / lying triangular prisms.
Run: python -m unittest aaltonomous.baselines.classical.test_prism_orientation -v
"""
import hashlib
import time
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

from .config import FLAGS, load_config
from .mock_world import Renderer, make_mock_env
from .perception import Perception
from .policy import ClassicalPolicy

SC, CAL = Scene.load(), Calibration.load()
DIMS = SC.object_dims("triangle")
ALT, SIDE, HSTAND = 3 * DIMS["apothem"], DIMS["side"], SC.object_height()
ON = {"perception.prism_orientation": True, "grasp.lying_prism": True, "perception.classify_shape": True}
POS = SC.mat_to_pelvis([0.40, 0.15, 0.0])
YAWS = np.radians(np.arange(0, 180, 20.0))
_REND = Renderer(SC, CAL)


def _faces(kind, length):
    """Outward half-spaces (n, d): n.x <= d in the prism's local frame (z = 0 on the counter, ridge along local x)."""
    if kind == "standing":   # triangle in xy (altitude ALT), extruded by `length` (= the standing height) along z
        f = [(np.array([0, 0, 1.0]), length), (np.array([0, 0, -1.0]), 0.0)]
        for k in range(3):
            phi = np.radians(60 + 120 * k)
            f.append((np.array([np.cos(phi), np.sin(phi), 0.0]), ALT / 3))
    else:                    # triangle in yz on a rectangular side, ridge along x, length `length`
        f = [(np.array([1.0, 0, 0]), length / 2), (np.array([-1.0, 0, 0]), length / 2), (np.array([0, 0, -1.0]), 0.0)]
        for sgn in (1, -1):
            n = np.array([0, ALT, sgn * SIDE / 2]) if False else np.array([0.0, sgn * ALT, SIDE / 2])
            n = n / np.linalg.norm(n)
            f.append((n, float(n @ np.array([0, sgn * SIDE / 2, 0.0]))))
    return f


def render(kind, yaw, length=HSTAND, pos=POS, noise_mm=1.0, seed=0):
    """rgb, depth (uint16 mm) of one red prism on the counter, ray-cast through the head camera."""
    rgb, zb = _REND._bg[0].copy(), _REND._bg[1].copy()
    c, s = np.cos(yaw), np.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])
    o = R.T @ (_REND.o - np.array([pos[0], pos[1], SC.counter_z]))
    r = _REND.rays @ R                                   # R.T applied to every ray
    t_in, t_out = np.full(zb.shape, -np.inf), np.full(zb.shape, np.inf)
    for n, d in _faces(kind, length):
        nr = r @ n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (d - o @ n) / nr
        t_in = np.where(nr < 0, np.maximum(t_in, t), t_in)
        t_out = np.where(nr > 0, np.minimum(t_out, t), t_out)
        t_out = np.where(nr == 0, np.where(o @ n > d, -np.inf, t_out), t_out)
    hit = (t_in <= t_out) & (t_in > 0) & (t_in < zb)
    zb[hit] = t_in[hit]
    rgb[hit] = np.asarray(SC.colors["red"], float)
    depth = np.where(np.isfinite(zb), zb * 1000.0, 0.0)
    depth = depth + np.random.default_rng(seed).normal(0, noise_mm, depth.shape)
    return np.clip(rgb, 0, 255).astype(np.uint8), np.clip(depth, 0, 65535).astype(np.uint16)


def detect(kind, yaw, overrides=ON, **kw):
    p = Perception(load_config(overrides=overrides), SC, CAL, log=lambda *_: None)
    dets = p.detect(*render(kind, yaw, **kw))
    return dets


def ang_err(a, b, period=np.pi):
    return abs((a - b + period / 2) % period - period / 2)


class TestClassification(unittest.TestCase):
    def test_standing_prisms_are_standing_at_every_yaw(self):
        for yaw in YAWS:
            d, = detect("standing", yaw)
            self.assertEqual(d.prism_orientation, "standing", f"yaw {np.degrees(yaw):.0f}: {d}")
            self.assertIsNone(d.ridge_yaw)

    def test_lying_prisms_are_lying_with_the_ridge_yaw(self):
        for length in (HSTAND, 0.060):                    # our scene length, and a longer prism (ridge = long axis)
            for yaw in YAWS:
                d, = detect("lying", yaw, length=length)
                self.assertEqual(d.prism_orientation, "lying", f"L={length} yaw {np.degrees(yaw):.0f}: {d}")
                self.assertLess(np.degrees(ang_err(d.ridge_yaw, yaw)), 10.0)
                self.assertEqual((d.shape, d.shape_path), ("triangle", "lying"))
                true = np.array(POS[:2])
                self.assertLess(np.linalg.norm(np.array(d.prism_xy) - true), 0.012)

    def test_cube_is_never_lying(self):
        rgb, depth = _REND.render([dict(color="red", shape="cube", yaw=0.3, size=SC.size, height=HSTAND,
                                         pos=np.r_[POS[:2], SC.counter_z + HSTAND / 2])])
        d, = Perception(load_config(overrides=ON), SC, CAL, log=lambda *_: None).detect(rgb, depth)
        self.assertNotEqual(d.prism_orientation, "lying")
        self.assertIsNone(d.ridge_yaw)

    def test_flag_off_leaves_detections_untouched(self):
        for kind in ("standing", "lying"):
            on = detect(kind, 0.4)
            off = detect(kind, 0.4, overrides={"perception.classify_shape": True})
            self.assertIsNone(off[0].prism_orientation)
            self.assertIsNone(off[0].ridge_yaw)
            np.testing.assert_array_equal(on[0].top, off[0].top)
            if kind == "standing":
                self.assertEqual((on[0].shape, on[0].shape_path), (off[0].shape, off[0].shape_path))

    def test_depth_height_cue_is_silent_when_the_heights_are_equal(self):
        self.assertAlmostEqual(ALT, HSTAND)     # our scene: altitude == length, only the two footprint cues vote
        d, = detect("lying", 0.2)
        self.assertEqual(d.prism_orientation, "lying")

    def test_taller_standing_prism_height_cue_agrees(self):
        d, = detect("standing", 0.5, length=0.070)
        self.assertEqual(d.prism_orientation, "standing")


class TestFlags(unittest.TestCase):
    def test_registered_and_default_off(self):
        cfg = load_config(overlay="none")
        for k in ("perception.prism_orientation", "grasp.lying_prism"):
            self.assertIn(k, FLAGS)
            self.assertIs(FLAGS[k][1], False)
        self.assertFalse(cfg["perception"]["prism_orientation"])
        self.assertFalse(cfg["grasp"]["lying_prism"])

    def test_r22_overlay_is_r21_plus_the_two_flags(self):
        r21, r22 = load_config(overlay="finals_r21"), load_config(overlay="finals_r22_prism")
        self.assertEqual(r22.pop("overlay"), "finals_r22_prism")
        r21.pop("overlay")
        self.assertTrue(r22["perception"]["prism_orientation"] and r22["grasp"]["lying_prism"])
        r22["perception"]["prism_orientation"] = r21["perception"]["prism_orientation"]
        r22["grasp"]["lying_prism"] = r21["grasp"]["lying_prism"]
        self.assertEqual(r22, r21)


def _planner(**over):
    pol = ClassicalPolicy(config=load_config(overlay="finals_r21", overrides=over), log=None)
    return pol, pol.planner


def _lying_det(yaw, pos=POS):
    d, = detect("lying", yaw, pos=pos)
    return d


class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.pol, cls.pl = _planner(**{"perception.prism_orientation": True, "grasp.lying_prism": True})
        cls.spec = JointSpec()
        cls.arm, cls.hand = cls.pol.home.copy(), cls.pol.post["open"].copy()

    def test_hand_heading_is_across_the_ridge_within_10_deg(self):
        n_ok = 0
        for yaw in YAWS:
            d = _lying_det(yaw)
            p = self.pl.plan(d, self.arm, self.hand, relocate=False)
            if not p.ok:
                continue
            if "lying prism" not in p.reason:     # heading outside the arm's natural range: the stock path took over
                continue
            n_ok += 1
            h = self.pl.finger_heading(p)
            err = np.degrees(ang_err(h, d.ridge_yaw + np.pi / 2))     # palm x = the closing axis = across the ridge
            self.assertLess(err, 10.0, f"ridge {np.degrees(d.ridge_yaw):.0f}: heading {np.degrees(h):.0f}")
            np.testing.assert_allclose(p.waypoints["grasp"][:2], d.prism_xy, atol=0.006)
            self.assertAlmostEqual(p.waypoints["grasp"][2] - SC.counter_z, ALT / 3, delta=0.012)
        self.assertGreaterEqual(n_ok, 4, "too few reachable lying-prism yaws")

    def test_headings_are_ridge_plus_minus_90(self):
        for side in ("left", "right"):
            for yaw in YAWS:
                for a in self.pl.lying_headings(yaw, side):
                    self.assertAlmostEqual(np.degrees(ang_err(a - yaw, np.pi / 2)), 0.0, places=6)

    def test_joint_targets_inside_urdf_limits(self):
        n = 0
        for yaw in YAWS[::2]:
            p = self.pl.plan(_lying_det(yaw), self.arm, self.hand, relocate=False)
            if not p.ok:
                continue
            n += 1
            self.assertTrue(np.all(p.arm >= self.spec.arm_lower - 1e-9) and np.all(p.arm <= self.spec.arm_upper + 1e-9))
            self.assertTrue(np.all(p.hand >= self.spec.hand_lower - 1e-9) and np.all(p.hand <= self.spec.hand_upper + 1e-9))
            self.assertTrue(np.all(np.isfinite(p.arm)) and np.all(np.isfinite(p.hand)))
        self.assertGreater(n, 0)

    def test_jaw_squeezes_the_section_at_the_grasp_height(self):
        """URDF FK: closed_tight jaw gap (thumb vs fingers, palm x) < the lying section width at the grasp height."""
        k = self.pl.kin
        t = k.tips_in_palm("left", self.pol.post["closed_tight"])
        gap = abs(t[0, 0] - t[2, 0])
        z = ALT / 3
        width = SIDE * (1 - z / ALT)
        self.assertLess(gap, width)
        self.assertLess(width - gap, 0.012)               # a squeeze, not a crush
        self.assertAlmostEqual(abs(t[0, 1] - t[2, 1]), 0.0, delta=0.01)   # pads at the same palm height

    def test_standing_prism_plan_is_unchanged_by_the_flags(self):
        d, = detect("standing", 0.3)
        a = self.pl.plan(d, self.arm, self.hand, relocate=False)
        _, base = _planner()
        b = base.plan(d, self.arm, self.hand, relocate=False)
        self.assertEqual(a.ok, b.ok)
        if a.ok:
            np.testing.assert_array_equal(a.arm, b.arm)
            np.testing.assert_array_equal(a.hand, b.hand)

    def test_lying_flag_off_ignores_the_orientation_fields(self):
        d = _lying_det(0.5)
        _, base = _planner()
        plain = _planner()[1]
        import copy
        d0 = copy.copy(d)
        d0.prism_orientation = d0.ridge_yaw = d0.prism_xy = None
        a = base.plan(d, self.arm, self.hand, relocate=False)
        b = plain.plan(d0, self.arm, self.hand, relocate=False)
        self.assertEqual(a.ok, b.ok)
        if a.ok:
            np.testing.assert_array_equal(a.arm, b.arm)
            np.testing.assert_array_equal(a.hand, b.hand)


def _stream(overrides, steps=60):
    env = make_mock_env(seed=0)
    pol = ClassicalPolicy(config=load_config(overrides=overrides), log=None, safety=env.filter.cfg)
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    h = hashlib.sha256()
    for _ in range(steps):
        a = pol.act(obs)
        h.update(np.ascontiguousarray(a["arm_q_target"]).tobytes() + np.ascontiguousarray(a["hand_q_target"]).tobytes())
        obs, info = env.step(a)
        if info["episode_over"]:
            break
    return h.hexdigest()


class TestEpisode(unittest.TestCase):
    def test_flags_on_change_nothing_without_prisms(self):
        """The mock scene has cubes only: the stream with both flags on must equal the flags-off stream byte for byte."""
        self.assertEqual(_stream({}), _stream({"perception.prism_orientation": True, "grasp.lying_prism": True}))

    def test_act_never_blocks_with_a_lying_prism_in_view(self):
        env = make_mock_env(seed=0)
        cfg = load_config(overlay="finals_r22_prism")
        pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
        obs = env.reset()
        pol.reset({"instruction": "Put the red objects in the left bin"})
        rgb, depth = render("lying", 0.5)
        times = []
        for _ in range(150):
            obs = dict(obs, head_rgb=rgb, head_depth=depth)
            t0 = time.perf_counter()
            a = pol.act(obs)
            times.append(time.perf_counter() - t0)
            obs, info = env.step(a)
            if info["episode_over"]:
                break
        times = np.array(times)
        self.assertLess(times.max(), 1 / 27, f"act() max {1000 * times.max():.1f} ms")
        self.assertTrue(any(m for _, m in pol.events))


if __name__ == "__main__":
    unittest.main()
