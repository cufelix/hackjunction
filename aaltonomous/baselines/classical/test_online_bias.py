"""grasp.online_bias: per-hand grasp xy bias learned from misses (default OFF).
Run: python -m unittest aaltonomous.baselines.classical.test_online_bias -v
"""
import unittest

import numpy as np

from aaltonomous.scene.scene import Scene

from . import online_bias as ob
from .config import hand_postures, home_arm, load_config
from .perception import Detection
from .planner import Planner
from .policy import ClassicalPolicy


def simulate(true, start, hand="left", model="air", max_miss=12, tol=0.004, per_episode=False):
    """Drive OnlineBias with a hand whose real offset is `true`; held iff |b - true| <= tol. 'air': a miss never
    moves the object; 'contact': a miss within 3 cm pushes it toward the grasp centre (0.5 x the error)."""
    s = ob.OnlineBias({hand: start}, log=lambda m: None)
    true = np.asarray(true, float)
    for n in range(max_miss + 1):
        err = s.get(hand) - true
        assert np.linalg.norm(s.get(hand)) <= ob.CLAMP_M + 1e-12
        if np.linalg.norm(err) <= tol:
            s.success(hand)
            return s, n
        delta = 0.5 * err if (model == "contact" and np.linalg.norm(err) < 0.03) else np.zeros(2)
        s.miss(hand, delta)
        if per_episode:
            s.begin_episode()
    return s, None


class TestUpdateRule(unittest.TestCase):
    def test_converges_from_zero_to_minus_two_cm(self):
        for model in ("air", "contact"):
            s, n = simulate((0.0, -0.02), (0.0, 0.0), model=model)
            self.assertIsNotNone(n, model)
            self.assertLessEqual(n, 6, model)
            np.testing.assert_allclose(s.get("left"), (0.0, -0.02), atol=0.004)

    def test_right_hand_probes_the_other_way_first(self):
        s, n = simulate((0.0, 0.02), (0.0, 0.0), hand="right")
        self.assertIsNotNone(n)
        self.assertLessEqual(n, 6)

    def test_recovers_from_a_wrong_start(self):
        for model, lim in (("contact", 6), ("air", 8)):
            s, n = simulate((0.0, -0.02), (0.0, 0.02), model=model, max_miss=14, per_episode=True)
            self.assertIsNotNone(n, model)
            self.assertLessEqual(n, lim, model)

    def test_clamp_and_episode_cap(self):
        s = ob.OnlineBias(log=lambda m: None)
        for _ in range(30):
            s.miss("left", (0.0, 0.05))   # object keeps being pushed +y: bias walks -y, must stop at the caps
            self.assertLessEqual(np.linalg.norm(s.get("left")), ob.CLAMP_M + 1e-12)
            self.assertLessEqual(np.linalg.norm(s.get("left") - s.ep0["left"]), ob.EPISODE_CAP_M + 1e-12)
        self.assertAlmostEqual(float(np.linalg.norm(s.get("left"))), ob.EPISODE_CAP_M, places=9)
        s.begin_episode()
        for _ in range(30):
            s.miss("left", (0.0, 0.05))
        self.assertAlmostEqual(float(np.linalg.norm(s.get("left"))), ob.CLAMP_M, places=9)

    def test_success_keeps_bias_and_decays_alpha(self):
        s = ob.OnlineBias({"left": (0.0, -0.01)}, log=lambda m: None)
        a0 = s.alpha("left")
        for _ in range(8):
            s.success("left")
        np.testing.assert_allclose(s.get("left"), (0.0, -0.01))
        self.assertLess(s.alpha("left"), a0)
        self.assertGreaterEqual(s.alpha("left"), ob.ALPHA_MIN)

    def test_hands_are_independent_and_log_line(self):
        lines = []
        s = ob.OnlineBias(log=lines.append)
        s.miss("left")
        self.assertTrue(np.all(s.get("right") == 0.0))
        self.assertTrue(lines[-1].startswith("[online_bias] hand=left b=("), lines)
        self.assertIn("reason=", lines[-1])


class TestWiring(unittest.TestCase):
    def test_flag_off_is_inert(self):
        pol = ClassicalPolicy(log=None)
        self.assertIsNone(pol._obias)
        self.assertIsNone(getattr(pol.planner, "online_bias", None))

    def test_state_persists_across_reset_and_starts_at_configured_offset(self):
        pol = ClassicalPolicy(log=None, overrides={"grasp.online_bias": True,
                                                   "grasp.arm_xy_offset": {"left": [0.0, 0.02], "right": [0.0, 0.0]}})
        np.testing.assert_allclose(pol._obias.get("left"), (0.0, 0.02))
        pol._obias.miss("left")
        learned = pol._obias.get("left")
        pol._ob_pending.append({"color": "red", "side": "left", "obj_xy": np.zeros(2), "held": False, "done": False})
        pol.reset()
        np.testing.assert_allclose(pol._obias.get("left"), learned)
        self.assertEqual(pol._ob_pending, [])
        np.testing.assert_allclose(pol._obias.ep0["left"], learned)   # cap re-based
        self.assertIs(pol.planner.online_bias, pol._obias)

    def test_retry_redetection_feeds_the_update_before_planning(self):
        pol = ClassicalPolicy(log=None, overrides={"grasp.online_bias": True})
        rec = {"color": "red", "side": "left", "obj_xy": np.array([0.30, 0.10]), "held": False, "done": False}
        pol._ob_pending.append(rec)
        d = Detection("red", np.array([0.30, 0.10 + 0.02, 0.8]), 500, 400, "depth")   # pushed +y by 2 cm
        pol._ob_resolve(d)
        np.testing.assert_allclose(pol._obias.get("left"), (0.0, -ob.STEP_M * pol._obias.alpha("left")), atol=1e-12)
        self.assertEqual(pol._ob_pending, [])

    def test_only_the_grasp_point_is_biased(self):
        cfg = load_config()
        cfg["grasp"]["online_bias"] = True
        sc = Scene.load()
        post, home = hand_postures(cfg), home_arm(cfg)
        pl = Planner(cfg, sc, post, home)
        top = np.r_[0.31, 0.10, sc.counter_z + sc.object_height()]
        det = Detection("red", top, 1000, 1000, "depth", shape="cube")
        base = pl._plan_side(det, "left", "left", home, post["open"])
        pl.online_bias = ob.OnlineBias({"left": (0.0, -0.01)}, log=lambda m: None)
        biased = pl._plan_side(det, "left", "left", home, post["open"])
        self.assertTrue(base.ok and biased.ok, (base.reason, biased.reason))
        np.testing.assert_allclose(biased.waypoints["grasp"][:2] - base.waypoints["grasp"][:2], (0.0, -0.01), atol=1e-9)
        np.testing.assert_allclose(biased.waypoints["drop"], base.waypoints["drop"])


if __name__ == "__main__":
    unittest.main()
