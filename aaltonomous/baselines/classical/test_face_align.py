"""grasp.face_align: cube finger heading so the pinch line is perpendicular to a pair of faces.
Run: python -m unittest aaltonomous.baselines.classical.test_face_align -v"""
import unittest

import numpy as np

from aaltonomous.scene.scene import Scene

from .config import check_flag_values, hand_postures, home_arm, load_config
from .perception import Detection, _yaw
from .planner import Planner

KEYS = ("face_align", "face_align_sides", "face_align_tol_deg")


def _planner(**g):
    cfg = load_config(overlay="ab_lh_z_p10")
    cfg["grasp"].update(g)
    sc = Scene.load()
    return Planner(cfg, sc, hand_postures(cfg), home_arm(cfg)), sc, cfg


def _cube(sc, yaw, y=0.10):
    return Detection("red", np.r_[0.31, y, sc.counter_z + sc.object_height()], 1000, 1000, "depth",
                     yaw=float(yaw), shape="cube")


def _pinch_angle(pl, plan, side):
    e = next(e for name, s, e in plan.phases if name == "descend")
    pl.kin._fk(plan.arm[e - 1], pl.post[pl.cfg["grasp"].get("close_posture", "closed")])
    t = pl.kin._tips(side)
    v = t[2] - t[0]
    return float(np.arctan2(v[1], v[0]))


def _off_face_normal_deg(angle, yaw):
    d = (angle - yaw) % (np.pi / 2)
    return float(np.degrees(min(d, np.pi / 2 - d)))


class TestFaceAlign(unittest.TestCase):
    def test_default_off_and_validation(self):
        g = load_config()["grasp"]
        self.assertFalse(g["face_align"])
        self.assertEqual(g["face_align_sides"], ["left"])
        check_flag_values(load_config())

    def test_cube_yaw_estimate(self):
        import cv2
        for deg in (-40, -10, 0, 15, 35):
            a = np.radians(deg)
            R = np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])
            g = np.linspace(-0.0225, 0.0225, 12)
            pts = np.array([[x, y] for x in g for y in g]) @ R.T
            self.assertAlmostEqual(np.degrees(_yaw(pts)), deg, delta=1.0)

    def test_pinch_offset_left_about_minus_44(self):
        pl, _, _ = _planner()
        off = np.degrees(pl.pinch_offset("left"))
        self.assertTrue(-60 < off < -30, off)
        self.assertAlmostEqual(np.degrees(pl.pinch_offset("right")), -off, delta=1.0)
        self.assertAlmostEqual(off, -44.0, delta=3.0)

    def test_pinch_line_on_face_normal(self):
        pl, sc, _ = _planner(face_align=True, face_align_sides=["left"], yaw_max_dev_deg=180.0)
        for deg in (0, 20, 45, 70):
            yaw = (np.radians(deg) + np.pi / 4) % (np.pi / 2) - np.pi / 4
            home = home_arm(pl.cfg)
            plan = pl._plan_side_hints(_cube(sc, yaw), "left", "center", home, pl.post["open"])
            self.assertTrue(plan.ok, (deg, plan.reason))
            self.assertIn("face-aligned", plan.reason or "", deg)
            err = _off_face_normal_deg(_pinch_angle(pl, plan, "left"), yaw)
            self.assertLessEqual(err, 5.0, (deg, err))

    def test_flags_off_plans_identical(self):
        base, sc, _ = _planner()
        stripped, _, cfg = _planner()
        for k in KEYS:
            stripped.cfg["grasp"].pop(k, None)
        home = home_arm(base.cfg)
        for side, y in (("left", 0.10), ("right", -0.10)):
            for yaw in (0.0, 0.5):
                a = base._plan_side_hints(_cube(sc, yaw, y), side, "center", home, base.post["open"])
                b = stripped._plan_side_hints(_cube(sc, yaw, y), side, "center", home, base.post["open"])
                self.assertEqual(np.asarray(a.arm).tobytes(), np.asarray(b.arm).tobytes())

    def test_on_other_side_cylinder_unchanged(self):
        off, sc, _ = _planner()
        on, _, _ = _planner(face_align=True, face_align_sides=["left"])
        home = home_arm(off.cfg)
        a = off._plan_side_hints(_cube(sc, 0.3, -0.10), "right", "center", home, off.post["open"])
        b = on._plan_side_hints(_cube(sc, 0.3, -0.10), "right", "center", home, off.post["open"])
        self.assertEqual(np.asarray(a.arm).tobytes(), np.asarray(b.arm).tobytes())
        cyl = _cube(sc, 0.3)
        cyl.shape = "cylinder"
        a = off._plan_side_hints(cyl, "left", "center", home, off.post["open"])
        b = on._plan_side_hints(cyl, "left", "center", home, off.post["open"])
        self.assertEqual(np.asarray(a.arm).tobytes(), np.asarray(b.arm).tobytes())

    def test_overlays_resolve(self):
        g = load_config(overlay="ab_lh_face")["grasp"]
        self.assertTrue(g["face_align"]); self.assertEqual(g["face_align_sides"], ["left"])
        g = load_config(overlay="ab_lh_face_both")["grasp"]
        self.assertEqual(g["face_align_sides"], ["left", "right"])
        g = load_config(overlay="ab_lh_face_y")["grasp"]
        self.assertEqual(g["arm_xy_offset"], {"left": [0.0, -0.01], "right": [0.0, 0.0]})
        self.assertEqual(g["left_hand_fix"]["shift_z"], 0.01)


if __name__ == "__main__":
    unittest.main()
