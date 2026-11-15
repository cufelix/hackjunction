"""Synthetic pan (20 cm disc + 12 cm handle): the novel path must grasp the handle across its width."""
from __future__ import annotations

import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm
from .mock_world import Renderer
from .perception import Perception
from .planner import Planner
from .test_novel_objects import novel_config


SC, CAL = Scene.load(), Calibration.load()
RAD, HL, HW = 0.10, 0.12, 0.03          # disc radius, handle length, handle width
COL = [30, 190, 50]


def pan_frame(theta_deg=None, centre=(.45, .17)):
    """Disc centre (default 0.45, 0.17) in mat coords (nearer rows cut off the camera frame; bins at y>=0.26 mask the far side); handle angle theta in mat coords (0 = +x, 90 = +y)."""
    renderer = Renderer(SC, CAL)
    rgb, zbuf = renderer._bg[0].copy(), renderer._bg[1].copy()
    dc = SC.mat_to_pelvis([centre[0], centre[1], .02])
    renderer._cylinder(rgb, zbuf, dc, RAD, .015, COL)
    dir_p = perp_p = None
    if theta_deg is not None:
        th = np.radians(theta_deg)
        dir_p = np.array([np.sin(th), np.cos(th), 0.0])     # linear part of mat_to_pelvis (x<->y swap)
        perp_p = np.array([-dir_p[1], dir_p[0], 0.0])
        R = np.column_stack([dir_p, perp_p, [0.0, 0.0, 1.0]])
        renderer._box(rgb, zbuf, dc + (RAD + HL / 2) * dir_p, R, np.array([HL / 2, HW / 2, .0175]), COL)
    depth = np.where(np.isfinite(zbuf), zbuf * 1000.0, 0.0).astype(np.uint16)
    return np.clip(rgb, 0, 255).astype(np.uint8), depth, dc, dir_p, perp_p


def fold_pi(a):
    return float(abs((a + np.pi / 2) % np.pi - np.pi / 2))


class TestPanHandleGrasp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = novel_config()
        # geometry test; no pan placement in the FP filter bands was found (sweep), FP filter covered in test_novel_objects
        cls.cfg["perception"]["novel"]["fp_filter"] = False
        cls.cfg["perception"]["novel"]["max_area_px"] = 60000  # a 20 cm disc is ~35-41k px here; the universal overlay gate is 60000
        cls.planner = Planner(cls.cfg, SC, hand_postures(cls.cfg), home_arm(cls.cfg))

    def novels(self, theta, centre=(.45, .17)):
        rgb, depth, dc, dir_p, perp_p = pan_frame(theta, centre)
        perception = Perception(self.cfg, SC, CAL, log=lambda *_: None)
        perception.hand_opening_m = self.planner.generic_hand_opening
        dets = perception.detect(rgb, depth)
        return [d for d in dets if getattr(d, "novel", False)], dc, dir_p, perp_p

    def test_plain_disc_is_not_a_handle(self):
        novel, _, _, _ = self.novels(None)
        self.assertEqual(len(novel), 1)
        self.assertNotEqual(novel[0].shape, "with-handle")
        self.assertTrue(novel[0].ungraspable)
        self.assertEqual(novel[0].ungraspable_reason, "wider than the hand, no handle")

    def test_disc_cut_by_image_edge_keeps_visible_handle_graspable(self):
        # default centre: the disc touches the bottom image edge, the handle is fully in view -> no border penalty
        novel, _, _, _ = self.novels(0)
        self.assertEqual(len(novel), 1)
        self.assertTrue(novel[0].graspable)
        self.assertEqual(novel[0].grasp_kind, "handle")

    def test_handle_grasp_clears_the_rim_or_is_ungraspable(self):
        # 0 deg: full handle visible -> grasp on the handle >= rim + 3 cm.
        novel, dc, dir_p, perp_p = self.novels(0)
        self.assertEqual(len(novel), 1)
        d = novel[0]
        self.assertEqual((d.shape, d.grasp_kind), ("with-handle", "handle"))
        self.assertIsNone(d.ungraspable_reason)   # border-touch confidence is a separate, pre-existing gate
        rel = d.top[:2] - dc[:2]
        s, t = float(rel @ dir_p[:2]), float(rel @ perp_p[:2])
        self.assertGreaterEqual(s, RAD + 0.03, "grasp inside rim + 3 cm")
        self.assertLessEqual(s, RAD + HL + 0.01, "grasp past the handle tip")
        self.assertLessEqual(abs(t), HW / 2 + 0.01, "grasp outside the handle width")
        handle_ang = float(np.arctan2(dir_p[1], dir_p[0]))
        print(f"\n[theta=0] grasp xy={d.top[:2].round(3)} s={s:.3f} t={t:.3f} "
              f"yaw_vs_handle={np.degrees(fold_pi(d.yaw - handle_ang)):.1f}deg")
        # 45 deg (handle truncated by the frame / bin mask): grasp >= true disc radius + 3 cm from the TRUE centre, or ungraspable as "handle too short/occluded"
        novel, dc, dir_p, perp_p = self.novels(45)
        self.assertEqual(len(novel), 1)
        d = novel[0]
        if d.ungraspable_reason != "handle too short/occluded":
            rel = d.top[:2] - dc[:2]
            self.assertGreaterEqual(float(np.linalg.norm(rel)), RAD + 0.03, "grasp inside the true rim + 3 cm")
        print(f"[theta=45] ungraspable={d.ungraspable} reason={d.ungraspable_reason} xy={d.top[:2].round(3)}")
        # 90 deg: the handle is masked (far side), only the disc is seen -> ungraspable, never a disc-centre grasp
        novel, _, _, _ = self.novels(90)
        self.assertEqual(len(novel), 1)
        print(f"[theta=90] ungraspable={novel[0].ungraspable} reason={novel[0].ungraspable_reason} "
              f"shape={novel[0].shape} xy={novel[0].top[:2].round(3)}")
        self.assertTrue(novel[0].ungraspable)
        self.assertEqual(novel[0].ungraspable_reason, "wider than the hand, no handle")

    def test_handle_is_grasped_across_its_width(self):
        for theta in (0,):
            with self.subTest(theta=theta):
                novel, dc, dir_p, perp_p = self.novels(theta)
                self.assertEqual(len(novel), 1)
                d = novel[0]
                self.assertEqual(d.shape, "with-handle")
                self.assertEqual(d.grasp_kind, "handle")
                rel = d.top[:2] - dc[:2]
                s, t = float(rel @ dir_p[:2]), float(rel @ perp_p[:2])
                self.assertGreater(float(np.linalg.norm(rel)), RAD + 0.01, "grasp on the disc")
                self.assertGreaterEqual(s, RAD + 0.01, "grasp not beyond the disc rim")
                self.assertLessEqual(s, RAD + HL + 0.01, "grasp past the handle tip")
                self.assertLessEqual(abs(t), HW / 2 + 0.01, "grasp outside the handle width")
                handle_ang = float(np.arctan2(dir_p[1], dir_p[0]))
                yaw_vs_handle = np.degrees(fold_pi(d.yaw - handle_ang))
                print(f"\n[theta={theta}] grasp xy={d.top[:2].round(3)} s={s:.3f} t={t:.3f} "
                      f"tip_dist={RAD + HL - s:.3f} yaw_vs_handle={yaw_vs_handle:.1f}deg")

    def _plan_pan_handle(self, kind="handle"):
        novel, dc, dir_p, perp_p = self.novels(90, centre=(.35, .0))
        d = novel[0]
        d.generic_confidence, d.graspable = 1.0, True  # disc touches the image border here (flagged ungraspable): checks geometry only
        if kind != "handle":
            d.grasp_kind = kind
        plan = self.planner.plan(d, home_arm(self.cfg), hand_postures(self.cfg)["open"], bin_name="left")
        return d, plan

    def _finger_heading(self, plan):
        e = next(e for name, _, e in plan.phases if name == "descend")
        self.planner.kin._fk(plan.arm[e - 1])
        x = self.planner.kin.data.oMf[self.planner.kin.palm[plan.side]].rotation[:, 0]
        return float(np.arctan2(x[1], x[0]))

    def test_planner_closes_across_handle(self):
        d, plan = self._plan_pan_handle()
        self.assertTrue(plan.ok, plan.reason)
        finger_ang = self._finger_heading(plan)
        handle_ang = float(np.arctan2(d.handle_axis[1], d.handle_axis[0]))
        perp_err = float(np.degrees(np.pi / 2 - fold_pi(finger_ang - handle_ang)))   # 0 = perpendicular, 90 = along the bar
        print(f"\n[handle plan] side={plan.side} finger_vs_perp={perp_err:.1f}deg")
        self.assertLessEqual(perp_err, 20.0, "fingers do not close across the handle (within 20 deg of perpendicular)")

    def test_handle_grasp_height_not_above_stock(self):
        d, plan = self._plan_pan_handle()
        self.assertTrue(plan.ok, plan.reason)
        g = self.planner.cfg["grasp"]
        saved = g.get("handle_fingertip_clearance_m")
        try:
            g["handle_fingertip_clearance_m"] = g["fingertip_clearance"]   # stock clearance, handle-only key disabled
            _, stock = self._plan_pan_handle()
        finally:
            if saved is None:
                g.pop("handle_fingertip_clearance_m", None)
            else:
                g["handle_fingertip_clearance_m"] = saved
        self.assertTrue(stock.ok, stock.reason)
        z, z_stock = plan.waypoints["grasp"][2], stock.waypoints["grasp"][2]
        print(f"\n[handle plan] grasp z={1000 * z:.1f} mm stock z={1000 * z_stock:.1f} mm")
        self.assertLessEqual(z, z_stock + 1e-6, "handle grasp point higher than the stock-height grasp")

    def test_non_handle_grasp_keeps_stock_heading(self):
        # same geometry, not a handle: the yaw candidates stay [yaw, yaw + pi] (fingers along the major axis)
        d, plan = self._plan_pan_handle(kind="disc")
        self.assertTrue(plan.ok, plan.reason)
        finger_ang = self._finger_heading(plan)
        self.assertLessEqual(float(np.degrees(fold_pi(finger_ang - float(d.yaw)))), 15.0)


class TestUnknownNameHandle(unittest.TestCase):
    """The same pan, called 'wok': the name is unknown, so only the VLM set is offered; the handle grasp must come from shape alone."""

    @classmethod
    def setUpClass(cls):
        cls.cfg = novel_config()
        cls.cfg["perception"]["novel"]["fp_filter"] = False
        cls.cfg["perception"]["novel"]["max_area_px"] = 60000
        cls.planner = Planner(cls.cfg, SC, hand_postures(cls.cfg), home_arm(cls.cfg))

    def novels(self, theta, centre=(.45, .17)):
        rgb, depth, dc, dir_p, perp_p = pan_frame(theta, centre)
        perception = Perception(self.cfg, SC, CAL, log=lambda *_: None)
        perception.hand_opening_m = self.planner.generic_hand_opening
        dets = perception.detect(rgb, depth)
        return [d for d in dets if getattr(d, "novel", False)], dc, dir_p, perp_p

    def test_wok_is_grounded_and_grasped_by_its_handle(self):
        from .novel_objects import match_instruction, novel_noun_named

        novel, dc, _, _ = self.novels(0)
        self.assertEqual(len(novel), 1)
        d = novel[0]
        # the name is not a known noun, so the deterministic path abstains and offers the visible novel set to the brain
        self.assertFalse(novel_noun_named("put the wok in the left bin"))
        m = match_instruction("put the wok in the left bin", novel)
        self.assertIsNone(m.matched)
        self.assertEqual(m.unresolved_noun, "wok")
        self.assertEqual(m.target_bin, "left")
        self.assertIn(d, m.brain_candidates)
        # geometry the robot uses for that candidate; the handle decision comes from shape only, the name is not an input
        self.assertEqual(d.grasp_kind, "handle")
        self.assertLess(d.generic_width, 0.05)
        self.assertTrue(d.graspable)
        gap = float(np.linalg.norm(d.top[:2] - dc[:2]))
        print(f"\n[wok] grasp xy={d.top[:2].round(3)} dist_from_disc_centre={gap:.3f} "
              f"generic_width={d.generic_width:.3f}")
        self.assertGreaterEqual(gap, RAD + 0.02, "grasp point not outside the disc rim")


if __name__ == "__main__":
    unittest.main()
