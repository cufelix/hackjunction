"""Offline grasp check for hand tools lying flat on the counter (screwdriver, adjustable wrench, hammer, pliers).

Synthetic head frames only, no VLM. The mock renderer has no horizontal cylinder, so round handles are
rendered as square-section boxes. Centres and yaws are pelvis frame (yaw 0 = pelvis +x).
"""
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
GREY, GREEN = [120, 120, 120], [30, 190, 50]
CENTRE = (0.30, 0.0)   # pelvis xy of the tool centre (free strip x 0.27..0.33)
# part: (label, s along tool, t across, h above counter, half_s, half_t, half_h, rot_deg, colour), metres
SPECS = {
    "screwdriver": [("handle", -.04, 0, .015, .05, .015, .015, 0, GREEN),
                    ("shaft", .05, 0, .015, .04, .003, .003, 0, GREY)],
    "wrench": [("bar", 0, 0, .004, .10, .0125, .004, 0, GREY),
               ("head", .125, 0, .004, .025, .02, .004, 0, GREY)],
    "hammer": [("handle", -.125, 0, .0125, .125, .0125, .0125, 0, GREEN),
               ("head", .0125, 0, .0125, .0125, .05, .0125, 0, GREY)],
    "pliers": [("bar", .06, .0104, .006, .06, .006, .006, 10, GREY),
               ("bar", .06, -.0104, .006, .06, .006, .006, -10, GREY),
               ("jaw", .13, 0, .006, .008, .025, .006, 0, GREY)],
}


def _axes(yaw_deg, rot_deg=0.0):
    a = np.radians(yaw_deg + rot_deg)
    return np.array([np.cos(a), np.sin(a), 0.0]), np.array([-np.sin(a), np.cos(a), 0.0])


def tool_frame(name, yaw_deg, centre=CENTRE):
    renderer = Renderer(SC, CAL)
    rgb, zbuf = renderer._bg[0].copy(), renderer._bg[1].copy()
    c0 = np.array([centre[0], centre[1], SC.counter_z])
    for _, s, t, h, hs, ht, hh, rot, col in SPECS[name]:
        du, dv = _axes(yaw_deg, rot)
        ctr = c0 + s * _axes(yaw_deg)[0] + t * _axes(yaw_deg)[1] + np.array([0, 0, h])
        R = np.column_stack([du, dv, [0.0, 0.0, 1.0]])
        renderer._box(rgb, zbuf, ctr, R, np.array([hs, ht, hh]), col)
    depth = np.where(np.isfinite(zbuf), zbuf * 1000.0, 0.0).astype(np.uint16)
    return np.clip(rgb, 0, 255).astype(np.uint8), depth


def part_at(name, yaw_deg, xyz, centre=CENTRE):
    """Nearest tool part to a pelvis point, in tool coordinates: (label, s, t)."""
    d = np.asarray(xyz[:2], float) - np.asarray(centre, float)
    du, dv = _axes(yaw_deg)[0][:2], _axes(yaw_deg)[1][:2]
    s, t = float(d @ du), float(d @ dv)
    best = min(SPECS[name], key=lambda p: max(abs(s - p[1]) - p[4], 0) ** 2 + max(abs(t - p[2]) - p[5], 0) ** 2)
    return best[0], round(s, 3), round(t, 3)


class TestToolGrasp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = novel_config()
        cls.cfg["perception"]["novel"]["fp_filter"] = False
        cls.cfg["perception"]["novel"]["max_area_px"] = 60000
        cls.planner = Planner(cls.cfg, SC, hand_postures(cls.cfg), home_arm(cls.cfg))

    def case(self, name, yaw, side="left"):
        """Detect the tool, plan with one arm into the left bin. Returns (detection list, plan, part, grasp z)."""
        rgb, depth = tool_frame(name, yaw)
        perception = Perception(self.cfg, SC, CAL, log=lambda *_: None)
        perception.hand_opening_m = self.planner.generic_hand_opening
        dets = [d for d in perception.detect(rgb, depth) if getattr(d, "novel", False)]
        if len(dets) != 1:
            return dets, None, None, None
        d = dets[0]
        plan = self.planner._plan_side(d, side, "left", home_arm(self.cfg), hand_postures(self.cfg)["open"])
        g = plan.waypoints.get("grasp")
        part = part_at(name, yaw, g)[0] if g is not None else None
        gz = float(g[2] - SC.counter_z) if g is not None else None
        print(f"\n[{name} yaw={yaw} {side}] shape={d.shape} kind={d.grasp_kind} graspable={d.graspable} "
              f"reason={d.ungraspable_reason} | plan.ok={plan.ok} reason={plan.reason!r} part={part} grasp_z={gz}")
        return dets, plan, part, gz

    # --- screwdriver (yaw 90 is the one that works)
    def test_screwdriver_yaw90_left_grasps_handle(self):
        dets, plan, part, gz = self.case("screwdriver", 90)
        self.assertEqual(len(dets), 1)
        self.assertTrue(dets[0].graspable)
        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual(part, "handle")
        self.assertGreaterEqual(gz, 0.02)
        self.assertLessEqual(gz, 0.04)

    @unittest.expectedFailure
    def test_screwdriver_yaw0_grasps_handle_body(self):
        # FAIL: at yaw 0 the 6 mm shaft is taken as the 'handle' protrusion (with-handle), so the grasp lands at s=+0.06 on the
        # shaft. Fix: reject a protrusion narrower than ~1.5 cm in _handle_pixels so the shaft is not a handle.
        dets, plan, part, _ = self.case("screwdriver", 0)
        self.assertEqual(len(dets), 1)
        self.assertEqual(part, "handle")

    # --- adjustable wrench
    @unittest.expectedFailure
    def test_wrench_is_detected(self):
        # FAIL: no novel blob at either yaw. The bar top is 8 mm above the counter: the height mask (novel_objects height_m
        # lower bound 0.01) drops it, and gate (c) needs height98 >= min_height_m 0.015. Fix: lower min_height_m to ~0.006
        # (or keep the bar in the height mask), then re-check the 2.5 cm width gate.
        dets, _, _, _ = self.case("wrench", 0)
        self.assertEqual(len(dets), 1)

    # --- hammer
    def test_hammer_yaw90_detects_handle(self):
        dets, _, part, _ = self.case("hammer", 90)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].shape, "with-handle")
        self.assertEqual(dets[0].grasp_kind, "handle")
        self.assertEqual(part, "handle")

    @unittest.expectedFailure
    def test_hammer_yaw90_left_plan_ok(self):
        # FAIL: 'hand 1.6 cm above counter during to_pregrasp (need 2.0)'; grasp z 2.1 cm, handle top 2.5 cm.
        # Fix: raise the handle pre-grasp hover to clear 2.0 cm, or grasp the handle at its top (z ~2.4 cm).
        _, plan, _, _ = self.case("hammer", 90)
        self.assertTrue(plan.ok, plan.reason)

    @unittest.expectedFailure
    def test_hammer_yaw0_grasps_handle(self):
        # FAIL: at yaw 0 the blob reads box-like (no handle), pinch grasp lands at s=0.00, the handle/head junction.
        # Fix: run the handle finder on box-like blobs with a head at one end, or ungraspable 'head-junction' for pinch.
        dets, _, part, _ = self.case("hammer", 0)
        self.assertEqual(len(dets), 1)
        self.assertEqual(dets[0].grasp_kind, "handle")
        self.assertEqual(part, "handle")

    # --- pliers
    def test_pliers_detected(self):
        dets, _, _, _ = self.case("pliers", 90)
        self.assertEqual(len(dets), 1)

    @unittest.expectedFailure
    def test_pliers_yaw90_left_plan_ok(self):
        # FAIL: 'generic object height 13 mm outside safe range' (planner, generic height check); the 12 cm V reads as a
        # 1.3 cm tall pinch object. Fix: widen the generic safe-height range to cover 13 mm, or skip for pliers via the 2-bar shape.
        _, plan, _, _ = self.case("pliers", 90)
        self.assertTrue(plan.ok, plan.reason)

    @unittest.expectedFailure
    def test_pliers_yaw0_left_plan_ok(self):
        # FAIL: same 13 mm generic height reason at yaw 0.
        _, plan, _, _ = self.case("pliers", 0)
        self.assertTrue(plan.ok, plan.reason)


if __name__ == "__main__":
    unittest.main()
