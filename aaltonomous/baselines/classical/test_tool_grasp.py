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
    "drill": [("body", 0, 0, .025, .09, .025, .025, 0, [60, 60, 60]),
              ("chuck", .11, 0, .015, .02, .012, .012, 0, GREY),
              ("grip", -.03, -.07, .015, .018, .045, .015, 0, GREEN),
              ("battery", -.03, -.14, .025, .04, .03, .025, 0, [60, 60, 60])],
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

    def test_screwdriver_yaw0_grasps_handle_body(self):
        # FAIL: at yaw 0 the 6 mm shaft is taken as the 'handle' protrusion (with-handle), so the grasp lands at s=+0.06 on the
        # shaft. Fix: reject a protrusion narrower than ~1.5 cm in _handle_pixels so the shaft is not a handle.
        dets, plan, part, _ = self.case("screwdriver", 0)
        self.assertEqual(len(dets), 1)
        self.assertEqual(part, "handle")

    # --- pistol-grip drill (flat on its side: body along s, grip and battery along t)
    # Never grasp the chuck (a 4 cm stub on a big body). Abstaining (not graspable, or plan refused) is the expected outcome
    # where no grip/body grasp exists; a grasp that is planned must land on grip or body.
    def _drill(self, yaw, side):
        dets, plan, part, gz = self.case("drill", yaw, side)
        self.assertEqual(len(dets), 1)
        if dets[0].graspable and plan.ok:
            self.assertIn(part, {"grip", "body"})
        return gz

    def test_drill_yaw0_left(self):
        self._drill(0, "left")

    @unittest.expectedFailure
    # FAIL: with-handle shape (4 cm chuck stub), plan ok, grasp lands on the chuck. Needs a stub-vs-body rule (see abstain notes).
    def test_drill_yaw90_left(self):
        self._drill(90, "left")

    def test_drill_yaw180_left(self):
        self._drill(180, "left")

    def test_drill_yaw270_left(self):
        self._drill(270, "left")

    def test_drill_yaw90_right(self):
        self._drill(90, "right")

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


# --- kitchen objects: pelvis centre y +0.12 (left arm, left bin) / -0.12 (right arm, right bin). Sphere and
# horizontal-cylinder parts are approximated: round parts are vertical cylinders (same silhouette from above),
# the spatula handle is a box (15 x 2 x 2 cm), the mug handle is a solid box (no loop hole).
WHITE, BROWN = [230, 230, 230], [120, 80, 40]
CENTRES = {"left": (0.30, 0.12), "right": (0.30, -0.12)}
OBJ_SPECS = {
    "mug_plus": [("body", 0, 0, .045, .04, .04, .045, 0, WHITE),
                 ("handle", 0, .055, .045, .01, .015, .025, 0, WHITE)],
    "mug_minus": [("body", 0, 0, .045, .04, .04, .045, 0, WHITE),
                  ("handle", 0, -.055, .045, .01, .015, .025, 0, WHITE)],
    "lime": [("body", 0, 0, .025, .025, .025, .025, 0, GREEN)],
    "apple": [("body", 0, 0, .0375, .0375, .0375, .0375, 0, GREEN)],
    "knife": [("handle", -.05, 0, .01, .05, .0125, .01, 0, BROWN),
              ("blade", .06, 0, .001, .06, .0125, .001, 0, GREY)],
    "spatula": [("handle", -.075, 0, .01, .075, .01, .01, 0, GREY),
                ("head", .04, 0, .0015, .04, .03, .0015, 0, WHITE)],
    "cup": [("body", 0, 0, .05, .035, .035, .05, 0, WHITE)],
}
ROUND = {("mug_plus", "body"), ("mug_minus", "body"), ("lime", "body"), ("apple", "body"), ("cup", "body")}
SPECS.update(OBJ_SPECS)


def obj_frame(name, centre):
    renderer = Renderer(SC, CAL)
    rgb, zbuf = renderer._bg[0].copy(), renderer._bg[1].copy()
    c0 = np.array([centre[0], centre[1], SC.counter_z])
    for label, s, t, h, hs, ht, hh, _, col in OBJ_SPECS[name]:
        ctr = c0 + np.array([s, t, h])
        if (name, label) in ROUND:
            renderer._cylinder(rgb, zbuf, ctr, hs, hh, col)
        else:
            renderer._box(rgb, zbuf, ctr, np.eye(3), np.array([hs, ht, hh]), col)
    depth = np.where(np.isfinite(zbuf), zbuf * 1000.0, 0.0).astype(np.uint16)
    return np.clip(rgb, 0, 255).astype(np.uint8), depth


class TestKitchenGrasp(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = novel_config()
        cls.cfg["perception"]["novel"]["fp_filter"] = False
        cls.cfg["perception"]["novel"]["max_area_px"] = 60000
        cls.planner = Planner(cls.cfg, SC, hand_postures(cls.cfg), home_arm(cls.cfg))

    def case(self, name, side):
        """Detect the object at its arm's centre, plan with the near arm into the same-side bin."""
        centre = CENTRES[side]
        rgb, depth = obj_frame(name, centre)
        perception = Perception(self.cfg, SC, CAL, log=lambda *_: None)
        perception.hand_opening_m = self.planner.generic_hand_opening
        dets = [d for d in perception.detect(rgb, depth) if getattr(d, "novel", False)]
        if len(dets) != 1:
            print(f"\n[{name} {side}] detections={len(dets)}")
            return dets, None, None, None
        d = dets[0]
        plan = self.planner._plan_side(d, side, side, home_arm(self.cfg), hand_postures(self.cfg)["open"])
        g = plan.waypoints.get("grasp")
        part = part_at(name, 0, g, centre=centre)[0] if g is not None else None
        gz = float(g[2] - SC.counter_z) if g is not None else None
        print(f"\n[{name} {side}] label={d.shape} kind={d.grasp_kind} graspable={d.graspable} "
              f"reason={d.ungraspable_reason} | plan.ok={plan.ok} reason={plan.reason!r} part={part} grasp_z={gz}")
        return dets, plan, part, gz

    def expect(self, name, side, want_part):
        dets, plan, part, gz = self.case(name, side)
        self.assertEqual(len(dets), 1)
        self.assertTrue(dets[0].graspable, dets[0].ungraspable_reason)
        self.assertTrue(plan.ok, plan.reason)
        self.assertEqual(part, want_part)

    # mug: detector says box-like, planner rejects 91 mm object height (safe range)
    @unittest.expectedFailure
    def test_mug_handle_plus_left(self):
        self.expect("mug_plus", "left", "body")

    # mug: detector says box-like, planner rejects 91 mm object height (safe range)
    @unittest.expectedFailure
    def test_mug_handle_plus_right(self):
        self.expect("mug_plus", "right", "body")

    # mug: detector says box-like, planner rejects 91 mm object height (safe range)
    @unittest.expectedFailure
    def test_mug_handle_minus_left(self):
        self.expect("mug_minus", "left", "body")

    # mug: detector says box-like, planner rejects 91 mm object height (safe range)
    @unittest.expectedFailure
    def test_mug_handle_minus_right(self):
        self.expect("mug_minus", "right", "body")

    def test_lime_left(self):
        self.expect("lime", "left", "body")

    def test_lime_right(self):
        self.expect("lime", "right", "body")

    def test_apple_left(self):
        self.expect("apple", "left", "body")

    def test_apple_right(self):
        self.expect("apple", "right", "body")

    # knife: detector graspable=False, 'handle too short/occluded' (2.5 cm handle, flat blade)
    @unittest.expectedFailure
    def test_knife_left(self):
        self.expect("knife", "left", "handle")

    # knife: detector graspable=False, 'handle too short/occluded' (2.5 cm handle, flat blade)
    @unittest.expectedFailure
    def test_knife_right(self):
        self.expect("knife", "right", "handle")

    # spatula: detector graspable=False, 'handle too short/occluded' (box-approximated 2 cm handle)
    @unittest.expectedFailure
    def test_spatula_left(self):
        self.expect("spatula", "left", "handle")

    # spatula: detector graspable=False, 'handle too short/occluded' (box-approximated 2 cm handle)
    @unittest.expectedFailure
    def test_spatula_right(self):
        self.expect("spatula", "right", "handle")

    # cup: planner rejects 101 mm object height (safe range)
    @unittest.expectedFailure
    def test_cup_left(self):
        self.expect("cup", "left", "body")

    # cup: planner rejects 101 mm object height (safe range)
    @unittest.expectedFailure
    def test_cup_right(self):
        self.expect("cup", "right", "body")
