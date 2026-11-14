"""policy.insert_shapes (rung 3): shape -> hole, hole poses, yaw alignment, abstain cases, flag off = unchanged.

CPU only; the planner tests need pinocchio + the G1 URDF (as test_yaw_align):
    python3 -m unittest aaltonomous.baselines.classical.test_insert_shapes -v
"""
from __future__ import annotations

import itertools
import unittest
from unittest import mock

import numpy as np

from aaltonomous.interface.tasks import COLORS, INSERT_INSTRUCTION, SHAPES
from aaltonomous.scene.scene import Scene, in_convex

from . import insert_shapes as ins
from .config import FLAGS, hand_postures, home_arm, load_config
from .perception import Detection
from .planner import Plan, Planner
from .policy import ClassicalPolicy

SC = Scene.load()


def _det(shape="cube", conf=0.9, path="metric", yaw=0.1, apex_yaw=None, apex_conf=0.0, xy=(0.30, 0.10), **kw):
    d = Detection("red", np.array([xy[0], xy[1], SC.counter_z + SC.size]), 100, 100, "depth", yaw)
    d.shape, d.shape_conf, d.shape_path, d.apex_yaw, d.apex_conf = shape, conf, path, apex_yaw, apex_conf
    for k, v in kw.items():
        setattr(d, k, v)
    return d


class Stub:
    """Just enough of ClassicalPolicy for _episode_mapping / _insert_shapes_mapping."""

    def __init__(self, **policy):
        self.cfg = {"policy": dict(policy), "instruction": {}}
        self.logged = []
        self.log = self.logged.append
        self.scene = SC
        self.fixed_mapping = {}
        self._insert_guard_mapping = lambda info: ClassicalPolicy._insert_guard_mapping(self, info)
        self._instruction_mapping = lambda info: {"default": None}
        self._log_fixed_conflict = lambda *a: None


class TestMappingAndPoses(unittest.TestCase):
    def test_shape_to_hole(self):
        self.assertEqual(ins.SHAPE_HOLE, {"cube": "square", "cylinder": "circle", "triangle": "triangle"})
        self.assertEqual(set(ins.SHAPE_HOLE), set(SHAPES))
        self.assertEqual(ins.SYMMETRY_DEG, {"cube": 90.0, "cylinder": None, "triangle": 120.0})

    def test_hole_poses_match_scene_yaml_numbers(self):
        """Independent re-derivation from scene.yaml (bin centre mat (0.71, 0.365), 45 mm objects, +10 mm clearance)."""
        t = ins.hole_targets(SC)
        a = 0.0225 + 0.005                       # cube / cylinder hole apothem
        yu = 0.102 - 0.005 - a                   # upper row, bin inner half-width 0.102
        cx, cy = 0.71, 0.365
        want = {"cylinder": (cx - yu, cy + yu), "cube": (cx + yu, cy + yu)}
        for shape, (mx, my) in want.items():
            np.testing.assert_allclose(t[shape]["xy"], [0.14 + my, -0.45 + mx], atol=1e-9)
        tri_mat_y = cy + (yu - a - 0.005) - (0.015 + 0.005)       # base 5 mm under the upper row, apothem 20 mm
        np.testing.assert_allclose(t["triangle"]["xy"], [0.14 + tri_mat_y, -0.45 + cx], atol=1e-9)
        self.assertAlmostEqual(t["cylinder"]["xy"][0], 0.5745, places=4)
        self.assertAlmostEqual(t["triangle"]["xy"][0], 0.522, places=3)
        for h in t.values():
            self.assertAlmostEqual(h["margin"], 0.005)
            self.assertAlmostEqual(h["z_top"], 0.073)             # counter 0.01 + wall 0.06 + lid 0.003

    def test_hole_yaw(self):
        t = ins.hole_targets(SC)
        self.assertAlmostEqual(t["cube"]["yaw"], 0.0)
        self.assertAlmostEqual(abs(t["triangle"]["yaw"]), np.pi)  # apex towards the robot (-x)
        # the polygon's apex vertex really lies on that pelvis direction
        poly = [h for h in SC.lid_holes() if h["shape"] == "triangle"][0]
        c = poly["center"]
        apex = poly["poly"][np.argmin(poly["poly"][:, 1])] - c    # most -y_mat vertex
        v = SC.mat_to_pelvis(np.r_[c + apex, 0.0])[:2] - SC.mat_to_pelvis(np.r_[c, 0.0])[:2]
        self.assertAlmostEqual(np.cos(np.arctan2(v[1], v[0]) - t["triangle"]["yaw"]), 1.0, places=6)

    def test_centres_are_inside_their_own_polygon_only(self):
        for h in SC.lid_holes():
            self.assertTrue(in_convex(h["poly"], h["center"]))
            for o in SC.lid_holes():
                if o["shape"] != h["shape"]:
                    self.assertFalse(in_convex(o["poly"], h["center"]), (h["shape"], o["shape"]))


class TestYawAlignment(unittest.TestCase):
    def test_cube_final_yaw_matches_the_hole_for_every_object_yaw(self):
        t = ins.hole_targets(SC)["cube"]
        for yaw in np.radians(np.arange(-44, 45, 4)):
            for a_g in ins.grasp_headings("cube", yaw, "left"):
                a_d = ins.drop_heading("cube", yaw, t["yaw"], a_g, "left")
                self.assertIsNotNone(a_d)
                self.assertLess(ins.yaw_error_deg("cube", yaw, a_g, a_d, t["yaw"]), 1e-6)
                # a cube grasped across a pair of faces: heading = yaw + k * 90
                self.assertAlmostEqual(abs(ins.fold(a_g - yaw, 90.0)), 0.0, places=9)

    def test_triangle_final_apex_matches_the_hole_for_every_apex_yaw(self):
        t = ins.hole_targets(SC)["triangle"]
        for apex in np.radians(np.arange(0, 360, 15)):
            for a_g in ins.grasp_headings("triangle", apex, "left"):
                a_d = ins.drop_heading("triangle", apex, t["yaw"], a_g, "left")
                if a_d is None:
                    continue
                self.assertLess(ins.yaw_error_deg("triangle", apex, a_g, a_d, t["yaw"]), 1e-6)

    def test_headings_stay_inside_the_wrist_range(self):
        for shape in ("cube", "triangle"):
            for yaw in np.radians(np.arange(-90, 90, 7)):
                for a in ins.grasp_headings(shape, yaw, "left", 25.0, 75.0):
                    self.assertLessEqual(abs(ins._wrap(a + np.radians(25.0))), np.radians(75.0) + 1e-9)

    def test_cylinder_needs_no_yaw(self):
        self.assertEqual(ins.grasp_headings("cylinder", None, "left"), [None])
        self.assertIsNone(ins.drop_heading("cylinder", 0.0, 0.0, None, "left"))
        self.assertEqual(ins.yaw_error_deg("cylinder", 0.0, 0.0, 1.0, 0.0), 0.0)

    def test_unreachable_wrist_heading_is_none(self):
        # an extreme grasp heading whose hole-aligned release heading falls outside the 75 deg window -> abstain
        self.assertIsNone(ins.drop_heading("cube", 0.0, 0.0, np.radians(20.0), "left", 25.0, 20.0))

    def test_release_gap_window(self):
        self.assertTrue(ins.release_gap_ok(0.073 + 0.006, 0.073))
        self.assertFalse(ins.release_gap_ok(0.073 + 0.001, 0.073))     # scrapes the lid
        self.assertFalse(ins.release_gap_ok(0.073 + 0.030, 0.073))     # long fall, tips over the edge


class TestAbstain(unittest.TestCase):
    IC = {"min_shape_conf": 0.7, "require_metric": True}

    def test_confident_metric_shapes_pass(self):
        for s in SHAPES:
            self.assertEqual(ins.decide(_det(s), self.IC), (True, ""))

    def test_abstain_cases(self):
        bad = [_det(None), _det("sphere"), _det("cube", conf=0.69), _det("cube", conf=0.0),
               _det("cube", path="blob"), _det("cube", path=None), _det("triangle", prism_orientation="lying"),
               _det("cube", xy=(np.nan, 0.0))]
        for d in bad:
            ok, why = ins.decide(d, self.IC)
            self.assertFalse(ok, (d.shape, d.shape_conf, d.shape_path))
            self.assertTrue(why)

    def test_yaw_unknown_cases(self):
        self.assertIsNone(ins.object_yaw(_det("cube", yaw=None), "cube"))
        self.assertIsNone(ins.object_yaw(_det("cube", yaw=np.nan), "cube"))
        self.assertIsNone(ins.object_yaw(_det("triangle", apex_yaw=None), "triangle"))
        self.assertIsNone(ins.object_yaw(_det("triangle", apex_yaw=1.0, apex_conf=0.3), "triangle"))
        self.assertEqual(ins.object_yaw(_det("triangle", apex_yaw=1.0, apex_conf=0.8), "triangle"), 1.0)
        self.assertEqual(ins.object_yaw(_det("cylinder", yaw=None), "cylinder"), 0.0)


class TestPolicyWiring(unittest.TestCase):
    INFO = {"task": "insert_shapes", "instruction": INSERT_INSTRUCTION}

    def test_flag_registered_and_off(self):
        self.assertIn("policy.insert_shapes", FLAGS)
        cfg = load_config(overlay="none")
        self.assertFalse(cfg["policy"]["insert_shapes"])
        self.assertFalse(load_config(overlay="finals_r23")["policy"]["insert_shapes"])

    def test_flag_off_is_the_guard_path(self):
        for guard in (True, False):
            st = Stub(insert_shapes=False, insert_guard=guard)
            m, guarded = ClassicalPolicy._episode_mapping(st, dict(self.INFO))
            self.assertEqual(m, {"default": "abstain"} if guard else {"default": None})
            self.assertEqual(guarded, {"default": "abstain"} if guard else None)
            self.assertFalse(st._insert_episode)

    def test_flag_on_maps_every_colour_to_the_lid_bin_and_beats_the_guard(self):
        for guard in (True, False):
            st = Stub(insert_shapes=True, insert_guard=guard)
            m, guarded = ClassicalPolicy._episode_mapping(st, dict(self.INFO))
            self.assertEqual(set(m), set(COLORS) | {"default"})
            self.assertEqual(set(m.values()), {SC.lid_bin})
            self.assertIsNone(guarded)
            self.assertTrue(st._insert_episode)

    def test_detection_by_sentence_and_task_only(self):
        for info, want in ((self.INFO, True), ({"instruction": INSERT_INSTRUCTION}, True),
                           ({"task": "insert_shapes"}, True),
                           ({"task": "sort_fixed", "instruction": "Put the red objects in the left bin."}, False),
                           ({"task": "sort_language", "instruction": "Put the cube in the hole."}, False),
                           ({"instruction": "Put the lid on the left bin."}, False), ({}, False)):
            st = Stub(insert_shapes=True)
            ClassicalPolicy._episode_mapping(st, dict(info))
            self.assertEqual(st._insert_episode, want, info)

    def test_explicit_mapping_wins(self):
        st = Stub(insert_shapes=True)
        m, _ = ClassicalPolicy._episode_mapping(st, dict(self.INFO, mapping={"red": "center"}))
        self.assertEqual(m, {"red": "center"})
        self.assertFalse(st._insert_episode)

    def test_never_raises(self):
        st = Stub(insert_shapes=True)
        st.scene = None
        with mock.patch.object(Scene, "load", side_effect=RuntimeError("boom")):
            self.assertIsNone(ClassicalPolicy._insert_shapes_mapping(st, self.INFO))


class TestPrintedLid(unittest.TestCase):
    """insert.hole_offsets_mat: the printed 200 mm lid has its square / circle holes in the NEAR row (reachable)."""
    OFF = {"cube": [-0.048, -0.035], "cylinder": [0.048, -0.035]}

    def test_offsets_move_the_targets(self):
        t = ins.hole_targets(SC, offsets_mat=self.OFF, margin=0.0075)
        np.testing.assert_allclose(t["cube"]["xy"], [0.505 - 0.035, 0.26 - 0.048], atol=1e-9)
        np.testing.assert_allclose(t["cylinder"]["xy"], [0.505 - 0.035, 0.26 + 0.048], atol=1e-9)
        self.assertAlmostEqual(t["cube"]["margin"], 0.0075)
        np.testing.assert_allclose(t["triangle"]["xy"], ins.hole_targets(SC)["triangle"]["xy"])   # not overridden
        self.assertTrue(np.allclose(ins.hole_targets(SC, offsets_mat=None)["cube"]["xy"], ins.hole_targets(SC)["cube"]["xy"]))

    def test_end_to_end_plan_over_the_near_row_hole(self):
        """The whole plan_insert path on the real IK: pick a cube / cylinder, carry, lower over the printed-lid hole."""
        cfg = load_config(overlay="finals_r23")
        cfg["policy"]["insert_shapes"] = True
        cfg["insert"]["hole_offsets_mat"] = self.OFF
        cfg["insert"]["hole_margin_m"] = 0.0075
        pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
        pl.insert_active = True
        arm0, hand0 = home_arm(cfg), np.zeros(14)
        ok = []
        for d in (_det("cylinder", xy=(0.30, 0.15)), _det("cube", yaw=0.0, xy=(0.30, 0.15)),
                  _det("cube", yaw=0.3, xy=(0.30, 0.20))):
            p = pl.plan_insert(d, arm0, hand0)
            ok.append(p.ok)
            if p.ok:
                self.assertEqual(p.side, "left")
                self.assertEqual(p.safety_clips, 0)
                tgt = ins.hole_targets(SC, offsets_mat=self.OFF, margin=0.0075)[d.shape]
                e = next(e for n, s, e in p.phases if n == "carry")
                pt, axis = pl.kin.grasp_pose("left", p.arm[e - 1])
                self.assertLess(np.linalg.norm(pt[:2] - tgt["xy"]), 0.004)
                self.assertLess(np.degrees(np.arccos(-axis[2])), 8.0)
        self.assertTrue(any(ok), "no near-row plan found at all")


class TestPlanner(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(overlay="finals_r23")
        cls.cfg["policy"]["insert_shapes"] = True

    def planner(self):
        return Planner(self.cfg, SC, hand_postures(self.cfg), home_arm(self.cfg))

    def test_inactive_planner_is_stock(self):
        pl = self.planner()
        self.assertFalse(pl.insert_active)
        self.assertIsNone(pl.insert_target)
        with mock.patch.object(pl, "plan_insert") as pi, mock.patch.object(pl, "_plan_core", return_value="stock") as core:
            self.assertEqual(pl.plan(_det("cube"), np.zeros(14), np.zeros(14), bin_name="left"), "stock")
        pi.assert_not_called()
        core.assert_called_once()

    def test_active_planner_routes_to_plan_insert(self):
        pl = self.planner()
        pl.insert_active = True
        with mock.patch.object(pl, "plan_insert", return_value="ins") as pi:
            self.assertEqual(pl.plan(_det("cube"), np.zeros(14), np.zeros(14)), "ins")
        pi.assert_called_once()

    def test_abstains_on_bad_detections_without_touching_ik(self):
        pl = self.planner()
        with mock.patch.object(pl, "_plan_side") as side, mock.patch.object(pl, "_plan_side_legacy") as leg:
            for d in (_det(None), _det("cube", conf=0.2), _det("cube", path="blob"), _det("cube", yaw=None),
                      _det("triangle", apex_yaw=None)):
                p = pl.plan_insert(d, home_arm(self.cfg), np.zeros(14))
                self.assertFalse(p.ok)
                self.assertTrue(p.reason.startswith("insert:"), p.reason)
        side.assert_not_called()
        leg.assert_not_called()

    def test_state_is_restored_after_planning(self):
        pl = self.planner()
        hint, w = pl.kin.hint.copy(), pl.kin.w_hint
        fail = Plan(False, reason="IK grasp left")
        with mock.patch.object(pl, "_plan_side", return_value=fail), mock.patch.object(pl, "_plan_side_legacy", return_value=fail):
            for d in (_det("cube"), _det("cylinder"), _det("triangle", apex_yaw=0.4, apex_conf=0.9)):
                self.assertFalse(pl.plan_insert(d, home_arm(self.cfg), np.zeros(14)).ok)
        self.assertIsNone(pl.insert_target)
        self.assertIsNone(pl._carry_hint)
        np.testing.assert_array_equal(pl.kin.hint, hint)
        self.assertEqual(pl.kin.w_hint, w)

    def test_a_passing_plan_is_rejected_when_a_budget_fails(self):
        """_insert_check is the last gate: a stock-OK plan that misses the hole is not released over the lid."""
        pl = self.planner()
        tgt = ins.hole_targets(SC)["cylinder"]
        arm = home_arm(self.cfg)
        p = Plan(True, side="left", arm=np.tile(arm, (5, 1)), hand=np.zeros((5, 14)),
                 phases=[("descend", 0, 2), ("carry", 2, 5)])
        ic, g = self.cfg["insert"], self.cfg["grasp"]
        why = pl._insert_check(p, "left", tgt, "cylinder", 0.0, None, None, tgt["z_top"], ic, g)
        self.assertIn("off the hole centre", why)      # home pose is nowhere near the hole

    def test_real_reach_every_hole_is_out_of_reach(self):
        """Documents the measured fact behind the abstentions: with the real IK no held object reaches any hole centre."""
        pl = self.planner()
        arm0 = home_arm(self.cfg)
        for shape in SHAPES:
            tgt = ins.hole_targets(SC)[shape]
            z = tgt["z_top"] + 0.006 + 0.5 * SC.size
            _, ok, err, _ = pl.kin.solve("left", np.r_[tgt["xy"], z], arm0, top_down=True,
                                         extra_seeds=[arm0[:7]])
            self.assertFalse(ok, shape)
            self.assertGreater(err, 0.02, shape)       # >= 2 cm short (tolerance is 6 mm)

    def test_end_to_end_abstains_today(self):
        """Full plan_insert on a real IK: every shape abstains with an IK reason (0 pts, no lid release)."""
        pl = self.planner()
        arm0, hand0 = home_arm(self.cfg), np.zeros(14)
        for d in (_det("cube", yaw=0.2, xy=(0.30, 0.15)), _det("cylinder", xy=(0.30, 0.15)),
                  _det("triangle", apex_yaw=1.0, apex_conf=0.9, xy=(0.30, 0.15))):
            p = pl.plan_insert(d, arm0, hand0)
            self.assertFalse(p.ok, d.shape)
            self.assertTrue(p.reason.startswith("insert:"))


if __name__ == "__main__":
    unittest.main()
