"""Kinematic and wiring tests for grasp.cage_mode and grasp.tri_vertex_align.

Run: ../v16/bin/python -m pytest aaltonomous/baselines/classical/test_cage_grasp.py -q
"""
import xml.etree.ElementTree as ET
from types import SimpleNamespace

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import DEFAULT_URDF, JointSpec
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .perception import Detection, Perception
from .planner import Planner


SCENE = Scene.load()


def _planner(overlay="finals_r22_grasp", **overrides):
    cfg = load_config(overlay=overlay, overrides=overrides)
    return cfg, Planner(cfg, SCENE, hand_postures(cfg), home_arm(cfg))


def _det(shape, **kw):
    d = dict(shape=shape, shape_conf=0.9, shape_path="metric", apex_gate=(0.9, 60.0))
    d.update(kw)
    return SimpleNamespace(**d)


class TestCageKinematics:
    def setup_method(self):
        self.cfg = load_config(overlay="finals_r22_grasp")
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)
        self.kin = ArmKinematics(self.cfg, self.spec)

    def test_shape_commands_are_clipped_inside_urdf_limits(self):
        limits = {}
        for joint in ET.parse(DEFAULT_URDF).getroot().iter("joint"):
            lim = joint.find("limit")
            if lim is not None and joint.get("name") in self.spec.hand_names:
                limits[joint.get("name")] = (float(lim.get("lower")), float(lim.get("upper")))
        assert set(limits) == set(self.spec.hand_names)

        base = np.asarray(self.cfg["grasp"]["cage_finger_posture"], float)
        closed = np.asarray(self.cfg["hand"]["closed"], float)
        for shape in ("cylinder", "triangle"):
            command = self.post[f"cage_{shape}"]
            raw_left = base.copy()
            raw_left[1] = closed[1] + self.cfg["grasp"]["cage_thumb_extra_rad"][shape]
            np.testing.assert_allclose(command[:7], np.clip(raw_left, self.spec.hand_lower[:7],
                                                             self.spec.hand_upper[:7]))
            for name, q in zip(self.spec.hand_names, command):
                lo, hi = limits[name]
                assert lo - 1e-12 <= q <= hi + 1e-12, (shape, name, q, lo, hi)
        assert "cage_cube" not in self.post

    def test_point_is_closed_pad_midpoint_minus_half_nominal_object_width(self):
        point = np.asarray(self.cfg["grasp"]["cage_point_in_palm"], float)
        closed_tips = self.kin.tips_in_palm("left", self.post["closed"])
        pad_midpoint = closed_tips[:2].mean(0)
        nominal_width = np.mean([0.040, 0.060])
        derived = pad_midpoint - np.array([nominal_width / 2.0, 0.0, 0.0])
        # The configured point is the FK result rounded to the requested millimetre.
        np.testing.assert_allclose(point, np.round(derived, 3), atol=1e-12)

    def test_40_to_60_mm_object_centres_stay_within_ten_mm_of_fk_gap_centre(self):
        left_point = np.asarray(self.cfg["grasp"]["cage_point_in_palm"], float)
        for side, point in (("left", left_point), ("right", left_point * [1.0, -1.0, 1.0])):
            pads = self.kin.tips_in_palm(side, self.post["closed"])[:2]
            pad_midpoint = pads.mean(0)
            for width in np.linspace(0.040, 0.060, 21):
                fk_centre = pad_midpoint - np.array([width / 2.0, 0.0, 0.0])
                assert np.max(np.abs(point - fk_centre)) <= 0.010
                assert abs(fk_centre[2] - pads[:, 2].mean()) <= 1e-12
                assert pads[:, 2].min() < fk_centre[2] < pads[:, 2].max()

    def test_finger_posture_is_closed_tight_plus_ten_mm_fk_flexion(self):
        base = np.asarray(self.cfg["hand"]["closed_tight"], float)
        target = np.asarray(self.cfg["grasp"]["cage_finger_posture"], float)
        np.testing.assert_allclose(target[[0, 1, 2, 4, 6]], base[[0, 1, 2, 4, 6]])
        assert target[3] == target[5] < base[3] == base[5]
        base_tips = self.kin.tips_in_palm("left", np.r_[base, -base[[0, 1, 2, 5, 6, 3, 4]]])
        target_tips = self.kin.tips_in_palm("left", np.r_[target, -target[[0, 1, 2, 5, 6, 3, 4]]])
        inward = base_tips[:2, 0] - target_tips[:2, 0]
        np.testing.assert_allclose(inward, 0.010, atol=0.0001)

    def test_thumb_extras_command_eight_to_twelve_mm_at_shape_surface(self):
        closed = self.post["closed"].copy()
        closed_thumb = self.kin.tips_in_palm("left", closed)[2]
        # At the rear centre the cylinder normal is palm x. With a triangle vertex line along the finger gap, the
        # contacted equilateral-prism side has palm-x normal component cos(30 deg).
        normal_x = {"cylinder": 1.0, "triangle": np.cos(np.radians(30.0))}
        for shape, projection in normal_x.items():
            thumb = self.kin.tips_in_palm("left", self.post[f"cage_{shape}"])[2]
            penetration = projection * (thumb[0] - closed_thumb[0])
            assert 0.008 <= penetration <= 0.012
            np.testing.assert_allclose(penetration, 0.010, atol=0.0001)


class TestCagePlanning:
    def test_shape_choice_and_point_offset(self):
        cfg, planner = _planner()
        cyl = planner._posture_choice(_det("cylinder"))
        tri = planner._posture_choice(_det("triangle"))
        cube = planner._posture_choice(_det("cube"))
        assert (cyl[0], tri[0], cube[0]) == ("cage_cylinder", "cage_triangle", "closed_tight")
        np.testing.assert_allclose(cyl[1], cfg["grasp"]["cage_point_in_palm"])
        np.testing.assert_allclose(tri[1], cfg["grasp"]["cage_point_in_palm"])
        assert cube[1] is None
        assert planner._posture_choice(_det("cylinder", shape_conf=0.49))[0] == "closed_tight"
        assert planner._posture_choice(_det("triangle", shape_path="blob"))[0] == "closed_tight"

    def test_enabled_shapes_reach_cage_commands_and_triangle_heading(self):
        cfg, planner = _planner()
        post = hand_postures(cfg)
        layout = SCENE.cfg["layout"]
        top = SCENE.mat_to_pelvis([np.mean(layout["region_x"]) + 0.08, np.mean(layout["region_y"]),
                                   SCENE.object_height()])
        for shape, yaw in (("cylinder", None), ("triangle", 0.37)):
            det = Detection("yellow", top.copy(), 0, 0, "depth", yaw=yaw, shape=shape, shape_conf=0.9,
                            shape_path="metric", apex_gate=(0.9, 60.0))
            plan = planner.plan(det, home_arm(cfg), post["open"])
            assert plan.ok, plan.reason
            assert plan.posture[0] == f"cage_{shape}"
            close_end = next(end for name, _start, end in plan.phases if name == "close") - 1
            sl = ArmKinematics.sl(plan.side)
            np.testing.assert_array_equal(plan.hand[close_end][sl], post[f"cage_{shape}"][sl])
            if shape == "triangle":
                assert plan.reason.startswith("vertex-aligned")

    def test_cube_plan_is_identical_with_both_new_flags_on(self):
        cfg21, p21 = _planner("finals_r21")
        cfg22, p22 = _planner("finals_r22_grasp")
        layout = SCENE.cfg["layout"]
        top = SCENE.mat_to_pelvis([np.mean(layout["region_x"]) + 0.08, np.mean(layout["region_y"]),
                                   SCENE.object_height()])
        det = Detection("yellow", top, 0, 0, "depth", yaw=0.37, shape="cube", shape_conf=0.9,
                        shape_path="metric")
        a = p21.plan(det, home_arm(cfg21), hand_postures(cfg21)["open"])
        b = p22.plan(det, home_arm(cfg22), hand_postures(cfg22)["open"])
        assert a.ok and b.ok, (a.reason, b.reason)
        assert a.phases == b.phases and a.waypoints.keys() == b.waypoints.keys()
        np.testing.assert_array_equal(a.arm, b.arm)
        np.testing.assert_array_equal(a.hand, b.hand)
        for key in a.waypoints:
            np.testing.assert_array_equal(a.waypoints[key], b.waypoints[key])

    def test_flags_off_add_no_postures_or_triangle_dispatch(self):
        cfg, planner = _planner("none")
        assert not any(name.startswith("cage_") for name in hand_postures(cfg))
        assert not planner._yaw_align_on(_det("triangle", yaw=0.4, apex_yaw=None, apex_conf=0.0))


class TestTriangleVertexAlign:
    @staticmethod
    def _wrap_period(a, period):
        return (a + period / 2) % period - period / 2

    def test_synthetic_yaw_aligns_gap_with_a_vertex_mod_120(self):
        _cfg, planner = _planner()
        yaw = np.radians(37.0)
        det = _det("triangle", yaw=yaw, apex_yaw=None, apex_conf=0.0)
        assert planner._yaw_align_on(det)
        resolved, vertex_mode = planner._triangle_yaw(det)
        assert vertex_mode
        assert abs(self._wrap_period(resolved - yaw, 2 * np.pi / 3)) < 1e-12
        for side, gap_from_x in (("left", -np.pi / 2), ("right", np.pi / 2)):
            hints = planner.vertex_hint_angles(resolved, side)
            assert hints
            for heading in hints:
                gap_yaw = heading + gap_from_x
                assert abs(self._wrap_period(gap_yaw - resolved, 2 * np.pi / 3)) < 1e-12

    def test_confident_pose_precedes_rectangle_and_top_face_yaw(self):
        _cfg, planner = _planner()
        det = _det("triangle", apex_yaw=0.8, apex_conf=0.9, vertex_yaw=0.4, yaw=0.2)
        assert planner._triangle_yaw(det) == (0.8, True)
        det.apex_conf = 0.1
        assert planner._triangle_yaw(det) == (0.4, True)
        det.vertex_yaw = None
        assert planner._triangle_yaw(det) == (0.2, True)

    def test_blob_min_area_rect_fallback_produces_finite_robot_yaw(self):
        cfg = load_config(overlay="finals_r22_grasp")
        perception = Perception(cfg, SCENE, Calibration.load(), log=None)
        mask = np.zeros((480, 640), np.uint8)
        vertices = np.array([[320, 180], [270, 285], [385, 275]], np.int32)
        import cv2
        cv2.fillConvexPoly(mask, vertices, 1)
        yaw = perception._blob_vertex_yaw(mask, SCENE.counter_z + SCENE.object_height())
        assert yaw is not None and np.isfinite(yaw)
        assert 0.0 <= yaw < 2.0 * np.pi / 3.0
