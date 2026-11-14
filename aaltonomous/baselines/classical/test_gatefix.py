"""Synthetic regressions for the RGB recall gates in finals_r22_gate."""
from __future__ import annotations

import copy
import time
import unittest

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import FLAGS, _get, hand_postures, load_config
from .mock_world import Renderer, make_mock_env
from .perception import Detection, Perception
from .policy import ClassicalPolicy


SC, CAL = Scene.load(), Calibration.load()
H_OBJ = SC.object_height()
NEW_FLAGS = {
    "perception.reject_log", "perception.gate_below_bins", "perception.tape_value_cut",
    "perception.split_by_solidity", "perception.color_gate.sat_floor_params.blue",
    "perception.color_gate.fk_silhouette", "perception.color_gate.fk_dilate_px",
    "perception.sim_center_principal_point", "perception.aim_z_frac", "perception.still_vote",
}


def rgb_frame(draw):
    hsv = np.zeros((480, 640, 3), np.uint8)
    draw(hsv)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def fake_perception(**overrides):
    base = {"perception.hsv_set": "robust_v2", "perception.classify_shape": False,
            "perception.color_gate.enabled": False, "perception.hue_guard.enabled": False}
    base.update(overrides)
    p = Perception(load_config(profile="none", overlay="none", overrides=base), SC, CAL, log=lambda _m: None)
    p._reject = lambda _det, bins_first=False: None
    p._from_plane = lambda color, uv, area, n: Detection(
        color, np.array([0.30, 0.0, SC.counter_z + H_OBJ]), area, n, "plane", uv=tuple(map(float, uv)))
    return p


class TestGatefixFlags(unittest.TestCase):
    def test_registered_default_off_and_overlay_adds_exactly_the_new_flags(self):
        base, gate = load_config(overlay="finals_r22"), load_config(overlay="finals_r22_gate")
        self.assertTrue(NEW_FLAGS <= set(FLAGS))
        changed = {key for key in FLAGS if _get(base, key) != _get(gate, key)}
        self.assertEqual(changed, NEW_FLAGS)

    def test_flags_off_detection_is_byte_identical_when_new_config_is_absent(self):
        cfg = load_config(overlay="finals_r21")
        legacy = copy.deepcopy(cfg)
        for key in ("sim_center_principal_point", "aim_z_frac", "split_by_solidity",
                    "split_by_solidity_params", "tape_value_cut", "tape_value_cut_params",
                    "gate_below_bins", "gate_below_bins_params", "reject_log", "still_vote",
                    "still_vote_params"):
            legacy["perception"].pop(key, None)
        legacy["perception"]["color_gate"].pop("fk_silhouette", None)
        obj = dict(color="yellow", shape="cube", yaw=0.0, size=SC.size, height=H_OBJ,
                   pos=SC.mat_to_pelvis([0.45, 0.15, H_OBJ / 2]))
        rgb, depth = Renderer(SC, CAL).render([obj])
        a = Perception(cfg, SC, CAL, log=lambda _m: None).detect(rgb, depth, debug=True)
        b = Perception(legacy, SC, CAL, log=lambda _m: None).detect(rgb, depth, debug=True)
        self.assertEqual([repr(d) for d in a[0]], [repr(d) for d in b[0]])
        self.assertEqual([(repr(d), why) for d, why in a[1]], [(repr(d), why) for d, why in b[1]])
        self.assertEqual(np.asarray([d.top for d in a[0]]).tobytes(), np.asarray([d.top for d in b[0]]).tobytes())


class TestRGBGates(unittest.TestCase):
    def test_tape_is_rejected_but_hue_alone_never_rejects_a_thick_object(self):
        p = fake_perception(**{"perception.tape_value_cut": True, "perception.reject_log": True})
        tape = rgb_frame(lambda hsv: cv2.rectangle(hsv, (80, 300), (300, 309), (10, 170, 115), -1))
        dets, rejected = p.detect(tape, None, debug=True)
        self.assertEqual(dets, [])
        self.assertIn("tape_value_cut", [why for _, why in rejected])
        self.assertEqual(p.rejection_table()["red"]["tape_value_cut"], 1)
        thick = rgb_frame(lambda hsv: cv2.rectangle(hsv, (160, 260), (209, 309), (10, 170, 115), -1))
        self.assertEqual(len(p.detect(thick, None)), 1)

    def test_bin_rim_rejected_and_shifted_far_row_object_accepted(self):
        p = fake_perception(**{"perception.gate_below_bins": True})

        def draw(hsv):
            cv2.rectangle(hsv, (40, 90), (600, 219), (5, 110, 110), -1)   # observed bin/tape mass
            cv2.rectangle(hsv, (280, 230), (329, 269), (115, 180, 200), -1)  # object below a +10 cm-shifted row

        dets, rejected = p.detect(rgb_frame(draw), None, debug=True)
        self.assertEqual([d.color for d in dets], ["blue"])
        self.assertIn("gate_below_bins", [why for _d, why in rejected])
        self.assertTrue(p._image_bin_gate_active)

    def test_reject_log_includes_components_below_the_detection_area_floor(self):
        p = fake_perception(**{"perception.reject_log": True})
        tiny = rgb_frame(lambda hsv: cv2.rectangle(hsv, (300, 350), (307, 357), (10, 200, 200), -1))
        self.assertEqual(p.detect(tiny, None), [])
        self.assertEqual(p.rejection_table()["red"]["min_area"], 1)

    def test_touching_cube_and_prism_split_into_two_and_reclassified(self):
        p = fake_perception(**{"perception.split_by_solidity": True})

        def draw(hsv):
            cv2.rectangle(hsv, (100, 340), (134, 374), (10, 200, 200), -1)
            cv2.fillConvexPoly(hsv, np.array([[134, 340], [180, 357], [134, 374]], np.int32), (10, 200, 200))

        dets = p.detect(rgb_frame(draw), None)
        self.assertEqual(len(dets), 2)
        self.assertEqual([d.color for d in dets], ["red", "red"])

    def test_compact_blob_still_uses_the_existing_area_splitter(self):
        p = fake_perception(**{"perception.split_by_solidity": True, "perception.split_blobs": True})
        p.expected_area_px = lambda *_args: 1000.0
        pair = rgb_frame(lambda hsv: cv2.rectangle(hsv, (100, 330), (179, 369), (10, 200, 200), -1))
        self.assertGreater(len(p.detect(pair, None)), 1)

    def test_blue_floor_is_only_the_registered_parameter_variant(self):
        base = Perception(load_config(overlay="finals_r21"), SC, CAL, log=lambda _m: None)
        gate = Perception(load_config(overlay="finals_r22_gate"), SC, CAL, log=lambda _m: None)
        det = Detection("blue", np.array([0.3, 0.0, SC.counter_z + H_OBJ]), 2000, 0, "plane", sat_med=90)
        self.assertTrue(base._color_gate_sat_low(det, base.color_gate_cfg))
        self.assertFalse(gate._color_gate_sat_low(det, gate.color_gate_cfg))
        self.assertEqual(gate.color_gate_cfg["sat_floor_params"]["yellow"], 45)


class TestGeometryAndVoting(unittest.TestCase):
    def test_filled_fk_silhouette_keeps_all_25_visible_spawn_objects(self):
        # The mock table sits at the nominal height; finals overlays carry the real-robot no-depth prior (0.15 m),
        # which a single look (apply_height not yet committed) would use and so mask every object out.
        # near_margin_m 0.10 (r21/r22 New H2/M1) drops the x=0.31 row on the mock bins; isolated like test_recall.
        SC.counter_z_prior = None
        cfg = load_config(overlay="finals_r22_gate", overrides={
            "perception.still_vote": False, "perception.gate_below_bins": False,
            "perception.split_by_solidity": False, "perception.tape_value_cut": False,
            "scene.counter_z_prior": None, "perception.color_gate.near_margin_m": 0.0})   # isolate FK from fixfk's near-bin hedge
        arm = np.asarray(cfg["home_arm_q"], float)
        hand = hand_postures(cfg)["open"]
        renderer = Renderer(SC, CAL)
        seen = 0
        for i, x in enumerate(np.linspace(0.23, 0.31, 5)):
            for j, y in enumerate(np.linspace(-0.20, 0.20, 5)):
                mat = SC.pelvis_to_mat(np.array([x, y, SC.counter_z]))[:2]
                obj = dict(color=("red", "yellow", "blue")[(i + j) % 3], shape="cube", yaw=0.0,
                           size=SC.size, height=H_OBJ,
                           pos=SC.mat_to_pelvis([float(mat[0]), float(mat[1]), H_OBJ / 2]))
                rgb, depth = renderer.render([obj])
                p = Perception(cfg, SC, CAL, log=lambda _m: None)
                dets = p.detect(rgb, depth, robot_q=(arm, hand))
                seen += len(dets) == 1
                self.assertGreater(p.fk_mask_px, 0)
        self.assertEqual(seen, 25)

    def test_no_depth_speckle_floor_counts_only_the_visible_object_footprint(self):
        cfg = load_config(overlay="finals_r22_gate", overrides={"perception.still_vote": False})
        p = Perception(cfg, SC, CAL, log=lambda _m: None)
        p._color_gate_depth = lambda _top: None
        p._last_arm_mask = np.zeros((480, 640), dtype=bool)
        p._last_arm_mask[284:316, 296:328] = True
        det = Detection("red", np.array([0.30, 0.0, SC.counter_z + H_OBJ]), 400, 0, "plane", uv=(320, 300))
        self.assertIsNone(p._color_gate_why(det, p.color_gate_cfg))
        p.fk_silhouette = False
        self.assertEqual(p._color_gate_why(det, p.color_gate_cfg), "color_gate:speckle_no_z")

    def test_sim_center_is_319_5_239_5_and_real_keeps_file_intrinsics(self):
        sim = Perception(load_config(profile="sim", overlay="none",
                                     overrides={"perception.sim_center_principal_point": True}), SC, CAL)
        real = Perception(load_config(profile="none", overlay="none",
                                      overrides={"perception.sim_center_principal_point": True}), SC, CAL)
        self.assertEqual(tuple(sim.cal.cams[sim.cam].K[:2, 2]), (319.5, 239.5))
        np.testing.assert_array_equal(real.cal.cams[real.cam].K, CAL.cams[real.cam].K)

    def test_aim_fraction_moves_plane_intersection_toward_top_face(self):
        a = Perception(load_config(profile="none", overlay="none"), SC, CAL)
        b = Perception(load_config(profile="none", overlay="none",
                                   overrides={"perception.aim_z_frac": 0.1}), SC, CAL)
        da, db = a._from_plane("red", (320, 300), 1000, 0), b._from_plane("red", (320, 300), 1000, 0)
        self.assertEqual(da.top[2], db.top[2])
        self.assertGreater(np.linalg.norm(da.top[:2] - db.top[:2]), 0.001)

    def test_three_frame_vote_is_non_blocking_and_requires_two_hits(self):
        p = fake_perception(**{"perception.still_vote": True})
        present = rgb_frame(lambda hsv: cv2.rectangle(hsv, (280, 340), (329, 389), (10, 200, 200), -1))
        empty = np.zeros_like(present)
        t0 = time.perf_counter()
        self.assertIsNone(p.detect(present, None, still_frame=True))
        self.assertIsNone(p.detect(empty, None, still_frame=True))
        voted = p.detect(present, None, still_frame=True)
        self.assertEqual(len(voted), 1)
        self.assertLess((time.perf_counter() - t0) * 1000, 100.0)

    def test_act_stays_above_27_hz_while_vote_detection_runs(self):
        env = make_mock_env(seed=0)
        obs = env.reset()
        policy = ClassicalPolicy(config=load_config(overlay="finals_r22_gate"), log=None, safety=env.filter.cfg)
        policy.reset({"instruction": obs["instruction"]})
        policy.state, policy._bv_done, policy.pitch_calibration = "look", True, True
        policy._held_cal_done = True   # rc7 held empty-close calibration needs a moving obs clock; this test freezes it
        calls = []

        def three_frame_vote(*_args, **_kwargs):
            time.sleep(0.03)
            calls.append(1)
            return None if len(calls) < 3 else []

        policy.perception.detect = three_frame_vote
        elapsed = []
        for _ in range(100):
            start = time.perf_counter()
            policy.act(obs)
            elapsed.append(time.perf_counter() - start)
            if policy.state == "done":
                break
            time.sleep(0.005)
        self.assertEqual(len(calls), 3)
        self.assertEqual(policy.state, "done")
        self.assertLess(max(elapsed), 1.0 / 27.0)

    def test_production_overlay_added_cost_under_five_ms_per_frame(self):
        obj = dict(color="yellow", shape="cube", yaw=0.0, size=SC.size, height=H_OBJ,
                   pos=SC.mat_to_pelvis([0.45, 0.15, H_OBJ / 2]))
        rgb, depth = Renderer(SC, CAL).render([obj])

        def production(overlay):
            cfg = load_config(overlay=overlay, overrides={"perception.still_vote": False, "scene.counter_z_prior": None})
            robot_q = (np.asarray(cfg["home_arm_q"], float), hand_postures(cfg)["open"])
            return Perception(cfg, SC, CAL, log=lambda _m: None), robot_q

        base, base_q = production("finals_r22")
        gate, gate_q = production("finals_r22_gate")
        for _ in range(5):
            base.detect(rgb, depth, robot_q=base_q)
            gate.detect(rgb, depth, robot_q=gate_q)

        def median_ms(p, robot_q):
            samples = []
            for _ in range(20):
                start = time.perf_counter()
                p.detect(rgb, depth, robot_q=robot_q)
                samples.append((time.perf_counter() - start) * 1000)
            return float(np.median(samples))

        self.assertLess(median_ms(gate, gate_q) - median_ms(base, base_q), 5.0)


if __name__ == "__main__":
    unittest.main()
