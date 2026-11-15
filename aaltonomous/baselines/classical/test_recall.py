"""Synthetic regression tests for the five finals_r22 recall flags.

Run: python -m unittest aaltonomous.baselines.classical.test_recall -v
"""
from __future__ import annotations

from dataclasses import fields
import pickle
import statistics
import time
import types
import unittest

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .perception import Detection, Perception
from .policy import ClassicalPolicy


SC, CAL = Scene.load(), Calibration.load()
H, W = 480, 640
CLEAR_TOP = SC.mat_to_pelvis([0.45, 0.155, SC.object_height()])


def _perception(**overrides):
    # Synthetic frames are drawn at the nominal counter with no depth: pin the r21 additions that change that geometry
    # (counter_z_prior 0.15 plane, fixfk near-bin margin / FK dilation) so each test isolates its recall flag.
    base = {"scene.counter_z_prior": None, "perception.color_gate.near_margin_m": 0.0,
            "perception.color_gate.fk_dilate_px": 0}
    cfg = load_config(overlay="finals_r21", overrides={**base, **overrides})
    return Perception(cfg, SC, CAL, log=lambda *_: None)


def _canvas():
    hsv = np.zeros((H, W, 3), np.uint8)
    hsv[..., 2] = 180
    return hsv


def _rgb(hsv):
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)


def _uv(p, top=CLEAR_TOP):
    u, v, _ = p._project(top)
    return int(round(u)), int(round(v))


class TestRecallSyntheticFrames(unittest.TestCase):
    def test_all_recall_flags_off_match_config_without_recall_keys(self):
        self.assertEqual([f.name for f in fields(Detection)],
                         ["color", "top", "n_px", "n_depth", "method", "yaw", "uv", "shape", "apex_yaw",
                          "apex_conf", "vertex_yaw", "shape_conf", "apex_gate", "shape_path", "generic_required",
                          "generic_width", "generic_confidence", "shape_diag", "color_conf", "hsv_med", "sat_med", "prism_like", "border_touch", "hue_mixed",
                          "novel", "label", "footprint", "height_m", "ungraspable", "ungraspable_reason", "prism_orientation", "ridge_yaw",
                          "prism_xy", "prism_conf"])
        cfg = load_config(overlay="finals_r21")
        old = load_config(overlay="finals_r21")
        for key in ("sat_floor_adaptive", "sat_floor_adaptive_params", "area_floor_geometric",
                    "area_floor_geometric_params", "topface_mode", "topface_min_px",
                    "partial_ok_near_hand", "partial_ok_near_hand_params", "relook_before_done",
                    "relook_fresh_frames"):
            old["perception"].pop(key, None)
        current = Perception(cfg, SC, CAL, log=lambda *_: None)
        legacy = Perception(old, SC, CAL, log=lambda *_: None)
        hsv = _canvas()
        u, v = _uv(current)
        cv2.rectangle(hsv, (u - 26, v - 26), (u + 26, v + 26), (111, 120, 190), -1)
        a = current.detect(_rgb(hsv), None, debug=True)
        b = legacy.detect(_rgb(hsv), None, debug=True)
        self.assertEqual(pickle.dumps(a), pickle.dumps(b))
        self.assertEqual([repr(d) for d in a[0]], [repr(d) for d in b[0]])
        self.assertEqual([(repr(d), why) for d, why in a[1]], [(repr(d), why) for d, why in b[1]])

    def test_low_s_purple_detected_only_with_adaptive_floor(self):
        fixed = _perception()
        adaptive = _perception(**{"perception.sat_floor_adaptive": True})
        hsv = _canvas()
        u, v = _uv(fixed)
        cv2.rectangle(hsv, (u - 26, v - 26), (u + 26, v + 26), (111, 70, 190), -1)
        rgb = _rgb(hsv)
        old, rejected = fixed.detect(rgb, None, debug=True)
        self.assertFalse(any(d.color == "blue" for d in old))
        self.assertIn("color_gate:sat_floor", [why for _, why in rejected])
        found = adaptive.detect(rgb, None)
        self.assertEqual([d.color for d in found], ["blue"])
        self.assertAlmostEqual(adaptive._sat_floors["blue"], 45.0)

    def test_adaptive_floor_measures_purple_hued_counter_background(self):
        p = _perception(**{"perception.sat_floor_adaptive": True})
        hsv = np.zeros((20, 20, 3), np.uint8)
        hsv[...] = (111, 80, 180)
        hsv[8:12, 8:12] = (111, 100, 190)
        p._counter_image_mask = lambda shape: np.ones(shape, bool)
        masks = p._adaptive_saturation_masks(hsv, np.full((20, 20), 255, np.uint8))
        self.assertAlmostEqual(p._sat_floors["blue"], 90.0)
        self.assertTrue(masks["blue"][9, 9])
        self.assertFalse(masks["blue"][0, 0])
        background = Detection("blue", CLEAR_TOP.copy(), 3000, 0, "plane", sat_med=80.0)
        self.assertTrue(p._color_gate_sat_low(background, p.color_gate_cfg))

    def test_adaptive_flag_alone_does_not_enable_yellow_fixed_floor(self):
        p = _perception(**{"perception.color_gate.sat_floor": False,
                           "perception.sat_floor_adaptive": True})
        yellow = Detection("yellow", CLEAR_TOP.copy(), 3000, 0, "plane", sat_med=20.0)
        blue = Detection("blue", CLEAR_TOP.copy(), 3000, 0, "plane", sat_med=20.0)
        p._sat_floors["blue"] = 45.0
        self.assertFalse(p._color_gate_sat_low(yellow, p.color_gate_cfg))
        self.assertTrue(p._color_gate_sat_low(blue, p.color_gate_cfg))

    def test_small_orange_prism_detected_by_geometric_area(self):
        legacy = _perception()
        geometric = _perception(**{"perception.area_floor_geometric": True})
        hsv = _canvas()
        u, v = _uv(legacy)
        # About 894 px after opening: below the legacy square-based floor but
        # above 0.35 of the projected triangular-prism top face.
        tri = np.array([[u, v - 20], [u - 22, v + 18], [u + 22, v + 18]], np.int32)
        cv2.fillConvexPoly(hsv, tri, (12, 190, 210))
        rgb = _rgb(hsv)
        old, rejected = legacy.detect(rgb, None, debug=True)
        self.assertEqual(old, [])
        self.assertIn("color_gate:speckle", [why for _, why in rejected])
        found = geometric.detect(rgb, None)
        self.assertEqual(len(found), 1)
        self.assertEqual(found[0].color, "red")

    def test_projected_area_at_counter_z_015_matches_calibrated_pinhole(self):
        p = _perception(**{"perception.area_floor_geometric": True})
        top = np.array([0.305, 0.0, 0.15 + 0.045])
        cube = p._projected_top_area_px(top, 0.045 ** 2)
        prism = p._projected_top_area_px(top, 0.045 ** 2 / np.sqrt(3.0))
        self.assertAlmostEqual(cube, 3901.07, delta=0.1)
        self.assertAlmostEqual(prism, 2252.29, delta=0.1)

    def test_specular_yellow_cylinder_uses_whole_blob_vote(self):
        top35 = _perception(**{"perception.area_floor_geometric": True})
        whole = _perception(**{"perception.area_floor_geometric": True,
                               "perception.topface_mode": "whole_blob"})
        hsv = _canvas()
        u, v = _uv(top35)
        cv2.circle(hsv, (u, v), 20, (40, 80, 170), -1)
        # Bright red/orange glare occupies >half of the brightest 35%, but <half of the whole component.
        cv2.rectangle(hsv, (u - 10, v - 15), (u + 10, v - 1), (12, 160, 255), -1)
        rgb = _rgb(hsv)
        old, old_rej = top35.detect(rgb, None, debug=True)
        self.assertNotIn("yellow", [d.color for d in old] + [d.color for d, why in old_rej if why.startswith("hue_guard")])
        found, rej = whole.detect(rgb, None, debug=True)
        # whole_blob relabels the candidate yellow; H1 (fixh1h7) still abstains on a part whose source pixels hold
        # both red and yellow, so it is rejected as ambiguous (0, never -1) instead of being placed.
        self.assertEqual(found, [])
        self.assertEqual([(d.color, why) for d, why in rej if why.startswith("hue_guard")],
                         [("yellow", "hue_guard:ambiguous")])

    def test_low_s_thin_tape_stays_rejected(self):
        p = _perception(**{"perception.sat_floor_adaptive": True,
                           "perception.area_floor_geometric": True,
                           "perception.topface_mode": "whole_blob"})
        hsv = _canvas()
        u, v = _uv(p)
        cv2.rectangle(hsv, (u - 55, v - 2), (u + 55, v + 2), (111, 40, 160), -1)
        self.assertEqual(p.detect(_rgb(hsv), None), [])

    def test_bin_rim_stays_rejected(self):
        p = _perception(**{"perception.sat_floor_adaptive": True,
                           "perception.area_floor_geometric": True,
                           "perception.topface_mode": "whole_blob"})
        hsv = _canvas()
        cx, cy = SC.cfg["bins"]["centers"]["center"]
        rim = SC.mat_to_pelvis([cx, cy, SC.cfg["bins"]["wall_height"]])
        u, v = _uv(p, rim)
        cv2.circle(hsv, (u, v), 22, (12, 190, 190), -1)
        self.assertEqual(p.detect(_rgb(hsv), None), [])

    def test_fk_boundary_fragment_is_tentative_and_never_normal_without_flag(self):
        off = _perception()
        on = _perception(**{"perception.sat_floor_adaptive": True,
                            "perception.area_floor_geometric": True,
                            "perception.partial_ok_near_hand": True})
        hsv = _canvas()
        u, v = _uv(on)
        cv2.rectangle(hsv, (u - 9, v - 5), (u + 9, v + 5), (12, 190, 210), -1)
        rgb = _rgb(hsv)

        def mask(shape, *_):
            m = np.zeros(shape, bool)
            m[:, u:] = True
            return m

        off._fk_arm_mask = mask
        on._fk_arm_mask = mask
        robot_q = (np.zeros(14), np.zeros(14))
        self.assertEqual(off.detect(rgb, None, robot_q=robot_q), [])
        found = on.detect(rgb, None, robot_q=robot_q)
        self.assertEqual(len(found), 1)
        self.assertTrue(found[0].tentative)
        self.assertEqual(found[0].tentative_reason, "fk_mask_boundary")

    def test_added_detection_time_below_three_ms_at_640x480(self):
        base = _perception()
        recall = _perception(**{"perception.sat_floor_adaptive": True,
                                "perception.area_floor_geometric": True,
                                "perception.topface_mode": "whole_blob",
                                "perception.partial_ok_near_hand": True})
        hsv = _canvas()
        u, v = _uv(base)
        cv2.rectangle(hsv, (u - 26, v - 26), (u + 26, v + 26), (111, 90, 190), -1)
        rgb = _rgb(hsv)
        fk = np.zeros((H, W), bool)
        fk[100:300, :80] = True
        for p in (base, recall):
            p._fk_arm_mask = lambda *_: fk
            for _ in range(3):
                p.detect(rgb, None, robot_q=(np.zeros(14), np.zeros(14)))
        old, new = [], []
        for i in range(20):
            order = ((base, old), (recall, new)) if i % 2 == 0 else ((recall, new), (base, old))
            for p, samples in order:
                t0 = time.perf_counter()
                p.detect(rgb, None, robot_q=(np.zeros(14), np.zeros(14)))
                samples.append(time.perf_counter() - t0)
        added = statistics.median(new) - statistics.median(old)
        self.assertLess(added, 0.003, f"added detection time {1000 * added:.3f} ms")


class TestRecallPolicyGuards(unittest.TestCase):
    def test_tentative_candidate_is_not_given_to_picker(self):
        p = object.__new__(ClassicalPolicy)
        p.cfg = {"perception": {"partial_ok_near_hand": True,
                                "partial_ok_near_hand_params": {"max_relooks": 2}},
                 "policy": {"retry_pass": False}}
        p._partial_relooks = 0
        p._slots_on = lambda: False
        seen = []
        p._choose = lambda ds, obs, verify: (seen.extend(ds) or ("none", None, len(ds)))
        d = Detection("red", np.array([0.3, 0.0, 0.05]), 80, 0, "plane")
        d.tentative, d.tentative_reason = True, "fk_mask_boundary"
        res = ClassicalPolicy._choose_retry(p, [d], {"t": 0.0}, False)
        self.assertEqual(res[0], "partial")
        self.assertEqual(seen, [])

        d.tentative = False
        d.tentative_reason = None
        p._partial_relooks = 1
        res = ClassicalPolicy._choose_retry(p, [d], {"t": 1.0}, False)
        self.assertEqual(res[0], "none")
        self.assertEqual(seen, [d])

    def test_tentative_bin_fragment_cannot_be_corrected_or_cleaned_up(self):
        p = object.__new__(ClassicalPolicy)
        p.cfg = {"policy": {"verify_yellow_unknown": False}}
        p.mapping = {"red": "right"}
        p.verification_pending = None
        p.last_verification = None
        d = Detection("red", np.array([0.5, 0.0, 0.05]), 80, 0, "plane")
        d.tentative, d.tentative_reason = True, "fk_mask_boundary"
        self.assertEqual(ClassicalPolicy._review_bins(p, [(d, "left")], 0.0), [])
        self.assertIsNone(ClassicalPolicy._plan_corrections(p, [(d, "left", "right")], [d]))

    def test_nothing_left_starts_one_nonblocking_retracted_relook(self):
        p = object.__new__(ClassicalPolicy)
        p.cfg = {"perception": {"relook_before_done": True},
                 "motion": {"chunk": 10, "hold_ticks": 1},
                 "policy": {"episode_summary_log": False}}
        p.last_verification = None
        p._final_relook_used = False
        p._summary_logged = False
        p._shell = False
        p._stall = None
        p._rel = None
        p._gdiag = None             # release: _pop checks the carry read window
        p.arm_cmd = p.home = np.zeros(14)
        p.hand_cmd = np.zeros(14)
        p.post = {"open": np.zeros(14)}
        p.planner = types.SimpleNamespace(go_to=lambda *_: (np.empty((0, 14)), np.empty((0, 14))))
        p.events = []
        p.log = lambda *_: None
        first = ClassicalPolicy._after_look(p, ("none", None, 0), 1.0)
        self.assertFalse(first["done"])
        self.assertEqual(p.state, "relook_settle")
        self.assertTrue(p._final_relook_used)
        second = ClassicalPolicy._after_look(p, ("none", None, 0), 2.0)
        self.assertTrue(second["done"])
        self.assertEqual(p.state, "done")

    def test_final_relook_waits_for_two_fresh_frames_before_detection(self):
        p = ClassicalPolicy(config=load_config(overlay="finals_r22_recall"), log=lambda *_: None)
        p.arm_cmd, p.hand_cmd = p.home.copy(), p.post["open"].copy()
        p.queue_a = p.queue_h = None
        p.state = "relook_fresh"
        p._relook_last_t = None
        p._relook_fresh = 0
        p._relook_reason = "nothing-left guard"
        p._final_relook_used = True
        p.threaded = False
        p.pitch_calibration = object()  # keep unrelated first-look calibration out of this state test
        p._bv_done = True               # r22 base: bin.vote ON; its first-look frames are unrelated to this test
        calls = []
        p._look = lambda obs: (calls.append(float(obs["t"])) or ("none", None, 0))
        obs = {"t": 1.0, "arm_q": p.home.copy(), "hand_q": p.post["open"].copy(),
               "head_rgb": np.zeros((H, W, 3), np.uint8), "head_depth": None}

        first = p._act(obs)
        self.assertFalse(first["done"])
        self.assertEqual((p.state, p._relook_fresh, calls), ("relook_fresh", 1, []))
        obs["t"] = 2.0
        second = p._act(obs)
        self.assertTrue(second["done"])
        self.assertEqual((p.state, p._relook_fresh, calls), ("done", 2, [2.0]))
        self.assertTrue(p.perception.sat_floor_adaptive)
        self.assertTrue(p.perception.area_floor_geometric)
        self.assertEqual(p.perception.topface_mode, "whole_blob")


if __name__ == "__main__":
    unittest.main()
