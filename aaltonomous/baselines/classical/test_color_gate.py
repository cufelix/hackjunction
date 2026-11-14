"""perception.color_gate (FLAG, default off): post-filter over the accepted detections, plus the triangle shape
fallback and the planner's shape-keyed lookup. Synthetic Detections only, no images. Run:
    python -m unittest aaltonomous.baselines.classical.test_color_gate -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import Renderer
from .perception import BinCalibration, Detection, Perception
from .planner import lookup_bin

SC, CAL = Scene.load(), Calibration.load()
CAM = load_config(overlay="none")["perception"]["camera"]
OUTER = np.asarray(SC.cfg["bins"]["outer_size"], float)
HALF = OUTER / 2
BIN = np.asarray(SC.cfg["bins"]["centers"]["center"], float)
REGION_XY = np.array([0.45, 0.155])          # mat xy on the counter, clear of every bin footprint (checked in tests)
H_TOP = SC.object_height()


lines = []


def _perception(enabled=True):
    cfg = load_config(overlay="none", overrides={"perception.color_gate.enabled": enabled})
    return Perception(cfg, SC, CAL, log=lines.append)


def _top(mat_xy, h=H_TOP):
    return SC.mat_to_pelvis([float(mat_xy[0]), float(mat_xy[1]), h])


def _Z(top):
    T = CAL._T(CAM)
    return float(((np.asarray(top, float) - T[:3, 3]) @ T[:3, :3])[2])


def _floor(top, area_frac=0.4):
    fx = CAL.cams[CAM].K[0, 0]
    return area_frac * (fx * 0.045 / _Z(top)) ** 2


def _det(top, n_px=None, method="depth", n_depth=500, shape=None, shape_conf=0.0, color="red"):
    if n_px is None:
        n_px = int(2 * _floor(top)) + 1
    return Detection(color, np.asarray(top, float), int(n_px), int(n_depth), method, shape=shape,
                     shape_conf=shape_conf)


class TestColorGate(unittest.TestCase):
    def setUp(self):
        lines.clear()
        self.p = _perception()
        self.assertTrue(np.all(np.isfinite(_top(REGION_XY))))
        self.assertGreater(_Z(_top(REGION_XY)), 0)

    def test_region_point_is_clear_of_bins_and_in_front_of_camera(self):
        self.assertTrue(np.any(np.abs(REGION_XY - BIN) > HALF))   # outside the footprint on at least one axis

    def test_flag_is_off_by_default_and_registered(self):
        from .config import FLAGS
        self.assertIn("perception.color_gate.enabled", FLAGS)
        self.assertFalse(load_config(overlay="none")["perception"]["color_gate"]["enabled"])
        self.assertFalse(_perception(enabled=False).color_gate)

    def test_flag_off_detect_never_runs_the_gate_and_matches_no_config_block(self):
        off = _perception(enabled=False)

        def no_gate(*_):
            raise AssertionError("gate ran with the flag off")
        off._color_gate = no_gate
        cfg = load_config(overlay="none")
        cfg["perception"].pop("color_gate")
        stock = Perception(cfg, SC, CAL, log=lambda *_: None)
        objects = [dict(color=c, shape="cube", yaw=0.0, size=SC.size, height=H_TOP,
                        pos=_top(xy, H_TOP / 2)) for c, xy in
                   zip(("red", "yellow", "blue"), ((0.30, 0.13), (0.45, 0.16), (0.60, 0.18)))]
        rgb, depth = Renderer(SC, CAL).render(objects)
        dets_off, rej_off = off.detect(rgb, depth, debug=True)
        dets_stock, rej_stock = stock.detect(rgb, depth, debug=True)
        self.assertEqual([repr(d) for d in dets_off], [repr(d) for d in dets_stock])
        self.assertEqual([(repr(d), w) for d, w in rej_off], [(repr(d), w) for d, w in rej_stock])
        self.assertFalse(off.color_gate_degraded)

    def test_flag_on_detect_resets_degraded_each_call(self):
        self.p.color_gate_degraded = True
        rgb, depth = Renderer(SC, CAL).render([])
        self.p.detect(rgb, depth)
        self.assertFalse(self.p.color_gate_degraded)

    def test_speckle_below_floor_rejected_and_big_blob_kept(self):
        floor = _floor(_top(REGION_XY))
        small = _det(_top(REGION_XY), n_px=max(1, int(0.5 * floor)))
        big = _det(_top(REGION_XY), n_px=int(2 * floor) + 1)
        rejected = []
        kept = self.p._color_gate([small, big], rejected)
        self.assertEqual(kept, [big])
        self.assertEqual([(d.n_px, w) for d, w in rejected], [(small.n_px, "color_gate:speckle")])

    def test_no_z_uses_the_1000_px_floor(self):
        self.p._color_gate_depth = lambda top: None
        low, ok = _det(_top(REGION_XY), n_px=999), _det(_top(REGION_XY), n_px=1000)
        rejected = []
        kept = self.p._color_gate([low, ok], rejected)
        self.assertEqual(kept, [ok])
        self.assertTrue(rejected[0][1].startswith("color_gate:"))

    def test_inside_bin_footprint_rejected_in_bin(self):
        d = _det(_top(BIN))
        rejected = []
        self.assertEqual(self.p._color_gate([d], rejected), [])
        self.assertEqual(rejected[0][1], "color_gate:workspace")   # rule 2b fires first: behind the bin row near edge
        self.p.color_gate_cfg["ws_x"] = None                       # 2b off: the bin footprint rule
        rejected = []
        self.assertEqual(self.p._color_gate([d], rejected), [])
        self.assertEqual(rejected[0][1], "color_gate:in_bin")

    def test_just_outside_edge_with_plane_method_kept(self):
        # no depth cannot tell a rim from a real object right in front of a bin: keep (Codex H2)
        d = _det(_top(BIN - [0, HALF[1] + 0.015]), method="plane", n_depth=0)
        self.assertEqual(self.p._color_gate([d], []), [d])

    def test_just_outside_edge_depth_at_rim_height_rejected(self):
        # a 6 cm rim top just outside the footprint is not an object (4.5 cm): the band stops at 5.5 cm (Codex H4)
        d = _det(_top(BIN - [0, HALF[1] + 0.015], h=0.06), method="depth", n_depth=500)
        rejected = []
        self.assertEqual(self.p._color_gate([d], rejected), [])
        self.assertEqual(rejected[0][1], "color_gate:bin_edge")

    def test_just_outside_edge_with_depth_top_5cm_high_kept(self):
        d = _det(_top(BIN - [0, HALF[1] + 0.01], h=0.05), method="depth", n_depth=500)
        self.assertEqual(self.p._color_gate([d], []), [d])

    def test_just_outside_edge_depth_but_wrong_height_rejected(self):
        d = _det(_top(BIN - [0, HALF[1] + 0.01], h=0.10), method="depth", n_depth=500)
        rejected = []
        self.assertEqual(self.p._color_gate([d], rejected), [])
        self.assertEqual(rejected[0][1], "color_gate:bin_edge")

    def test_in_front_of_bin_beyond_edge_m_kept(self):
        d = _det(_top(BIN - [0, HALF[1] + 0.03]), method="plane", n_depth=0)
        self.assertEqual(self.p._color_gate([d], []), [d])

    def test_more_than_six_survivors_all_passed_on_largest_first_and_degrade(self):
        # nothing truncated in perception: an unplannable big blob must never hide a real object (Codex C1)
        pts = [REGION_XY + np.array([0.02 * i, 0.0]) for i in range(8)]
        dets = [_det(_top(xy), n_px=5000 + 100 * i) for i, xy in enumerate(pts)]
        rejected = []
        kept = self.p._color_gate(dets, rejected)
        self.assertEqual([d.n_px for d in kept], [5700, 5600, 5500, 5400, 5300, 5200, 5100, 5000])
        self.assertTrue(self.p.color_gate_degraded)
        self.assertEqual(rejected, [])
        self.assertTrue(any("degraded" in line for line in lines))

    def test_six_survivors_are_not_degraded(self):
        dets = [_det(_top(REGION_XY + np.array([0.02 * i, 0.0])), n_px=5000) for i in range(6)]
        kept = self.p._color_gate(dets, [])
        self.assertEqual(len(kept), 6)
        self.assertFalse(self.p.color_gate_degraded)

    def test_triangle_without_depth_loses_shape(self):
        d = _det(_top(REGION_XY), method="plane", n_depth=0, shape="triangle", shape_conf=0.9)
        (out,) = self.p._color_gate([d], [])
        self.assertIsNone(out.shape)
        self.assertEqual(out.shape_conf, 0.0)
        self.assertEqual(d.shape, "triangle")   # the input is not mutated

    def test_triangle_with_depth_keeps_shape(self):
        d = _det(_top(REGION_XY), method="depth", shape="triangle", shape_conf=0.9)
        (out,) = self.p._color_gate([d], [])
        self.assertEqual((out.shape, out.shape_conf), ("triangle", 0.9))

    def test_shape_keyed_lookup_with_shape_none_returns_none(self):
        cfg = {"policy": {"bin_lookup": "resolve_bin"}}
        mapping = {"cube": "left", "triangle": "right"}
        self.assertIsNone(lookup_bin(cfg, mapping, "red", None))
        self.assertEqual(lookup_bin(cfg, mapping, "red", "cube"), "left")


class TestColorGateWorkspaceAndCeiling(unittest.TestCase):
    """Rule 2b (positive workspace: the counter in front of the bin row) and rule 1b (area ceiling)."""

    def setUp(self):
        self.p = _perception()
        self.g = self.p.color_gate_cfg

    def _pel(self, x, y, h=H_TOP, **kw):
        top = np.array([x, y, SC.counter_z + h])
        return _det(top, n_px=int(2 * _floor(top)) + 1, **kw)

    def test_workspace_bounds(self):
        near_x = float(SC.mat_to_pelvis(np.array([0.0, min(float(c[1]) for c in SC.cfg["bins"]["centers"].values())
                                                    - HALF[1]]))[0])
        self.assertAlmostEqual(near_x, 0.40, places=2)                              # bin row near edge (pelvis x)
        self.assertIsNone(self.p._color_gate_why(self._pel(0.30, 0.0), self.g))     # in front of the bins
        self.assertIsNone(self.p._color_gate_why(self._pel(0.385, 0.2), self.g))    # just before near edge - 1 cm
        for x, y in ((0.395, 0.0), (0.45, 0.0), (0.10, 0.0), (0.30, 0.41), (0.30, -0.41)):
            self.assertEqual(self.p._color_gate_why(self._pel(x, y), self.g), "color_gate:workspace", (x, y))

    def test_area_ceiling(self):
        d = self._pel(0.30, 0.0)
        Z = _Z(d.top)
        K = CAL.cams[CAM].K
        expected = K[0, 0] * K[1, 1] * (SC.cfg["objects"]["size"] / Z) ** 2
        big = _det(d.top, n_px=int(3.2 * expected))
        self.assertEqual(self.p._color_gate_why(big, self.g), "color_gate:too_big")
        two = _det(d.top, n_px=int(2.2 * expected))                                 # two touching objects: kept
        self.assertIsNone(self.p._color_gate_why(two, self.g))

    def test_area_ceiling_without_depth(self):
        p = _perception()
        p._color_gate_depth = lambda top: None
        d = self._pel(0.30, 0.0)
        self.assertEqual(p._color_gate_why(_det(d.top, n_px=6500), p.color_gate_cfg), "color_gate:too_big_no_z")
        self.assertIsNone(p._color_gate_why(_det(d.top, n_px=3000), p.color_gate_cfg))

    def test_bad_tuning_fails_at_load(self):
        for k, v in (("max_cands", 0), ("area_frac", float("nan")), ("edge_h", [0.07, 0.03]), ("ws_x", [0.1]),
                     ("max_area_frac", "x"), ("fk_dilate_px", 1.5)):
            with self.assertRaises(ValueError, msg=(k, v)):
                load_config(overlay="none", overrides={"perception.color_gate.enabled": True,
                                                       f"perception.color_gate.{k}": v})

    def test_plane_triangle_clears_all_shape_metadata(self):
        d = _det(_top(REGION_XY), method="plane", n_depth=0, shape="triangle", shape_conf=0.9)
        out = self.p._color_gate_shape(d)
        self.assertEqual((out.shape, out.shape_conf, out.shape_path, out.apex_yaw), (None, 0.0, "abstain_color_gate", None))


if __name__ == "__main__":
    unittest.main()


class TestColorGateLiveBins(unittest.TestCase):
    """live-bin footprints hedge each bin from planner.bin_conf, never from a nonempty shift dictionary."""
    G = None

    def _p(self, live=True, margin=0.10):
        cfg = load_config(overlay="none", overrides={"perception.color_gate.enabled": True,
                                                     "perception.color_gate.live_bins": live,
                                                     "perception.color_gate.near_margin_m": margin})
        p = Perception(cfg, SC, CAL, log=lambda *_: None)
        return p, p.color_gate_cfg

    def _near(self, dy):   # mat point on the centre bin's axis, dy in front of its NEAR edge (towards the robot)
        return _det(_top([BIN[0], BIN[1] - HALF[1] - dy]), method="plane", n_depth=0)

    def test_near_margin_is_a_default_off_flag_and_finals_uses_the_full_vote_displacement(self):
        from .config import FLAGS

        self.assertIn("perception.color_gate.near_margin_m", FLAGS)
        self.assertEqual(load_config(overlay="none")["perception"]["color_gate"]["near_margin_m"], 0.0)
        cfg = load_config(overlay="finals_r21")
        self.assertEqual(cfg["perception"]["color_gate"]["near_margin_m"], cfg["bin"]["vote"]["max_shift_m"])

    def test_missing_confidence_grows_the_near_side_by_full_allowed_10cm(self):
        p, g = self._p()
        self.assertEqual(p._color_gate_why(self._near(0.05), g), "color_gate:in_bin")   # 5 cm in front: inside the margin
        self.assertIsNone(p._color_gate_why(self._near(0.15), g))                       # beyond margin + edge band

    def test_flag_off_is_the_nominal_footprint(self):
        p, g = self._p(live=False)
        self.assertIsNone(p._color_gate_why(self._near(0.05), g))

    def test_calibration_success_does_not_substitute_for_per_bin_vote_confidence(self):
        p, g = self._p()
        p.calibration = BinCalibration(True, True, 0.9, np.zeros(2), 100, 0.002, 1.0, "ok")
        self.assertEqual(p._color_gate_why(self._near(0.05), g), "color_gate:in_bin")

    def test_prior_or_disagree_shift_dict_still_gets_the_union_hedge(self):
        for confidence in (0.5, 0.3):
            p, g = self._p()
            p.bin_shift_src = type("Pl", (), {"bin_shift": {"center": np.array([-0.07, 0.0])},
                                                "bin_conf": {"center": confidence}})()
            g["ws_x"] = None
            self.assertEqual(p._color_gate_why(self._near(0.05), g), "color_gate:in_bin", confidence)

    def test_confident_bin_vote_uses_only_the_tight_shifted_footprint(self):
        p, g = self._p()
        p.bin_shift_src = type("Pl", (), {"bin_shift": {"center": np.array([-0.07, 0.0])},
                                           "bin_conf": {"center": 1.0}})()   # pelvis dx -7 cm
        self.assertEqual(p._color_gate_why(self._near(0.05), g), "color_gate:workspace")   # 2b edge follows the shift
        far = _det(_top([BIN[0], BIN[1] + HALF[1] - 0.03]), method="plane", n_depth=0)  # nominal far strip, now outside
        self.assertEqual(p._color_gate_why(far, g), "color_gate:workspace")             # behind the bin row: 2b
        g["ws_x"] = None
        self.assertEqual(p._color_gate_why(self._near(0.05), g), "color_gate:in_bin")   # inside the SHIFTED box
        self.assertIsNone(p._color_gate_why(far, g))

    def test_hedge_is_per_bin_and_uses_nominal_to_full_allowed_displacement_union(self):
        p, g = self._p()
        p.bin_shift_src = type("Pl", (), {"bin_shift": {"center": np.array([0.07, 0.0])},
                                           "bin_conf": {"left": 1.0, "center": 0.5, "right": 0.3}})()
        boxes = dict(zip(SC.bin_names, p._color_gate_boxes(g)))
        nominal = p.bin_geometry(gate=False)
        for name in ("center", "right"):
            lo, hi = boxes[name]
            np.testing.assert_allclose(lo, nominal[name]["outer_lo"] - [0.0, 0.10])
            np.testing.assert_allclose(hi, nominal[name]["outer_hi"])
        np.testing.assert_allclose(boxes["left"][0], nominal["left"]["outer_lo"])
        np.testing.assert_allclose(boxes["left"][1], nominal["left"]["outer_hi"])
