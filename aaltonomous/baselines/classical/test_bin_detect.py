"""perception.detect_bins (bin_detect.py). Run:
    python -m pytest aaltonomous/baselines/classical/test_bin_detect.py -q
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import Renderer, _yaw_R
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
CFG_OFF = load_config(overlay="none")
CFG_ON = load_config(overlay="none", overrides={"perception.detect_bins": True})


def true_layout(shift=(0.0, 0.0), yaw=0.0):
    """Nominal row rotated by `yaw` about its centre, then shifted (pelvis xy). -> {name: (xy, yaw)}."""
    b = SC.cfg["bins"]
    nom = {n: SC.mat_to_pelvis(np.r_[c, 0.0])[:2] for n, c in b["centers"].items()}
    row_c = np.mean(list(nom.values()), axis=0)
    R = _yaw_R(yaw)[:2, :2]
    return {n: (row_c + (xy - row_c) @ R.T + np.asarray(shift, float), float(yaw)) for n, xy in nom.items()}


class MovedBinsRenderer(Renderer):
    """Mock renderer with the bins at given pelvis poses {name: (xy, yaw)} (missing names are not drawn)."""

    def __init__(self, poses, **kw):
        self.poses = poses
        super().__init__(SC, CAL, **kw)

    def _background(self):
        b = self.scene.cfg["bins"]
        saved = b["centers"]
        b["centers"] = {}
        try:
            rgb, zb = super()._background()        # counter + mat, no bins
        finally:
            b["centers"] = saved
        ox, oy = np.asarray(b["outer_size"], float) / 2
        wt, wh, ft = b["wall_thickness"], b["wall_height"], b["floor_thickness"]
        parts = [((0, 0, ft / 2), (oy, ox, ft / 2)),
                 ((oy - wt / 2, 0, wh / 2), (wt / 2, ox, wh / 2)), ((-oy + wt / 2, 0, wh / 2), (wt / 2, ox, wh / 2)),
                 ((0, ox - wt / 2, wh / 2), (oy, wt / 2, wh / 2)), ((0, -ox + wt / 2, wh / 2), (oy, wt / 2, wh / 2))]
        for name, (xy, yaw) in self.poses.items():
            R = _yaw_R(yaw)
            c = np.r_[xy, self.scene.counter_z]
            for off, half in parts:
                self._box(rgb, zb, c + R @ np.asarray(off, float), R, np.asarray(half), list(b["rgb"]))
        return rgb, zb


def render(poses, objects=(), seed=0):
    rng = np.random.default_rng(seed)
    return MovedBinsRenderer(poses).render(list(objects), rng, depth_noise_mm=1.5, hole_frac=0.05)


def cube(color, xy, size=None):
    s = size or SC.size
    return dict(color=color, shape="cube", yaw=0.3, size=s, height=s, pos=np.r_[xy, SC.counter_z + s / 2])


class TestBinDetect(unittest.TestCase):
    def test_flag_off_is_unchanged_and_layout_nominal(self):
        poses = true_layout((0.06, -0.05), np.radians(4))
        rgb, depth = render(poses, [cube("red", [0.32, 0.05])])
        p = Perception(CFG_OFF, SC, CAL, log=lambda *a: None)
        q = Perception(load_config(overlay="none"), SC, CAL, log=lambda *a: None)
        d1, r1 = p.detect(rgb, depth, debug=True)
        self.assertIsNone(p.bin_detection)
        self.assertTrue(all(v["source"] == "nominal" for v in p.bin_layout.values()))
        d2, r2 = q.detect(rgb, depth, debug=True)
        self.assertEqual([repr(d) for d in d1], [repr(d) for d in d2])
        self.assertEqual([w for _, w in r1], [w for _, w in r2])

    def test_shifted_rotated_row_recovered_within_1cm(self):
        for k, (shift, deg) in enumerate([((0.08, 0.0), 0), ((-0.06, 0.07), 5), ((0.05, -0.08), -5),
                                          ((0.10, 0.05), 3)]):
            poses = true_layout(shift, np.radians(deg))
            rgb, depth = render(poses, [cube("red", [0.30, 0.10]), cube("blue", [0.31, -0.12])], seed=k)
            p = Perception(CFG_ON, SC, CAL, log=lambda *a: None)
            p.detect(rgb, depth)
            res = p.bin_detection
            self.assertTrue(res.success, res.reason)
            for name, (xy, yaw) in poses.items():
                v = p.bin_layout[name]
                self.assertEqual(v["source"], "perceived")
                self.assertLess(np.linalg.norm(v["xy"] - xy), 0.01, (shift, deg, name, v["xy"], xy))
                self.assertLess(abs(np.degrees(v["yaw"] - yaw)), 1.5, (shift, deg, name))
            self.assertLess(abs(np.degrees(res.row_yaw) - deg), 1.0)

    def test_moved_bins_no_phantoms_and_objects_kept(self):
        """Moved row: flag OFF sees bin-wall phantoms / loses the object next to a bin; flag ON does not."""
        poses = true_layout((0.08, 0.0), 0.0)
        near = cube("blue", [0.33, 0.26])                 # in front of the moved left bin, inside the nominal hull
        rgb, depth = render(poses, [near])
        on = Perception(CFG_ON, SC, CAL, log=lambda *a: None).detect(rgb, depth)
        self.assertEqual([d.color for d in on], ["blue"])
        self.assertLess(np.linalg.norm(on[0].top[:2] - near["pos"][:2]), 0.01)

    def test_one_bin_missing_falls_back_to_nominal(self):
        poses = true_layout((0.05, 0.0), 0.0)
        poses.pop("right")
        rgb, depth = render(poses)
        p = Perception(CFG_ON, SC, CAL, log=lambda *a: None)
        p.detect(rgb, depth)
        self.assertFalse(p.bin_detection.success)
        self.assertIn("right", p.bin_detection.reason)
        self.assertTrue(all(v["source"] == "nominal" for v in p.bin_layout.values()))
        self.assertFalse(p._bins_perceived)

    def test_out_of_gate_falls_back(self):
        cfg = load_config(overlay="none", overrides={"perception.detect_bins": True,
                                                     "perception.bin_detect.max_offset_m": 0.03})
        rgb, depth = render(true_layout((0.08, 0.0), 0.0))
        p = Perception(cfg, SC, CAL, log=lambda *a: None)
        p.detect(rgb, depth)
        self.assertFalse(p.bin_detection.success)
        self.assertTrue(all(v["source"] == "nominal" for v in p.bin_layout.values()))

    def test_no_depth_retries_then_latches_and_resets(self):
        rgb, depth = render(true_layout((0.05, 0.02), 0.0))
        p = Perception(CFG_ON, SC, CAL, log=lambda *a: None)
        p.detect(rgb, np.zeros_like(depth))
        self.assertFalse(p.bin_detection.valid_frame)
        self.assertFalse(p._bins_done)
        p.detect(rgb, depth)
        self.assertTrue(p.bin_detection.success, p.bin_detection.reason)
        self.assertLess(p.bin_detection.elapsed_ms, 150.0)   # warm ~25 ms; loose for CI noise
        p.reset_calibration()
        self.assertIsNone(p.bin_detection)
        self.assertEqual(p.bin_layout["left"]["source"], "nominal")

    def test_flat_orange_counter_rejected(self):
        """No walls (no bins at all): nominal, never a perceived layout."""
        rgb, depth = render({})
        p = Perception(CFG_ON, SC, CAL, log=lambda *a: None)
        p.detect(rgb, depth)
        self.assertFalse(p.bin_detection.success)


if __name__ == "__main__":
    unittest.main()
