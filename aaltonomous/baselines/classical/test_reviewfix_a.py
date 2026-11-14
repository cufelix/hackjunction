"""Regression tests for the review-fix A batch (int-a#4, real#1, real#2, real#3). Run:
    python -m unittest aaltonomous.baselines.classical.test_reviewfix_a -v
"""
import unittest
from types import SimpleNamespace

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .novel_objects import augment_detections, match_instruction
from .perception import Detection, Perception
from .policy import ClassicalPolicy
from .test_gate_failopen import _perc
from .test_novel_objects import novel_config, novel_frame

SC, CAL = Scene.load(), Calibration.load()
NOOP = lambda *_: None


def _novel_red_setup():
    rgb, depth, _ = novel_frame()
    per = Perception(novel_config(), SC, CAL, log=NOOP)
    dets = per.detect(rgb, depth)
    red = next(d for d in dets if getattr(d, "novel", False) and d.color == "red")
    return per, rgb, depth, red


class TestNovelPairingAndSupersede(unittest.TestCase):
    def test_colour_pairs_with_following_bin(self):
        _, _, _, red = _novel_red_setup()
        m = match_instruction("blue left, red one right", [red])
        self.assertIs(m.matched, red)
        self.assertEqual(m.target_bin, "right")

    def test_novel_never_supersedes_known_cube(self):
        per, rgb, depth, red = _novel_red_setup()
        cube = Detection("yellow", np.array([0.30, 0.10, 0.04]), 3000, 0, "depth", 0.0, red.uv, shape="cube")
        out = augment_detections(per, rgb, depth, [cube])
        self.assertTrue(any(d is cube for d in out))   # a yellow cube under a red novel blob
        self.assertFalse(any(getattr(d, "novel", False) and d.color == "red" for d in out))


class TestYellowRelabelWithoutHueGuard(unittest.TestCase):
    def _perc(self, yellow_fix):
        cfg = load_config(overlay="none", overrides={"perception.hsv_set": "robust_v2",
                                                     "perception.hue_guard.yellow_fix": yellow_fix})
        return Perception(cfg, SC, CAL, log=NOOP)

    def _det(self):
        det = Detection("red", np.zeros(3), 3000, 0, "depth")
        det.hsv_med = (40.0, 80.0, 240.0)   # yellow top-face hue, labelled red by the component split
        return det

    def test_relabel_runs_with_hue_guard_off(self):
        p = self._perc(True)
        self.assertFalse(p.hue_guard)
        self.assertEqual(p._yellow_relabel([self._det()])[0].color, "yellow")

    def test_flag_off_is_default(self):
        self.assertFalse(self._perc(False).hue_guard_yellow_fix)


class TestWorkspaceReprojectionWritesTop(unittest.TestCase):
    def _plane_det(self, p, mat_xy, counter_z):
        T = p._T()
        o = T[:3, 3]
        true = SC.mat_to_pelvis([float(mat_xy[0]), float(mat_xy[1]), 0.0])
        true = np.array([true[0], true[1], counter_z + 0.75 * p.height])
        z_top = SC.nominal_counter_z + p.height
        z_aim = z_top - 0.25 * p.height
        s = (z_aim - o[2]) / (true[2] - o[2])
        q = o + s * (true - o)
        return Detection("yellow", np.array([q[0], q[1], z_top]), 3000, 0, "plane")

    def test_prior_applies_no_extra_shift(self):
        # a live counter_z_prior already puts the plane at the assumed height: dz is 0, so the gate matches flag-off
        p_on, p_off = _perc(True), _perc(False)
        for p in (p_on, p_off):
            p.scene.counter_z_prior, p.scene.counter_z_runtime = 0.15, None
        d_on = self._plane_det(p_on, [0.45, 0.155], 0.15)
        d_off = self._plane_det(p_off, [0.45, 0.155], 0.15)
        z0 = float(d_on.top[2])
        self.assertEqual(p_on._color_gate_why(d_on, p_on.color_gate_cfg),
                         p_off._color_gate_why(d_off, p_off.color_gate_cfg))
        self.assertAlmostEqual(float(d_on.top[2]), z0, places=9)

    def test_no_prior_writes_reprojected_xy_to_top(self):
        p = _perc(True)
        p.scene.counter_z_runtime = None
        d = self._plane_det(p, [0.45, 0.155], 0.15)
        xy0 = np.array(d.top[:2], float)
        p._color_gate_why(d, p.color_gate_cfg)
        self.assertFalse(np.allclose(d.top[:2], xy0))


class TestRimZCommitsWithLook(unittest.TestCase):
    def test_look_job_commits_rim_z_and_state(self):
        worker = SimpleNamespace(
            _look=lambda obs: "ok",
            scene=SimpleNamespace(counter_z_runtime=0.15, rim_z_runtime={"wall": 0.20}),
            perception=SimpleNamespace(_rz_est={"wall": [0.20]}, _rz_tries=1))
        fake = SimpleNamespace(_look_generation=0, log=NOOP,
                               _LOOK_COMMIT_FIELDS=ClassicalPolicy._LOOK_COMMIT_FIELDS,
                               _LOOK_PERCEPTION_FIELDS=ClassicalPolicy._LOOK_PERCEPTION_FIELDS,
                               scene=SimpleNamespace(counter_z_runtime=None, rim_z_runtime=None),
                               perception=SimpleNamespace(_rz_est={}, _rz_tries=0))
        job = {"kind": "look", "obs": None, "generation": 0, "result": None}
        ClassicalPolicy._look_job(fake, job, worker=worker, logs=[])
        ClassicalPolicy._commit_look_result(fake, job["result"])
        self.assertEqual(fake.scene.rim_z_runtime, {"wall": 0.20})
        self.assertEqual(fake.perception._rz_tries, 1)
        self.assertEqual(fake.perception._rz_est, {"wall": [0.20]})


if __name__ == "__main__":
    unittest.main()
