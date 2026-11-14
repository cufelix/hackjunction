"""perception.color_gate survival under the finals_r19 overlay (live_bins, fk_mask, hue_guard, robust_v2): first look at
the sim home pose, no bin.vote estimate, synthetic renders (mock_world.Renderer draws no arms). Run:
    python -m unittest aaltonomous.baselines.classical.test_color_gate_survival -v
"""
import re
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import hand_postures, load_config
from .mock_world import Renderer
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
H_OBJ = SC.object_height()
# pelvis (x, y, colour) inside the head camera's view and the spawn region (x 0.19-0.35)
SPAWN = [(0.23, -0.20, "red"), (0.27, 0.15, "yellow"), (0.31, -0.15, "blue"), (0.35, -0.15, "red")]
CENTRE_BIN_FRONT = [(0.33, 0.00, "red")]   # 7 cm in front of the centre bin's nominal near edge (pelvis x 0.40)


def _pos(x, y, h):
    m = SC.pelvis_to_mat(np.array([x, y, SC.counter_z]))[:2]
    return SC.mat_to_pelvis([float(m[0]), float(m[1]), h])


def _objects(spots):
    return [dict(color=c, shape="cube", yaw=0.0, size=SC.size, height=H_OBJ, pos=_pos(x, y, H_OBJ / 2))
            for x, y, c in spots]


class TestColorGateSurvival(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config(overlay="finals_r19")
        self.arm = np.asarray(self.cfg["home_arm_q"], float)
        self.hand = hand_postures(self.cfg)["open"]
        self.lines = []

    def _detect(self, spots, overrides=None):
        cfg = load_config(overlay="finals_r19", overrides=overrides)
        self.lines.clear()
        p = Perception(cfg, SC, CAL, log=self.lines.append)
        rgb, depth = Renderer(SC, CAL).render(_objects(spots))
        dets, rej = p.detect(rgb, depth, debug=True, robot_q=(self.arm, self.hand))
        return p, dets, rej

    def test_overlay_settings_are_the_finals_r19_ones(self):
        g = self.cfg["perception"]["color_gate"]
        self.assertEqual(self.cfg["overlay"], "finals_r19")
        self.assertTrue(g["enabled"] and g["live_bins"] and g["fk_mask"])
        self.assertTrue(self.cfg["perception"]["hue_guard"]["enabled"])
        self.assertEqual(self.cfg["perception"]["hsv_set"], "robust_v2")

    def test_gate_summary_line_counts_in_and_out(self):
        _, dets, rej = self._detect(SPAWN)
        hits = [l for l in self.lines if l.startswith("[color_gate] in=")]
        self.assertEqual(len(hits), 1)
        m = re.fullmatch(r"\[color_gate\] in=(\d+) out=(\d+) reasons=\{(.*)\}", hits[0])
        self.assertIsNotNone(m, hits[0])
        self.assertEqual(int(m.group(1)), len(dets) + len(rej))
        self.assertEqual(int(m.group(2)), len(dets))

    def test_fk_mask_coverage_at_home_pose(self):
        p, _, _ = self._detect(SPAWN)
        cov = p.fk_mask_px / (480 * 640)
        self.assertTrue(0.20 < cov < 0.30, cov)   # measured 25.7 % (dilated 15 px), not the ~23 % in the notes

    def test_spawn_survivors_without_fk_mask(self):
        _, dets, _ = self._detect(SPAWN, {"perception.color_gate.fk_mask": False})
        self.assertEqual(len(dets), 4)

    @unittest.expectedFailure
    def test_all_four_spawn_objects_survive_overlay(self):
        # REAL BUG (owner decision): the FK arm mask (dilated 15 px) removes the object at pelvis (0.35, -0.15), whose
        # centre pixel lies in the mask: 'color_gate:speckle'. fk_dilate_px 5 or 0 keeps all four. See report.
        _, dets, rej = self._detect(SPAWN)
        self.assertEqual(len(dets), 4, [w for _, w in rej])

    def test_centre_bin_front_object_survives_without_hedge(self):
        _, dets, _ = self._detect(CENTRE_BIN_FRONT, {"perception.color_gate.near_margin_m": 0.0})
        self.assertEqual(len(dets), 1)

    @unittest.expectedFailure
    def test_centre_bin_front_object_survives_overlay(self):
        # REAL BUG (owner decision): with no bin estimate the live footprint is grown 10 cm towards the robot
        # (near_margin_m), so the centre bin's box starts at pelvis x 0.30 and the object at x 0.33 is 'color_gate:in_bin'.
        _, dets, rej = self._detect(CENTRE_BIN_FRONT)
        self.assertEqual(len(dets), 1, [w for _, w in rej])


if __name__ == "__main__":
    unittest.main()
