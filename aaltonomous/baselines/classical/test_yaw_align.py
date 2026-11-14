"""Wiring tests for the triangle yaw-align path (no IK solves: planner methods are mocked, perception uses a stub
camera). Run: python -m pytest aaltonomous/baselines/classical/test_yaw_align.py -q"""
import copy
import unittest
from unittest import mock

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config
from . import perception
from .perception import Detection, Perception
from .prism_pose import TRIANGLE_MIN_POINTS
from .planner import Planner

SC = Scene.load()


def _planner(**grasp):
    cfg = load_config()
    cfg["grasp"].update(grasp)
    return Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))


def _tri(**kw):
    d = dict(shape="triangle", apex_yaw=0.3, apex_conf=0.9)
    d.update(kw)
    return Detection("red", np.array([0.3, 0.1, 0.9]), 0, 0, "depth", **d)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


class TestYawHints(unittest.TestCase):
    def test_right_arm_mirrors_left(self):
        pl = _planner(yaw_max_dev_deg=180.0, yaw_max_hints=3)
        phi = np.radians(pl.cfg["grasp"]["yaw_phi_deg"])
        for apex in np.radians([-170, -90, -30, 0, 45, 90, 150]):
            L = pl.yaw_hint_angles(apex, "left")
            R = pl.yaw_hint_angles(apex, "right")
            self.assertEqual((len(L), len(R)), (3, 3))
            for a in L:   # left: apex + phi + k*120
                self.assertAlmostEqual(abs(_wrap((a - apex - phi) * 3)) % (2 * np.pi), 0, places=6)
            for a in R:   # right: apex - phi + k*120
                self.assertAlmostEqual(abs(_wrap((a - apex + phi) * 3)) % (2 * np.pi), 0, places=6)
            # mirror: right(apex) == -left(-apex) as a set of angles
            mir = sorted(_wrap(-a) % (2 * np.pi) for a in pl.yaw_hint_angles(-apex, "left"))
            self.assertTrue(np.allclose(sorted(a % (2 * np.pi) for a in R), mir, atol=1e-9))

    def test_default_filter_and_order(self):
        pl = _planner()
        for side, s in (("left", 1.0), ("right", -1.0)):
            ref = -s * np.radians(25.0)
            for apex in np.radians(np.arange(-180, 180, 20)):
                out = pl.yaw_hint_angles(apex, side)
                self.assertLessEqual(len(out), 2)
                dev = [abs(_wrap(a - ref)) for a in out]
                self.assertTrue(all(d <= np.radians(75.0) + 1e-9 for d in dev))
                self.assertEqual(dev, sorted(dev))


class TestYawDispatch(unittest.TestCase):
    def _run(self, planner, det, yaw_result):
        args = (det, "left", "left", np.zeros(14), np.zeros(14))
        with mock.patch.object(planner, "_plan_side_yaw", return_value=yaw_result) as y, \
                mock.patch.object(planner, "_plan_side_legacy", return_value="legacy") as l:
            r = planner._plan_side_hints(*args)
        return r, y, l

    def test_yaw_first_when_on(self):
        r, y, l = self._run(_planner(yaw_align=True), _tri(), "yaw")
        self.assertEqual(r, "yaw")
        y.assert_called_once()
        l.assert_not_called()

    def test_fallback_to_legacy_when_yaw_fails(self):
        r, y, l = self._run(_planner(yaw_align=True), _tri(), None)
        self.assertEqual(r, "legacy")
        y.assert_called_once()
        l.assert_called_once()

    def test_skipped_when_off_or_not_triangle_or_low_conf(self):
        for pl, det in ((_planner(yaw_align=False), _tri()),
                        (_planner(yaw_align=True), _tri(shape="cube")),
                        (_planner(yaw_align=True), _tri(shape=None)),
                        (_planner(yaw_align=True), _tri(apex_yaw=None)),
                        (_planner(yaw_align=True), _tri(apex_conf=0.1))):
            r, y, l = self._run(pl, det, "yaw")
            self.assertEqual(r, "legacy")
            y.assert_not_called()

    def test_yaw_plan_tries_hint_angles_in_order(self):
        """_plan_side_yaw calls _plan_side once per candidate with kin.hint = that finger direction."""
        pl = _planner(yaw_align=True)
        seen = []

        def fake(d2, side, *a, **k):
            seen.append(np.arctan2(pl.kin.hint[1], pl.kin.hint[0]))
            return mock.Mock(ok=False)

        det = _tri(apex_yaw=0.0)
        with mock.patch.object(pl, "_plan_side", side_effect=fake):
            self.assertIsNone(pl._plan_side_yaw(det, "left", "left", np.zeros(14), np.zeros(14)))
        exp = pl.yaw_hint_angles(0.0, "left")
        self.assertEqual(len(seen), len(exp))
        np.testing.assert_allclose([_wrap(a - b) for a, b in zip(seen, exp)], 0, atol=1e-9)


class _StubCal:
    def pixel_to_pelvis(self, cam, u, v, d):
        return np.stack([u, v, d], -1).astype(float)   # "pixels" already metres


class TestPerceptionWiring(unittest.TestCase):
    def _perc(self, **flags):
        cfg = load_config()
        cfg["perception"].update(flags)
        return Perception(cfg, SC, _StubCal())

    def _cloud(self, apex=0.4, n=400, seed=0):
        rng = np.random.default_rng(seed)
        R = 0.045 * 2 / 3
        V = R * np.stack([np.cos(apex + 2 * np.pi * np.arange(3) / 3), np.sin(apex + 2 * np.pi * np.arange(3) / 3)], -1)
        u = rng.random((n, 2)); f = u.sum(1) > 1; u[f] = 1 - u[f]
        xy = V[0] + u[:, :1] * (V[1] - V[0]) + u[:, 1:] * (V[2] - V[0]) + np.array([0.3, 0.1])
        return xy

    def _det(self, perc, label=None, xy=None, conf=0.9):
        """_from_depth with the shape seam answering (label, conf): the only way a shape enters perception."""
        xy = self._cloud() if xy is None else xy
        # pixel_to_pelvis stub takes (u, v, d); d is z in "metres" here, depth weight uses d/1000 (uniform)
        with mock.patch.object(perception, "classify_top_face", return_value=(label, conf)) as seam:
            d = perc._from_depth("red", xy[:, 0], xy[:, 1], np.full(len(xy), 0.9), 500, None)
        self.seam_args = seam.call_args
        return d

    def test_flags_off_unchanged(self):
        p = self._perc()
        with mock.patch.object(perception, "classify_top_face", side_effect=AssertionError("seam called")):
            d = p._from_depth("red", *self._cloud().T, np.full(400, 0.9), 500, None)
            self.assertIsNone(p._shape_of(None, np.ones((10, 10), np.uint8)))
        self.assertIsNone(d.shape)
        self.assertIsNone(d.apex_yaw)
        self.assertEqual(d.apex_conf, 0.0)

    def test_seam_receives_top_face_and_blob(self):
        p = self._perc(classify_shape=True)
        blob = np.zeros((20, 30), np.uint8)
        with mock.patch.object(perception, "classify_top_face", return_value=("cube", 1.0)) as seam:
            p._from_depth("red", *self._cloud().T, np.full(400, 0.9), 500, blob)
        top, b, shape = seam.call_args[0]
        self.assertEqual(top.shape, (400, 2))
        self.assertIs(b, blob)
        self.assertEqual(shape, (20, 30))

    def test_label_comes_only_from_seam(self):
        p = self._perc(classify_shape=True)
        for lab in ("cube", "cylinder", "triangle", None):
            self.assertEqual(self._det(p, lab).shape, lab)

    def test_triangle_pose_set(self):
        p = self._perc(triangle_pose=True)
        d = self._det(p, "triangle")
        self.assertEqual(d.shape, "triangle")
        self.assertGreater(d.apex_conf, 0.5)
        err = abs(_wrap((d.apex_yaw - 0.4) * 3)) / 3   # modulo 120 deg
        self.assertLess(np.degrees(err), 6.0)
        np.testing.assert_allclose(d.top[:2], [0.3, 0.1], atol=0.004)

    def test_cube_untouched_by_triangle_flag(self):
        base = self._det(self._perc(), "cube")
        on = self._det(self._perc(triangle_pose=True), "cube")
        np.testing.assert_array_equal(base.top, on.top)
        self.assertIsNone(on.apex_yaw)

    def test_wrong_triangle_label_degrades_to_fallback(self):
        """Seam says triangle for a cube / cylinder: apex_conf stays below yaw_min_conf, planner does not use yaw."""
        pl = _planner(yaw_align=True)
        p = self._perc(triangle_pose=True)
        rng = np.random.default_rng(1)
        cube = np.array([0.3, 0.1]) + rng.uniform(-0.0225, 0.0225, (400, 2))
        a, r = rng.uniform(0, 2 * np.pi, 400), 0.0225 * np.sqrt(rng.uniform(0, 1, 400))
        cyl = np.array([0.3, 0.1]) + np.stack([r * np.cos(a), r * np.sin(a)], -1)
        for xy in (cube, cyl):
            d = self._det(p, "triangle", xy=xy)
            self.assertLess(d.apex_conf, pl.cfg["grasp"]["yaw_min_conf"])
            self.assertFalse(pl._yaw_align_on(d))

    def test_too_few_points_no_apex(self):
        p = self._perc(triangle_pose=True)
        d = self._det(p, "triangle", xy=self._cloud(n=TRIANGLE_MIN_POINTS - 1))
        self.assertIsNone(d.apex_yaw)
        self.assertEqual(d.apex_conf, 0.0)
        self.assertFalse(_planner(yaw_align=True)._yaw_align_on(d))

    def test_plane_fallback_asks_seam_with_blob_only(self):
        p = self._perc(classify_shape=True)
        blob = np.zeros((20, 30), np.uint8)
        with mock.patch.object(perception, "classify_top_face", return_value=("triangle", 0.8)) as seam:
            lab = p._shape_of(None, blob)
        self.assertEqual(lab, "triangle")
        self.assertIsNone(seam.call_args[0][0])

    def test_no_other_shape_logic(self):
        """classify_blob / area-ratio thresholds exist only in shape_id."""
        import pathlib
        d = pathlib.Path(perception.__file__).parent
        for f in ("perception.py", "planner.py", "prism_pose.py"):
            src = (d / f).read_text()
            self.assertNotIn("classify_blob", src, f)
            self.assertNotIn("RATIO_C", src, f)


if __name__ == "__main__":
    unittest.main()
