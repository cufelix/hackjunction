"""FK proof that the LEFT and RIGHT hand are exact mirrors for the grasp (closed_tight, point_in_palm, IK at mirrored
targets): fingertips and thumb land at y-mirrored positions relative to the object, and the planner's right/left
vectors are the sign-flipped mirror. Rules out a posture-sign / grasp-point-mirror / TCP bug as the cause of the
left-hand empty grasps (the asymmetry is not kinematic). Also: grasp.prefer_right_arm plans the right arm first."""
import unittest

import numpy as np

try:
    import pinocchio  # noqa: F401
except ImportError:  # pragma: no cover
    pinocchio = None

from aaltonomous.scene.scene import Scene
from .config import hand_postures, home_arm, load_config
from .kinematics import ArmKinematics
from .perception import Detection
from .planner import Planner

SC = Scene.load()


@unittest.skipIf(pinocchio is None, "needs pinocchio")
class TestHandMirror(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(profile="sim")
        cls.post = hand_postures(cls.cfg)
        cls.home = home_arm(cls.cfg)
        cls.K = ArmKinematics(cls.cfg)
        cls.hc = cls.post[cls.cfg["grasp"]["close_posture"]]

    def _tips(self, side, x, y):
        z = SC.counter_z + self.cfg["grasp"]["height_frac"] * 0.045
        tgt = np.array([x, y, z])
        q, ok, err, tilt = self.K.solve(side, tgt, self.home, top_down=True, extra_seeds=[self.home[ArmKinematics.sl(side)]])
        self.assertTrue(ok, (side, x, y))
        tips = self.K.hand_points(side, q, self.hc)[-3:]      # index, middle, thumb
        return tips - tgt, tgt

    def test_postures_are_sign_mirrors(self):
        for name in ("open", "closed", "closed_tight", "release"):
            v = self.post[name]
            np.testing.assert_allclose(v[7:10], -v[0:3], atol=1e-9)           # thumb joints flip sign
            np.testing.assert_allclose(v[10:12], -v[5:7], atol=1e-9)          # right index = -left index
            np.testing.assert_allclose(v[12:14], -v[3:5], atol=1e-9)          # right middle = -left middle

    def test_fingertips_mirror_and_straddle_object(self):
        half = 0.0225                                                         # 45 mm cube
        for x, y in ((0.27, 0.05), (0.30, 0.10), (0.32, 0.15)):
            l, _ = self._tips("left", x, y)
            r, _ = self._tips("right", x, -y)
            # mirror: x and z equal, y negated (index, middle, thumb)
            np.testing.assert_allclose(l * [1, -1, 1], r, atol=3e-3)
            # contact finger (index) and thumb are on opposite sides of the object centre in x (fore/aft pinch) and
            # in the lateral axis, both within the object footprint -> the claw closes on the object, not beside it
            for t in (l, r):
                idx, mid, th = t
                self.assertGreater(idx[0], 0.0)
                self.assertLess(th[0], 0.0)
                self.assertLess(idx[1] * th[1], 0.0)
                self.assertLess(abs(idx[1]), half)
                self.assertLess(abs(th[1]), half)

    def test_prefer_right_arm_plans_right_first(self):
        cfg = load_config(profile="sim", overrides={"grasp.prefer_right_arm": True})
        pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
        order = []
        orig = pl._plan_side_hints

        def spy(det, side, *a, **k):
            order.append(side)
            return orig(det, side, *a, **k)
        pl._plan_side_hints = spy
        p0 = SC.mat_to_pelvis(np.array([[0.5, 0.2, 0.0]]))[0, :2]
        pl.plan(Detection("yellow", np.r_[p0, SC.counter_z + 0.045], 0, 0, "depth"), self.home, self.post["open"], relocate=False)
        self.assertEqual(order[0], "right")


if __name__ == "__main__":
    unittest.main()
