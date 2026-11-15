"""planner.lift_from_contact (real-only): a table-resting start is lifted vertically before the home route."""
import unittest

import numpy as np

from .config import load_config
from .kinematics import ArmKinematics
from .policy import ClassicalPolicy

TOL = 1e-3


def contact_start(pol, y_left):
    """Both grasp points 1 cm above the counter at (0.30, +-0.20): hands resting on the table."""
    a = pol.home.copy()
    for side, y in (("left", y_left), ("right", -y_left)):
        p = np.array([0.30, y, pol.scene.counter_z + 0.01])
        q, ok, _, _ = pol.planner.kin.solve(side, p, a.copy(), top_down=False, restarts=True)
        assert ok, f"IK failed for {side}"
        a[ArmKinematics.sl(side)] = q[ArmKinematics.sl(side)]
    return a


class TestLiftFromContact(unittest.TestCase):
    def setUp(self):
        self.pol_off = ClassicalPolicy(config=load_config(), log=None)
        self.pol_on = ClassicalPolicy(config=load_config(overrides={"planner.lift_from_contact": True}), log=None)

    def test_flag_off_start_is_unsafe_and_route_fails(self):
        hand = self.pol_off.post["open"]
        a = contact_start(self.pol_off, 0.20)
        route = self.pol_off.planner.safe_home_route(a, hand)
        self.assertFalse(route.ok)
        self.assertIn("keepout", route.reason)

    def test_flag_on_lifts_then_routes_home(self):
        for y in (0.20, -0.20):
            with self.subTest(y=y):
                pl = self.pol_on.planner
                hand = self.pol_on.post["open"]
                a = contact_start(self.pol_on, y)
                route = pl.safe_home_route(a, hand)
                self.assertTrue(route.ok, route.reason)
                self.assertEqual(route.phases[0][0], "lift_from_contact")
                end = route.phases[0][2]
                hmin = np.array([min(pl.kin.hand_points(s, route.arm[k], route.hand[k])[:, 2].min()
                                     for s in ("left", "right")) for k in range(len(route.arm))])
                self.assertTrue(np.all(np.diff(hmin[:end + 1]) >= -TOL))
                # the lift is vertical: grasp xy of both sides stays put
                for s in ("left", "right"):
                    xy = np.array([pl.kin.grasp_pose(s, route.arm[k])[0][:2] for k in range(end)])
                    self.assertLess(np.abs(xy - xy[0]).max(), TOL)
                # first lateral phase only after the hand clears the table keepout
                home_start = next(st for name, st, _ in route.phases if name == "home")
                self.assertGreaterEqual(hmin[home_start], pl.counter_keepout_z + 0.003 - TOL)


if __name__ == "__main__":
    unittest.main()
