"""fsm layer (real-only flags): hands-on-table start, zero-detection re-look, descend contact guard."""
import unittest

import numpy as np

from .config import load_config
from .kinematics import ArmKinematics
from .planner import SafeRoute
from .policy import ClassicalPolicy

FLAGS = {"planner.lift_from_contact": True, "calib.plane.apply_height": True, "policy.contact_start_unblock": True,
         "policy.zero_det_relook": True, "policy.descend_contact_guard": True}


def contact_start(pol, y_left, clr=0.001):
    """Both hands resting ON the counter: the lowest hand point of each side is clr above the counter surface."""
    a = pol.home.copy()
    hand = pol.post["open"]
    for side, y in (("left", y_left), ("right", -y_left)):
        z = pol.scene.counter_z + 0.05
        for _ in range(6):
            q, ok, _, _ = pol.planner.kin.solve(side, np.array([0.30, y, z]), a.copy(), top_down=False, restarts=True)
            assert ok, f"IK failed for {side}"
            a[ArmKinematics.sl(side)] = q[ArmKinematics.sl(side)]
            z -= pol.planner.clearance(side, a, hand) - clr
    return a


def make():
    pol = ClassicalPolicy(config=load_config(overrides=FLAGS), log=None)
    pol.reset({"instruction": "x", "task": "sort_fixed", "mapping": {"yellow": "left", "blue": "center", "red": "right"}})
    return pol


def obs(pol, a, t):
    return {"arm_q": a, "hand_q": pol.post["open"].copy(), "t": t, "head_rgb": None, "head_depth": None}


class TestHandsOnTableStart(unittest.TestCase):
    def drive(self, pol, a, n=60):
        rows, t = [], 0.0
        pol.state = "init"
        for _ in range(n):
            r = pol.act(obs(pol, a, t))
            A = np.asarray(r["arm_q_target"])
            if len(A):
                rows.append(A)
                a = A[-1].astype(float)
                t += len(A) / 30.0
            else:
                t += 1 / 30.0
            if sum(len(x) for x in rows) > 150:
                break
        return rows

    def test_palms_at_table_height_lift_straight_up(self):
        pol = make()
        a0 = contact_start(pol, 0.20)
        self.assertLess(min(pol.planner.clearance(s, a0, pol.post["open"]) for s in ("left", "right")), 0.003)
        rows = self.drive(pol, a0)
        A = np.vstack(rows)
        self.assertGreater(len(A), 60, "no motion: planning is blocked by the on-table start")
        pl = pol.planner
        k = min(len(A) - 1, 45)   # first 1.5 s: vertical only
        for s in ("left", "right"):
            xy = np.array([pl.kin.grasp_pose(s, A[i])[0][:2] for i in range(k)])
            z = np.array([pl.kin.grasp_pose(s, A[i])[0][2] for i in range(k)])
            self.assertLess(np.abs(xy - xy[0]).max(), 2e-3, f"{s} hand moved laterally while on the table")
            self.assertGreaterEqual(z[-1], z[0] - 1e-3)

    def test_failed_proof_never_blocks_forever(self):
        pol = make()
        pol.planner.safe_home_route = lambda *a, **k: SafeRoute(False, reason="forced: keepout")
        a0 = contact_start(pol, 0.20)
        rows = self.drive(pol, a0, n=80)
        A = np.vstack(rows) if rows else np.zeros((0, 14))
        self.assertGreater(len(A), 30, "permanent hold: a clearance check blocked the start")
        pl = pol.planner
        for s in ("left", "right"):
            xy = np.array([pl.kin.grasp_pose(s, A[i])[0][:2] for i in range(min(len(A), 30))])
            self.assertLess(np.abs(xy - xy[0]).max(), 2e-3)


class TestZeroDetRelook(unittest.TestCase):
    def test_zero_dets_relook_until_margin(self):
        pol = make()
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        pol.time_limit = 120.0
        pol.state = "look"
        r = pol._after_look(("none", None, 0), 10.0)
        self.assertEqual(pol.state, "settle")
        self.assertFalse(r["done"])
        pol._after_look(("none", None, 0), 40.0)
        self.assertEqual(pol.state, "settle")
        pol._after_look(("none", None, 0), 118.0)   # inside the end margin: stock stop
        self.assertEqual(pol.state, "done")

    def test_after_success_clear_counter_ends(self):
        pol = make()
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        pol._n_held_yes = 2
        for t in (10.0, 11.5, 13.2):
            pol.state = "look"
            pol._after_look(("none", None, 0), t)
        self.assertEqual(pol.state, "done")


if __name__ == "__main__":
    unittest.main()
