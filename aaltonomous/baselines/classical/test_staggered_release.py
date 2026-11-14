"""CPU-only tests for the thumb-first Dex3 release trajectory."""
import unittest
from types import MethodType, SimpleNamespace

import numpy as np

from .config import hand_postures, load_config
from aaltonomous.interface.safety import SafetyConfig

from .planner import Planner, dex3_release_groups, ramp, staggered_release


class TestStaggeredRelease(unittest.TestCase):
    def test_thumb_then_index_and_middle_for_each_hand(self):
        cfg = load_config(overlay="none")
        post = hand_postures(cfg)
        dt = 1.0 / 30.0
        delay_s = cfg["grasp"]["stagger_delay_s"]
        settle_s = cfg["grasp"]["release_settle_s"]
        delay_ticks = int(np.ceil(delay_s / dt - 1e-12))
        settle_ticks = int(np.ceil(settle_s / dt - 1e-12))
        expected = {
            "left": {"thumb": (0, 1, 2), "middle": (3, 4), "index": (5, 6)},
            "right": {"thumb": (7, 8, 9), "index": (10, 11), "middle": (12, 13)},
        }

        for side in ("left", "right"):
            with self.subTest(side=side):
                groups = dex3_release_groups(side)
                self.assertEqual(groups, expected[side])
                active = np.asarray(groups["thumb"] + groups["index"] + groups["middle"])
                inactive = np.asarray([i for i in range(14) if i not in active])
                start = post["closed"].copy()
                target = start.copy()
                target[active] = post["release"][active]
                h = staggered_release(start, target, side, cfg["motion"]["hand_max_vel"], dt,
                                      delay_s, settle_s)

                first_change = {}
                for name, indices in groups.items():
                    moving = [i for i in indices if target[i] != start[i]]
                    first_change[name] = min(np.flatnonzero(np.any(h[:, moving] != start[moving], axis=1)))
                self.assertEqual(first_change["thumb"], 0)
                self.assertEqual(first_change["index"], delay_ticks)
                self.assertEqual(first_change["middle"], delay_ticks)
                self.assertTrue(np.array_equal(h[:, inactive], np.repeat(start[None, inactive], len(h), axis=0)))

                all_open = np.flatnonzero(np.all(np.isclose(h[:, active], target[active], atol=1e-12), axis=1))
                self.assertTrue(len(all_open))
                self.assertEqual(len(h) - int(all_open[0]) - 1, settle_ticks)
                lo, hi = np.minimum(start, target), np.maximum(start, target)
                self.assertTrue(np.all((h >= lo) & (h <= hi)))

    def test_planner_open_phase_holds_arm_runs_safety_and_preserves_stock_bytes(self):
        """Exercise the actual planner wiring without the third-party URDF by replacing only its geometry/IK."""
        for enabled in (False, True):
            cfg = load_config(overlay="none", overrides={"grasp.staggered_release": enabled})
            post = hand_postures(cfg)
            scene = SimpleNamespace(counter_z=0.0, cfg={"bins": {"wall_height": 0.12}},
                                    object_height=lambda: 0.045)
            kin = SimpleNamespace(hint=None, w_hint=0.0)
            planner = Planner(cfg, scene, post, np.zeros(14), kin=kin, safety=SafetyConfig(workspace=False))

            def ik(self, side, point, seed, name, plan):
                plan.ik[name] = (True, 0.0, 0.0)
                return np.zeros(14), True

            def cartesian(self, side, p0, p1, q0, name, plan):
                plan.ik[name] = (True, 0.0, 0.0)
                return [np.zeros(14)]

            planner._ik = MethodType(ik, planner)
            planner._cartesian = MethodType(cartesian, planner)
            planner.clearance = MethodType(lambda self, *args, **kwargs: np.inf, planner)
            planner.bin_clearance = MethodType(lambda self, *args, **kwargs: np.inf, planner)
            planner.hits_bin_wall = MethodType(lambda self, *args, **kwargs: False, planner)
            planner.drop_xy = MethodType(lambda self, *args, **kwargs: np.zeros(2), planner)
            det = SimpleNamespace(color="red", top=np.array([0.4, 0.0, 0.045]))

            for side in ("left", "right"):
                with self.subTest(enabled=enabled, side=side):
                    plan = planner._plan_side_inner(det, side, "left", np.zeros(14), post["open"])
                    self.assertTrue(plan.ok, plan.reason)
                    self.assertEqual(plan.safety_clips, 0)
                    start, end = next((s, e) for name, s, e in plan.phases if name == "open")
                    self.assertTrue(np.array_equal(plan.arm[start:end],
                                                   np.repeat(plan.arm[start][None], end - start, axis=0)))
                    if not enabled:
                        target = plan.hand[start - 1].copy()
                        active = slice(0, 7) if side == "left" else slice(7, 14)
                        target[active] = post["release"][active]
                        legacy = ramp(plan.hand[start - 1], target, cfg["motion"]["hand_max_vel"], planner.dt)
                        legacy = np.vstack([legacy, np.repeat(target[None],
                                                             int(round(cfg["motion"]["open_hold_s"] / planner.dt)),
                                                             axis=0)])
                        self.assertTrue(np.array_equal(plan.hand[start:end], legacy))


if __name__ == "__main__":
    unittest.main()
