"""Two-stage, non-blocking Dex3 release trajectory tests."""
import unittest

import numpy as np

from .planner import staggered_release_ramp


class TestStaggeredRelease(unittest.TestCase):
    def test_second_finger_starts_after_configured_delay_and_finishes_open(self):
        start = np.array([0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6,
                          0.0, 0.0, -1.2, 1.0, 0.6, 1.0, 0.6])
        opened = np.array([0.0, -0.5, 1.5, -0.05, -0.05, -0.05, -0.05,
                           0.0, 0.5, -1.5, 0.05, 0.05, 0.05, 0.05])
        dt, delay = 1.0 / 30.0, 0.25

        for side, off in (("left", 0), ("right", 7)):
            path, second_tick = staggered_release_ramp(start, opened, side, 1.2, dt, delay)
            self.assertEqual(second_tick, 8)  # ceil(0.25 s * 30 Hz): never starts early
            np.testing.assert_array_equal(path[:second_tick, off + 5:off + 7],
                                          np.repeat(start[None, off + 5:off + 7], second_tick, axis=0))
            self.assertFalse(np.array_equal(path[0, off:off + 5], start[off:off + 5]))
            self.assertFalse(np.array_equal(path[second_tick, off + 5:off + 7],
                                             start[off + 5:off + 7]))
            np.testing.assert_allclose(path[-1], opened, atol=0.0)

    def test_planner_records_stage_ticks_and_ends_release_phase_at_release_posture(self):
        from aaltonomous.scene.scene import Scene
        from .config import hand_postures, home_arm, load_config
        from .perception import Detection
        from .planner import Planner

        cfg = load_config(overrides={"grasp.staggered_release": True,
                                     "grasp.staggered_release_delay_s": 0.30})
        scene = Scene.load()
        post, home = hand_postures(cfg), home_arm(cfg)
        layout = scene.cfg["layout"]
        top = scene.mat_to_pelvis([np.mean(layout["region_x"]) + 0.08, np.mean(layout["region_y"]),
                                   scene.object_height()])
        plan = Planner(cfg, scene, post, home).plan(Detection("yellow", top, 0, 0, "depth"),
                                                    home, post["open"].copy())
        self.assertTrue(plan.ok, plan.reason)
        first, second, ramp_end = plan.release_stages
        open_start, open_end = next((s, e) for name, s, e in plan.phases if name == "open")
        self.assertEqual(first, open_start)
        self.assertEqual(second - first, 9)
        self.assertLessEqual(ramp_end, open_end)
        hand_slice = slice(0, 7) if plan.side == "left" else slice(7, 14)
        np.testing.assert_allclose(plan.hand[open_end - 1, hand_slice], post["release"][hand_slice], atol=0.0)


if __name__ == "__main__":
    unittest.main()
