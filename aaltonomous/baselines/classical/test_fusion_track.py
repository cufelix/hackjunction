"""perception.depth_fusion + perception.track (Oliver fusion). Run:
    python -m unittest aaltonomous.baselines.classical.test_fusion_track -v
"""
import unittest

import numpy as np

from .config import FLAGS, _get, load_config
from .fusion import DepthFuser, Tracker
from .fusion_bench import SC, CAL, Sequence
from .perception import Detection, Perception
from .policy import ClassicalPolicy
from .mock_world import make_mock_env

FUSE = {"perception.depth_fusion.enabled": True}
TRACK = {"perception.track.enabled": True}


def cfg(**ov):
    return load_config(overlay="none", overrides=ov)


def det(color, x, y, z=0.8):
    return Detection(color, np.array([x, y, z]), 100, 100, "depth")


class TestFlagsOff(unittest.TestCase):
    def test_registered_and_off(self):
        for k in ("perception.depth_fusion.enabled", "perception.track.enabled"):
            self.assertIn(k, FLAGS)
            self.assertIs(_get(load_config(overlay="none"), k), False)
            self.assertIs(_get(load_config(overlay="submission"), k), False)

    def test_off_is_the_stock_pipeline(self):
        p = Perception(cfg(), SC, CAL)
        self.assertFalse(p.temporal)
        seq = Sequence(np.random.default_rng(0))
        rgb, depth = seq.frame()
        a = p.detect(rgb, depth)
        p.observe(rgb, depth)                        # no-op when off
        b = Perception(cfg(), SC, CAL).detect(rgb, depth)
        self.assertEqual(len(a), len(b))
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.top, y.top)
            self.assertIsNone(x.track_id)

    def test_enabled_with_no_history_equals_stock(self):
        rgb, depth = Sequence(np.random.default_rng(1)).frame()
        a = Perception(cfg(), SC, CAL).detect(rgb, depth)
        b = Perception(cfg(**FUSE), SC, CAL).detect(rgb, depth)      # empty buffer -> unchanged
        for x, y in zip(a, b):
            np.testing.assert_array_equal(x.top, y.top)


class TestDepthFuser(unittest.TestCase):
    def setUp(self):
        self.f = DepthFuser({"frames": 5, "min_valid": 3, "tol_mm": 8, "range_mm": (150, 2000)})
        self.roi = np.zeros((20, 30), bool)
        self.roi[5:15, 5:25] = True

    def _frames(self, n=4, base=600):
        rng = np.random.default_rng(0)
        return [(base + rng.normal(0, 2, (20, 30))).astype(np.uint16) for _ in range(n)]

    def test_median_rejects_flying_pixel_and_fills_hole(self):
        past = self._frames()
        for d in past:
            self.f.push(d)
        cur = self._frames(1)[0].copy()
        cur[10, 10] = 300                  # flying pixel in the current frame
        cur[8, 8] = 0                      # hole in the current frame
        ref = cur.copy()
        out = self.f.fuse(cur, self.roi)
        np.testing.assert_array_equal(cur, ref)            # input untouched
        self.assertLess(abs(int(out[10, 10]) - 600), 8)
        self.assertLess(abs(int(out[8, 8]) - 600), 8)
        np.testing.assert_array_equal(out[~self.roi], cur[~self.roi])   # outside the ROI: unchanged

    def test_too_few_valid_samples_become_holes(self):
        past = self._frames()
        for d in past:
            d[10, 10] = 0
            self.f.push(d)
        cur = self._frames(1)[0]
        self.assertEqual(int(self.f.fuse(cur, self.roi)[10, 10]), 0)   # only 1 valid sample of 5 < 3

    def test_inconsistent_samples_are_dropped(self):
        for v in (600, 700, 800, 900):                 # no 3 samples agree within 8 mm
            self.f.push(np.full((20, 30), v, np.uint16))
        out = self.f.fuse(np.full((20, 30), 1000, np.uint16), self.roi)
        self.assertEqual(int(out[10, 10]), 0)

    def test_reset_empties(self):
        self.f.push(self._frames(1)[0])
        self.f.reset()
        d = self._frames(1)[0]
        self.assertIs(self.f.fuse(d, self.roi), d)


class TestFusionLowersError(unittest.TestCase):
    def test_fused_z_and_jitter_beat_single_frame(self):
        stats = {}
        for name, ov in (("single", {}), ("fused", FUSE)):
            ez, pos = [], {}
            for s in range(3):
                seq = Sequence(np.random.default_rng(10 + s), sigma_mm=3.0, hole=0.10)
                p = Perception(cfg(**ov), SC, CAL)
                for f in range(10):
                    rgb, depth = seq.frame()
                    dets = p.detect(rgb, depth)
                    if p.fuser is not None:
                        p.fuser.push(depth)
                    if f < 5:
                        continue
                    for col, top in seq.truth():
                        d = [d for d in dets if d.color == col]
                        self.assertEqual(len(d), 1, (name, f, col))
                        ez.append(abs(d[0].top[2] - top[2]))
                        pos.setdefault((s, col), []).append(d[0].top[:2])
            jit = np.mean([np.linalg.norm(np.std(v, axis=0)) for v in pos.values()])
            stats[name] = (np.mean(ez), jit)
        self.assertLess(stats["fused"][0], stats["single"][0])
        self.assertLess(stats["fused"][1], 0.5 * stats["single"][1])


class TestTracker(unittest.TestCase):
    def tr(self, **kw):
        return Tracker({"passthrough_cold": False, "confirm_m": 3, "window_n": 5, "drop_after": 3, **kw})

    def test_real_object_confirmed_on_the_mth_frame_and_id_is_stable(self):
        t = self.tr()
        outs = [t.update([det("red", 0.40, 0.10)]) for _ in range(4)]
        self.assertEqual([len(o) for o in outs], [0, 0, 1, 1])      # M = 3 -> third frame
        self.assertEqual(outs[2][0].track_id, outs[3][0].track_id)

    def test_one_frame_phantom_is_never_confirmed(self):
        t = self.tr()
        for f in range(12):
            d = [det("red", 0.40, 0.10)]
            if f == 7:
                d.append(det("blue", 0.35, -0.12))
            out = t.update(d)
            self.assertTrue(all(o.color == "red" for o in out), f)
        # even a phantom that shows up twice (not consecutive) stays below 3 of 5
        t = self.tr()
        for f in range(10):
            d = [det("red", 0.40, 0.10)] + ([det("blue", 0.3, 0.0)] if f in (5, 7) else [])
            self.assertTrue(all(o.color == "red" for o in t.update(d)))

    def test_smoothing_reduces_jitter(self):
        rng = np.random.default_rng(0)
        t = self.tr()
        raw, sm = [], []
        for f in range(30):
            xy = np.array([0.4, 0.1]) + rng.normal(0, 0.003, 2)
            o = t.update([det("red", *xy)])
            if f >= 5:
                raw.append(xy)
                sm.append(o[0].top[:2])
        self.assertLess(np.std(sm, axis=0).max(), 0.6 * np.std(raw, axis=0).max())

    def test_two_same_colour_objects_keep_distinct_ids_and_vanished_one_is_dropped(self):
        t = self.tr()
        for _ in range(4):
            out = t.update([det("red", 0.40, 0.10), det("red", 0.40, 0.20)])
        self.assertEqual(len({o.track_id for o in out}), 2)
        for _ in range(4):                                  # one picked / gone
            out = t.update([det("red", 0.40, 0.10)])
        self.assertEqual([round(o.top[1], 2) for o in out], [0.10])
        self.assertEqual(len(t.tracks), 1)                  # the other track was forgotten
        out = t.update([det("red", 0.40, 0.10), det("red", 0.40, 0.20)])   # reappears: must be re-confirmed
        self.assertEqual(len(out), 1)

    def test_far_apart_is_a_new_object(self):
        t = self.tr()
        for _ in range(3):
            t.update([det("red", 0.40, 0.10)])
        out = t.update([det("red", 0.40, 0.14)])            # 4 cm jump > 2 cm radius
        self.assertEqual(out, [])

    def test_cold_start_passes_through_so_the_policy_never_sees_nothing(self):
        t = Tracker({"confirm_m": 3, "window_n": 5})        # passthrough_cold default on
        self.assertEqual(len(t.update([det("red", 0.4, 0.1)])), 1)
        t.update([det("red", 0.4, 0.1)])
        t.update([det("red", 0.4, 0.1)])                    # warm now
        self.assertEqual(t.update([det("blue", 0.3, 0.0)]), [])

    def test_reset(self):
        t = self.tr()
        for _ in range(3):
            t.update([det("red", 0.4, 0.1)])
        t.reset()
        self.assertEqual(t.update([det("red", 0.4, 0.1)]), [])


class TestTrackedPerception(unittest.TestCase):
    def test_phantom_frame_filtered_end_to_end(self):
        seq = Sequence(np.random.default_rng(5), sigma_mm=2.5, hole=0.07)
        p = Perception(cfg(**TRACK, **{"perception.track.passthrough_cold": False, "perception.track.confirm_m": 3,
                                       "perception.track.window_n": 5}), SC, CAL)
        got = []
        for f in range(8):
            if f == 5:
                (rgb, depth), ph = seq.phantom_frame()
            else:
                rgb, depth = seq.frame()
            got.append(len(p.detect(rgb, depth)))
        self.assertEqual(got[:2], [0, 0])
        self.assertEqual(got[2:], [3] * 6)                  # confirmed at frame 3, the phantom never shows

    def test_policy_warms_up_with_short_holds_then_plans_and_resets(self):
        c = cfg(**FUSE, **TRACK)
        env = make_mock_env(seed=0, cfg=c)
        pol = ClassicalPolicy(config=c, log=None, safety=env.filter.cfg)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        look_acts = 0
        for _ in range(200):
            was = pol.state
            obs, info = env.step(pol.act(obs))
            if was == "look":
                look_acts += 1
            if was != "look":                                        # any act() outside 'look' clears the history
                self.assertEqual(len(pol.perception.fuser.buf), 0)
            if pol.state == "executing":
                break
        self.assertEqual(pol.state, "executing")
        # frames=3, confirm 2-of-3: 2 warm-up holds (3 ticks each = 0.2 s) + the look act itself
        self.assertEqual(look_acts, 3)
        self.assertGreaterEqual(pol.perception.tracker.frames, 3)
        pol.reset({"instruction": obs["instruction"]})
        self.assertEqual(pol.perception.tracker.frames, 0)
        self.assertEqual(len(pol.perception.fuser.buf), 0)

    def test_flags_off_policy_looks_on_the_first_look_act(self):
        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(log=None, safety=env.filter.cfg)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        look_acts = 0
        for _ in range(200):
            was = pol.state
            obs, info = env.step(pol.act(obs))
            look_acts += was == "look"
            if pol.state == "executing":
                break
        self.assertEqual(look_acts, 1)

    def test_full_mock_episode_with_both_flags_on(self):
        c = cfg(**FUSE, **TRACK)
        env = make_mock_env(seed=1, cfg=c)
        pol = ClassicalPolicy(config=c, log=None, safety=env.filter.cfg)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        while True:
            obs, info = env.step(pol.act(obs))
            if info["episode_over"]:
                break
        out = env.backend.outcomes(pol.mapping)
        self.assertNotIn("wrong", out)
        self.assertGreaterEqual(out.count("correct"), 1, out)


if __name__ == "__main__":
    unittest.main()
