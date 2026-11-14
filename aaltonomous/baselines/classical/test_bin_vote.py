"""bin.vote: the 2-of-3 rule, failure modes, planner integration. Pure / synthetic."""
import unittest

import numpy as np

from .bin_vote import combine, resolve, rim_votes, vote_bins
from .config import load_config
from .planner import Planner

V = {"agree_m": 0.04, "search_m": 0.10, "max_shift_m": 0.10, "low_conf_extra_m": 0.04, "min_rim_points": 180,
     "min_tape_px": 60, "tape_hsv": [[3, 150, 70], [8, 255, 180]]}
Z = np.zeros(2)


def xy(x, y=0.0):
    return np.array([x, y])


class TestCombine(unittest.TestCase):
    def test_two_of_three_agree(self):
        p, n, e = combine(Z, xy(0.0, 0.07), xy(0.0, 0.06), V)
        np.testing.assert_allclose(p, [0.0, 0.065])
        self.assertEqual((n, e), (2, 0.0))

    def test_all_agree(self):
        p, n, e = combine(Z, xy(0.0, 0.01), xy(0.0, 0.02), V)
        self.assertEqual((n, e), (3, 0.0))
        np.testing.assert_allclose(p, [0.0, 0.01])

    def test_all_disagree_median_plus_extra(self):
        p, n, e = combine(Z, xy(0.0, 0.09), xy(0.0, -0.05), V)
        np.testing.assert_allclose(p, Z)
        self.assertEqual((n, e), (0, 0.04))

    def test_rim_missing_tape_and_prior(self):
        p, n, e = combine(Z, None, xy(0.0, 0.03), V)
        self.assertEqual((n, e), (2, 0.0))
        p, n, e = combine(Z, None, xy(0.0, 0.08), V)           # disagree: median of 2 = mean, extra
        np.testing.assert_allclose(p, [0.0, 0.04])
        self.assertEqual((n, e), (0, 0.04))

    def test_only_prior(self):
        self.assertEqual(combine(Z, None, None, V)[1:], (0, 0.04))

    def test_garbage_votes_ignored(self):
        p, n, e = combine(Z, xy(np.nan, 0.0), None, V)
        np.testing.assert_allclose(p, Z)
        self.assertEqual((n, e), (0, 0.04))

    def test_shift_over_max_keeps_prior(self):
        p, n, e = combine(Z, xy(0.0, 0.12), xy(0.0, 0.13), V)
        np.testing.assert_allclose(p, Z)
        self.assertEqual((n, e), (0, 0.04))

    def test_ordering_violation_falls_back_to_prior(self):
        pri = {"left": xy(0.3, 0.12), "center": xy(0.3, 0.0), "right": xy(0.3, -0.12)}
        ok = {"left": xy(0.3, 0.13), "center": xy(0.3, 0.01), "right": xy(0.3, -0.11)}
        sh, ex, _ = resolve(pri, ok, ok, V)
        self.assertAlmostEqual(sh["left"][1], 0.01 * 2 / 3)   # all agree: mean of prior + 2 votes
        bad = dict(ok, left=xy(0.3, 0.07), center=xy(0.3, 0.09))   # centre ends up left of the left bin
        sh, ex, _ = resolve(pri, bad, bad, V)
        self.assertTrue(all(np.allclose(s, 0) for s in sh.values()))
        self.assertTrue(all(e == 0.04 for e in ex.values()))


class TestDriver(unittest.TestCase):
    def test_nan_and_garbage_depth_never_raise(self):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(overrides={"bin.vote.enabled": True})
        for depth in (np.full((48, 64), np.nan), np.zeros((48, 64)), None, "junk", np.full((48, 64), 1e12)):
            sh, ex = vote_bins(pol.perception, pol.planner, [{"head_rgb": None, "head_depth": depth}] * 2, pol.cfg,
                               log=lambda m: None)
            self.assertTrue(all(np.allclose(s, 0) for s in sh.values()), depth)
            self.assertTrue(all(e == 0.04 for e in ex.values()))
        self.assertTrue(all(v is None for v in rim_votes(pol.perception, None, {"left": Z}, pol.cfg["bin"]["vote"]).values()))


def _plan(pl, bin_name="center"):
    from .config import hand_postures
    from .perception import Detection
    sc = pl.scene
    top = sc.mat_to_pelvis([0.45, 0.15, sc.object_height()])
    return pl.plan(Detection("yellow", top, 0, 0, "depth"), pl.home, pl.post["open"], bin_name=bin_name)


class TestPlanner(unittest.TestCase):
    def setUp(self):
        from .config import hand_postures, home_arm
        self.cfg = load_config(overlay="none")
        from .policy import ClassicalPolicy
        self.pl = ClassicalPolicy(config=self.cfg).planner

    def test_shift_applies_to_bin_drop_and_slots(self):
        pl = self.pl
        n = pl.scene.bin_names[0]
        base = (pl.bin_xy(n).copy(), pl.drop_xy(n).copy(), [s.copy() for s in pl.drop_slots(n)])
        pl.bin_shift = {n: xy(0.01, -0.02)}
        np.testing.assert_allclose(pl.bin_xy(n), base[0] + [0.01, -0.02])
        np.testing.assert_allclose(pl.drop_xy(n), base[1] + [0.01, -0.02])
        for a, b in zip(pl.drop_slots(n), base[2]):
            np.testing.assert_allclose(a, b + [0.01, -0.02])

    def test_drop_z_rises_by_exactly_drop_extra(self):
        pl = self.pl
        a = _plan(pl)
        self.assertTrue(a.ok, a.reason)
        pl.drop_extra = {"center": 0.02}
        b = _plan(pl)
        self.assertTrue(b.ok, b.reason)
        self.assertAlmostEqual(b.waypoints["drop"][2] - a.waypoints["drop"][2], 0.02, places=9)
        pl.drop_extra = {"center": 1.0}                     # capped at the carry height
        c = _plan(pl)
        self.assertAlmostEqual(c.waypoints["drop"][2], c.waypoints["lift"][2], places=9)

    def test_empty_dicts_plan_is_byte_identical(self):
        pl = self.pl
        a = _plan(pl)
        pl.bin_shift, pl.drop_extra = {}, {}
        b = _plan(pl)
        self.assertEqual(a.arm.tobytes(), b.arm.tobytes())
        self.assertEqual(a.hand.tobytes(), b.hand.tobytes())

    def test_flag_off_by_default(self):
        self.assertFalse(load_config(overlay="none")["bin"]["vote"]["enabled"])


if __name__ == "__main__":
    unittest.main()
