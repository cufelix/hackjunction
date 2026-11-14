"""bin.vote: the 2-of-3 rule, failure modes, planner integration, tape blobs. Pure / synthetic."""
import threading
import time
import unittest
from unittest import mock

import numpy as np

from .bin_vote import _bin_blob_vote, _blob_vote, combine, resolve, rim_votes, tape_votes, vote_bins
from .config import load_config
from .planner import Planner

V = {"agree_m": 0.04, "search_m": 0.10, "max_shift_m": 0.10, "low_conf_extra_m": 0.04, "min_rim_points": 180,
     "min_tape_px": 60, "tape_hsv": [[3, 150, 70], [8, 255, 180]], "tape_elong_aspect": 3.0}
Z = np.zeros(2)
CONF_AGREE, CONF_PRIOR, CONF_DISAGREE = 1.0, 0.5, 0.3


def xy(x, y=0.0):
    return np.array([x, y])


class TestCombine(unittest.TestCase):
    def test_two_of_three_agree(self):
        p, n, e, c = combine(Z, xy(0.0, 0.07), xy(0.0, 0.06), V)
        np.testing.assert_allclose(p, [0.0, 0.065])
        self.assertEqual((n, e, c), (2, 0.0, CONF_AGREE))

    def test_all_agree(self):
        p, n, e, c = combine(Z, xy(0.0, 0.01), xy(0.0, 0.02), V)
        self.assertEqual((n, e, c), (3, 0.0, CONF_AGREE))
        np.testing.assert_allclose(p, [0.0, 0.01])

    def test_full_disagreement_keeps_prior_with_extra_and_low_conf(self):
        p, n, e, c = combine(Z, xy(0.0, 0.09), xy(0.0, -0.05), V)
        np.testing.assert_array_equal(p, Z)            # shift exactly 0, no median
        self.assertEqual((n, e), (0, 0.04))
        self.assertLess(c, 0.5)
        self.assertEqual(c, CONF_DISAGREE)

    def test_rim_missing_tape_and_prior(self):
        p, n, e, c = combine(Z, None, xy(0.0, 0.03), V)
        self.assertEqual((n, e, c), (2, 0.0, CONF_AGREE))
        p, n, e, c = combine(Z, None, xy(0.0, 0.08), V)       # prior and tape disagree: prior kept, extra, low conf
        np.testing.assert_array_equal(p, Z)
        self.assertEqual((n, e, c), (0, 0.04, CONF_DISAGREE))

    def test_only_prior(self):
        self.assertEqual(combine(Z, None, None, V)[1:], (0, 0.04, CONF_PRIOR))

    def test_garbage_votes_ignored(self):
        p, n, e, c = combine(Z, xy(np.nan, 0.0), None, V)
        np.testing.assert_allclose(p, Z)
        self.assertEqual((n, e, c), (0, 0.04, CONF_PRIOR))

    def test_shift_over_max_keeps_prior(self):
        p, n, e, c = combine(Z, xy(0.0, 0.12), xy(0.0, 0.13), V)   # 2-of-3 agree, but mean 0.125 > max 0.10
        np.testing.assert_allclose(p, Z)
        self.assertEqual((n, e, c), (0, 0.04, CONF_PRIOR))

    def test_max_shift_cap_does_not_touch_disagreement(self):
        # disagreement is always shift 0; the cap is only about agreed results
        p, n, e, c = combine(Z, xy(0.0, 0.09), xy(0.0, -0.05), V)
        np.testing.assert_array_equal(p, Z)
        self.assertEqual(c, CONF_DISAGREE)

    def test_full_disagreement_via_resolve_shift_zero(self):
        pri = {"left": xy(0.3, 0.12), "center": xy(0.3, 0.0), "right": xy(0.3, -0.12)}
        rims = {"left": xy(0.3, 0.20)}
        tapes = {"left": xy(0.3, 0.04)}                 # three pairwise gaps > agree_m: full disagreement
        sh, ex, info = resolve(pri, rims, tapes, V)
        np.testing.assert_array_equal(sh["left"], Z)
        self.assertEqual(ex["left"], 0.04)
        self.assertLess(info["left"][2], 0.5)
        self.assertEqual(info["left"][2], CONF_DISAGREE)
        self.assertEqual(info["center"][2], CONF_PRIOR)

    def test_ordering_violation_falls_back_to_prior(self):
        pri = {"left": xy(0.3, 0.12), "center": xy(0.3, 0.0), "right": xy(0.3, -0.12)}
        ok = {"left": xy(0.3, 0.13), "center": xy(0.3, 0.01), "right": xy(0.3, -0.11)}
        sh, ex, _ = resolve(pri, ok, ok, V)
        self.assertAlmostEqual(sh["left"][1], 0.01 * 2 / 3)   # all agree: mean of prior + 2 votes
        bad = dict(ok, left=xy(0.3, 0.07), center=xy(0.3, 0.09))   # centre ends up left of the left bin
        sh, ex, _ = resolve(pri, bad, bad, V)
        self.assertTrue(all(np.allclose(s, 0) for s in sh.values()))
        self.assertTrue(all(e == 0.04 for e in ex.values()))


class TestTapeBlobs(unittest.TestCase):
    def test_elongated_blob_votes_inner_end_not_centroid(self):
        # 10 cm x 2 cm strip centred at (0.40, 0.20) in pelvis xy: PCA aspect 5, inner end is x ~0.35 (near robot)
        X, Y = np.meshgrid(np.linspace(0.35, 0.45, 21), np.linspace(0.19, 0.21, 5))
        Q = np.c_[X.ravel(), Y.ravel()]
        v = _blob_vote(Q, V["tape_elong_aspect"])
        self.assertLess(abs(v[0] - 0.36), 0.01)
        self.assertLess(abs(v[1] - 0.20), 0.005)
        self.assertGreater(abs(v[0] - 0.40), 0.03)      # not the centroid

    def test_round_blob_votes_centroid(self):
        X, Y = np.meshgrid(np.linspace(0.38, 0.42, 9), np.linspace(0.18, 0.22, 9))
        v = _blob_vote(np.c_[X.ravel(), Y.ravel()], V["tape_elong_aspect"])
        np.testing.assert_allclose(v, [0.40, 0.20], atol=1e-6)

    def test_aspect_not_computable_gives_no_vote(self):
        self.assertIsNone(_blob_vote(np.array([[0.3, 0.2], [0.3, 0.21]]), 3.0))   # too few points
        self.assertIsNone(_blob_vote(np.tile([0.3, 0.2], (5, 1)), 3.0))           # all on one spot
        self.assertIsNone(_blob_vote(np.array([[np.nan, 0.2]] * 5), 3.0))

    def test_ambiguous_bin_gives_no_vote(self):
        big = np.c_[np.linspace(0.3, 0.4, 30), np.full(30, 0.2)]
        similar = np.c_[np.linspace(0.3, 0.4, 25), np.full(25, 0.3)]
        self.assertIsNone(_bin_blob_vote([big, similar], 3.0))
        small = np.c_[np.linspace(0.3, 0.31, 5), np.full(5, 0.3)]
        self.assertIsNotNone(_bin_blob_vote([big, small], 3.0))   # second blob < half the first: the big one votes
        self.assertIsNone(_bin_blob_vote([], 3.0))


class TestDriver(unittest.TestCase):
    def test_nan_and_garbage_depth_never_raise(self):
        from .policy import ClassicalPolicy
        pol = ClassicalPolicy(overrides={"bin.vote.enabled": True})
        for depth in (np.full((48, 64), np.nan), np.zeros((48, 64)), None, "junk", np.full((48, 64), 1e12)):
            sh, ex, cf = vote_bins(pol.perception, pol.planner, [{"head_rgb": None, "head_depth": depth}] * 2, pol.cfg,
                                   log=lambda m: None)
            self.assertTrue(all(np.allclose(s, 0) for s in sh.values()), depth)
            self.assertTrue(all(e == 0.04 for e in ex.values()))
            self.assertTrue(all(c in (CONF_PRIOR, CONF_DISAGREE, CONF_AGREE) for c in cf.values()))
        self.assertTrue(all(v is None for v in rim_votes(pol.perception, None, {"left": Z}, pol.cfg["bin"]["vote"]).values()))


def _edge_points(center, half, n=100):
    """Dense samples of both axes of one rectangle in pelvis xy."""
    t = np.linspace(-1.0, 1.0, n)
    return np.vstack((np.c_[np.full(n, center[0] - half[0]), center[1] + t * half[1]],
                      np.c_[np.full(n, center[0] + half[0]), center[1] + t * half[1]],
                      np.c_[center[0] + t * half[0], np.full(n, center[1] - half[1])],
                      np.c_[center[0] + t * half[0], np.full(n, center[1] + half[1])]))


class _SyntheticRimPerception:
    def __init__(self, planner, delta, cfg):
        self.scene, self.cfg = planner.scene, cfg["perception"]
        b = self.scene.cfg["bins"]
        outer = np.asarray(b["outer_size"], float)[::-1] / 2
        inner = outer - float(b["wall_thickness"])
        points = []
        for name in self.scene.bin_names:
            center = np.asarray(planner.bin_xy(name), float) + delta
            q = np.vstack((_edge_points(center, outer), _edge_points(center, inner)))
            points.append(np.c_[q, np.full(len(q), self.scene.counter_z + float(b["wall_height"]))])
        self.points = np.vstack(points)
        self.transforms = []

    def _counter_plane(self, depth, cfg, z0=None, transform_snapshot=None):
        self.transforms.append(transform_snapshot)
        return self.points, np.array([0.0, 0.0, self.scene.counter_z]), None


class TestSignedRimFit(unittest.TestCase):
    def test_signed_xy_rim_displacements_reach_final_drop_slots(self):
        from .policy import ClassicalPolicy
        cfg = load_config(overrides={"bin.vote.signed_rim_fit": True})
        planner = ClassicalPolicy(config=cfg, log=lambda *_: None).planner
        priors = {n: np.asarray(planner.bin_xy(n), float) for n in planner.scene.bin_names}
        v = cfg["bin"]["vote"]
        for delta in (xy(0.024, 0.0), xy(-0.024, 0.0), xy(0.0, 0.024), xy(0.0, -0.024)):
            with self.subTest(delta=delta):
                base_slots = {n: [p.copy() for p in planner.drop_slots(n)] for n in planner.scene.bin_names}
                perception = _SyntheticRimPerception(planner, delta, cfg)
                rims = rim_votes(perception, np.ones((2, 2)), priors, v)
                for name in planner.scene.bin_names:
                    np.testing.assert_allclose(rims[name], priors[name] + delta, atol=0.001)
                shift, _, _ = resolve(priors, rims, rims, v)
                planner.bin_shift = shift
                expected = 2.0 * delta / 3.0  # prior + agreeing rim+tape: mean of all three votes
                for name in planner.scene.bin_names:
                    np.testing.assert_allclose(shift[name], expected, atol=0.001)
                    for actual, nominal in zip(planner.drop_slots(name), base_slots[name]):
                        np.testing.assert_allclose(actual, nominal + expected, atol=0.001)
                planner.bin_shift = {}

    def test_legacy_rim_sign_is_unchanged_when_flag_is_off(self):
        from .policy import ClassicalPolicy
        cfg = load_config(overlay="none")
        planner = ClassicalPolicy(config=cfg, log=lambda *_: None).planner
        priors = {n: np.asarray(planner.bin_xy(n), float) for n in planner.scene.bin_names}
        delta = xy(0.02, -0.015)
        rims = rim_votes(_SyntheticRimPerception(planner, delta, cfg), np.ones((2, 2)), priors,
                         cfg["bin"]["vote"])
        for name in planner.scene.bin_names:
            np.testing.assert_allclose(rims[name], priors[name] - delta, atol=0.001)


class _SnapshotPerception(_SyntheticRimPerception):
    def __init__(self, planner, delta, cfg, expected_T, target):
        super().__init__(planner, delta, cfg)
        self.expected_T, self.target = expected_T, np.asarray(target, float)

    def _pixel_to_pelvis(self, u, v, depth, transform_snapshot=None):
        self.transforms.append(transform_snapshot)
        if transform_snapshot is not self.expected_T:
            raise AssertionError("tape did not receive the frame's transform snapshot")
        u, v = np.asarray(u), np.asarray(v)
        return np.c_[self.target[0] + (u - 7.5) * 0.0001,
                     self.target[1] + (v - 7.5) * 0.0001,
                     np.full(len(u), self.scene.counter_z + self.scene.cfg["bins"]["wall_height"])]


class TestFrameTransformSnapshot(unittest.TestCase):
    def test_nonzero_pitch_tape_and_rim_agree_on_one_snapshot(self):
        from .policy import ClassicalPolicy
        cfg = load_config(overrides={"bin.vote.signed_rim_fit": True,
                                     "bin.vote.frame_transform_snapshot": True,
                                     "bin.vote.min_tape_px": 20})
        planner = ClassicalPolicy(config=cfg, log=lambda *_: None).planner
        name = "center"
        prior = np.asarray(planner.bin_xy(name), float)
        delta = xy(0.018, -0.012)
        a = np.radians(6.0)
        T = np.array([[1, 0, 0, 0], [0, np.cos(a), -np.sin(a), 0],
                      [0, np.sin(a), np.cos(a), 0], [0, 0, 0, 1]], float)
        T.setflags(write=False)
        perception = _SnapshotPerception(planner, delta, cfg, T, prior + delta)
        rgb = np.zeros((24, 24, 3), np.uint8)
        rgb[3:13, 3:13] = [150, 52, 32]  # OpenCV HSV approximately [5, 201, 150], inside tape_hsv
        depth = np.full((24, 24), 500, np.uint16)
        centers = {n: np.asarray(planner.bin_xy(n), float) for n in planner.scene.bin_names}
        rim = rim_votes(perception, depth, centers, cfg["bin"]["vote"], T)[name]
        tape = tape_votes(perception, rgb, depth, centers, cfg["bin"]["vote"], T)[name]
        np.testing.assert_allclose(rim, prior + delta, atol=0.001)
        np.testing.assert_allclose(tape, prior + delta, atol=0.002)
        self.assertTrue(perception.transforms)
        self.assertTrue(all(snapshot is T for snapshot in perception.transforms))


class TestAsyncBinVote(unittest.TestCase):
    @staticmethod
    def _policy_and_obs():
        from .policy import ClassicalPolicy
        cfg = load_config(overrides={"bin.vote.enabled": True, "bin.vote.async_extract": True,
                                     "bin.vote.frame_transform_snapshot": True, "bin.vote.n_frames": 1})
        policy = ClassicalPolicy(config=cfg, log=lambda *_: None)
        policy.state = "look"
        policy.arm_cmd, policy.hand_cmd = policy.home.copy(), policy.post["open"].copy()
        obs = {"arm_q": policy.home.copy(), "hand_q": policy.post["open"].copy(), "t": 1.0,
               "head_rgb": np.zeros((32, 32, 3), np.uint8), "head_depth": np.full((32, 32), 500, np.uint16)}
        return policy, obs

    def test_act_latency_under_5ms_while_vote_computes_on_copied_frame(self):
        policy, obs = self._policy_and_obs()
        started, release, seen = threading.Event(), threading.Event(), []
        job = None

        def slow_vote(perception, planner, frames, cfg, log):
            seen.append(frames)
            started.set()
            release.wait(2.0)
            names = planner.scene.bin_names
            return ({n: np.zeros(2) for n in names}, {n: 0.0 for n in names}, {n: 1.0 for n in names})

        try:
            with mock.patch("aaltonomous.baselines.classical.policy.vote_bins", side_effect=slow_vote):
                policy.act(obs)
                self.assertTrue(started.wait(1.0))
                job = policy._bv_job
                self.assertIsNot(seen[0][0]["head_depth"], obs["head_depth"])
                self.assertFalse(np.shares_memory(seen[0][0]["head_depth"], obs["head_depth"]))
                self.assertFalse(seen[0][0]["head_depth"].flags.writeable)
                self.assertFalse(seen[0][0]["transform_snapshot"].flags.writeable)
                elapsed = []
                for _ in range(5):
                    t0 = time.perf_counter()
                    policy.act(obs)
                    elapsed.append(time.perf_counter() - t0)
                self.assertLess(max(elapsed), 0.005, elapsed)
        finally:
            release.set()
            if policy._bv_job is not None:
                policy._bv_job["done"].wait(2.0)
        self.assertIsNotNone(job["result"])
        self.assertIsInstance(job["result"].shift, tuple)
        with self.assertRaises(AttributeError):
            job["result"].generation = -1

    def test_reset_generation_discards_stale_vote_result(self):
        policy, obs = self._policy_and_obs()
        started, release = threading.Event(), threading.Event()

        def stale_vote(perception, planner, frames, cfg, log):
            started.set()
            release.wait(2.0)
            names = planner.scene.bin_names
            return ({n: np.array([0.08, 0.08]) for n in names},
                    {n: 0.0 for n in names}, {n: 1.0 for n in names})

        with mock.patch("aaltonomous.baselines.classical.policy.vote_bins", side_effect=stale_vote):
            policy.act(obs)
            self.assertTrue(started.wait(1.0))
            old_job = policy._bv_job
            policy.reset()
            release.set()
            self.assertTrue(old_job["done"].wait(2.0))
        self.assertIsNone(old_job["result"])
        self.assertIsNone(policy._bv_job)
        self.assertEqual(policy.planner.bin_shift, {})
        self.assertFalse(policy._bv_done)


def _plan(pl, bin_name="center"):
    from .perception import Detection
    sc = pl.scene
    top = sc.mat_to_pelvis([0.45, 0.15, sc.object_height()])
    return pl.plan(Detection("yellow", top, 0, 0, "depth"), pl.home, pl.post["open"], bin_name=bin_name)


class _FakeKin:
    """Stands in for ArmKinematics.hand_points: fixed hand points (pelvis frame)."""
    def __init__(self, pts):
        self.pts = np.asarray(pts, float)

    def hand_points(self, side, arm_q, hand_q):
        return self.pts


class TestPlanner(unittest.TestCase):
    def setUp(self):
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

    def test_bin_conf_defaults_empty(self):
        self.assertEqual(self.pl.bin_conf, {})

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

    def test_bin_clearance_uses_shifted_box(self):
        pl = self.pl
        b = pl.scene.cfg["bins"]
        c = np.asarray(b["centers"]["left"], float)
        rim = pl.scene.counter_z + float(b["wall_height"])
        # mat point 2 cm past the nominal box's +y edge (margin included): outside nominal, inside once shifted
        half = np.asarray(b["outer_size"], float) / 2 + 0.01
        m = np.array([c[0], c[1] + half[1] + 0.02, float(b["wall_height"]) + 0.02])   # mat z
        pl.kin = _FakeKin([pl.scene.mat_to_pelvis(m)])
        self.assertEqual(pl.bin_clearance("left", None, None), np.inf)
        pl.bin_shift = {"left": (0.03, 0.0)}                # pelvis x = mat y: the box centre moves +0.03 in mat y
        np.testing.assert_allclose(pl._mat_centre("left", c), c + [0.0, 0.03])
        self.assertAlmostEqual(pl.bin_clearance("left", None, None), 0.02, places=9)

    def test_hits_bin_wall_uses_shifted_wall(self):
        pl = self.pl
        b = pl.scene.cfg["bins"]
        c = np.asarray(b["centers"]["left"], float)
        rim = pl.scene.counter_z + float(b["wall_height"])
        outer = np.asarray(b["outer_size"], float) / 2
        m = np.array([c[0], c[1] + outer[1] + 0.02, float(b["wall_height"]) - 0.02])   # below rim, outside nominal wall
        pl.kin = _FakeKin([pl.scene.mat_to_pelvis(m)])
        self.assertFalse(pl.hits_bin_wall("left", None, None))
        pl.bin_shift = {"left": (0.03, 0.0)}
        self.assertTrue(pl.hits_bin_wall("left", None, None))

    def test_flag_off_by_default(self):
        self.assertFalse(load_config(overlay="none")["bin"]["vote"]["enabled"])


if __name__ == "__main__":
    unittest.main()
