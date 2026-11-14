"""grasp.soft_squeeze + held.soft_detector: a grip that works on soft (Dex3 kp 1.5) fingers.

The simulated finger is a first-order servo (time constant kd/kp, velocity-limited, with a stiction dead band) that an
object can stop at a given closing angle. Stiff = kp 100 / kd 2 (the sim default), soft = kp 1.5 / kd 0.2 (real Dex3).
"""
import time
import unittest

import numpy as np

from aaltonomous.interface.safety import JointSpec

from .config import hand_postures, load_config, soft_hand_active, soft_overclose
from .held import HeldDetector
from .planner import ramp

HZ, DT = 30.0, 1.0 / 30.0
STIFF, SOFT = (100.0, 2.0), (1.5, 0.2)       # (kp, kd)
FRICTION = 0.05                               # N m stiction: steady error = FRICTION / kp
SOFT_ON = {"grasp.soft_squeeze": True, "grasp.soft_hand": True, "held.soft_detector": True,
           "grasp.held_detector": "finger_sum"}


def cfg_soft(**extra):
    return load_config(overlay="none", overrides={**SOFT_ON, **extra})


def closing_sign(side):
    return -1.0 if side == "left" else 1.0


def sim_finger(gains, cmds, start, stop=None, q0=None, vmax=12.0):
    """Drive 14 joints through the commanded rows. `stop` (14 or None) = closing-positive limit from `rest` per joint
    (an object in the way): the finger cannot go past it. Returns [(cmd, q)] per tick."""
    kp, kd = gains
    q = np.array(start if q0 is None else q0, float)
    out = []
    for cmd in cmds:
        for _ in range(10):                                   # 10 sub-steps per 30 Hz tick
            err = cmd - q
            v = np.where(np.abs(kp * err) > FRICTION, np.clip(kp * err / kd, -vmax, vmax), 0.0)
            q = q + v * DT / 10
            if stop is not None:
                for j in range(14):
                    if np.isfinite(stop[j]):
                        q[j] = min(q[j], stop[j]) if cmd[j] > stop[j] else max(q[j], stop[j])
        out.append((cmd.copy(), q.copy()))
    return out


class Rig:
    """Config, postures, and the close / lift command streams of one hand."""

    def __init__(self, side="left", gains=SOFT, **cfg_extra):
        self.cfg = cfg_soft(**cfg_extra)
        self.spec = JointSpec()
        self.post = hand_postures(self.cfg, self.spec)
        self.side, self.gains = side, gains
        self.sl = slice(0, 7) if side == "left" else slice(7, 14)
        self.rest = self.post["preshape"].copy()
        base = self.post[self.cfg["grasp"]["close_posture"]]
        self.cls = soft_overclose(self.cfg, self.spec, base, self.rest)
        self.closed = base
        g, h = self.cfg["grasp"], self.cfg["held"]
        self.hold_s = max(self.cfg["motion"]["close_hold_s"], g["soft_lag_s"] + g["soft_settle_s"] + h["soft_read_after_s"])
        self.hv = float(self.cfg["motion"]["hand_max_vel"])

    def close_cmds(self, hold_s=None):
        up = ramp(self.rest, self.cls, self.hv, DT)
        return np.vstack([up, np.repeat(self.cls[None], int(round((hold_s or self.hold_s) / DT)), 0)])

    def object_stop(self, frac):
        """Closing-positive stop: the fingers stall at `frac` of the way from rest to the plain closed posture."""
        stop = np.full(14, np.nan)
        stop[self.sl] = (self.rest + frac * (self.closed - self.rest))[self.sl]
        return stop

    def episode_start_reference(self, det):
        det.start(0.0)
        for k in range(30):
            det.observe(k / 30.0, self.rest, self.rest)

    def detector(self, soft=True):
        det = HeldDetector("finger_sum", 0.045)
        if soft:
            h = self.cfg["held"]
            det.configure_soft(self.rest, h["soft_margin_rad"], h["soft_lift_margin_rad"], h["soft_prior_ratio"],
                               self.cfg["grasp"]["soft_lag_s"] + self.cfg["grasp"]["soft_settle_s"] + h["soft_read_after_s"])
        self.episode_start_reference(det)
        return det

    def calibrated(self):
        det = self.detector()
        trace = sim_finger(self.gains, self.close_cmds(), self.rest)
        for side in ("left", "right"):          # the baseline closes both hands; only self.side is simulated here
            det.soft_calibrate(side, trace[-4:])
        return det

    def close(self, stop=None):
        return sim_finger(self.gains, self.close_cmds(), self.rest, stop)

    def lift(self, after, stop=None, lift_s=0.9):
        cmds = np.repeat(after[-1][0][None], int(round(lift_s / DT)), 0)
        return sim_finger(self.gains, cmds, self.rest, stop, q0=after[-1][1])


class TestSimulatedFinger(unittest.TestCase):
    def test_soft_servo_lags_300_to_450_ms_and_is_much_slower_than_stiff(self):
        def t95(gains):
            step = np.zeros((60, 14)); step[:, 3] = -1.0
            tr = sim_finger(gains, step, np.zeros(14))
            return next(i for i, (c, q) in enumerate(tr) if q[3] <= 0.95 * -1.0 + 1e-9) * DT
        self.assertTrue(0.3 <= t95(SOFT) <= 0.7, t95(SOFT))
        self.assertLess(t95(STIFF), 0.1)


class TestSoftDetector(unittest.TestCase):
    def verdict(self, side, gains, obj_frac, soft=True, margin=None):
        rig = Rig(side, gains)
        det = rig.calibrated() if soft else rig.detector(soft=False)
        tr = rig.close(rig.object_stop(obj_frac) if obj_frac is not None else None)
        cmd, q = tr[-1]
        return det.held(side, cmd, q, margin)[0] if soft else det.held(side, cmd, q)[0]

    def test_empty_close_reads_empty_stiff_and_soft_both_hands(self):
        for side in ("left", "right"):
            for gains in (STIFF, SOFT):
                self.assertFalse(self.verdict(side, gains, None), (side, gains))

    def test_object_close_reads_held_stiff_and_soft_both_hands(self):
        for side in ("left", "right"):
            for gains in (STIFF, SOFT):
                for frac in (0.5, 0.7):
                    self.assertTrue(self.verdict(side, gains, frac), (side, gains, frac))

    def test_a_servo_still_converging_at_the_read_is_learned_by_the_baseline(self):
        rig = Rig("left", SOFT)
        det = rig.detector()
        slow = lambda stop=None: sim_finger(SOFT, rig.close_cmds(), rig.rest, stop, vmax=0.35)  # still moving at the read
        det.soft_calibrate("left", slow()[-4:])
        empty, held = slow()[-1], slow(rig.object_stop(0.6))[-1]
        self.assertGreater(det.held("left", *empty)[1]["raw_finger_sum"], 0.3)     # legacy would call this held
        self.assertFalse(det.held("left", *empty)[0])
        self.assertTrue(det.held("left", *held)[0])

    def test_legacy_finger_sum_misreads_the_soft_empty_close_as_held(self):
        # documents the bug this flag fixes: soft fingers lag, the legacy threshold sees the stall as an object
        self.assertTrue(self.verdict("left", SOFT, None, soft=False))
        self.assertFalse(self.verdict("left", STIFF, None, soft=False))

    def test_lift_check_slip_reads_empty_and_a_kept_object_reads_held(self):
        for side in ("left", "right"):
            for gains in (STIFF, SOFT):
                rig = Rig(side, gains)
                det = rig.calibrated()
                stop = rig.object_stop(0.6)
                close = rig.close(stop)
                self.assertTrue(det.held(side, *close[-1], det.soft["margin"])[0])
                kept = rig.lift(close, stop)
                self.assertTrue(det.held(side, *kept[-1], det.soft["lift_margin"])[0], (side, gains))
                slipped = rig.lift(close, None)              # the object left the hand: the fingers keep closing
                self.assertFalse(det.held(side, *slipped[-1], det.soft["lift_margin"])[0], (side, gains))

    def test_soft_detector_off_is_the_legacy_finger_sum(self):
        det = HeldDetector("finger_sum", 0.045)
        self.assertIsNone(det.soft)
        self.assertIn("finger_sum", det.held("left", np.zeros(14), np.zeros(14))[1])
        self.assertNotIn("excess_index", det.held("left", np.zeros(14), np.zeros(14))[1])

    def test_evaluation_tick_waits_lag_settle_read_after_from_the_hold_start(self):
        rig = Rig()
        det = rig.detector()
        need = int(np.ceil((0.38 + 0.4 + 0.4) * HZ))
        self.assertEqual(det.evaluation_tick(10, 20, HZ, hold_start=15), max(20, 15 + need))
        self.assertEqual(det.evaluation_tick(10, 10 + 200, HZ, hold_start=15), 10 + 200)


class TestOvershootLimits(unittest.TestCase):
    def test_every_posture_both_hands_stays_inside_the_urdf_limits(self):
        cfg, spec = cfg_soft(), JointSpec()
        post = hand_postures(cfg, spec)
        m = cfg["grasp"]["soft_limit_margin_rad"]
        for name in ("closed", "closed_tight", "closed_demo"):
            oc = soft_overclose(cfg, spec, post[name], post["preshape"])
            self.assertTrue(np.all(oc >= spec.hand_lower + m - 1e-9), name)
            self.assertTrue(np.all(oc <= spec.hand_upper - m + 1e-9), name)
            moved = np.abs(oc - post[name])
            self.assertGreater(moved[3:7].min(), 0.0)         # left fingers over-closed
            self.assertGreater(moved[10:14].min(), 0.0)       # right fingers over-closed (mirrored)
            self.assertEqual(moved[0], 0.0)                    # thumb0 never moves
            d = np.sign(post[name] - post["preshape"])
            self.assertTrue(np.all((oc - post[name]) * d >= -1e-12))   # only further along the closing direction

    def test_a_huge_requested_overshoot_is_clipped_to_the_limits(self):
        cfg = cfg_soft()
        cfg["grasp"]["soft_overshoot"] = [2.0] * 7
        spec = JointSpec()
        post = hand_postures(cfg, spec)
        oc = soft_overclose(cfg, spec, post["closed"], post["preshape"])
        m = cfg["grasp"]["soft_limit_margin_rad"]
        self.assertTrue(np.all(oc >= spec.hand_lower + m - 1e-9) and np.all(oc <= spec.hand_upper - m + 1e-9))

    def test_default_overshoot_torque_numbers(self):
        # SOFTGRIP.md: kp_soft x (overshoot + typical stall shortfall 0.2 rad) vs the 1.4 N m effort limit
        ov = np.array(cfg_soft()["grasp"]["soft_overshoot"])[3:7]
        self.assertAlmostEqual(1.5 * (ov[0] + 0.2), 1.05)
        self.assertGreaterEqual(1.5 * (ov[1] + 0.2), 1.4 - 1e-9)

    def test_soft_hand_rule(self):
        self.assertTrue(soft_hand_active(cfg_soft()))
        self.assertFalse(soft_hand_active(load_config(overlay="none", profile="sim")))
        self.assertFalse(soft_hand_active(load_config(overlay="none", profile="sim",
                                                      overrides={"grasp.soft_squeeze": True})))   # sim stiffness
        self.assertTrue(soft_hand_active(load_config(overlay="none", profile="none",
                                                     overrides={"grasp.soft_squeeze": True})))    # profile none = real
        self.assertFalse(soft_hand_active(load_config(overlay="none", profile="none")))           # flag off


class TestPlannerPhases(unittest.TestCase):
    @classmethod
    def plan_for(cls, overrides):
        from aaltonomous.scene.scene import Scene
        from .config import home_arm
        from .perception import Detection
        from .planner import Planner

        cfg = load_config(overlay="none", overrides=overrides)
        scene = Scene.load()
        post, home = hand_postures(cfg), home_arm(cfg)
        layout = scene.cfg["layout"]
        top = scene.mat_to_pelvis([np.mean(layout["region_x"]) + 0.08, np.mean(layout["region_y"]),
                                   scene.object_height()])
        plan = Planner(cfg, scene, post, home).plan(Detection("yellow", top, 0, 0, "depth"), home,
                                                    post["open"].copy())
        assert plan.ok, plan.reason
        return cfg, post, plan

    def test_flags_off_plan_is_byte_identical_and_soft_squeeze_needs_a_soft_hand(self):
        _, _, off = self.plan_for({})
        _, _, not_soft = self.plan_for({"grasp.soft_squeeze": True})          # profile sim, grasp.soft_hand false
        self.assertEqual(off.phases, not_soft.phases)
        np.testing.assert_array_equal(off.arm, not_soft.arm)
        np.testing.assert_array_equal(off.hand, not_soft.hand)
        self.assertNotIn("lift_check", [n for n, _, _ in off.phases])

    def test_soft_plan_overcloses_holds_for_the_lag_and_lifts_slowly_after(self):
        _, post, off = self.plan_for({})
        cfg, post, soft = self.plan_for({"grasp.soft_squeeze": True, "grasp.soft_hand": True})
        names = [n for n, _, _ in soft.phases]
        self.assertEqual(names[names.index("close") + 1:names.index("close") + 3], ["lift_check", "lift"])
        hs, he = soft.close_hold
        g, h = cfg["grasp"], cfg["held"]
        self.assertGreaterEqual((he - hs) / HZ + 1e-6, g["soft_lag_s"] + g["soft_settle_s"])
        self.assertGreaterEqual((he - hs) / HZ + 1e-6, cfg["motion"]["close_hold_s"])
        sl = slice(0, 7) if soft.side == "left" else slice(7, 14)
        oc = soft_overclose(cfg, JointSpec(), post[g["close_posture"]], post["preshape"])
        np.testing.assert_allclose(soft.hand[he - 1][sl], oc[sl], atol=1e-12)
        self.assertGreater(np.abs(oc - post[g["close_posture"]])[sl].max(), 0.1)
        # the arm does not move during the hold; lift_check is slower than the stock lift over the same distance
        np.testing.assert_array_equal(soft.arm[hs], soft.arm[he - 1])
        ph = {n: (s, e) for n, s, e in soft.phases}
        self.assertEqual(ph["close"][1], ph["lift_check"][0])
        off_lift = next((s, e) for n, s, e in off.phases if n == "lift")
        self.assertGreater(ph["lift_check"][1] - ph["lift_check"][0], 0)
        self.assertGreaterEqual((ph["lift_check"][1] - ph["lift_check"][0]) * 1.0, 1)
        self.assertGreater(off_lift[1] - off_lift[0], 0)


class TestPolicyWiring(unittest.TestCase):
    def policy(self):
        from .test_retry_vary import det as mkdet, make
        d = mkdet("red", 0.30, shape="cube")
        pol = make([d], overrides=SOFT_ON)
        return d, pol

    def test_soft_flags_configure_the_detector_and_baseline_start_is_queued_not_blocking(self):
        d, pol = self.policy()
        self.assertIsNotNone(pol.held_detector.soft)
        pol.arm_cmd, pol.hand_cmd = pol.home.copy(), pol.post["open"].copy()
        pol.queue_a = pol.queue_h = None
        t0 = time.perf_counter()
        pol._soft_baseline_start(0.0)
        self.assertLess(time.perf_counter() - t0, 0.05)
        self.assertGreater(len(pol.queue_a), 60)                        # a few seconds of queued ticks
        self.assertEqual(pol.queue_a.shape[0], pol.queue_h.shape[0])
        self.assertIsNotNone(pol._softcal)
        # drive the baseline from the simulated soft finger; afterwards the empty expectation is calibrated
        rig = Rig("left", SOFT)
        q = pol.post["open"].copy()
        pol._held_hand_lo, pol._held_hand_hi = np.full(14, -9.0), np.full(14, 9.0)
        while pol.queue_a is not None and len(pol.queue_a) and pol._softcal is not None:
            pol._pop()
            (cmd, q), = sim_finger(SOFT, pol.hand_cmd[None], pol.post["open"], q0=q)
            pol._soft_baseline_step({"hand_q": q})
        self.assertIsNone(pol._softcal)
        self.assertIn("left", pol.held_detector._soft_ratio)
        self.assertIn("right", pol.held_detector._soft_ratio)

    def test_baseline_never_runs_with_the_flag_off(self):
        from .test_retry_vary import det as mkdet, make
        pol = make([mkdet("red", 0.30, shape="cube")])
        self.assertIsNone(pol.held_detector.soft)
        self.assertFalse(pol._soft_det)

    def started(self, last_q_fn):
        """Policy mid-pick with a lift_check read pending; returns (pol, q_close, q_lift)."""
        d, pol = self.policy()
        rig = Rig("left", SOFT)
        det = rig.calibrated()
        pol.held_detector = det
        pol._held_hand_lo, pol._held_hand_hi = np.full(14, -9.0), np.full(14, 9.0)
        pol.planner.go_to = lambda *a, **k: (np.zeros((5, 14)), np.zeros((5, 14)))
        pol._start(pol._plan_pick(d, {}), 0.0)
        stop = rig.object_stop(0.6)
        close = rig.close(stop)
        lift = rig.lift(close, stop if last_q_fn == "kept" else None)
        pol._gdiag["lift_check"] = True
        pol._gdiag["samples"] = {"close": (20, close[-1][1], close[-1][0])}
        pol.hand_cmd = lift[-1][0]
        pol.queue_a, pol.queue_h = np.zeros((30, 14)), np.zeros((30, 14))
        return d, pol, lift[-1][1]

    def test_slip_during_lift_check_aborts_and_goes_to_the_retry_path(self):
        d, pol, q = self.started("slipped")
        logs = []
        pol.log = logs.append
        pol._grasp_diag({"hand_q": q})
        self.assertIsNone(pol._gdiag)
        self.assertEqual(len(pol.queue_a), 0)                  # carry cancelled
        self.assertIsNotNone(pol._empty_recovery)              # open -> vertical lift -> home runs as the rc7 background recovery
        self.assertIs(pol.vary_hist[-1]["held"], False)
        self.assertTrue(any("post-lift sample empty" in m for m in logs))

    def test_object_kept_through_lift_check_continues(self):
        d, pol, q = self.started("kept")
        pol._grasp_diag({"hand_q": q})
        self.assertEqual(len(pol.queue_a), 30)
        self.assertIsNot(pol.vary_hist[-1]["held"], False)

    def test_act_latency_is_unchanged_by_the_flags(self):
        def median_pop_ms(overrides):
            from .test_retry_vary import det as mkdet, make
            pol = make([mkdet("red", 0.30, shape="cube")], overrides=overrides)
            pol.queue_a, pol.queue_h = np.zeros((4000, 14)), np.zeros((4000, 14))
            ts = []
            for _ in range(300):
                t0 = time.perf_counter(); pol._act({"t": 1.0, "hand_q": np.zeros(14), "arm_q": np.zeros(14)})
                ts.append(time.perf_counter() - t0)
            return 1000 * float(np.median(ts))
        off, on = median_pop_ms({}), median_pop_ms(SOFT_ON)
        self.assertLess(on, off + 2.0, (off, on))             # ms: no extra work on the act() path


if __name__ == "__main__":
    unittest.main()
