"""grasp.pregrasp_relocalise (FLAG, default OFF): closed-loop re-localisation at the pre-grasp pose. Run:
    python -m unittest aaltonomous.baselines.classical.test_pregrasp -v

Mock world, synthetic frames: the first look sees the object displaced by an offset from where it really is (a
localisation error); once the policy has paused at its pre-grasp pose the frames show the true position. The robot
tracks its commands perfectly. Virtual time: sim time advances by the ticks returned, the planner's wall time is not
counted (as in test_six_objects). The hand is a painted rectangle + the FK arm mask patched to the same rectangle.
"""
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import FLAGS, load_config
from .mock_world import Renderer, make_mock_env
from .planner import Plan
from .policy import ClassicalPolicy

SC, CAL = Scene.load(), Calibration.load()
H = SC.object_height()
REND = Renderer(SC, CAL)
HZ = 30.0
ON, OFF = "finals_r22_pregrasp", "finals_r21"
EMPTY = np.zeros((480, 640), bool)
# (colour, mat xy) scenes the policy can plan: yellow -> left arm, red -> right arm, blue -> centre bin
YELLOW, RED, BLUE = ("yellow", (0.45, 0.16)), ("red", (0.45, 0.10)), ("blue", (0.35, 0.12))


def _frame(color, pos, yaw=0.0, shape="cube"):
    if pos is None:
        return REND.render([])
    return REND.render([dict(color=color, shape=shape, yaw=yaw, size=SC.size, height=H, pos=pos)])


def hand_over(color, pos, side, frac):
    """(rgb, depth, mask): the frame of the object with a grey 'hand' covering `frac` of its blob from `side`."""
    rgb, dep = (a.copy() for a in _frame(color, pos))
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    ys, xs = np.nonzero(Perception_masks(hsv)[color] > 0)
    x0, x1, y0, y1 = xs.min(), xs.max(), ys.min(), ys.max()
    w, h = x1 - x0 + 1, y1 - y0 + 1
    m = np.zeros((480, 640), bool)
    if side == "right":
        m[y0 - 30:y1 + 30, int(x1 - frac * w):x1 + 30] = True
    elif side == "left":
        m[y0 - 30:y1 + 30, x0 - 30:int(x0 + frac * w)] = True
    elif side == "top":
        m[y0 - 30:int(y0 + frac * h), x0 - 30:x1 + 30] = True
    elif side == "bottom":
        m[int(y1 - frac * h):y1 + 30, x0 - 30:x1 + 30] = True
    elif side == "all":
        m[y0 - 30:y1 + 30, x0 - 30:x1 + 30] = True
    rgb[m], dep[m] = (60, 60, 60), 450          # a dark hand, closer to the camera than the object
    return rgb, dep, m


def Perception_masks(hsv):
    from .perception import Perception
    cfg = load_config(overlay=ON)
    return Perception(cfg, SC, CAL, log=None)._masks_hsv(hsv)


class Run:
    """One pick at the pre-grasp pose. `off` = first-look estimate minus truth (pelvis xy, m)."""

    def __init__(self, scene=YELLOW, off=(0.0, 0.0), overlay=ON, overrides=None, truth_frame=None, mask=None,
                 gone=False, shape="cube", max_ticks=4000):
        self.color, mat = scene
        self.env = make_mock_env(seed=0)
        self.base = self.env.reset()
        ov = {"policy.safety_shell": False}      # inline planning: deterministic, virtual clock
        ov.update(overrides or {})
        self.pol = ClassicalPolicy(config=load_config(overlay=overlay, overrides=ov), safety=self.env.filter.cfg,
                                   log=None)
        self.pol.reset({"instruction": "", "time_limit": 120})
        self.true_p = SC.mat_to_pelvis([mat[0], mat[1], H / 2])
        first = _frame(self.color, self.true_p + np.array([off[0], off[1], 0.0]), shape=shape)
        truth = truth_frame or _frame(self.color, None if gone else self.true_p, shape=shape)
        self.mask = EMPTY if mask is None else mask
        self.starts, self.t_pause = [], None
        begin = self.pol._start

        def rec(plan, t, **kw):
            self.starts.append((float(t), plan, kw))
            return begin(plan, t, **kw)
        self.pol._start = rec
        self.acts = []
        t = 0.0
        pol = self.pol
        with mock.patch.object(pol.perception, "_fk_arm_mask",
                               side_effect=lambda *a, **k: self.mask if pol._pg is not None else EMPTY):
            for _ in range(max_ticks):
                rgb, dep = truth if pol._pg is not None else first
                if pol._pg is not None and self.t_pause is None:
                    self.t_pause = t
                a_q = self.base["arm_q"] if pol.arm_cmd is None else pol.arm_cmd
                h_q = self.base["hand_q"] if pol.hand_cmd is None else pol.hand_cmd
                res = pol.act(dict(self.base, head_rgb=rgb, head_depth=dep, t=t, arm_q=a_q, hand_q=h_q))
                self.acts.append(res)
                t += len(res["arm_q_target"]) / HZ
                if self.done(pol):
                    break
        self.t_end = t

    def done(self, pol):
        if not pol._pg_on():                       # flag off: the stock pick is one start
            return len(self.starts) >= 1 or pol.state == "done"
        return self.final is not None or pol.state == "done" or len(self.starts) >= 8

    @property
    def final(self):
        """(t, plan) of the pick that was actually started after the pre-grasp pause, else None."""
        for t, plan, kw in self.starts:
            if kw.get("pregrasp_done"):
                return t, plan
        return None

    def grasp_err(self):
        return float(np.linalg.norm(self.final[1].waypoints["grasp"][:2] - self.true_p[:2]))

    def notes(self):
        return [m for _, m in self.pol.events if "[pregrasp]" in m]


def oracle_duration(scene):
    """Stock plan (flag off) for the object at its true position: what the pick costs without any localisation error."""
    r = Run(scene, overlay=OFF)
    return r.starts[0][1].duration


class TestCorrection(unittest.TestCase):
    def test_offsets_2_4_6_cm_land_within_1_cm(self):
        for scene in (YELLOW, BLUE):
            for off in ((0.02, 0.0), (0.0, 0.04), (-0.03, 0.03), (0.0, -0.04), (0.0, 0.06), (-0.04, -0.045)):
                with self.subTest(scene=scene[0], off=off):
                    r = Run(scene, off)
                    self.assertIsNotNone(r.final, r.notes())
                    self.assertLess(r.grasp_err(), 0.010)
                    self.assertIn("corrected grasp", r.notes()[0])

    def test_red_right_arm_offsets_land_within_1_cm(self):
        for off in ((0.02, 0.0), (0.0, 0.04), (0.0, 0.06)):
            with self.subTest(off=off):
                r = Run(RED, off)
                self.assertLess(r.grasp_err(), 0.010)

    def test_flag_off_grasps_the_first_look_position(self):
        r = Run(YELLOW, (0.0, 0.04), overlay=OFF)
        self.assertFalse(r.notes())
        err = float(np.linalg.norm(r.starts[0][1].waypoints["grasp"][:2] - r.true_p[:2]))
        self.assertGreater(err, 0.03)        # the 4 cm localisation error the flag removes

    def test_small_shift_keeps_the_plan(self):
        r = Run(YELLOW, (0.004, 0.0))                  # 4 mm < min_shift 8 mm
        self.assertIn("within tolerance", r.notes()[0])
        t0, p0, _ = r.starts[0]
        _, tail = r.final
        np.testing.assert_array_equal(tail.arm, p0.arm[p0.phases[0][2]:])    # the original plan's own tail
        self.assertEqual([n for n, _, _ in tail.phases], [n for n, _, _ in p0.phases[1:]])
        self.assertEqual(tail.close_hold[0], p0.close_hold[0] - p0.phases[0][2])

    def test_every_correction_is_logged_with_dx_dy_dyaw(self):
        r = Run(YELLOW, (0.0, 0.04))
        line = r.notes()[0]
        for key in ("dx=", "dy=", "dyaw=", "shift="):
            self.assertIn(key, line)
        self.assertIn("dy=-", line)                    # first look 4 cm too far left: dy -40 mm back to the truth

    def test_corrected_plan_keeps_arm_bin_and_variant_bookkeeping(self):
        r = Run(YELLOW, (0.0, 0.04))
        p0, p1 = r.starts[0][1], r.final[1]
        self.assertEqual((p0.side, p0.target_bin, p0.drop_slot), (p1.side, p1.target_bin, p1.drop_slot))
        self.assertEqual(r.pol.n_started, 1)           # one pick, not two
        self.assertEqual(len(r.pol.attempts), 1)
        self.assertTrue(p1.ok and p1.safety_clips == 0)


class TestOcclusion(unittest.TestCase):
    def test_partial_occlusion_by_the_hand_still_corrects(self):
        pos = SC.mat_to_pelvis([YELLOW[1][0], YELLOW[1][1], H / 2])
        for side in ("right", "left", "top", "bottom"):
            for frac in (0.3, 0.5):
                for off in ((0.03, 0.0), (0.0, 0.03), (-0.02, -0.02)):
                    with self.subTest(side=side, frac=frac, off=off):
                        rgb, dep, m = hand_over("yellow", pos, side, frac)
                        r = Run(YELLOW, off, truth_frame=(rgb, dep), mask=m)
                        self.assertIsNotNone(r.final, r.notes())
                        self.assertLess(r.grasp_err(), 0.010)
                        self.assertIn("occluded", r.notes()[0])

    def test_object_under_the_hand_is_not_judged(self):
        pos = SC.mat_to_pelvis([YELLOW[1][0], YELLOW[1][1], H / 2])
        rgb, dep, m = hand_over("yellow", pos, "all", 1.0)
        r = Run(YELLOW, (0.0, 0.03), truth_frame=(rgb, dep), mask=m)
        self.assertIn("not observable", r.notes()[0])
        self.assertIsNotNone(r.final)                  # plan as made (no evidence either way), never an abort
        self.assertEqual(len(r.pol.failed), 0)


class TestGone(unittest.TestCase):
    def test_vanished_object_retreats_and_looks_again_never_grasps_the_empty_spot(self):
        r = Run(YELLOW, (0.0, 0.0), gone=True, max_ticks=6000)
        self.assertIsNone(r.final)
        self.assertTrue(all(not kw.get("pregrasp_done") for _, _, kw in r.starts))
        notes = r.notes()
        self.assertIn("target not found", notes[0])
        self.assertIn("retreat + re-look", notes[0])
        self.assertEqual(r.pol.n_started, 0)           # the aborted pre-grasp is not a try
        self.assertEqual(r.pol.attempts, [])

    def test_retreat_goes_home_with_the_open_hand_then_looks(self):
        r = Run(YELLOW, (0.0, 0.0), gone=True, max_ticks=6000)
        pol = r.pol
        sl = slice(0, 7)                               # yellow -> left arm
        np.testing.assert_allclose(pol.arm_cmd[sl], pol.home[sl], atol=1e-6)
        self.assertGreaterEqual(len([1 for _, m in pol.events if "to home" in m or "retreat" in m]), 1)

    def test_repeated_aborts_leave_the_object_on_the_counter(self):
        r = Run(YELLOW, (0.0, 0.0), gone=True, max_ticks=8000)
        self.assertEqual(sum("retreat + re-look" in n for n in r.notes()), 2)      # max_aborts_per_object
        self.assertEqual(len(r.pol.failed), 1)
        self.assertTrue(any("left on the counter" in n for n in r.notes()))
        self.assertEqual(r.pol.state, "done")
        self.assertTrue(all(not kw.get("pregrasp_done") for _, _, kw in r.starts))


class TestTooFar(unittest.TestCase):
    def test_correction_above_max_aborts_to_a_re_look(self):
        r = Run(YELLOW, (0.0, 0.09), overrides={"grasp.pregrasp.gate_m": 0.15}, max_ticks=6000)
        self.assertIn("correction above max", r.notes()[0])
        self.assertIn("retreat + re-look", r.notes()[0])
        self.assertIsNone(r.final)
        self.assertTrue(all(not kw.get("pregrasp_done") for _, _, kw in r.starts))

    def test_outside_the_gate_is_not_found_and_aborts(self):
        r = Run(YELLOW, (0.0, 0.10), max_ticks=6000)
        self.assertIn("target not found", r.notes()[0])
        self.assertIsNone(r.final)

    def test_infeasible_corrected_plan_retreats_unless_the_shift_is_tiny(self):
        bad = Plan(False, reason="mock infeasible")
        with mock.patch.object(ClassicalPolicy, "_pg_plan", return_value=bad):
            far = Run(YELLOW, (0.0, 0.03), max_ticks=6000)
            near = Run(YELLOW, (0.0, 0.010))
        self.assertIn("infeasible", far.notes()[0])
        self.assertIsNone(far.final)
        self.assertIn("grasp as planned", near.notes()[0])
        self.assertIsNotNone(near.final)


class TestFlagsOff(unittest.TestCase):
    def test_flag_registered_default_off_and_overlay_is_r21_plus_it(self):
        self.assertIn("grasp.pregrasp_relocalise", FLAGS)
        self.assertFalse(load_config(overlay="none")["grasp"]["pregrasp_relocalise"])
        self.assertFalse(load_config(overlay=OFF)["grasp"]["pregrasp_relocalise"])
        on, off = load_config(overlay=ON), load_config(overlay=OFF)
        self.assertTrue(on["grasp"]["pregrasp_relocalise"])
        on["grasp"]["pregrasp_relocalise"], on["overlay"] = False, off["overlay"]
        self.assertEqual(on, off)

    def test_off_never_splits_the_plan_and_never_touches_the_feature(self):
        with mock.patch("aaltonomous.baselines.classical.perception.Perception.relocalise",
                        side_effect=AssertionError("relocalise called with the flag off")):
            r = Run(YELLOW, (0.0, 0.04), overlay=OFF)
        self.assertIsNone(r.pol._pg)
        self.assertEqual(len(r.starts), 1)             # nothing but the stock start of the first object
        t, plan, kw = r.starts[0]
        self.assertEqual(kw, {})
        self.assertFalse(hasattr(plan, "pg_ctx") or hasattr(plan, "pg_vary"))
        # the whole plan is queued by the one start: the first chunks are the plan's own first ticks
        first = np.vstack([a["arm_q_target"] for a in r.acts if True])
        self.assertTrue(any(np.array_equal(plan.arm[:10], first[i:i + 10]) for i in range(0, len(first) - 9)))
        self.assertEqual(r.notes(), [])

    def test_episode_actions_are_deterministic_and_unchanged_by_the_new_code_path(self):
        def stream(overrides):
            env = make_mock_env(seed=0, n_objects=2)
            pol = ClassicalPolicy(config=load_config(overlay=OFF, overrides=overrides), safety=env.filter.cfg,
                                  time_limit=120.0, log=None)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"], "time_limit": 120.0})
            out, k = [], 0
            while k < 4000:
                job = getattr(pol, "_job", None)
                if job is not None and job.get("result") is None:
                    time.sleep(0.002)
                    continue
                a = pol.act(obs)
                out.append((a["arm_q_target"].copy(), a["hand_q_target"].copy()))
                obs, info = env.step(a)
                k += 1
                if info["episode_over"]:
                    break
            return out
        with mock.patch("aaltonomous.baselines.classical.perception.Perception.relocalise",
                        side_effect=AssertionError("called")):
            a, b = stream({}), stream({"grasp.pregrasp_relocalise": False, "grasp.pregrasp.gate_m": 0.2})
        self.assertEqual(len(a), len(b))
        for (x1, y1), (x2, y2) in zip(a, b):
            np.testing.assert_array_equal(x1, x2)
            np.testing.assert_array_equal(y1, y2)


class TestTime(unittest.TestCase):
    def test_pause_without_a_correction_is_one_hold(self):
        r = Run(YELLOW, (0.0, 0.0))
        t0, p0, _ = r.starts[0]
        t1, _ = r.final
        pause = t1 - t0 - p0.phases[0][2] / HZ
        self.assertLess(pause, 0.4)
        self.assertLess(pause, 0.8)

    def test_added_episode_time_per_pick_is_below_0_8_s(self):
        """Pick time with the flag (to_pregrasp + pause + corrected plan) minus the stock pick time for the object at its
        true position (no localisation error at all): the pause plus the re-targeting move of the arm."""
        added = []
        for scene, offs in ((YELLOW, ((0.0, 0.0), (0.02, 0.0), (0.0, 0.04), (0.0, -0.04))),
                            (BLUE, ((0.0, 0.0), (0.02, 0.0), (0.0, 0.04), (0.0, -0.04)))):
            ref = oracle_duration(scene)
            for off in offs:
                r = Run(scene, off)
                t0, p0, _ = r.starts[0]
                t1, p1 = r.final
                total = p0.phases[0][2] / HZ + (t1 - t0 - p0.phases[0][2] / HZ) + p1.duration
                added.append(total - ref)
                with self.subTest(scene=scene[0], off=off):
                    self.assertLess(total - ref, 0.8, f"{total - ref:.2f} s")
        self.assertLess(float(np.median(added)), 0.6)


class TestRealtime(unittest.TestCase):
    def test_act_never_blocks_threaded_and_the_pause_ends(self):
        """safety_shell (the finals) = threaded: re-detection and re-planning run in a thread, act() only holds."""
        env = make_mock_env(seed=0)
        base = env.reset()
        cfg = load_config(overlay=ON)
        self.assertTrue(cfg["policy"]["safety_shell"])
        pol = ClassicalPolicy(config=cfg, safety=env.filter.cfg, log=None)
        pol.reset({"instruction": "", "time_limit": 120})
        true_p = SC.mat_to_pelvis([YELLOW[1][0], YELLOW[1][1], H / 2])
        first, truth = _frame("yellow", true_p + np.array([0.0, 0.04, 0.0])), _frame("yellow", true_p)
        lat, pause_lat, t, in_pause, finished = [], [], 0.0, False, False
        with mock.patch.object(pol.perception, "_fk_arm_mask", return_value=EMPTY):
            for _ in range(20000):
                job = getattr(pol, "_job", None)
                pgj = None if pol._pg is None else pol._pg["job"]
                if (job is not None and job.get("result") is None) or (pgj is not None and pgj["result"] is None):
                    time.sleep(0.002)                  # look / re-localisation thread busy: virtual clock paused
                    continue
                rgb, dep = truth if pol._pg is not None else first
                a_q = base["arm_q"] if pol.arm_cmd is None else pol.arm_cmd
                h_q = base["hand_q"] if pol.hand_cmd is None else pol.hand_cmd
                paused = pol._pg is not None
                t0 = time.perf_counter()
                res = pol.act(dict(base, head_rgb=rgb, head_depth=dep, t=t, arm_q=a_q, hand_q=h_q))
                dt = time.perf_counter() - t0
                lat.append(dt)
                if paused:
                    pause_lat.append(dt)
                t += len(res["arm_q_target"]) / HZ
                if pol.n_started and pol._pg is None and any("corrected" in m for _, m in pol.events):
                    finished = True
                    break
        self.assertTrue(finished, [m for _, m in pol.events][-5:])
        self.assertTrue(pause_lat)
        # a pause tick is a hold (no IK, no detection in the control thread): same order as a plain hold / pop
        self.assertLess(float(np.percentile(pause_lat, 95)), 0.020)
        self.assertLess(max(pause_lat), 0.05)
        self.assertLess(float(np.median(pause_lat)), 0.005)
        self.assertLess(float(np.percentile(lat, 95)), 0.020)


class TestRelocalise(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from .perception import Perception
        cls.perc = Perception(load_config(overlay=ON), SC, CAL, log=None)
        cls.pos = SC.mat_to_pelvis([YELLOW[1][0], YELLOW[1][1], H / 2])
        rgb, dep = _frame("yellow", cls.pos)
        cls.prior = cls.perc.detect(rgb, dep)[0]

    def test_found_at_the_true_position(self):
        rgb, dep = _frame("yellow", self.pos + np.array([0.0, 0.03, 0.0]))
        r = self.perc.relocalise(rgb, dep, self.prior)
        self.assertEqual(r["kind"], "found")
        self.assertLess(np.linalg.norm(r["top"][:2] - (self.prior.top[:2] + [0.0, 0.03])), 0.004)
        self.assertFalse(r["occluded"])

    def test_other_colour_or_no_depth_is_never_the_object(self):
        rgb, dep = _frame("blue", self.pos)
        self.assertEqual(self.perc.relocalise(rgb, dep, self.prior)["kind"], "gone")
        rgb, dep = _frame("yellow", self.pos)
        self.assertEqual(self.perc.relocalise(rgb, None, self.prior)["kind"], "unsupported")

    def test_arm_mask_is_applied_even_with_color_gate_fk_mask_off(self):
        rgb, dep, m = hand_over("yellow", self.pos, "right", 0.4)
        with mock.patch.object(self.perc, "_fk_arm_mask", return_value=m) as fk:
            r = self.perc.relocalise(rgb, dep, self.prior, robot_q=(np.zeros(14), np.zeros(14)))
        fk.assert_called_once()
        self.assertEqual(r["kind"], "found")
        self.assertTrue(r["occluded"])


if __name__ == "__main__":
    unittest.main()
