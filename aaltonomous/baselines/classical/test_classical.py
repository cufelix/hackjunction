"""Offline tests of the classical baseline (no robot). Run:
    ~/miniforge3/envs/aalto/bin/python -m unittest aaltonomous/baselines/classical/test_classical.py -v
"""
import unittest

import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

from .config import hand_postures, home_arm, load_config, mirror_left_to_right
from .calibration_sweep import jittered_camera, run_sweep
from .kinematics import ArmKinematics
from .mock_world import Renderer, make_mock_env
from .perception import Perception
from .planner import Plan, Planner
from .policy import ClassicalPolicy

SC, CAL, CFG = Scene.load(), Calibration.load(), load_config()
# Opt-in features are default OFF (integration branch); their tests switch on exactly their own flag.
CAL_ON = {"perception.self_calibration.enabled": True}
VERIFY_ON = {"policy.verify_placement": True}
CFG_CAL = load_config(overrides=CAL_ON)
CFG_VERIFY = load_config(overrides=VERIFY_ON)


def scene_objects(rng, n=3, shapes=("cube", "cylinder")):
    L = SC.cfg["layout"]
    objs, used = [], []
    colors = list(rng.permutation(["red", "yellow", "blue"]))[:n]
    while len(objs) < n:
        x, y = rng.uniform(*L["region_x"]), rng.uniform(*L["region_y"])
        if any(np.hypot(x - a, y - b) < 0.09 for a, b in used):
            continue
        used.append((x, y))
        h = SC.object_height()
        objs.append(dict(color=colors[len(objs)], shape=shapes[len(objs) % len(shapes)], yaw=rng.uniform(-0.8, 0.8),
                         size=SC.size, height=h, pos=SC.mat_to_pelvis([x, y, h / 2])))
    return objs


class TestPerception(unittest.TestCase):
    def setUp(self):
        self.r = Renderer(SC, CAL)
        self.p = Perception(CFG, SC, CAL)

    def _check(self, rgb, depth, objs, tol):
        dets = self.p.detect(rgb, depth)
        self.assertEqual(sorted(d.color for d in dets), sorted(o["color"] for o in objs))
        for o in objs:
            d = next(d for d in dets if d.color == o["color"])
            top = o["pos"] + [0, 0, o["height"] / 2]
            err = np.linalg.norm(d.top - top)
            self.assertLess(err, tol, f"{o['color']} {o['shape']} err {err * 1000:.1f} mm ({d.method})")
        return dets

    def test_top_centre_within_1cm_with_noise_and_holes(self):
        rng = np.random.default_rng(0)
        for _ in range(5):
            objs = scene_objects(rng)
            rgb, depth = self.r.render(objs, rng, depth_noise_mm=1.5, hole_frac=0.15)
            dets = self._check(rgb, depth, objs, 0.01)
            self.assertTrue(all(d.method == "depth" for d in dets))

    def test_no_depth_falls_back_to_plane(self):
        rng = np.random.default_rng(1)
        objs = scene_objects(rng)
        rgb, _ = self.r.render(objs)
        dets = self._check(rgb, np.zeros(rgb.shape[:2], np.uint16), objs, 0.025)
        self.assertTrue(all(d.method == "plane" for d in dets))

    def test_bins_are_not_objects(self):
        rgb, depth = self.r.render([])
        self.assertEqual(self.p.detect(rgb, depth), [])

    def test_object_in_bin_is_exposed_only_for_verification(self):
        cfg = load_config()
        pl = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
        xy = pl.drop_slots("center")[0]
        lo, _ = SC.bin_inner_box("center")
        ob = dict(color="red", shape="cube", yaw=0.0, size=SC.size, height=SC.object_height(),
                  pos=np.r_[xy, SC.counter_z + lo[2] + SC.object_height() / 2])
        rgb, depth = self.r.render([ob])
        self.assertEqual(self.p.detect(rgb, depth), [])
        dets, rejected = self.p.detect(rgb, depth, debug=True, include_bins=True)
        self.assertEqual(dets, [])
        self.assertEqual([why for _, why in rejected], ["in_bin:center"])


class TestSelfCalibration(unittest.TestCase):
    def test_recovers_known_camera_translation(self):
        rng = np.random.default_rng(7)
        translation = np.array([0.008, -0.006, 0.004])
        actual = jittered_camera(CAL._T("head"), translation, np.zeros(3))
        objects = scene_objects(rng)
        rgb, depth = Renderer(SC, CAL, T_pelvis_optical=actual).render(
            objects, rng, depth_noise_mm=2.0, hole_frac=0.05)
        perception = Perception(CFG_CAL, SC, CAL)
        before = {d.color: d for d in perception.detect(rgb, depth)}
        result = perception.calibrate_bins(depth)
        after = {d.color: d for d in perception.detect(rgb, depth)}

        self.assertTrue(result.success, result)
        np.testing.assert_allclose(result.xy_offset, translation[:2], atol=0.0025)
        before_error = np.mean([np.linalg.norm(before[o["color"]].top[:2] - o["pos"][:2]) for o in objects])
        after_error = np.mean([np.linalg.norm(after[o["color"]].top[:2] - o["pos"][:2]) for o in objects])
        self.assertLess(after_error, 0.004)
        self.assertLess(after_error, 0.5 * before_error)

    def test_failure_keeps_configured_fallback(self):
        cfg = load_config(overrides={"perception.xy_offset": [0.004, -0.003], **CAL_ON})
        perception = Perception(cfg, SC, CAL)
        depth = np.full((480, 640), 500, np.uint16)
        result = perception.calibrate_bins(depth)
        self.assertTrue(result.valid_frame)
        self.assertFalse(result.success)
        np.testing.assert_allclose(perception.xy_offset, [0.004, -0.003])

    def test_random_pose_sweep_reduces_xy_error(self):
        out = run_sweep(trials=12, seed=2026)
        self.assertEqual(out["calibration_failures"], 0, out)
        self.assertEqual(out["detection_failures"], 0, out)
        self.assertLess(out["after"]["mean_mm"], 0.5 * out["before"]["mean_mm"])
        self.assertLess(out["after"]["p95_mm"], 7.0)
        self.assertLess(out["calibration_time_p95_ms"], 500.0)

    def test_policy_retries_invalid_frame_latches_valid_and_resets(self):
        env = make_mock_env(seed=3)
        policy = ClassicalPolicy(config=CFG_CAL, log=None, safety=env.filter.cfg)
        obs = env.reset()
        policy.reset({"instruction": obs["instruction"]})
        invalid = dict(obs, head_depth=np.zeros_like(obs["head_depth"]))

        policy.act(invalid)
        self.assertFalse(policy._calibration_done)
        policy.act(obs)
        self.assertTrue(policy._calibration_done)
        self.assertTrue(policy.calibration.success, policy.calibration)
        accepted = policy.calibration
        policy.act(invalid)
        self.assertIs(policy.calibration, accepted)

        policy.reset({"instruction": obs["instruction"]})
        self.assertFalse(policy._calibration_done)
        self.assertIsNone(policy.calibration)
        np.testing.assert_allclose(policy.perception.xy_offset, policy.perception.base_xy_offset)

    def test_policy_calibration_exception_uses_fallback(self):
        cfg = load_config(overrides={"perception.xy_offset": [0.004, -0.003], **CAL_ON})
        env = make_mock_env(seed=4, cfg=cfg)
        policy = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
        obs = env.reset()
        policy.reset({"instruction": obs["instruction"]})

        def broken_calibration(_depth):
            raise RuntimeError("synthetic calibration failure")

        policy.perception.calibrate_bins = broken_calibration
        action = policy.act(obs)
        self.assertFalse(action["done"])
        self.assertTrue(policy._calibration_done)
        self.assertFalse(policy.calibration.success)
        self.assertIn("synthetic calibration failure", policy.calibration.reason)
        np.testing.assert_allclose(policy.perception.xy_offset, [0.004, -0.003])


class TestShapeDiag(unittest.TestCase):
    """No shape from the 2D blob (plane path abstains) + the passive perception.shape_diag log."""
    FIELDS = {"source", "n_top", "ratio_cloud", "ratio_blob", "path", "label", "conf", "near_bin", "dz_top", "top_xy"}

    def _perc(self, **flags):
        import copy
        cfg = copy.deepcopy(CFG)
        cfg["perception"].update(flags)
        self.lines = []
        return Perception(cfg, SC, CAL, log=self.lines.append)

    def _scene(self, seed=0, depth=True):
        rng = np.random.default_rng(seed)
        objs = scene_objects(rng)
        rgb, d = Renderer(SC, CAL).render(objs, rng, depth_noise_mm=1.0, hole_frac=0.05)
        return rgb, (d if depth else np.zeros(rgb.shape[:2], np.uint16))

    def test_plane_path_gives_no_shape(self):
        rgb, depth = self._scene(1, depth=False)
        p = self._perc(classify_shape=True, triangle_pose=True, shape_diag=True, no_blob_shape=True)
        dets = p.detect(rgb, depth)
        self.assertEqual(len(dets), 3)
        for d in dets:
            self.assertEqual(d.method, "plane")
            self.assertIsNone(d.shape)
            self.assertIsNone(d.apex_yaw)
            self.assertEqual(d.shape_diag["source"], "plane")
            self.assertIn(d.shape_diag["path"], ("abstain_plane", "abstain_border"))
            self.assertIsNotNone(d.shape_diag["ratio_blob"])     # measured for the log, never used
            self.assertIsNone(d.shape_diag["top_xy"])

    def test_diag_has_all_fields_and_one_line_per_detection(self):
        rgb, depth = self._scene(0)
        p = self._perc(classify_shape=True, triangle_pose=True, shape_diag=True)
        dets, rej = p.detect(rgb, depth, debug=True)
        self.assertEqual(len(dets), 3)
        self.assertEqual(len(self.lines), len(dets) + sum(d is not None for d, _ in rej))
        self.assertTrue(all(l.startswith("[shape_diag]") for l in self.lines))
        for d in dets:
            g = d.shape_diag
            self.assertEqual(set(g), self.FIELDS)
            self.assertEqual(g["source"], "depth")
            self.assertEqual(g["label"], d.shape)
            self.assertEqual(len(g["top_xy"]), g["n_top"])
            self.assertIn(g["path"], ("metric", "blob", "abstain_few_points", "abstain_border", "abstain_near_bin"))
            self.assertAlmostEqual(g["dz_top"], d.top[2] - SC.counter_z)
            self.assertIsInstance(g["near_bin"], bool)

    def test_no_blob_flag_off_keeps_blob_fallback(self):
        """perception.no_blob_shape off (integration default): the plane path still labels from the 2D blob."""
        rgb, depth = self._scene(1, depth=False)
        self.assertFalse(CFG["perception"]["no_blob_shape"])
        dets = self._perc(classify_shape=True, shape_diag=True).detect(rgb, depth)
        self.assertEqual(len(dets), 3)
        self.assertTrue(all(d.method == "plane" for d in dets))
        self.assertTrue(any(d.shape is not None for d in dets))
        self.assertTrue(all(d.shape_diag["path"] in ("blob", "abstain_plane", "abstain_border") for d in dets))

    def test_diag_off_by_default_and_passive(self):
        self.assertFalse(CFG["perception"]["shape_diag"])
        rgb, depth = self._scene(0)
        for flags in ({}, {"classify_shape": True, "triangle_pose": True}):
            off = self._perc(**flags).detect(rgb, depth)
            self.assertEqual(self.lines, [])
            self.assertTrue(all(d.shape_diag is None for d in off))
            on = self._perc(shape_diag=True, **flags).detect(rgb, depth)
            self.assertEqual(len(self.lines), len(on))
            self.assertEqual([repr(d) for d in off], [repr(d) for d in on])
            for a, b in zip(off, on):
                self.assertEqual((a.shape, a.apex_yaw, a.apex_conf, a.yaw), (b.shape, b.apex_yaw, b.apex_conf, b.yaw))
                np.testing.assert_array_equal(a.top, b.top)


class TestPostures(unittest.TestCase):
    def test_mirror_and_limits(self):
        post = hand_postures(CFG)  # raises if outside joint_spec limits
        for v in post.values():
            np.testing.assert_allclose(v[7:], mirror_left_to_right(v[:7]))
        kin = ArmKinematics(CFG)
        for name, v in post.items():  # mirrored postures put the right fingertips at the mirrored place
            np.testing.assert_allclose(kin.tips_in_palm("right", v), kin.tips_in_palm("left", v) * [1, -1, 1], atol=1e-6)


class TestIK(unittest.TestCase):
    def test_reaches_points_check_layout_calls_reachable(self):
        """Points over the mat that check_layout's random palm cloud reaches (palm 8 cm above an object centre)
        must be solved by our IK (palm frame, any orientation) within 1 cm. (cloud tolerance 1 cm, not the 3 cm
        check_layout prints with, so 'reachable' means a sampled configuration really got there.)"""
        from aaltonomous.scene.check_layout import palm_cloud, reach_map
        cfg = load_config(overrides={"grasp.point_in_palm": [0.0, 0.0, 0.0]})
        kin = ArmKinematics(cfg)
        home = home_arm(cfg)
        rng = np.random.default_rng(0)
        W, D = SC.cfg["counter"]["mat_size"]
        mat = np.c_[rng.uniform(0, W, 300), rng.uniform(0, D, 300), np.full(300, SC.object_height() / 2)]
        P = SC.mat_to_pelvis(mat)
        ok = reach_map(SC, palm_cloud(40000), P, 0.08, 0.01)
        pts = P[ok][:40] + [0, 0, 0.08]
        self.assertGreater(len(pts), 10)
        hit = 0
        for p in pts:
            side = "left" if p[1] > 0 else "right"
            res = [kin.solve(s, p, home, top_down=False) for s in (side, "right" if side == "left" else "left")]
            hit += min(r[2] for r in res) < 0.01
        self.assertGreaterEqual(hit / len(pts), 0.95, f"{hit}/{len(pts)}")

    def test_top_down_grasp_over_cube_region(self):
        kin, home = ArmKinematics(CFG), home_arm(CFG)
        L = SC.cfg["layout"]
        n = ok = 0
        for x in np.linspace(*L["region_x"], 5):
            for y in np.linspace(*L["region_y"], 3):
                p = SC.mat_to_pelvis([x, y, 0.5 * SC.object_height()])
                n += 1
                ok += any(kin.solve(s, p, home)[1] for s in ("left", "right"))
        self.assertGreaterEqual(ok / n, 0.8, f"top-down grasp IK {ok}/{n} over the cube region")


class TestPlanner(unittest.TestCase):
    def test_plan_respects_speed_and_safety(self):
        from .perception import Detection
        post, home = hand_postures(CFG), home_arm(CFG)
        pl = Planner(CFG, SC, post, home)
        L = SC.cfg["layout"]
        top = SC.mat_to_pelvis([np.mean(L["region_x"]) + 0.08, np.mean(L["region_y"]), SC.object_height()])
        p = pl.plan(Detection("yellow", top, 0, 0, "depth"), home, post["open"])
        self.assertTrue(p.ok, p.reason)
        dq = np.abs(np.diff(np.vstack([home, p.arm]), axis=0)).max()
        self.assertLessEqual(dq, CFG["motion"]["arm_max_vel"] / 30 + 1e-9)
        dh = np.abs(np.diff(np.vstack([post["open"], p.hand]), axis=0)).max()
        self.assertLessEqual(dh, CFG["motion"]["hand_max_vel"] / 30 + 1e-9)
        self.assertEqual(p.safety_clips, 0)
        np.testing.assert_allclose(p.arm[-1], home, atol=1e-9)  # ends at home

    def test_all_drop_slots_are_separated_reachable_and_safe(self):
        from .perception import Detection
        post, home = hand_postures(CFG), home_arm(CFG)
        pl = Planner(CFG, SC, post, home)
        L = SC.cfg["layout"]
        top = SC.mat_to_pelvis([np.mean(L["region_x"]), np.mean(L["region_y"]), SC.object_height()])
        colors = {"left": "red", "center": "yellow", "right": "blue"}
        spacing = CFG["grasp"]["drop_slot_spacing"]
        for name in SC.bin_names:
            slots = pl.drop_slots(name)
            self.assertGreaterEqual(len(slots), 2)
            for i, a in enumerate(slots):
                for b in slots[i + 1:]:
                    self.assertGreaterEqual(np.linalg.norm(a - b), spacing - 1e-9)
                p = pl.plan(Detection(colors[name], top, 1000, 1000, "depth"), home, post["open"],
                            bin_name=name, drop_slot=i, relocate=False)
                self.assertTrue(p.ok, f"{name} slot {i}: {p.reason}")
                self.assertEqual(p.safety_clips, 0)
                np.testing.assert_allclose(p.waypoints["drop"][:2], a, atol=1e-9)


    def test_cross_body_object_is_relocated(self):
        """Relocation fallback: the right arm puts a red (left bin) object down on the counter where the left arm has a
        full plan from home. With the published box row the left arm reaches the whole object band directly, so the
        fallback is called explicitly here."""
        from .perception import Detection
        cfg = load_config(profile="none")
        post, home = hand_postures(cfg), home_arm(cfg)
        pl = Planner(cfg, SC, post, home)
        top = SC.mat_to_pelvis([0.30, 0.18, SC.object_height()])
        det = Detection("red", top, 0, 0, "depth")
        direct = pl.plan(det, home, post["open"])
        self.assertTrue(direct.ok, direct.reason)
        self.assertEqual((direct.side, direct.target_bin), ("left", "left"))
        p = pl.plan_relocation(det, "left", home, post["open"])
        self.assertTrue(p.ok, p.reason)
        self.assertEqual((p.side, p.target_bin), ("right", "counter"))
        self.assertEqual(p.safety_clips, 0)
        spot = p.waypoints["drop"]
        self.assertLess(abs(spot[2] - p.waypoints["grasp"][2]), 0.01)  # put down at grasp height, not dropped
        fin = pl.plan(Detection("red", np.r_[spot[:2], top[2]], 0, 0, "depth"), home, post["open"], relocate=False)
        self.assertTrue(fin.ok, fin.reason)
        self.assertEqual((fin.side, fin.target_bin), ("left", "left"))


class TestEpisode(unittest.TestCase):
    def test_full_mock_episodes(self):
        total_correct = 0
        for seed in (0, 1, 2):
            env = make_mock_env(seed=seed)
            pol = ClassicalPolicy(log=None, safety=env.filter.cfg)
            obs = env.reset()
            pol.reset({"instruction": obs["instruction"]})
            while True:
                obs, info = env.step(pol.act(obs))
                if info["episode_over"]:
                    break
            self.assertIn(info["end_reason"], ("done", "time_limit"))
            self.assertEqual(info["safety_summary"]["clipped_steps"], 0, info["safety_summary"])
            out = env.backend.outcomes(pol.mapping)
            self.assertNotIn("wrong", out)
            self.assertNotIn("off", out)
            total_correct += out.count("correct")
        self.assertGreaterEqual(total_correct, 3)

    def test_confirm_gate_holds_until_confirmed(self):
        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(log=None, require_confirm=True)
        obs = env.reset()
        pol.reset()
        obs, info = env.step(pol.act(obs))
        self.assertTrue(pol.waiting)          # first: confirm the move to home
        self.assertIn("home", pol.pending_text)
        pol.confirm()
        for _ in range(400):
            obs, info = env.step(pol.act(obs))
            if pol.waiting:
                break
        self.assertTrue(pol.waiting)          # then: confirm the first object
        self.assertIsNotNone(pol.pending)
        q = obs["arm_q"].copy()
        k = int(pol.cfg["motion"]["hold_ticks"])
        for _ in range(60 // k):  # 2 s of waiting: arms hold (k-tick hold chunks), no watchdog stop
            a = pol.act(obs)
            self.assertEqual(a["arm_q_target"].shape, (k, 14))
            self.assertEqual(np.ptp(a["arm_q_target"], axis=0).max(), 0.0)
            obs, info = env.step(a)
        self.assertEqual(info["safety_mode"], "ok")
        self.assertLess(np.abs(obs["arm_q"] - q).max(), 1e-3)
        pol.confirm()
        obs, info = env.step(pol.act(obs))
        self.assertFalse(pol.waiting)
        self.assertEqual(pol.state, "executing")

    def test_delayed_confirmation_rechecks_time_budget(self):
        pol = ClassicalPolicy(config=CFG_VERIFY, log=None, require_confirm=True, time_limit=20.0)
        pol.arm_cmd = pol.home.copy()
        pol.hand_cmd = pol.post["open"].copy()
        plan = Plan(True, color="red", target_bin="left", drop_slot=0,
                    waypoints={"grasp": np.zeros(3)}, arm=np.zeros((300, 14)), hand=np.zeros((300, 14)))
        pol.pending, pol.waiting, pol.state = plan, True, "await"
        pol.confirm()
        action = pol.act({"t": 15.0, "arm_q": pol.arm_cmd, "hand_q": pol.hand_cmd})
        self.assertEqual(pol.state, "done")
        self.assertTrue(action["done"])


class TestPlacementVerification(unittest.TestCase):
    def setUp(self):
        self.renderer = Renderer(SC, CAL)
        self.pol = ClassicalPolicy(config=CFG_VERIFY, log=None)
        self.pol.arm_cmd = self.pol.home.copy()
        self.pol.hand_cmd = self.pol.post["open"].copy()

    @staticmethod
    def _object(color, pos):
        return dict(color=color, shape="cube", yaw=0.0, size=SC.size, height=SC.object_height(),
                    pos=np.asarray(pos, float))

    def _obs(self, objects):
        rgb, depth = self.renderer.render(objects)
        return {"head_rgb": rgb, "head_depth": depth}

    def _bin_object(self, color, bin_name, slot):
        xy = self.pol.planner.drop_slots(bin_name)[slot]
        lo, _ = SC.bin_inner_box(bin_name)
        return self._object(color, np.r_[xy, SC.counter_z + lo[2] + SC.object_height() / 2])

    def _pending(self, color, bin_name, slot):
        self.pol.reserved_drop_slots[bin_name].add(slot)
        self.pol.verification_pending = {"color": color, "target_bin": bin_name,
                                         "drop_slot": slot, "corrective": False}

    def test_object_left_on_counter_after_release_is_retried(self):
        self._pending("red", "left", 0)
        pos = SC.mat_to_pelvis([0.40, 0.15, SC.object_height() / 2])
        result = self.pol._look(self._obs([self._object("red", pos)]))
        self.assertEqual(result[0], "plan")
        self.assertIn("red", self.pol.retry_colors)
        self.assertEqual((result[1].target_bin, result[1].drop_slot), ("left", 1))
        self.assertEqual(result[1].safety_clips, 0)

    def test_object_in_wrong_bin_schedules_corrective_pick(self):
        result = self.pol._look(self._obs([self._bin_object("red", "center", 0)]))
        self.assertEqual(len(self.pol.scheduled_corrections), 1)
        self.assertEqual(result[0], "plan")
        plan = result[1]
        self.assertTrue(plan.corrective)
        self.assertEqual((plan.source_bin, plan.target_bin), ("center", "left"))
        self.assertEqual(plan.safety_clips, 0)

    def test_wrong_bin_with_full_target_is_removed_to_counter(self):
        self.pol.used_drop_slots["left"].update(range(3))
        result = self.pol._look(self._obs([self._bin_object("red", "center", 0)]))
        self.assertEqual(result[0], "plan")
        plan = result[1]
        self.assertTrue(plan.corrective)
        self.assertEqual((plan.source_bin, plan.target_bin), ("center", "counter"))
        self.assertEqual(plan.safety_clips, 0)

    def test_unseen_release_gets_one_settled_relook_before_done(self):
        self._pending("red", "left", 0)
        result = self.pol._look(self._obs([]))
        self.assertEqual(result[0], "relook")
        self.assertIn("red", self.pol.retry_colors)
        action = self.pol._after_look(result, 10.0)
        self.assertEqual(self.pol.state, "settle")
        self.assertFalse(action["done"])

    def test_unseen_release_relooks_before_sorting_unrelated_object(self):
        self._pending("red", "left", 0)
        other = self._object("blue", SC.mat_to_pelvis([0.40, 0.15, SC.object_height() / 2]))
        result = self.pol._look(self._obs([other]))
        self.assertEqual(result[0], "relook")
        self.assertFalse(any(p.color == "blue" for p in self.pol.plans))

    def test_second_object_uses_a_different_verified_slot(self):
        self._pending("red", "left", 0)
        placed = self._bin_object("red", "left", 0)
        waiting = self._object("red", SC.mat_to_pelvis([0.40, 0.15, SC.object_height() / 2]))
        result = self.pol._look(self._obs([placed, waiting]))
        self.assertEqual(self.pol.last_verification[0], "ok")
        self.assertEqual(result[0], "plan")
        self.assertEqual(result[1].drop_slot, 1)
        slots = self.pol.planner.drop_slots("left")
        self.assertGreaterEqual(np.linalg.norm(slots[1] - slots[0]), CFG["grasp"]["drop_slot_spacing"] - 1e-9)


if __name__ == "__main__":
    unittest.main()
