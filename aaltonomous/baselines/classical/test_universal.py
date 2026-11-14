"""Universal policy (finals_r23_universal): perception U VLM finds, parser-first instruction, brain validated, EV gate.
Mock world + scripted/garbage brain stubs, no network:
    PYTHONPATH=. python -m pytest aaltonomous/baselines/classical/test_universal.py -q
calib.plane.apply_height is forced OFF in every test overlay: timi/stagingfix (init staging_hold deadlock when it is ON) has
not landed in this base yet.
"""
from __future__ import annotations

import json
import time
import unittest

import numpy as np

from aaltonomous.baselines.classical.brain import PlannerBackend
from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.baselines.classical.policy import ClassicalPolicy

OVERLAY = "finals_r23_universal"
# Mock-compat overrides.  The finals overlay is tuned to the real Dex3/camera (pinch_v1 grasp, finger_sum held detector, depth
# plane + bin-vote calibration); the render mock cannot hold such a grasp (every pick aborts as "empty") and its table height is
# the nominal one.  Perception/brain/novel/EV flags stay exactly as in finals_r23_universal.
MOCK_OV = {"calib.plane.apply_height": False, "scene.counter_z_prior": None, "bin.vote.enabled": False,
           "perception.online_pitch_cal": False, "grasp.held_detector": "stall", "grasp.close_posture": "closed",
           "grasp.point_in_palm": [0.099, -0.084, 0.015], "grasp.close_posture_by_shape": False,
           "grasp.posture_apex_gate": False, "hand.closed_tight": [0.0, -0.1, 1.2, -1.0, -0.6, -1.0, -0.6],
           "motion.close_hold_s": 0.4, "hand.mirror_index_middle": False}
OV = dict(MOCK_OV, **{"brain.debug_wait": True, "brain.prefetch": False})
COLORS = {"grey": [95.0, 95.0, 100.0], "green": [40.0, 170.0, 60.0], "black": [20.0, 20.0, 22.0],
          "white": [235.0, 235.0, 235.0]}


class Brain(PlannerBackend):
    """Scripted planner backend. mode: ok (rules by object name) | garbage | empty | notlist | badbin | badid | inject |
    slow | boom | override (tries to move a resolver-certain object to another bin) | extrakeys | huge."""

    def __init__(self, truth, rules=None, mode="ok", delay=0.0, phrase="the objects"):
        self.truth, self.rules, self.mode, self.delay, self.phrase = truth, rules or {}, mode, delay, phrase
        self.reqs = []

    def decide(self, request_json):
        return json.dumps({"choice": json.loads(request_json)["classical"], "confidence": 1.0})

    def name_of(self, o):
        xy = np.asarray(o["xy"], float)
        best = min(self.truth, key=lambda n: np.linalg.norm(self.truth[n][:2] - xy))
        return best if np.linalg.norm(self.truth[best][:2] - xy) < 0.05 else None

    def plan(self, request_json):
        req = json.loads(request_json)
        self.reqs.append(req)
        if self.delay:
            time.sleep(self.delay)
        objs = req["world"]["objects"]
        m = self.mode
        if m == "boom":
            raise RuntimeError("backend exploded")
        if m == "garbage":
            return "<<<>>> rm -rf / ; not json at all \x00\x01"
        if m == "empty":
            return ""
        if m == "notlist":
            return json.dumps({"left": "o0"})
        if m == "huge":
            return "[" + ",".join(['{"tool":"wait","args":{}}'] * 5000) + "]"
        if m == "badbin":
            return json.dumps([{"tool": "pick_place", "args": {"obj_id": o["id"], "bin": "top"}} for o in objs] +
                              [{"tool": "done", "args": {}}])
        if m == "badid":
            return json.dumps([{"tool": "pick_place", "args": {"obj_id": "o99", "bin": "left"}},
                               {"tool": "done", "args": {}}])
        if m == "inject":
            return json.dumps([{"tool": "set_joints", "args": {"q": [0.0] * 14}},
                               {"tool": "pick_place", "args": {"obj_id": objs[0]["id"], "bin": "left", "q": [9.9]}},
                               {"tool": "exec", "args": {"cmd": "rm -rf /"}}])
        if m == "extrakeys":
            return json.dumps([{"tool": "pick_place", "args": {"obj_id": o["id"], "bin": "left"}, "arm_q": [0.0] * 7}
                               for o in objs])
        grounded = bool(req["world"].get("grounded_only"))
        found = req["world"].get("found") or {}
        if m == "hallucinate":        # a blind planner that "sees" everything: picks every object left, no find()
            return json.dumps([{"tool": "pick_place", "args": {"obj_id": o["id"], "bin": "left"}} for o in objs] +
                              [{"tool": "done", "args": {}}])
        if grounded and not found:    # open-vocabulary instruction: locate first, then pick only what find() returned
            return json.dumps([{"tool": "find", "args": {"description": self.phrase}}, {"tool": "wait", "args": {}}])
        ok_ids = {i for ids in found.values() for i in ids} if grounded else None
        steps = []
        for o in objs:
            if ok_ids is not None and o["id"] not in ok_ids:
                continue
            name = self.name_of(o)
            b = self.rules.get(name)
            if m == "override":
                b = {"left": "right", "right": "left", "center": "left"}.get(o.get("resolver_bin"), b)
            if b is not None and o.get("graspable") is not False:
                steps.append({"tool": "pick_place", "args": {"obj_id": o["id"], "bin": b}})
        steps.append({"tool": "done", "args": {}})
        return json.dumps(steps)


SPOTS = [np.array([0.29, 0.15, 0.0]), np.array([0.29, 0.05, 0.0]), np.array([0.29, -0.05, 0.0]),
         np.array([0.29, -0.15, 0.0])]


def build(instr, objs, backend_mode="ok", rules=None, delay=0.0, overrides=None, brain=True, overlay=OVERLAY,
          phrase="the objects"):
    """objs: [(name, colour, shape, size, height)] at the first len(objs) mock spawn positions."""
    env = make_mock_env(seed=0)
    ov = dict(OV)
    ov.update(overrides or {})
    if not brain:
        ov["brain.enabled"] = False
    lines = []
    pol = ClassicalPolicy(log=lines.append, safety=env.filter.cfg, time_limit=120.0, overlay=overlay, overrides=ov)
    obs = env.reset()
    be = env.backend
    for k, v in COLORS.items():
        be.renderer.colors[k] = np.array(v)
    base = SPOTS     # fixed, well separated counter spots (pelvis frame): novel keep-outs must not fire on the layout itself
    truth, scene = {}, []
    for (name, col, shape, size, h), p in zip(objs, base):
        scene.append({"color": col, "shape": shape, "yaw": 0.0, "size": size, "height": h,
                      "pos": np.array([p[0], p[1], 0.01 + h / 2]), "name": name})
        truth[name] = scene[-1]["pos"]
    be.objects = scene
    be._cache = None
    stub = Brain(truth, rules, backend_mode, delay, phrase)
    if pol.plan_brain is not None:
        pol.plan_brain.adapter.backend = stub
        pol.plan_brain.backend = stub
    if pol.brain is not None:
        pol.brain.backend = stub
    pol.reset({"instruction": instr, "time_limit": 120.0})
    return env, pol, obs, lines, scene, stub


def drive(env, pol, obs, cap=4000):
    dts = []
    for _ in range(cap):
        t0 = time.perf_counter()
        act = pol.act(obs)
        dts.append(time.perf_counter() - t0)
        obs, info = env.step(act)
        end = time.monotonic() + 5.0
        while (getattr(pol, "_job", None) or {}).get("result", True) is None and time.monotonic() < end:
            time.sleep(0.0005)
        if info["episode_over"]:
            break
    return dts


def final_bins(env, scene):
    sc = env.scene
    return {o["name"]: sc.bin_of(sc.pelvis_to_mat(o["pos"])) for o in scene}


def project(pol, pos):
    P = pol.perception
    cam, T = P.cal.cams[P.cam], P._T()
    p = (np.asarray(pos, float) - T[:3, 3]) @ T[:3, :3]
    return cam.K[0, 0] * p[0] / p[2] + cam.K[0, 2], cam.K[1, 1] * p[1] / p[2] + cam.K[1, 2]


CUBE = ("cube", "red", "cube", 0.045, 0.045)
BALL = ("ball", "green", "cylinder", 0.06, 0.06)
APPLE = ("apple", "white", "cylinder", 0.06, 0.06)
BLOCK = ("block", "grey", "cube", 0.065, 0.05)
PAN = ("pan", "black", "cylinder", 0.07, 0.03)
BOX = ("box", "white", "cube", 0.06, 0.05)


def finds(*names):
    """VLM finder stub: a point on each named scene object for ANY query (the 'located by find()' set)."""
    def make(pol, scene):
        pts = [project(pol, o["pos"] + [0, 0, o["height"] / 2]) for o in scene if o["name"] in names]
        return lambda rgb, q: [{"point": p, "bbox": None, "label": q} for p in pts]
    return make


def run(instr, objs, rules=None, mode="ok", delay=0.0, finder=None, **kw):
    env, pol, obs, lines, scene, stub = build(instr, objs, mode, rules, delay, **kw)
    if finder is not None:
        pol._brain_find_fn = finder(pol, scene)
    dts = drive(env, pol, obs)
    return final_bins(env, scene), dts, pol, lines, stub


class TestUnknownObjectInstructions(unittest.TestCase):
    """The three unknown-object instruction classes; per-object expected final bin (None = left on the counter)."""

    def check(self, got, expect, must=(), msg=""):
        """Never a wrong bin: every object ends in its expected bin or is left (None).  `must` names must be placed (a
        physical miss of a cluttered mock spot is a missed +2, never a -1; the keep-outs are what protect the neighbours)."""
        for name, b in expect.items():
            self.assertIn(got[name], (b, None), f"WRONG BIN {name}: {got} {msg}")
        for name in must:
            self.assertEqual(got[name], expect[name], f"{name} not placed: {got} {msg}")

    def test_put_the_ball_in_the_left_bin_parser_alone(self):
        # the deterministic parser/synonym match names the only round blob; the brain is never asked for a plan answer
        got, dts, pol, lines, stub = run("put the ball in the left bin", [CUBE, BALL, BLOCK], {"ball": "left"})
        self.check(got, {"ball": "left", "cube": None, "block": None}, must=("ball",))
        self.assertTrue(any("[novel] matched green round -> left" in l for l in lines), lines[:12])

    def test_put_the_ball_left_vlm_confirms(self):
        def finder(pol, scene):
            ball = next(o for o in scene if o["name"] == "ball")
            return lambda rgb, q: [{"point": project(pol, ball["pos"] + [0, 0, 0.04]), "bbox": None, "label": q}]
        got, _, pol, lines, _ = run("put the ball in the left bin", [CUBE, BALL, BLOCK], {"ball": "left"}, finder=finder)
        self.check(got, {"ball": "left", "cube": None, "block": None}, must=("ball",))
        self.assertTrue(any("confirms the parser match" in l for l in lines), lines[:12])

    def test_ball_parser_match_contradicted_by_vlm_leaves_the_object(self):
        # "ball" -> the only round blob is a pan body; the VLM points at nothing near it: 0 beats -1
        def finder(pol, scene):
            return lambda rgb, q: []
        got, _, pol, lines, _ = run("put the ball in the left bin", [CUBE, PAN, BLOCK], {}, finder=finder)
        self.check(got, {"pan": None, "cube": None, "block": None})
        self.assertTrue(any("CONTRADICTS" in l for l in lines), lines[:12])

    def test_sort_the_fruit_right_everything_else_center_via_brain(self):
        # open vocabulary: the plan may only pick what find("fruit") located; the cube / block (the "everything else" part) are
        # not grounded, so they stay on the counter: 0 points, never a -1 guess
        rules = {"ball": "right", "apple": "right", "cube": "center", "block": "center"}
        got, dts, pol, lines, stub = run("sort the fruit to the right, everything else center",
                                         [CUBE, BALL, APPLE, BLOCK], rules, finder=finds("ball", "apple"), phrase="fruit")
        self.assertTrue(stub.reqs, "the parser abstains on 'fruit' -> the brain was asked")
        self.check(got, {"ball": "right", "apple": "right", "cube": None, "block": None}, must=("ball",))

    def test_the_pan_goes_in_the_middle_via_brain(self):
        got, dts, pol, lines, stub = run("the pan goes in the middle", [CUBE, PAN, BOX], {"pan": "center"},
                                         finder=finds("pan"), phrase="pan")
        self.assertTrue(stub.reqs)
        self.check(got, {"pan": "center", "cube": None, "box": None}, must=("pan",))

    def test_known_objects_still_go_by_the_parser_not_the_brain(self):
        # parser certain (red -> left): a brain that tries to send it right is rejected by bin-authority
        got, _, pol, lines, stub = run("put the red cube in the left bin", [CUBE, BALL], {"cube": "right"}, mode="override")
        self.assertIn(got["cube"], ("left", None), got)
        self.assertNotEqual(got["ball"], "left")


class TestGarbageBrainNeverWrongBin(unittest.TestCase):
    """Whatever the brain returns, no object may end in a bin other than the (parser/brain-validated) expected one."""
    MODES = ("garbage", "empty", "notlist", "badbin", "badid", "inject", "extrakeys", "huge", "boom")

    def test_garbage_modes_leave_everything_on_the_counter(self):
        for mode in self.MODES:
            with self.subTest(mode=mode):
                got, dts, pol, lines, stub = run("sort the fruit to the right, everything else center",
                                                 [CUBE, BALL, BLOCK], {"ball": "right"}, mode=mode)
                self.assertTrue(stub.reqs, mode)
                self.assertEqual(got, {"cube": None, "ball": None, "block": None}, (mode, got))

    def test_malicious_json_never_reaches_the_arm(self):
        got, dts, pol, lines, stub = run("the pan goes in the middle", [CUBE, PAN, BOX], {"pan": "center"}, mode="inject")
        self.assertEqual(got, {"cube": None, "pan": None, "box": None})
        self.assertFalse(any("set_joints" in l and '"valid":true' in l for l in lines))

    def test_brain_overriding_a_certain_bin_is_rejected(self):
        got, dts, pol, lines, stub = run("put the red cube in the left bin", [CUBE, BALL], {}, mode="override")
        self.assertIn(got["cube"], ("left", None))

    def test_hallucinating_blind_planner_is_not_grounded_so_nothing_moves(self):
        # "all the fruit" but there is no fruit: a text-only plan picks every object; grounded_picks rejects all of it
        got, dts, pol, lines, stub = run("put all the fruit in the left bin", [CUBE, BALL, BLOCK], {}, mode="hallucinate")
        self.assertTrue(stub.reqs)
        self.assertEqual(got, {"cube": None, "ball": None, "block": None})

    def test_find_returning_nothing_means_nothing_is_picked(self):
        got, dts, pol, lines, stub = run("put all the fruit in the left bin", [CUBE, BALL, BLOCK], {"ball": "left"},
                                         finder=finds(), phrase="fruit")
        self.assertEqual(got, {"cube": None, "ball": None, "block": None})

    def test_ev_gate_leaves_a_low_confidence_brain_bin(self):
        # backend reports p_bin 0.2 for its answer -> EV = p_grasp*(3*0.2-1) < 0 -> leave (0 beats -1)
        env, pol, obs, lines, scene, stub = build("the pan goes in the middle", [CUBE, PAN, BOX], "ok", {"pan": "center"})
        stub.last_plan_conf = 0.2
        pol._brain_find_fn = finds("pan")(pol, scene)
        stub.phrase = "pan"
        drive(env, pol, obs)
        self.assertEqual(final_bins(env, scene), {"cube": None, "pan": None, "box": None})
        self.assertTrue(any("[ev] leave" in l for l in lines), lines[-8:])

    def test_ev_gate_passes_a_confident_brain_bin(self):
        env, pol, obs, lines, scene, stub = build("the pan goes in the middle", [CUBE, PAN, BOX], "ok", {"pan": "center"})
        stub.last_plan_conf = 0.95
        pol._brain_find_fn = finds("pan")(pol, scene)
        stub.phrase = "pan"
        drive(env, pol, obs)
        self.assertEqual(final_bins(env, scene)["pan"], "center")


class TestLatency(unittest.TestCase):
    LIMIT = 0.037

    def test_act_never_waits_on_a_slow_brain(self):
        for mode, delay in (("ok", 3.0), ("garbage", 0.0), ("boom", 0.0)):
            with self.subTest(mode=mode, delay=delay):
                env, pol, obs, lines, scene, stub = build(
                    "sort the fruit to the right, everything else center", [CUBE, BALL, BLOCK], mode,
                    {"ball": "right"}, delay, overrides={"brain.debug_wait": False})
                dts = []
                for _ in range(900):
                    t0 = time.perf_counter()
                    act = pol.act(obs)
                    dts.append(time.perf_counter() - t0)
                    obs, info = env.step(act)
                    if info["episode_over"]:
                        break
                self.assertLess(max(dts), self.LIMIT, (mode, max(dts)))

    def test_act_budget_during_universal_pick(self):
        got, dts, pol, lines, stub = run("put the ball in the left bin", [CUBE, BALL, BLOCK], {"ball": "left"})
        self.assertLess(max(dts), self.LIMIT, max(dts))


class TestKnownTaskNoRegression(unittest.TestCase):
    """finals_r23_universal with the brain UNAVAILABLE (closed loopback port) == finals_r23: same picks, same outcomes."""

    @staticmethod
    def episode(overlay, seed, extra=None):
        env = make_mock_env(seed=seed, n_objects=3 + seed % 3)
        ov = dict(MOCK_OV)
        ov.update(extra or {})
        pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=120.0, overlay=overlay, overrides=ov)
        starts = []
        orig = pol._start
        pol._start = lambda plan, t: (starts.append((plan.color, plan.target_bin, bool(plan.ok))), orig(plan, t))[1]
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"], "time_limit": 120.0})
        dts, n = [], 0
        while n < 6000:
            t0 = time.perf_counter()
            act = pol.act(obs)
            dts.append(time.perf_counter() - t0)
            n += 1
            obs, info = env.step(act)
            end = time.monotonic() + 5.0
            while (getattr(pol, "_job", None) or {}).get("result", True) is None and time.monotonic() < end:
                time.sleep(0.0005)
            if info["episode_over"]:
                break
        return starts, env.backend.outcomes(pol.mapping), max(dts)

    def test_same_picks_and_outcomes_as_finals_r23(self):
        down = {"brain.url": "http://127.0.0.1:9/completion", "brain.request_timeout_s": 0.2, "brain.prefetch": False}
        for seed in range(4):
            with self.subTest(seed=seed):
                a = self.episode("finals_r23", seed)
                b = self.episode(OVERLAY, seed, down)
                self.assertEqual(a[0], b[0], f"picks differ at seed {seed}")
                self.assertEqual(a[1], b[1])
                self.assertLess(b[2], 0.037)
        # sanity: the comparison is not vacuous
        self.assertTrue(a[0] and "correct" in a[1])


if __name__ == "__main__":
    unittest.main()
