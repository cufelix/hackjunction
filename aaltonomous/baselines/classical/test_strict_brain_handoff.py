"""instruction.strict + brain.planner: the planner sees the objects the strict mapping abstains on, and no planner
output can move an object off its strict bin. Stub-only, mock world, no network:
    PYTHONPATH=. python3 -m unittest aaltonomous.baselines.classical.test_strict_brain_handoff -v
"""
from __future__ import annotations

import json
import sys
import time
import unittest
from pathlib import Path
from unittest import mock

from aaltonomous.baselines.classical.brain import BrainLayer, PlanLayer, PlannerBackend
from aaltonomous.baselines.classical.mock_world import make_mock_env
from aaltonomous.baselines.classical.policy import ClassicalPolicy
from aaltonomous.interface.tasks import BINS, FIXED_MAPPING

INSTR = "Put the red objects in the left bin."       # strict binds red only; blue and yellow abstain
FLAGS = {"instruction.resolve": True, "instruction.strict": True, "brain.enabled": True, "brain.planner": True,
         "brain.debug_wait": True, "brain.prefetch": False}


def pick(oid, b):
    return {"tool": "pick_place", "args": {"obj_id": oid, "bin": b}}


class Planner(PlannerBackend):
    """Planner-only fake: records every world it is shown and answers with steps(world). Brain looks stay classical."""

    def __init__(self, steps=lambda w: []):
        self.worlds, self.steps = [], steps

    def decide(self, request_json):
        return json.dumps({"choice": json.loads(request_json)["classical"], "confidence": 1.0})

    def plan(self, request_json):
        world = json.loads(request_json)["world"]
        self.worlds.append(world)
        return json.dumps(self.steps(world))


def ids(world, color):
    return [o["id"] for o in world["objects"] if o["color"] == color]


def handoff_steps(w):
    """Abstained objects get their own bins; the strict red objects go left (the only legal bin for them)."""
    return ([pick(i, "right") for i in ids(w, "blue")] + [pick(i, "center") for i in ids(w, "yellow")] +
            [pick(i, "left") for i in ids(w, "red")])


def run(backend=None, instr=INSTR, overrides=FLAGS, cap=3000):
    env = make_mock_env(seed=0)                        # scene: yellow, blue, blue, red, red
    lines = []
    pol = ClassicalPolicy(log=lines.append, safety=env.filter.cfg, time_limit=120.0, overrides=dict(overrides))
    if backend is not None:
        pol.plan_brain.adapter.backend = backend      # only the whole-plan request is faked
    obs = env.reset()
    pol.reset({"instruction": instr, "time_limit": 120.0})
    for _ in range(cap):
        obs, info = env.step(pol.act(obs))
        deadline = time.monotonic() + 5.0          # look/choose threads: no timing race, never a hang
        while (getattr(pol, "_job", None) or {}).get("result", True) is None and time.monotonic() < deadline:
            time.sleep(0.0005)
        if info["episode_over"]:
            break
    return pol, lines


class TestStrictBrainHandoff(unittest.TestCase):
    def test_planner_is_asked_about_abstained_objects_and_red_keeps_its_bin(self):
        backend = Planner(handoff_steps)
        pol, _ = run(backend)
        self.assertTrue(pol._planner_needed)
        self.assertTrue(backend.worlds, "the brain planner was never asked")
        world = backend.worlds[0]
        authority = {o["color"]: o["resolver_bin"] for o in world["objects"]}
        self.assertEqual({c for c, b in authority.items() if b is None}, {"blue", "yellow"})
        self.assertEqual({c for c, b in authority.items() if b is not None}, {"red"})
        self.assertTrue(all(o["resolver_bin"] == "left" for o in world["objects"] if o["color"] == "red"))
        executed = {(p.color, p.target_bin) for p in pol.plans}
        self.assertIn(("blue", "right"), executed)         # the abstained objects are acted on by the planner
        self.assertIn(("yellow", "center"), executed)
        self.assertEqual({b for c, b in executed if c == "red"}, {"left"})

    def test_hostile_plan_cannot_move_a_strict_red_object(self):
        def hostile(w):   # every red: first a wrong bin (must be rejected), then its strict bin
            return [step for i in ids(w, "red") for step in (pick(i, "right"), pick(i, "left"))]
        backend = Planner(hostile)
        pol, lines = run(backend)
        self.assertTrue(any("bin-authority" in l and "is certainly left, not right" in l for l in lines), lines[-5:])
        self.assertNotIn("right", {p.target_bin for p in pol.plans if p.color == "red"})
        self.assertIn("left", {p.target_bin for p in pol.plans if p.color == "red"})

    def test_out_of_set_bin_for_an_abstained_object_is_rejected_not_a_crash(self):
        def bad(w):
            return [pick(i, "back") for i in ids(w, "blue")] + [pick(i, "center") for i in ids(w, "yellow")]
        backend = Planner(bad)
        pol, lines = run(backend)
        self.assertTrue(any(f"not in {BINS}" in l for l in lines), lines[-5:])
        self.assertFalse(any("select failed" in l for l in lines), [l for l in lines if "select failed" in l])
        self.assertTrue(all(p.target_bin in BINS for p in pol.plans))
        self.assertNotIn("back", {p.target_bin for p in pol.plans})

    def test_no_abstention_means_the_planner_is_never_asked(self):
        backend = Planner(handoff_steps)
        pol, _ = run(backend, instr="Red goes left, yellow goes right, and blue goes centre.")
        self.assertFalse(pol._planner_needed)
        self.assertEqual(backend.worlds, [])

    def test_everything_off_is_today_and_the_brain_is_never_called(self):
        sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "tools"))
        from equivalence_check import episode_digest
        env = make_mock_env(seed=0)
        pol = ClassicalPolicy(log=None, safety=env.filter.cfg, time_limit=120.0)
        self.assertIsNone(pol.brain)
        self.assertIsNone(pol.plan_brain)
        off = {"instruction.strict": False, "instruction.resolve": False, "brain.enabled": False, "brain.planner": False}
        with mock.patch.object(BrainLayer, "request", side_effect=AssertionError("brain called")), \
                mock.patch.object(PlanLayer, "request", side_effect=AssertionError("planner called")):
            self.assertEqual(episode_digest(0, 120.0), episode_digest(0, 120.0, off))
            pol, _ = run(None, overrides=off)
        self.assertEqual(pol.mapping, dict(FIXED_MAPPING))


if __name__ == "__main__":
    unittest.main()
