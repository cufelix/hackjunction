"""Whole-instruction planner tests; no model or network required."""
from __future__ import annotations

import json
import time
import unittest
from pathlib import Path

from .brain import PlanLayer, PlannerBackend, instruction_needs_plan, validate_plan


WORLD = {
    "instruction": "put the purple cube next to the yellow one",
    "objects": [
        {"id": "o0", "color": "purple", "shape": "cube", "xy": [0.29, -0.08], "size": 0.045,
         "resolver_bin": None},
        {"id": "o1", "color": "yellow", "shape": "novel", "xy": [0.29, 0.08], "size": 0.045,
         "resolver_bin": "center"},
        {"id": "o2", "color": "red", "shape": "cylinder", "xy": [0.32, 0.16], "size": 0.045,
         "resolver_bin": "left"},
    ],
    "bins": [{"name": "left", "xy": [0.505, 0.26]}, {"name": "center", "xy": [0.505, 0.0]},
             {"name": "right", "xy": [0.505, -0.26]}],
}


def place_ok(_oid, xy):
    margin = min(xy[0] - 0.26, 0.35 - xy[0], xy[1] + 0.18, 0.18 - xy[1])
    return margin >= 0.02, "outside region", margin


class Scripted(PlannerBackend):
    def __init__(self, plan, delay=0.0):
        self.value, self.delay = plan, delay

    def decide(self, _request_json):
        return '{"choice":0,"confidence":1}'

    def plan(self, _request_json):
        if self.delay:
            time.sleep(self.delay)
        return json.dumps(self.value)


class TestValidation(unittest.TestCase):
    def test_flag_off_is_the_default_and_does_not_construct_plan_path(self):
        text = (Path(__file__).with_name("config.yaml")).read_text()
        brain = text.split("\nbrain:\n", 1)[1].split("\nmotion:\n", 1)[0]
        self.assertIn("  planner: false", brain)
        self.assertIn("  enabled: false", brain)

    def test_novel_next_to_becomes_validated_place_at(self):
        raw = [{"tool": "place_at", "args": {"obj_id": "o0", "table_xy": [0.31, 0.0]}},
               {"tool": "done", "args": {}}]
        steps, errors = validate_plan(raw, WORLD, place_fn=place_ok, classical_has_work=True)
        self.assertFalse(errors, errors)
        self.assertEqual([s["tool"] for s in steps], ["place_at", "done"])
        self.assertTrue(all(s["valid"] for s in steps))

    def test_only_two_allows_exactly_two_moves_then_done(self):
        world = {**WORLD, "instruction": "only sort two objects"}
        raw = [{"tool": "pick_place", "args": {"obj_id": "o0", "bin": "right"}},
               {"tool": "pick_place", "args": {"obj_id": "o1", "bin": "center"}},
               {"tool": "done", "args": {}}]
        steps, errors = validate_plan(raw, world, reach_fn=lambda *_: (True, ""), classical_has_work=True)
        self.assertFalse(errors, errors)
        self.assertEqual(sum(s["tool"] == "pick_place" for s in steps), 2)

        duplicate = [raw[0], raw[0], raw[-1]]
        dup_steps, dup_errors = validate_plan(
            duplicate, world, reach_fn=lambda *_: (True, ""), classical_has_work=True)
        self.assertFalse(dup_steps[-1]["valid"])
        self.assertTrue(any("no-early-done" in e for e in dup_errors))

    def test_invalid_id_and_unreachable_xy_are_rejected_not_improvised(self):
        raw = [{"tool": "pick_place", "args": {"obj_id": "invented", "bin": "left"}},
               {"tool": "place_at", "args": {"obj_id": "o0", "table_xy": [0.9, 0.9]}}]
        steps, errors = validate_plan(raw, WORLD, place_fn=place_ok)
        self.assertEqual([s["valid"] for s in steps], [False, False])
        self.assertTrue(any("not in scene" in e for e in errors))
        self.assertTrue(any("unsafe place_at" in e for e in errors))

    def test_bin_authority_guard_holds(self):
        raw = [{"tool": "pick_place", "args": {"obj_id": "o1", "bin": "right"}},
               {"tool": "place_at", "args": {"obj_id": "o1", "table_xy": [0.31, 0.0]}}]
        steps, errors = validate_plan(raw, WORLD, reach_fn=lambda *_: (True, ""))
        self.assertFalse(steps[0]["valid"])
        self.assertFalse(steps[1]["valid"])
        self.assertEqual(sum("bin-authority" in e for e in errors), 2)

    def test_complex_trigger(self):
        for text in ("first red then blue", "only two", "swap them", "put X next to Y", "leave the cube",
                     "if a star exists put it right, otherwise wait"):
            self.assertTrue(instruction_needs_plan(text), text)
        self.assertFalse(instruction_needs_plan("red goes left"))


class TestAsyncFallback(unittest.TestCase):
    def test_timeout_returns_no_plan_so_classical_can_continue(self):
        layer = PlanLayer({"plan_timeout_s": 0.01}, backend=Scripted([], delay=0.08))
        layer.request(WORLD)
        time.sleep(0.02)
        self.assertIsNone(layer.take(WORLD, place_fn=place_ok))
        self.assertEqual(layer.req.status, "timeout")

    def test_stub_plan_is_taken_and_logged(self):
        plan = [{"tool": "place_at", "args": {"obj_id": "o0", "table_xy": [0.31, 0.0]}},
                {"tool": "done", "args": {}}]
        logs = []
        layer = PlanLayer({"plan_timeout_s": 1.0}, backend=Scripted(plan), log=logs.append)
        layer.request(WORLD)
        layer.wait(1.0)
        self.assertIsNotNone(layer.take(WORLD, place_fn=place_ok, classical_has_work=True))
        self.assertEqual(layer.next_call()["tool"], "place_at")
        layer.mark("ok")
        self.assertEqual(layer.next_call()["tool"], "done")
        self.assertTrue(layer.can_done(True))
        self.assertTrue(any(line.startswith("[planner] plan=") for line in logs))

    def test_only_one_replan_and_adoption_stays_fail_closed(self):
        plan = [{"tool": "wait", "args": {}}]
        layer = PlanLayer({"plan_timeout_s": 1.0, "step_timeout_s": 1.0}, backend=Scripted(plan))
        layer.request(WORLD)
        layer.wait(1.0)
        layer.take(WORLD)
        self.assertTrue(layer.adopted)
        self.assertTrue(layer.replan(WORLD, layer.next_call(), "failed"))
        self.assertTrue(layer.adopted)
        self.assertFalse(layer.replan(WORLD, layer.next_call(), "again"))


if __name__ == "__main__":
    unittest.main()
