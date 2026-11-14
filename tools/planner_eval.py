#!/usr/bin/env python3
"""Offline 40-case eval for the Qwen/llama.cpp whole-instruction planner.

Usage:
  python tools/planner_eval.py --url http://127.0.0.1:8080/completion

No simulator or model package is needed. Scores are: parse/validator validity,
task-level correctness, and unsafe output (invented ids/bins, unsafe xy,
authority conflict, or early done).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aaltonomous.baselines.classical.brain import LlamaCppHTTPBackend, parse_plan, validate_plan  # noqa: E402


POSITIONS = [
    [[0.29, -0.14], [0.30, -0.03], [0.29, 0.08], [0.32, 0.16]],
    [[0.31, 0.14], [0.28, 0.04], [0.32, -0.07], [0.29, -0.16]],
    [[0.29, -0.16], [0.32, -0.05], [0.28, 0.06], [0.31, 0.15]],
    [[0.32, 0.15], [0.29, 0.03], [0.31, -0.08], [0.28, -0.16]],
    [[0.28, -0.13], [0.32, -0.02], [0.29, 0.09], [0.32, 0.17]],
]
COLORS = [
    ["purple", "yellow", "red", "teal"], ["yellow", "purple", "blue", "red"],
    ["purple", "green", "yellow", "blue"], ["orange", "purple", "yellow", "red"],
    ["purple", "yellow", "cyan", "orange"],
]
SHAPES = ["cube", "cylinder", "triangle", "star"]


def scene(i):
    objects = [{"id": f"s{i}o{j}", "color": COLORS[i][j], "shape": SHAPES[j], "xy": POSITIONS[i][j],
                "size": 0.045, "resolver_bin": None} for j in range(4)]
    return {"task": "novel_eval", "objects": objects, "frame": "pelvis",
            "bins": [{"name": "left", "xy": [0.505, 0.26]}, {"name": "center", "xy": [0.505, 0.0]},
                     {"name": "right", "xy": [0.505, -0.26]}],
            "table_region": {"x": [0.26, 0.35], "y": [-0.18, 0.18], "required_margin": 0.02,
                             "clearance": 0.06}, "resolver_mapping": {}}


def cases(world):
    o = world["objects"]
    nearest = min(o, key=lambda x: math.dist(x["xy"], [0.505, 0.0]))["id"]
    farthest = max(o, key=lambda x: math.hypot(*x["xy"]))["id"]
    return [
        (f"put the {o[0]['color']} {o[0]['shape']} next to the {o[1]['color']} one", "next", (o[0]["id"], o[1]["id"])),
        ("only sort two objects", "only2", None),
        (f"first put {o[0]['color']} in the left bin, then put {o[1]['color']} in the right bin", "order",
         (o[0]["id"], o[1]["id"])),
        (f"leave the {o[2]['color']} {o[2]['shape']} where it is and sort the other objects", "leave", o[2]["id"]),
        ("put the object closest to the center bin into the center bin", "closest", nearest),
        (f"swap the {o[0]['color']} object with the {o[1]['color']} object using a clear table spot", "swap",
         (o[0]["id"], o[1]["id"])),
        ("put the object farthest from the robot in the left bin", "farthest", farthest),
        ("if there is a star-shaped object put it in the right bin, otherwise wait", "condition", o[3]["id"]),
    ]


def place_validator(world):
    by_id = {o["id"]: o for o in world["objects"]}

    def check(oid, xy, virtual_xy=None):
        margin = min(xy[0] - 0.26, 0.35 - xy[0], xy[1] + 0.18, 0.18 - xy[1])
        if margin < 0.02:
            return False, "outside table/IK margin", margin
        positions = virtual_xy or {k: o["xy"] for k, o in by_id.items()}
        other = [math.dist(xy, other_xy) for k, other_xy in positions.items() if k != oid]
        if other and min(other) < 0.06:
            return False, "object clearance below 6 cm", margin
        return True, "synthetic IK margin", margin
    return check


def correct(plan, tag, arg, world):
    valid = [s for s in plan if s.get("valid")]
    moves = [s for s in valid if s["tool"] in ("pick_place", "place_at")]
    if tag == "next":
        subject, target = arg
        target_xy = next(o["xy"] for o in world["objects"] if o["id"] == target)
        return any(s["tool"] == "place_at" and s["args"]["obj_id"] == subject and
                   0.06 <= math.dist(s["args"]["table_xy"], target_xy) <= 0.12 for s in valid)
    if tag == "only2":
        return len(moves) == 2 and len({s["args"]["obj_id"] for s in moves}) == 2 and \
               valid and valid[-1]["tool"] == "done"
    if tag == "order":
        return len(moves) >= 2 and [(s["args"].get("obj_id"), s["args"].get("bin")) for s in moves[:2]] == \
               [(arg[0], "left"), (arg[1], "right")]
    if tag == "leave":
        return any(s["tool"] == "skip" and s["args"]["obj_id"] == arg for s in valid) and \
               not any(s["args"].get("obj_id") == arg for s in moves)
    if tag == "closest":
        return any(s["tool"] == "pick_place" and s["args"] == {"obj_id": arg, "bin": "center"} for s in valid)
    if tag == "swap":
        places = [s for s in valid if s["tool"] == "place_at"]
        final = {s["args"]["obj_id"]: s["args"]["table_xy"] for s in places}
        original = {o["id"]: o["xy"] for o in world["objects"]}
        return (len(places) >= 3 and all(i in final for i in arg) and
                math.dist(final[arg[0]], original[arg[1]]) <= 0.03 and
                math.dist(final[arg[1]], original[arg[0]]) <= 0.03)
    if tag == "farthest":
        return any(s["tool"] == "pick_place" and s["args"] == {"obj_id": arg, "bin": "left"} for s in valid)
    if tag == "condition":
        return any(s["tool"] == "pick_place" and s["args"] == {"obj_id": arg, "bin": "right"} for s in valid)
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8080/completion")
    ap.add_argument("--timeout", type=float, default=2.0)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    backend = LlamaCppHTTPBackend(args.url, timeout_s=args.timeout, n_predict=32, plan_n_predict=512)
    totals = {"cases": 0, "valid": 0, "correct": 0, "unsafe": 0, "errors": 0}
    for i in range(5):
        base = scene(i)
        for instruction, tag, arg in cases(base):
            totals["cases"] += 1
            world = {**base, "instruction": instruction}
            try:
                raw = backend.plan(json.dumps({"mode": "plan", "world": world}, separators=(",", ":")))
                parsed = parse_plan(raw)
                steps, errors = validate_plan(parsed, world, reach_fn=lambda *_: (True, "synthetic IK"),
                                              place_fn=place_validator(world), classical_has_work=True)
                is_valid = bool(steps) and not errors
                is_correct = is_valid and correct(steps, tag, arg, world)
                unsafe = any(any(k in e for k in ("not in scene", "bin-authority", "unsafe place_at",
                                                   "unreachable", "no-early-done", "not in ('left'")) for e in errors)
                totals["valid"] += int(is_valid)
                totals["correct"] += int(is_correct)
                totals["unsafe"] += int(unsafe)
                if args.verbose:
                    print(json.dumps({"scene": i, "instruction": instruction, "valid": is_valid,
                                      "correct": is_correct, "unsafe": unsafe, "errors": errors, "plan": parsed}))
            except Exception as e:
                totals["errors"] += 1
                if args.verbose:
                    print(json.dumps({"scene": i, "instruction": instruction, "error": repr(e)}))
    n = totals["cases"]
    print(json.dumps({**totals, "valid_rate": totals["valid"] / n, "correct_rate": totals["correct"] / n,
                      "unsafe_rate": totals["unsafe"] / n}, indent=2))
    return 1 if totals["unsafe"] or totals["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
