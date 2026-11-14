"""Tool whitelist, macros, closed-set call generation, decision schema and GBNF. Data only: the model never writes a
call, it picks an INDEX into a list the code generated from the state (no free text, no code)."""
from __future__ import annotations

import copy

BINS = ("left", "center", "right")
VARIANTS = ("same", "closed_tight", "demo")      # retry(obj, variant): grasp posture variant
POINTS = ("after_look", "after_fail", "after_place")

# arg types: "obj_id" (id from the world JSON), "bin" (BINS), "variant" (VARIANTS), "str" (<= 80 chars)
TOOLS = {
    "pick_place": {"args": {"obj_id": "obj_id", "bin": "bin"}, "doc": "pick the object, release it in the bin"},
    "skip": {"args": {"obj_id": "obj_id"}, "doc": "leave the object on the counter (0 points, never -1)"},
    "retry": {"args": {"obj_id": "obj_id", "variant": "variant"}, "doc": "same object again, grasp variant"},
    "look": {"args": {}, "doc": "another head-camera look"},
    "find": {"args": {"query": "query"}, "doc": "locate objects matching a phrase in the head image (VLM grounding); "
                                                "behind brain.find, information only: it moves nothing"},
    "verify": {"args": {"obj_id": "obj_id"}, "doc": "check the placement of the object"},
    "done": {"args": {}, "doc": "nothing left to do"},
}

# Macro = named JSON sequence of existing tools ("$p" = macro parameter). Expanded, never executed as code.
MACROS = {
    "pick_and_verify": {"params": {"obj_id": "obj_id", "bin": "bin"},
                        "steps": [{"tool": "pick_place", "args": {"obj_id": "$obj_id", "bin": "$bin"}},
                                  {"tool": "verify", "args": {"obj_id": "$obj_id"}}]},
}


def arg_spec(name):
    return TOOLS[name]["args"] if name in TOOLS else MACROS[name]["params"]


def callable_names():
    return tuple(TOOLS) + tuple(MACROS)


def expand_macro(call, macros=None):
    """{"tool": macro, "args"} -> list of tool calls (tools pass through as a 1-list)."""
    macros = MACROS if macros is None else macros
    if call.get("tool") in TOOLS or call.get("tool") not in macros:
        return [call]
    a = call.get("args", {})
    return [{"tool": s["tool"], "args": {k: (a.get(v[1:]) if isinstance(v, str) and v.startswith("$") else v)
                                         for k, v in s["args"].items()}}
            for s in copy.deepcopy(macros[call["tool"]]["steps"])]


def allowed_calls(point, state):
    """Closed set of tool calls for a decision point, deterministic order. state keys:
    candidates [obj_id] (after_look), obj_id (after_fail / after_place), can_look (bool, default True),
    queries [phrase] (after_look, only with brain.find: adds find(q) per phrase).
    after_look : pick_place(o, b) for each bin, skip(o) per candidate; look(); done()
    after_fail : retry(o, variant) per variant, skip(o); look(); done()
    after_place: verify(o); look(); done()"""
    if point not in POINTS:
        raise ValueError(f"unknown decision point {point!r}")
    out = []
    if point == "after_look":
        for o in state.get("candidates", []):
            out += [{"tool": "pick_place", "args": {"obj_id": o, "bin": b}} for b in BINS]
            out.append({"tool": "skip", "args": {"obj_id": o}})
        out += [{"tool": "find", "args": {"query": q}} for q in state.get("queries", [])]
    elif point == "after_fail":
        o = state["obj_id"]
        out += [{"tool": "retry", "args": {"obj_id": o, "variant": v}} for v in VARIANTS]
        out.append({"tool": "skip", "args": {"obj_id": o}})
    else:
        out.append({"tool": "verify", "args": {"obj_id": state["obj_id"]}})
    if state.get("can_look", True):
        out.append({"tool": "look", "args": {}})
    if state.get("allow_done", True):     # the policy only allows done when the classical planner would end too
        out.append({"tool": "done", "args": {}})
    return [{"i": i, **c} for i, c in enumerate(out)]


def decision_json_schema(n_choices):
    """The model's whole output: {"choice": index, "confidence": 0..1}."""
    return {"type": "object", "additionalProperties": False, "required": ["choice", "confidence"],
            "properties": {"choice": {"type": "integer", "minimum": 0, "maximum": max(n_choices - 1, 0)},
                           "confidence": {"type": "number", "minimum": 0, "maximum": 1}}}


def decision_gbnf(n_choices):
    """llama.cpp GBNF: the index is an enum of 0..n-1, confidence a 0..1 decimal. Nothing else can be generated."""
    idx = " | ".join(f'"{i}"' for i in range(max(n_choices, 1)))
    return ('root ::= "{" ws "\\"choice\\"" ws ":" ws idx ws "," ws "\\"confidence\\"" ws ":" ws conf ws "}"\n'
            f"idx ::= {idx}\n"
            'conf ::= "0" ("." [0-9]{1,3})? | "1" ("." "0"{1,3})?\n'
            "ws ::= [ \\n]*\n")
