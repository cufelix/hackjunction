"""Closed-set, whole-instruction planner for the optional ``brain.planner`` mode.

The model emits data, never code or robot joints.  Every step is checked here
and motion primitives are dry-run by the classical planner before execution.
HTTP work always runs through :class:`AsyncRequest`; polling never blocks the
control loop.
"""
from __future__ import annotations

import json
import math
import re

from .core import AsyncRequest, BackendWorker, make_backend, world_to_json
from .registry import BINS

PLAN_TOOLS = ("pick_place", "place_at", "find", "skip", "wait", "done", "skill")
SKILLS = ("flip",)   # closed set of motion skills; the policy sets planner.skill from it while planning
MAX_PLAN_STEPS = 32

_COMPLEX = re.compile(
    r"\b(?:first\b.+\bthen|then\b.+\bafter|only\s+(?:(?:sort|move|place|pick)\s+)?(?:one|two|three|\d+)|swap|"
    r"closest\s+to|nearest\s+to|next\s+to|beside|adjacent\s+to|leave\b|"
    r"do\s+not\s+(?:move|touch)|don't\s+(?:move|touch)|if\b|unless\b|otherwise\b|when\b)\b",
    re.I | re.S,
)
_ONLY = re.compile(r"\bonly\s+(?:(?:sort|move|place|pick)\s+)?(one|two|three|\d+)\b", re.I)
_NUM = {"one": 1, "two": 2, "three": 3}
_RELATION = re.compile(r"\b(?:swap|next to|beside|adjacent to)\b", re.I)   # the only reasons place_at may be planned
_LEAVE = re.compile(r"\b(?:leave|don't move|do not move|don't touch|do not touch)\b", re.I)
# Longer instruction text is not planned, and the policy's resolver hooks skip it (reset() runs under a 1.0 s watchdog;
# the resolver and the trigger regexes are super-linear on repeated text).
MAX_INSTRUCTION_CHARS = 2000


def instruction_needs_plan(instruction, resolver_abstains=False):
    """True for whole-instruction abstention or relations/order/counts a bin map cannot express."""
    text = str(instruction or "")
    if len(text) > MAX_INSTRUCTION_CHARS:
        return True
    return bool(resolver_abstains or _COMPLEX.search(text))


def plan_json_schema():
    """Human/tooling-facing schema. Semantic constraints are enforced by ``validate_plan``."""
    return {
        "type": "array", "maxItems": MAX_PLAN_STEPS,
        "items": {
            "type": "object", "additionalProperties": False, "required": ["tool", "args"],
            "properties": {
                "tool": {"enum": list(PLAN_TOOLS)},
                "args": {"type": "object"},
            },
            "if": {"properties": {"tool": {"const": "skill"}}},
            "then": {"properties": {"args": {"type": "object", "additionalProperties": False,
                                             "required": ["obj_id", "skill"],
                                             "properties": {"obj_id": {"type": "string"},
                                                            "skill": {"enum": list(SKILLS)}}}}},
        },
    }


def plan_gbnf():
    """llama.cpp grammar for the six primitives, with exact key order and JSON-only output."""
    return (r'''root ::= ws "[" ws (step (ws "," ws step)*)? ws "]" ws
step ::= pick | place | find | skip | wait | done | skill
pick ::= "{" ws "\"tool\"" ws ":" ws "\"pick_place\"" ws "," ws "\"args\"" ws ":" ws "{" ws "\"obj_id\"" ws ":" ws string ws "," ws "\"bin\"" ws ":" ws bin ws "}" ws "}"
place ::= "{" ws "\"tool\"" ws ":" ws "\"place_at\"" ws "," ws "\"args\"" ws ":" ws "{" ws "\"obj_id\"" ws ":" ws string ws "," ws "\"table_xy\"" ws ":" ws "[" ws number ws "," ws number ws "]" ws "}" ws "}"
find ::= "{" ws "\"tool\"" ws ":" ws "\"find\"" ws "," ws "\"args\"" ws ":" ws "{" ws "\"description\"" ws ":" ws string ws "}" ws "}"
skip ::= "{" ws "\"tool\"" ws ":" ws "\"skip\"" ws "," ws "\"args\"" ws ":" ws "{" ws "\"obj_id\"" ws ":" ws string ws "}" ws "}"
wait ::= "{" ws "\"tool\"" ws ":" ws "\"wait\"" ws "," ws "\"args\"" ws ":" ws "{" ws "}" ws "}"
done ::= "{" ws "\"tool\"" ws ":" ws "\"done\"" ws "," ws "\"args\"" ws ":" ws "{" ws "}" ws "}"
skill ::= "{" ws "\"tool\"" ws ":" ws "\"skill\"" ws "," ws "\"args\"" ws ":" ws "{" ws "\"obj_id\"" ws ":" ws string ws "," ws "\"skill\"" ws ":" ws skillname ws "}" ws "}"
bin ::= "\"left\"" | "\"center\"" | "\"right\""
number ::= "-"? ("0" | [1-9] [0-9]*) ("." [0-9]+)?
string ::= "\"" chars "\""
chars ::= ([^"\\] | "\\" (["\\/bfnrt] | "u" [0-9a-fA-F]{4}))*
ws ::= [ \t\n\r]*
'''
    + "skillname ::= " + " | ".join(f'"\\"{s}\\""' for s in SKILLS) + "\n")


def plan_prompt(req):
    """Concise prompt with diverse relational/count/order examples."""
    scene = world_to_json(req["world"])
    examples = [
        ('put the purple cube next to the yellow one',
         '[{"tool":"place_at","args":{"obj_id":"o0","table_xy":[0.30,0.00]}},{"tool":"done","args":{}}]'),
        ('only sort two objects',
         '[{"tool":"pick_place","args":{"obj_id":"o0","bin":"left"}},{"tool":"pick_place","args":{"obj_id":"o1","bin":"right"}},{"tool":"done","args":{}}]'),
        ('first put red left, then blue right',
         '[{"tool":"pick_place","args":{"obj_id":"o2","bin":"left"}},{"tool":"pick_place","args":{"obj_id":"o3","bin":"right"}},{"tool":"done","args":{}}]'),
        ('leave the cylinder; sort the other two',
         '[{"tool":"skip","args":{"obj_id":"o1"}},{"tool":"pick_place","args":{"obj_id":"o0","bin":"left"}},{"tool":"pick_place","args":{"obj_id":"o2","bin":"center"}},{"tool":"done","args":{}}]'),
        ('put the one closest to the center bin into it',
         '[{"tool":"pick_place","args":{"obj_id":"o2","bin":"center"}},{"tool":"done","args":{}}]'),
        ('find the striped object, but do not move anything yet',
         '[{"tool":"find","args":{"description":"striped object"}},{"tool":"wait","args":{}}]'),
        ('swap the two objects using a clear table spot',
         '[{"tool":"place_at","args":{"obj_id":"o0","table_xy":[0.30,0.15]}},{"tool":"place_at","args":{"obj_id":"o1","table_xy":[0.29,-0.12]}},{"tool":"place_at","args":{"obj_id":"o0","table_xy":[0.30,0.06]}},{"tool":"done","args":{}}]'),
        ('take the pan and pretend you are making pancakes',
         '[{"tool":"skill","args":{"obj_id":"o0","skill":"flip"}},{"tool":"done","args":{}}]'),
    ]
    shots = "\n".join(f"Instruction: {i}\nPlan: {p}" for i, p in examples)
    failure = req.get("failure")
    retry = f"\nPrevious step failed: {json.dumps(failure, separators=(',', ':'))}. Re-plan once." if failure else ""
    if req.get("agent"):
        hist = "\n".join(f"- {h['tool']}({json.dumps(h['args'], separators=(',', ':'))}) -> {h['outcome']}"
                         + (f" ({h['reason']})" if h.get("reason") else "") for h in req.get("history", [])) or "- none"
        retry += ("\nAgent loop: plan ONLY the remaining steps (at most 4). Objects with novel=true were found by find(); "
                  "their bin is yours to choose from the instruction; graspable=false means leave them. If the target "
                  "is not in Scene, call find(description) then wait.\nExecuted so far:\n" + hist)
    return (
        "You are a safe robot task planner. Return ONLY a JSON array. Use ONLY these primitives: "
        "pick_place(obj_id,bin), place_at(obj_id,table_xy), find(description), skip(obj_id), skill(obj_id,skill), wait, done. "
        "Never invent ids/bins, never output joints, poses, code, or extra keys. table_xy is pelvis-frame metres. "
        "Prefer skip/wait when uncertain. done is legal only after the instruction is satisfied.\n"
        + ("This instruction names things you cannot see in Scene: call find(description) first, wait, and pick_place ONLY "
           "ids returned by find (Scene.found); if find returns nothing, pick nothing.\n"
           if req["world"].get("grounded_only") else "") +
        f"Schema: {json.dumps(plan_json_schema(), separators=(',', ':'))}\n{shots}\n"
        f"Scene: {scene}\nInstruction: {req['world'].get('instruction', '')}{retry}\nPlan:"
    )


def parse_plan(raw):
    """Parse a JSON array despite harmless fences/prefixes; never evaluates model text."""
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8", "replace")
    if isinstance(raw, list):
        return raw
    if isinstance(raw, dict) and isinstance(raw.get("plan"), list):
        return raw["plan"]
    if not isinstance(raw, str):
        raise ValueError("plan is not JSON text or a list")
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.I)
    try:
        value = json.loads(text)
    except json.JSONDecodeError:
        value = None
        dec = json.JSONDecoder()
        for i, ch in enumerate(text):
            if ch != "[":
                continue
            try:
                value, _ = dec.raw_decode(text[i:])
                break
            except json.JSONDecodeError:
                pass
    if isinstance(value, dict):
        value = value.get("plan")
    if not isinstance(value, list):
        raise ValueError("model output contains no JSON plan array")
    return value


def _result(ok, reason="", margin=None):
    if isinstance(ok, tuple):
        vals = list(ok) + [None, None]
        return bool(vals[0]), str(vals[1] or ""), vals[2]
    return bool(ok), reason, margin


def _protected_ids(instruction, objects):
    """Objects the instruction says to leave alone ("leave the red ones", "do not move the cube")."""
    low = str(instruction or "").lower()
    if not _LEAVE.search(low):
        return set()
    return {oid for oid, o in objects.items()
            if any(str(o.get(k, "")).lower() in low and str(o.get(k, ""))
                   for k in ("colour", "color", "shape"))}


def _only_limit(instruction):
    m = _ONLY.search(str(instruction or ""))
    if not m:
        return None
    return _NUM.get(m.group(1).lower(), int(m.group(1)) if m.group(1).isdigit() else None)


def explicit_plan_satisfied(instruction, successful_steps, world_or_count):
    """Conservative plan/state proof used by the no-early-done guard.

    This deliberately recognizes only the relational forms that trigger plan
    mode.  An unrecognized condition can still finish when the classical
    planner independently reports that no work remains.
    """
    text = str(instruction or "")
    world = world_or_count if isinstance(world_or_count, dict) else {}
    objects = {o.get("id"): o for o in world.get("objects", []) if isinstance(o, dict)}
    n_objects = len(objects) if objects else int(world_or_count or 0)
    moves = [s for s in successful_steps if s.get("tool") in ("pick_place", "place_at")]
    moved_ids = [s.get("args", {}).get("obj_id") for s in moves]

    def described(oid, phrase):
        obj = objects.get(oid, {})
        words = [str(obj.get(k, "")).lower() for k in ("colour", "color", "shape")]
        return any(w and re.search(rf"\b{re.escape(w)}\b", phrase) for w in words)

    def step_matches(step, phrase):
        args = step.get("args", {})
        oid = args.get("obj_id")
        named_bin = next((b for b in BINS if re.search(rf"\b{b}\b", phrase)), None)
        return ((not objects or described(oid, phrase)) and
                (named_bin is None or args.get("bin") == named_bin))
    m = _ONLY.search(text)
    if m:
        want = _NUM.get(m.group(1).lower(), int(m.group(1)) if m.group(1).isdigit() else -1)
        return len(moves) == want and len(set(moved_ids)) == want
    low = text.lower()
    if "swap" in low:
        places = [s for s in moves if s["tool"] == "place_at"]
        ids = list(dict.fromkeys(s["args"]["obj_id"] for s in places))
        if len(ids) != 2 or len(places) < 3 or any(i not in objects for i in ids):
            return False
        final = {s["args"]["obj_id"]: s["args"]["table_xy"] for s in places}
        return (all(i in final and len(objects[i].get("xy", [])) == 2 for i in ids) and
                math.dist(final[ids[0]], objects[ids[1]]["xy"]) <= 0.03 and
                math.dist(final[ids[1]], objects[ids[0]]["xy"]) <= 0.03)
    if "next to" in low or "beside" in low or "adjacent to" in low:
        relation = next(r for r in ("next to", "beside", "adjacent to") if r in low)
        subject_phrase, target_phrase = low.split(relation, 1)
        for s in moves:
            if s["tool"] != "place_at" or (objects and not described(s["args"]["obj_id"], subject_phrase)):
                continue
            target = s["args"]["table_xy"]
            oid = s["args"]["obj_id"]
            # 6 cm is the collision floor; 14 cm remains meaningfully adjacent
            # for the synthetic/competition object scale without claiming a
            # far-away table placement satisfied the relation.
            if any(len(o.get("xy", [])) == 2 and described(other, target_phrase) and
                   0.06 <= math.dist(target, o["xy"]) <= 0.14
                   for other, o in objects.items() if other != oid):
                return True
        return False
    if "first" in low and "then" in low:
        first, second = low.split("then", 1)
        return (len(moves) >= 2 and moved_ids[0] != moved_ids[1] and
                step_matches(moves[0], first) and step_matches(moves[1], second))
    if "closest to" in low or "nearest to" in low:
        if not moves or not objects:
            return False
        named = next((b for b in BINS if re.search(rf"\b{b}\b", low)), None)
        bins = world.get("bins", {})
        if isinstance(bins, dict):
            centre = bins.get(named) if named else None
        else:
            centre = next((b.get("xy") for b in bins if isinstance(b, dict) and b.get("name") == named), None)
        if not isinstance(centre, (list, tuple)) or len(centre) != 2:
            return False
        candidates = [(math.dist(o["xy"], centre), oid) for oid, o in objects.items()
                      if isinstance(o.get("xy"), (list, tuple)) and len(o["xy"]) == 2]
        return bool(candidates) and moves[0]["args"].get("obj_id") == min(candidates)[1]
    if "farthest from the robot" in low:
        candidates = [(math.hypot(*o["xy"]), oid) for oid, o in objects.items()
                      if isinstance(o.get("xy"), (list, tuple)) and len(o["xy"]) == 2]
        return bool(moves and candidates) and moves[0]["args"].get("obj_id") == max(candidates)[1]
    if re.search(r"\b(?:if|unless|otherwise)\b", low):
        mentioned = {oid for oid in objects if described(oid, low)}
        if mentioned:
            return bool(moves) and moves[0]["args"].get("obj_id") in mentioned
        return any(s.get("tool") == "wait" for s in successful_steps)
    if re.search(r"\b(?:leave|don't move|do not move|don't touch|do not touch)\b", low):
        protected = _protected_ids(low, objects)
        if protected and protected.intersection(moved_ids):
            return False
        return len(set(moved_ids)) >= max(0, n_objects - 1)
    return False


def forced_skill_plan(env_value, world, handle_ids=(), exclude=()):
    """AALTO_FORCE_SKILL demo override: a one-skill plan for the first graspable handle object (else the first
    graspable novel object). None when the value is not a known skill or nothing is eligible. Not in exclude."""
    skill = str(env_value or "").strip()
    if skill not in SKILLS:
        return None
    objs = {o.get("id"): o for o in (world or {}).get("objects", []) if isinstance(o, dict)}
    graspable = [oid for oid, o in objs.items() if o.get("graspable") is not False and oid not in exclude]
    pick = next((oid for oid in graspable if oid in handle_ids), None) \
        or next((oid for oid in graspable if objs[oid].get("novel")), None)
    if pick is None:
        return None
    return [{"tool": "skill", "args": {"obj_id": pick, "skill": skill}}, {"tool": "done", "args": {}}]


def validate_plan(raw, world, *, reach_fn=None, place_fn=None, find_enabled=True,
                  classical_has_work=True, max_steps=MAX_PLAN_STEPS, prior_steps=()):
    """Return ``(annotated_steps, errors)``. Invalid steps stay in-order and are skipped at execution.
    prior_steps: already executed successful {tool, args} (brain.closed_loop re-plans) - they count for the
    'only N' limit and for the no-early-done proof, exactly as if they were the start of this plan."""
    try:
        plan = parse_plan(raw)
    except Exception as e:
        return [], [str(e)]
    if len(plan) > int(max_steps):
        return [], [f"plan has {len(plan)} steps; limit is {max_steps}"]
    objects = {o.get("id"): o for o in (world or {}).get("objects", []) if isinstance(o, dict)}
    instruction = str((world or {}).get("instruction") or "")
    protected, limit = _protected_ids(instruction, objects), _only_limit(instruction)
    out, errors, valid_prefix = [], [], [dict(p) for p in prior_steps]
    virtual_xy = {oid: list(o.get("xy", [])) for oid, o in objects.items()}
    for i, step in enumerate(plan):
        err = []
        if not isinstance(step, dict) or set(step) != {"tool", "args"}:
            tool, args = None, None
            err.append("step must be exactly {tool,args}")
        else:
            tool, args = step.get("tool"), step.get("args")
        if tool not in PLAN_TOOLS:
            err.append(f"tool {tool!r} is not allowed")
        if not isinstance(args, dict):
            err.append("args must be an object")
            args = {}
        spec = {"pick_place": {"obj_id", "bin"}, "place_at": {"obj_id", "table_xy"},
                "find": {"description"}, "skip": {"obj_id"}, "skill": {"obj_id", "skill"},
                "wait": set(), "done": set()}.get(
                    tool if isinstance(tool, str) else None)
        if spec is not None and set(args) != spec:
            err.append(f"args must be exactly {sorted(spec)}")
        oid = args.get("obj_id")
        if not isinstance(oid, str):           # unhashable / non-string ids: never a dict key
            oid = None
        if tool in ("pick_place", "place_at", "skip", "skill") and oid not in objects:
            err.append(f"obj_id {oid!r} not in scene")
        if tool == "pick_place" and oid in objects:
            bin_name = args.get("bin")
            if bin_name not in BINS:
                err.append(f"bin {bin_name!r} not in {BINS}")
            known = objects[oid].get("resolver_bin")
            if known is not None and bin_name in BINS and bin_name != known:
                err.append(f"bin-authority: {oid} is certainly {known}, not {bin_name}")
            if not err and reach_fn is not None:
                ok, why, _ = _result(reach_fn(oid, bin_name))
                if not ok:
                    err.append(f"unreachable pick_place: {why}")
        if tool == "pick_place" and oid in objects and (world or {}).get("grounded_only"):
            grounded = {g for ids in ((world or {}).get("found") or {}).values() for g in (ids or [])}
            if oid not in grounded:
                err.append(f"{oid} was not located by find(): an open-vocabulary pick must be visually grounded")
        if tool in ("pick_place", "place_at", "skill") and oid in objects and objects[oid].get("graspable") is False:
            err.append(f"{oid} is ungraspable (too wide / flat): left on the counter")
        if tool in ("pick_place", "place_at", "skill") and oid in protected:
            err.append(f"{oid} is protected by the instruction (leave / do not move)")
        if tool in ("pick_place", "place_at") and limit is not None and \
                sum(s["tool"] in ("pick_place", "place_at") for s in valid_prefix) >= limit:
            err.append(f"the instruction allows only {limit} moves")
        if tool == "place_at" and oid in objects and not _RELATION.search(instruction):
            err.append("place_at needs a relation (swap / next to / beside / adjacent to) in the instruction")
        if tool == "place_at" and oid in objects:
            xy = args.get("table_xy")
            good_xy = (isinstance(xy, (list, tuple)) and len(xy) == 2 and
                       all(not isinstance(v, bool) and isinstance(v, (int, float)) and math.isfinite(v) for v in xy))
            if not good_xy:
                err.append("table_xy must be two finite numbers")
            elif objects[oid].get("resolver_bin") is not None:
                err.append(f"bin-authority: {oid} has certain bin {objects[oid]['resolver_bin']}, not the table")
            elif place_fn is not None:
                try:
                    checked = place_fn(oid, [float(xy[0]), float(xy[1])], dict(virtual_xy))
                except TypeError:  # backwards-compatible two-argument validator used by small tools/tests
                    checked = place_fn(oid, [float(xy[0]), float(xy[1])])
                ok, why, margin = _result(checked)
                if not ok:
                    err.append(f"unsafe place_at: {why}")
                elif margin is None or float(margin) < 0.02:
                    err.append("place_at IK/region margin is below 2 cm")
        if tool == "skill" and args.get("skill") not in SKILLS:
            err.append(f"skill {args.get('skill')!r} not in {SKILLS}")
        if tool == "find":
            desc = args.get("description")
            if not find_enabled:
                err.append("find is disabled")
            if not isinstance(desc, str) or not desc.strip() or len(desc) > 80:
                err.append("description must be 1..80 characters")
        if tool == "done":
            if i != len(plan) - 1:
                err.append("done must be the final step")
            if classical_has_work and not explicit_plan_satisfied(
                    (world or {}).get("instruction", ""), valid_prefix, world or len(objects)):
                err.append("no-early-done guard: instruction not proven satisfied")
        annotated = {"tool": tool, "args": args, "valid": not err, "errors": err, "outcome": None}
        out.append(annotated)
        if err:
            errors.extend(f"step {i}: {e}" for e in err)
        elif tool != "done":
            valid_prefix.append({"tool": tool, "args": args})
            if tool == "place_at":
                virtual_xy[oid] = [float(args["table_xy"][0]), float(args["table_xy"][1])]
            elif tool == "pick_place":
                virtual_xy.pop(oid, None)
    return out, errors


class _PlanAdapter:
    def __init__(self, backend):
        self.backend = backend

    def decide(self, request_json):
        return self.backend.plan(request_json)


class PlanLayer:
    """At most one initial plan plus one re-plan. All public polling methods are non-blocking."""

    def __init__(self, cfg, backend=None, log=None, clock=None):
        self.c = dict(cfg or {})
        self.log = log or (lambda *a: None)
        self.backend = backend or make_backend(self.c.get("backend", "stub"), self.c, self.log)
        self.adapter = _PlanAdapter(self.backend)
        self.worker = BackendWorker(self.adapter)
        self.clock = clock
        self.reset()

    def reset(self):
        if hasattr(self, "req"):
            self.req.reset()
        kw = {} if self.clock is None else {"clock": self.clock}
        self.req = AsyncRequest(self.adapter, float(self.c.get("plan_timeout_s", 2.0)), worker=self.worker, **kw)
        self.requested = False
        self.steps, self.index, self.history = None, 0, []
        self.skipped_ids = set()
        self.adopted = False
        self.replans = 0
        self.world = None

    def request(self, world, failure=None):
        if self.requested:
            return False
        timeout = self.c.get("plan_timeout_s", 2.0) if failure is None else self.c.get("step_timeout_s", 0.5)
        kw = {} if self.clock is None else {"clock": self.clock}
        self.req = AsyncRequest(self.adapter, float(timeout), worker=self.worker, **kw)
        payload = {"mode": "plan", "world": world}
        if failure is not None:
            payload["failure"] = failure
        self.req.submit(world_to_json(payload))
        self.requested, self.world = True, world
        return True

    def wait(self, seconds):
        return self.req.wait(seconds)

    def take(self, world, **validators):
        """Adopt a completed plan. None means pending/error/timeout; it never means execute arbitrary text."""
        if self.steps is not None:
            return self.steps
        raw = self.req.poll() if self.requested else None
        if raw is None:
            return None
        self.steps, errors = validate_plan(raw, world, max_steps=self.c.get("max_steps", MAX_PLAN_STEPS), **validators)
        # Once a model plan has been adopted, an empty/invalid retry remains
        # fail-closed rather than silently dropping back to classical guesses.
        self.adopted = self.adopted or bool(self.steps)
        self.index = 0
        shown = [{"tool": s["tool"], "args": s["args"], "valid": s["valid"]} for s in self.steps]
        self.log(f"[planner] plan={json.dumps(shown, separators=(',', ':'))}")
        for e in errors:
            self.log(f"[planner] rejected {e}")
        return self.steps

    def next_call(self):
        if self.steps is None:
            return None
        while self.index < len(self.steps):
            s = self.steps[self.index]
            if s["valid"]:
                return {"tool": s["tool"], "args": dict(s["args"])}
            self.log(f"[planner] step={self.index} outcome=invalid reason={'; '.join(s['errors'])}")
            s["outcome"] = "invalid"
            if isinstance(s.get("args"), dict) and isinstance(s["args"].get("obj_id"), str):
                self.skipped_ids.add(s["args"]["obj_id"])
            self.history.append(s)
            self.index += 1
        return None

    def mark(self, outcome, reason=""):
        if self.steps is None or self.index >= len(self.steps):
            return
        s = self.steps[self.index]
        s["outcome"] = outcome
        self.log(f"[planner] step={self.index} tool={s['tool']} outcome={outcome}" +
                 (f" reason={reason}" if reason else ""))
        self.history.append(s)
        self.index += 1

    def replan(self, world, failed_call, reason):
        if self.replans >= 1:
            return False
        self.replans += 1
        self.steps, self.index, self.requested = None, 0, False
        return self.request(world, {"call": failed_call, "reason": str(reason)[:160]})

    def can_done(self, classical_has_work):
        if not classical_has_work:
            return True
        successful = [s for s in self.history if s.get("valid") and s.get("outcome") == "ok"]
        return explicit_plan_satisfied((self.world or {}).get("instruction", ""), successful,
                                       self.world or 0)
