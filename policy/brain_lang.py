"""Closed-set tool chooser (offline Qwen3-8B via llama.cpp) layered over the regex resolver.

The model only picks an INDEX into an enumerated list of allowed tool calls plus a confidence;
the resolver (instruction.resolve) stays the fallback and has a veto on pick_place bins.
Standard library only. Dead/absent llama.cpp degrades to the resolver's behaviour.

World shape assumed: {"objects": [{"id": "obj_3", "color": "red", "shape": "cube", ...}, ...]}
(also accepts "colour"/"type"/"name" aliases, and "objects" as a dict keyed by id).
"""
from __future__ import annotations

import json
import math
import re
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from instruction import resolve  # noqa: E402

LLM_OVERRIDE_CONF = 0.95  # EV: guess only above 1/3; overriding the resolver needs far more.

TOOL_RE = re.compile(r"^\s*([A-Za-z_]\w*)\s*\(\s*([^()]*?)\s*\)\s*$")


def parse_tool(option: str) -> tuple[str, list[str]]:
    """'pick_place(obj_3, left)' -> ('pick_place', ['obj_3', 'left']). Raises ValueError if malformed."""
    m = TOOL_RE.match(option) if isinstance(option, str) else None
    if not m:
        raise ValueError(f"malformed tool call: {option!r}")
    inner = m.group(2)
    args = [a.strip() for a in inner.split(",")] if inner else []
    if any(not a for a in args):
        raise ValueError(f"empty argument in tool call: {option!r}")
    return m.group(1), args


# Canonical lookup (same as probe_families.py:11 / probe_structures.py:11). Bare colour outranks bare shape.
def resolve_bin(mapping, color, shape=None):
    for key in ((color, shape), color, shape, "default"):
        if key in mapping:
            return mapping[key]
    return None          # None == abstain


# ---------------------------------------------------------------- prompt / grammar / parsing

SYSTEM_PROMPT = (
    "You control a robot arm that sorts objects into three bins (left, center, right) from a "
    "typed English instruction. You are a closed-set chooser: pick exactly ONE of the numbered "
    "options. Scoring: a correct placement +2, a wrong placement -1, skipping 0. So guessing has "
    "expected value 3p-1 and skipping has 0: only guess when your probability of being right is "
    "above 1/3; otherwise choose a skip option if one is offered. Report your honest confidence "
    "(0..1) that the chosen option is correct. Reply with ONLY "
    '{"index": <int>, "confidence": <float 0..1>} and nothing else.'
)


# Few-shot examples. DEV MATERIAL ONLY: the sentences below are built from the organisers' eight
# public templates (docs/instruction-training-corpus.json) and our own dev suite. Nothing here comes
# from a sealed, blind or held-out set -- the challenge rules forbid looking at those, and a few-shot
# block is exactly where such a leak would hide.
#
# The three examples are chosen to teach the three behaviours we actually need, not to show off:
#   1. a plain colour mapping          -> confident placement
#   2. the instruction says nothing about this object's type -> skip, low confidence, NOT a guess.
#      NOTE the example had to be chosen carefully: "Red goes left and blue goes right" does NOT
#      abstain on yellow, because the resolver completes the third colour by elimination. Naming
#      ONE colour of three is a real abstain; naming two is not.
#   3. shape wording that is not the usual word ("wedges") -> still a confident placement
_FEWSHOT = (
    (
        '{"objects": [{"id": "obj_1", "color": "blue", "shape": "cube"}]}',
        "Put the red objects in the left bin, the blue objects in the center bin "
        "and the yellow objects in the right bin.",
        ["pick_place(obj_1, left)", "pick_place(obj_1, center)", "pick_place(obj_1, right)", "skip(obj_1)"],
        1, 0.99,
    ),
    (
        '{"objects": [{"id": "obj_2", "color": "yellow", "shape": "cylinder"}]}',
        "Put the red objects in the left bin.",
        ["pick_place(obj_2, left)", "pick_place(obj_2, center)", "pick_place(obj_2, right)", "skip(obj_2)"],
        3, 0.20,
    ),
    (
        '{"objects": [{"id": "obj_0", "color": "red", "shape": "triangle"}]}',
        "The wedges go in the right bin, everything else on the left.",
        ["pick_place(obj_0, left)", "pick_place(obj_0, center)", "pick_place(obj_0, right)", "skip(obj_0)"],
        2, 0.95,
    ),
)


def _render_example(world_json, instruction, options, index, conf):
    lines = "\n".join(f"{i}: {o}" for i, o in enumerate(options))
    return (f"World:\n{world_json}\n\nInstruction: {instruction}\n\n"
            f"Allowed tool calls:\n{lines}\n\n"
            f'{{"index": {index}, "confidence": {conf}}}')


def fewshot_block() -> str:
    """The few-shot examples as one prompt block. Dev material only (see _FEWSHOT)."""
    return "\n\n".join(_render_example(*e) for e in _FEWSHOT)


def brain_mapping(sentence: str) -> dict:
    """The mapping the BRAIN must reason over. Always use this, never bare `resolve(sentence)`.

    `instruction.resolve()` never abstains at the sentence level: when it parses nothing it returns
    FIXED_MAPPING (red left, yellow center, blue right). That is a sound prior for the shipped
    policy -- sort_fixed IS that mapping, and a failed parse of a colour-first instruction is
    usually still fixed-mapping-shaped -- so the policy keeps it.

    For the brain it is poison. Combined with the override guard ("the LLM may override only where
    the resolver abstains"), a fallback mapping looks like a CERTAIN resolver answer, so the guard
    protects it and the model is locked out of exactly the open-vocabulary sentences it exists for.
    Measured: on "The object farthest from the robot goes in the right bin.", a 0.99-confidence
    correct LLM answer is refused and the fixed mapping's wrong bin is taken instead -- +2 turned
    into -1.

    Passing fallback={} makes "parsed nothing" visible as abstention, which is what it actually is.
    """
    return resolve(sentence, fallback={})


def _num(v) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)


def geometry(world: dict) -> dict:
    """Return a NEW world with distances and ranks precomputed, so the model reads geometry instead
    of computing it. Never mutates `world`; anything malformed is passed through, never raised on."""
    objs = world.get("objects") if isinstance(world, dict) else None
    if not isinstance(objs, list) or not all(isinstance(o, dict) for o in objs):
        return world
    bins = world.get("bins")
    bins = bins if isinstance(bins, dict) else {}
    out = [dict(o) for o in objs]

    if bins and all(isinstance(b, dict) and _num(b.get("x")) and _num(b.get("y")) for b in bins.values()) \
            and all(_num(o.get("x")) and _num(o.get("y")) for o in out):
        for o in out:
            d = {n: round(math.hypot(o["x"] - b["x"], o["y"] - b["y"]), 3) for n, b in bins.items()}
            o["dist_to_bin"] = d
            o["nearest_bin"] = min(sorted(d), key=d.get)

    # (field, source key, sign): rank 1 = smallest of sign*value; ties break on id.
    for field, key, sign in (("rank_x", "x", 1), ("rank_y", "y", -1), ("size_rank", "size", -1)):
        if out and all(_num(o.get(key)) for o in out):
            order = sorted(range(len(out)), key=lambda i: (sign * out[i][key], str(out[i].get("id", ""))))
            for rank, i in enumerate(order, 1):
                out[i][field] = rank
    return {**world, "objects": out}


def build_prompt(world: dict, instruction: str, options: list[str]) -> tuple[str, str]:
    lines = "\n".join(f"{i}: {o}" for i, o in enumerate(options))
    user = (
        f"Worked examples:\n\n{fewshot_block()}\n\n"
        "Now the real one.\n\n"
        f"World:\n{json.dumps(geometry(world), sort_keys=True)}\n\n"
        f"Instruction: {instruction}\n\n"
        f"Allowed tool calls:\n{lines}\n\n"
        "Rules: a correct placement scores +2, a wrong one -1, skipping 0 -- guess only if "
        "confidence is above 1/3. "
        'Output ONLY {"index": <int>, "confidence": <float 0..1>}.'
    )
    return SYSTEM_PROMPT, user


def gbnf_for(n_options: int) -> str:
    if n_options < 1:
        raise ValueError("need at least one option")
    idx = " | ".join(f'"{i}"' for i in range(n_options))   # enum of exact literals, not [0-9]+
    return (
        'root ::= "{" ws "\\"index\\"" ws ":" ws idx ws "," ws "\\"confidence\\"" ws ":" ws conf ws "}"\n'
        f"idx ::= {idx}\n"
        'conf ::= "0" ( "." [0-9]+ )? | "1" ( "." "0"+ )?\n'
        'ws ::= [ \\t\\n]*\n'
    )


_NUM = r"(-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?)"


def parse_response(text: str, n_options: int) -> tuple[int, float] | None:
    try:
        if not isinstance(text, str):
            return None
        idx = conf = None
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            try:
                d = json.loads(m.group(0))
                if isinstance(d, dict):
                    idx, conf = d.get("index"), d.get("confidence")
            except ValueError:
                pass
        if idx is None or conf is None:   # regex fallback for slightly broken JSON
            mi = re.search(r'"?index"?\s*[:=]\s*"?(-?\d+(?:\.0+)?)', text)
            mc = re.search(r'"?confidence"?\s*[:=]\s*' + _NUM, text)
            if not (mi and mc):
                return None
            idx, conf = mi.group(1), mc.group(1)
        if isinstance(idx, bool) or isinstance(conf, bool):
            return None
        idx_f, conf_f = float(idx), float(conf)
        if not (math.isfinite(idx_f) and math.isfinite(conf_f)) or idx_f != int(idx_f):
            return None
        i = int(idx_f)
        # out-of-range confidence is rejected, not clamped: "95" must not become 1.0
        if not (0 <= i < n_options) or not (0.0 <= conf_f <= 1.0):
            return None
        return i, conf_f
    except Exception:
        return None


# ---------------------------------------------------------------- classical choice

def _objects(world: dict) -> dict:
    objs = (world or {}).get("objects", [])
    if isinstance(objs, dict):
        objs = [{"id": k, **v} for k, v in objs.items()]
    out = {}
    for o in objs:
        if not isinstance(o, dict):
            continue
        oid = o.get("id") or o.get("name")
        color = o.get("color") or o.get("colour")
        shape = o.get("shape") or o.get("type")
        if oid is not None:
            out[str(oid)] = (color, shape)
    return out


def _parsed(options):
    res = []
    for o in options:
        try:
            res.append(parse_tool(o))
        except ValueError:
            res.append(None)
    return res


def classical_choice(options: list[str], world: dict, mapping: dict) -> int | None:
    objs, parsed = _objects(world), _parsed(options)
    # 1st pass: pick_place into the resolver's bin for that object (first matching option wins).
    for i, p in enumerate(parsed):
        if p and p[0] == "pick_place" and len(p[1]) == 2 and p[1][0] in objs:
            if resolve_bin(mapping, *objs[p[1][0]]) == p[1][1]:
                return i
    # 2nd pass: resolver abstains for the object -> skip(obj) if offered.
    for i, p in enumerate(parsed):
        if p and p[0] == "skip" and len(p[1]) == 1 and p[1][0] in objs:
            if resolve_bin(mapping, *objs[p[1][0]]) is None:
                return i
    return None


# ---------------------------------------------------------------- the decision rule

def decide(options, world, mapping, llm=None) -> dict:
    """llm is (index, confidence) or None. See the numbered branches below."""
    classical = classical_choice(options, world, mapping)
    li, conf = llm if llm is not None else (None, 0.0)
    if li is not None and not (isinstance(li, int) and 0 <= li < len(options)):
        li = None                                   # unusable LLM answer == no LLM answer

    if li is not None and li == classical:
        # BRANCH 1 (agree): LLM and resolver pick the same option -> take it.
        return {"index": li, "source": "agree", "reason": "llm agrees with classical"}

    if li is not None:
        # BRANCH 2 (disagree): classical wins unless the LLM is >= 0.95 confident...
        if conf >= LLM_OVERRIDE_CONF:
            # BRANCH 3 (guard): ...and even then it may only override where the resolver
            # abstains for that object's type -- whatever tool it picked.
            ok, why = _guard(options[li], world, mapping, options)
            if ok:
                return {"index": li, "source": "llm",
                        "reason": f"conf {conf:.2f} >= {LLM_OVERRIDE_CONF}; {why}"}
            guard_note = f"llm refused ({why})"
        else:
            guard_note = f"llm conf {conf:.2f} < {LLM_OVERRIDE_CONF}"
        if classical is not None:
            return {"index": classical, "source": "classical",
                    "reason": f"disagree; {guard_note}"}
        # No classical opinion and the LLM was not allowed through -> falls to branch 4.
        return {"index": None, "source": "abstain", "reason": f"no classical choice; {guard_note}"}

    if classical is not None:
        # No usable LLM answer: today's resolver behaviour.
        return {"index": classical, "source": "classical", "reason": "no llm answer"}

    # BRANCH 4 (abstain): neither side has a choice.
    return {"index": None, "source": "abstain", "reason": "no classical choice and no llm answer"}


def _guard(option: str, world: dict, mapping: dict, options=None) -> tuple[bool, str]:
    """May a confident LLM override the classical choice? Only where the resolver abstains.

    Widened on 2026-11-14 on the team lead's call. Until then this covered `pick_place` alone,
    so a 0.99-confidence `skip()` could veto a resolver that was certain and turn +2 into 0.
    That protected nothing -- skipping is never unsafe -- and it contradicted the claim we make
    to judges, that abstentions fell because the resolver understands more, not because it
    gambles less. The rule now reads: whatever tool the model picks, it may only override where
    the resolver has no opinion. The model fills gaps; it never overrules a certain answer.
    """
    try:
        name, args = parse_tool(option)
    except ValueError:
        return False, "unparseable option"
    objs = _objects(world)
    # Which object is under decision: the one this call names, else the one the sibling options
    # name (every option at a decision point concerns the same object).
    oid = next((a for a in args if a in objs), None)
    if oid is None:
        for sib in (options or ()):
            try:
                _, sib_args = parse_tool(sib)
            except ValueError:
                continue
            oid = next((a for a in sib_args if a in objs), None)
            if oid is not None:
                break
    if oid is None:
        # Nothing of ours is at stake (e.g. a bare look()): there is no resolver opinion to protect.
        return True, "no object under decision"
    want = resolve_bin(mapping, *objs[oid])
    if want is None:
        return True, "resolver abstains for this type"
    if name == "pick_place" and len(args) == 2 and args[1] == want:
        return True, "bin matches resolver"
    return False, f"resolver is certain ({want}); only an abstaining resolver may be overridden"


# ---------------------------------------------------------------- backends

def _reply_text(payload):
    """Pull the generated text out of a llama.cpp reply, whichever endpoint answered.

    `/completion` returns {"content": ...}, but llama-server also serves an OpenAI-compatible
    `/v1/...` that returns {"choices": [{"text": ...}]} or {"choices":[{"message":{"content":...}}]}.
    Reading only "content" means a wrong URL yields None for EVERY decision -- which looks exactly
    like "the model abstained on everything" in the results table, instead of failing loudly.
    Accepting all three costs three lines and removes a silent, misreadable failure at the finals.
    """
    if not isinstance(payload, dict):
        return ""
    if isinstance(payload.get("content"), str):
        return payload["content"]
    choices = payload.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        c = choices[0]
        if isinstance(c.get("text"), str):
            return c["text"]
        msg = c.get("message")
        if isinstance(msg, dict) and isinstance(msg.get("content"), str):
            return msg["content"]
    return ""


class StubBackend:
    """No model: returns the classical choice with confidence 1.0 (so decide() reports 'agree')."""

    def choose(self, world, instruction, options):
        try:
            c = classical_choice(options, world, resolve(instruction))
        except Exception:
            return None
        return None if c is None else (c, 1.0)


class LlamaCppHTTPBackend:
    def __init__(self, url="http://127.0.0.1:8080/completion", timeout=20.0, n_predict=48):
        self.url, self.timeout, self.n_predict = url, timeout, n_predict

    def choose(self, world, instruction, options):
        try:
            system, user = build_prompt(world, instruction, options)
            body = json.dumps({
                "prompt": f"{system}\n\n{user}\n",
                "grammar": gbnf_for(len(options)),
                "temperature": 0,
                "n_predict": self.n_predict,
            }).encode()
            req = urllib.request.Request(self.url, data=body, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                content = _reply_text(json.loads(r.read().decode()))
            return parse_response(content, len(options))
        except Exception:
            return None   # dead server / timeout / bad JSON -> caller falls back to the resolver


# ---------------------------------------------------------------- self-check

def _selfcheck():
    world = {"objects": [
        {"id": "obj_1", "color": "red", "shape": "cube"},
        {"id": "obj_2", "color": "blue", "shape": "cylinder"},
        {"id": "obj_3", "color": "yellow", "shape": "triangle"},
    ]}
    opts = ["pick_place(obj_1, left)", "pick_place(obj_1, right)", "pick_place(obj_2, right)",
            "pick_place(obj_2, left)", "pick_place(obj_3, center)", "skip(obj_3)", "look()"]
    assert parse_tool(opts[0]) == ("pick_place", ["obj_1", "left"])
    assert parse_tool("look()") == ("look", [])
    for bad in ("pick_place(obj_1", "", "x y", "f(a,,b)"):
        try:
            parse_tool(bad); raise SystemExit(f"should reject {bad!r}")
        except ValueError:
            pass

    m = {"red": "left", "blue": "right", "yellow": "center"}
    c = classical_choice(opts, world, m)
    assert c == 0, c
    # bare colour outranks bare shape
    assert resolve_bin({"red": "left", "cube": "right"}, "red", "cube") == "left"

    # Stub + decide reproduce classical on the 3-object scene (real resolver, real text)
    text = "put the red ones left, blue right, yellow center"
    mapping = resolve(text)
    d = decide(opts, world, mapping, StubBackend().choose(world, text, opts))
    assert d["index"] == classical_choice(opts, world, mapping) and d["source"] == "agree", d
    assert decide(opts, world, mapping, None)["source"] == "classical"

    # Below 0.95 the classical choice always stands.
    d = decide(opts, world, m, (5, 0.94)); assert d["index"] == 0 and d["source"] == "classical", d

    # At and above 0.95 the LLM may override ONLY where the resolver abstains -- whatever tool it
    # picks. Option 5 is a skip. The resolver is certain about this object, so the skip is refused:
    # before the guard was widened this returned index 5, silently trading a +2 for a 0.
    d = decide(opts, world, m, (5, 0.96)); assert d["index"] == 0 and d["source"] == "classical", d
    d = decide(opts, world, m, (5, 0.95)); assert d["index"] == 0 and d["source"] == "classical", d

    # 0.99 pick_place into a bin contradicting a non-abstaining resolver: REFUSED
    d = decide(opts, world, m, (1, 0.99)); assert d["index"] == 0 and d["source"] == "classical", d
    # same, resolver abstains for red: ACCEPTED (and classical is None -> llm free)
    m2 = {"blue": "right", "yellow": "center"}
    d = decide(opts, world, m2, (1, 0.99)); assert d["index"] == 1 and d["source"] == "llm", d
    # ...and where the resolver genuinely abstains, a confident LLM IS allowed through.
    # m3 places obj_2 (blue) but has no opinion on obj_3 (yellow triangle), so the classical
    # choice is 2 and the LLM's 4 is a real override that the guard must permit.
    m3 = {"blue": "right"}
    assert classical_choice(opts, world, m3) == 2
    d = decide(opts, world, m3, (4, 0.99)); assert d["index"] == 4 and d["source"] == "llm", d
    # the same override below the threshold is refused
    d = decide(opts, world, m3, (4, 0.80)); assert d["index"] == 2 and d["source"] == "classical", d

    # abstain branch
    assert decide(opts, world, m2, None)["index"] == classical_choice(opts, world, m2)
    d = decide(["look()"], world, {}, None); assert d["source"] == "abstain" and d["index"] is None
    # resolver abstains for obj_3 and skip offered -> skip preferred
    assert classical_choice(["pick_place(obj_3, left)", "skip(obj_3)"], world, {"red": "left"}) == 1

    # _reply_text: a wrong endpoint must not look like "the model abstained on everything"
    assert _reply_text({"content": "x"}) == "x"
    assert _reply_text({"choices": [{"text": "y"}]}) == "y"
    assert _reply_text({"choices": [{"message": {"content": "z"}}]}) == "z"
    for junk in (None, [], {}, {"choices": []}, {"choices": [{}]}, {"content": 5}):
        assert _reply_text(junk) == "", junk

    g = gbnf_for(4)
    for k in "0123":
        assert f'"{k}"' in g
    assert '"4"' not in g and "[0-9]+ )? | \"1\"" in g
    assert parse_response('{"index": 7, "confidence": 0.9}', 4) is None
    assert parse_response('{"index": 2, "confidence": 0.9}', 4) == (2, 0.9)
    assert parse_response('noise {"index": 3, "confidence": 1}', 4) == (3, 1.0)
    for junk in ("", "nope", '{"index": "x"}', '{"index": 1, "confidence": 95}', None, '{', '{"index": -1, "confidence": .5}'):
        assert parse_response(junk, 4) is None, junk

    s, u = build_prompt(world, "x", opts)
    assert "0: pick_place(obj_1, left)" in u and "1/3" in s and '"index"' in u

    assert LlamaCppHTTPBackend(url="http://127.0.0.1:1/completion", timeout=1.0).choose(world, "x", opts) is None
    # geometry: expected numbers worked out by hand (3-4-5 triangles), not produced by geometry()
    import copy
    gw = {"frame": "f", "bins": {"a": {"x": 0.0, "y": 0.0}, "b": {"x": 0.3, "y": 0.4}},
          "objects": [{"id": "o1", "x": 0.3, "y": 0.4, "size": 0.05},
                      {"id": "o2", "x": 0.6, "y": 0.8, "size": 0.03},
                      {"id": "o3", "x": 0.0, "y": -0.5, "size": 0.04}]}
    before = copy.deepcopy(gw)
    g = geometry(gw)
    assert gw == before, "geometry mutated its input"
    o1, o2, o3 = g["objects"]
    assert o1["dist_to_bin"] == {"a": 0.5, "b": 0.0} and o1["nearest_bin"] == "b", o1
    assert o2["dist_to_bin"] == {"a": 1.0, "b": 0.5} and o2["nearest_bin"] == "b", o2
    assert o3["dist_to_bin"] == {"a": 0.5, "b": 0.949} and o3["nearest_bin"] == "a", o3  # sqrt(.9)
    # rank_x 1 = smallest x (o3); rank_y 1 = LARGEST y (o2); both differ if the sign is flipped
    assert (o1["rank_x"], o2["rank_x"], o3["rank_x"]) == (2, 3, 1)
    assert (o1["rank_y"], o2["rank_y"], o3["rank_y"]) == (2, 1, 3)
    assert (o1["size_rank"], o2["size_rank"], o3["size_rank"]) == (1, 3, 2)
    assert o1["id"] == "o1" and o1["size"] == 0.05  # existing keys kept
    assert json.dumps(geometry(gw), sort_keys=True) == json.dumps(g, sort_keys=True)
    # ties break on id, regardless of list order
    tw = {"objects": [{"id": "b", "x": 1}, {"id": "a", "x": 1}]}
    assert [o["rank_x"] for o in geometry(tw)["objects"]] == [2, 1]
    # malformed worlds never raise
    assert geometry({}) == {} and geometry({"objects": "x"}) == {"objects": "x"}
    pw = {"bins": {"a": {"x": 0, "y": 0}}, "objects": [{"id": "p", "x": 1, "y": 2, "size": 1}, {"id": "q", "x": 2, "y": 3}]}
    pg = geometry(pw)["objects"]
    assert all("size_rank" not in o and "rank_x" in o and "dist_to_bin" in o for o in pg), pg
    assert "dist_to_bin" not in geometry({"objects": [{"id": "p", "x": 1, "y": 2}], "bins": {}})["objects"][0]
    for junk in (None, [], {"objects": [None]}, {"objects": [{"x": "a"}], "bins": 5}):
        geometry(junk)

    print("brain_lang selfcheck OK")


if __name__ == "__main__":
    _selfcheck()
