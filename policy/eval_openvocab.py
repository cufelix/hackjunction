"""Open-vocabulary DEV eval: spatial / ordinal / relational sentences a colour-shape regex cannot resolve.

Coordinate convention (the team's world JSON, ROBOT PELVIS frame):
    x = FORWARD distance from the robot (larger x = farther away)
    y = LATERAL: positive = robot's LEFT, negative = robot's RIGHT
    bins: left (x=0.45, y=+0.21), center (x=0.45, y=0.00), right (x=0.45, y=-0.21)
    objects: x in [0.28, 0.40], y in [-0.19, +0.19], distinct positions, min 0.06 apart
"Closest to the robot" = smallest x.  "Farthest" = largest x.  "Leftmost" = largest y.
"Rightmost" = smallest y.  "Biggest" = largest size (metres).

Gold is ALWAYS computed from the scene by the rule functions below (never typed). Objects the
sentence does not name are absent from gold. Scenes are regenerated until the deciding quantity
has a clear margin (no ties). "Odd one out" has ONE definition: the object whose COLOUR no other
object shares; every such scene has exactly one. "Next to bin B" = smallest Euclidean distance
to that bin's centre.

DEV material only: sentences are invented here. Nothing is taken from any sealed/held-out set.
Standard library only.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
try:
    from brain_lang import (LlamaCppHTTPBackend, StubBackend, brain_mapping,  # noqa: F401
                            build_prompt, decide, resolve_bin)
except ImportError:  # pragma: no cover
    from policy.brain_lang import (LlamaCppHTTPBackend, StubBackend, brain_mapping,  # noqa: F401
                                   build_prompt, decide, resolve_bin)
from instruction import resolve  # noqa: E402

OUT = HERE / "openvocab_dev.jsonl"
SEED = 20261114
BINS = {"left": {"x": 0.45, "y": 0.21}, "center": {"x": 0.45, "y": 0.0}, "right": {"x": 0.45, "y": -0.21}}
FRAME = ("x = forward distance from robot (larger = farther); y = lateral, positive = robot's left, "
         "negative = right; bins at x=0.45, left y=+0.21, center y=0.00, right y=-0.21")
BIN_NAMES = tuple(BINS)
COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
MARGIN = 0.02       # min gap (m) on a deciding quantity; sizes use SIZE_MARGIN
SIZE_MARGIN = 0.004


# ---------------------------------------------------------------- rules: gold computed from scene
def _ranked(objs, key, reverse=False):
    return sorted(objs, key=key, reverse=reverse)


def _gap(vals, k):
    """Gap between the k-th and (k+1)-th smallest value (k items are selected)."""
    s = sorted(vals)
    return s[k] - s[k - 1] if len(s) > k else 1.0


def _dist(o, b):
    return math.hypot(o["x"] - BINS[b]["x"], o["y"] - BINS[b]["y"])


def g_closest(objs, p):
    return {_ranked(objs, lambda o: o["x"])[0]["id"]: p["to"]}


def g_farthest(objs, p):
    return {_ranked(objs, lambda o: o["x"], True)[0]["id"]: p["to"]}


def g_next_to_bin(objs, p):
    return {_ranked(objs, lambda o: _dist(o, p["ref"]))[0]["id"]: p["to"]}


def g_leftmost(objs, p):   # +y is the robot's left
    return {_ranked(objs, lambda o: o["y"], True)[0]["id"]: p["to"]}


def g_rightmost(objs, p):
    return {_ranked(objs, lambda o: o["y"])[0]["id"]: p["to"]}


def g_two_rightmost(objs, p):
    return {o["id"]: p["to"] for o in _ranked(objs, lambda o: o["y"])[:2]}


def g_two_leftmost(objs, p):
    return {o["id"]: p["to"] for o in _ranked(objs, lambda o: o["y"], True)[:2]}


def g_biggest(objs, p):
    return {_ranked(objs, lambda o: o["size"], True)[0]["id"]: p["to"]}


def g_smallest(objs, p):
    return {_ranked(objs, lambda o: o["size"])[0]["id"]: p["to"]}


def g_nearest_bin(objs, p):
    return {o["id"]: min(BIN_NAMES, key=lambda b: _dist(o, b)) for o in objs}


def g_odd_color(objs, p):
    cs = [o["color"] for o in objs]
    return {o["id"]: p["to"] for o in objs if cs.count(o["color"]) == 1}


# margin(objs, p): smallest gap on the deciding quantity; must be >= its threshold (0 => ambiguous)
def m_x_min(objs, p): return _gap([o["x"] for o in objs], 1)
def m_x_max(objs, p): return _gap([-o["x"] for o in objs], 1)
def m_y_min(objs, p): return _gap([o["y"] for o in objs], 1)
def m_y_max(objs, p): return _gap([-o["y"] for o in objs], 1)
def m_y_min2(objs, p): return _gap([o["y"] for o in objs], 2)
def m_y_max2(objs, p): return _gap([-o["y"] for o in objs], 2)
def m_s_max(objs, p): return _gap([-o["size"] for o in objs], 1)
def m_s_min(objs, p): return _gap([o["size"] for o in objs], 1)
def m_bin(objs, p): return _gap([_dist(o, p["ref"]) for o in objs], 1)


def m_nearest_bin(objs, p):
    gaps = []
    for o in objs:
        d = sorted(_dist(o, b) for b in BIN_NAMES)
        gaps.append(d[1] - d[0])
    return min(gaps)


def m_odd(objs, p):
    cs = [o["color"] for o in objs]
    return 1.0 if sum(cs.count(c) == 1 for c in cs) == 1 else 0.0


# name: (family, gold_fn, margin_fn, margin_threshold, templates)
RULES = {
    "closest_to_robot": ("spatial", g_closest, m_x_min, MARGIN, [
        "The object closest to the robot goes in the {to} bin.",
        "Put whatever is nearest to the robot in the {to} bin.",
        "The nearest object to you goes {to}."]),
    "farthest_from_robot": ("spatial", g_farthest, m_x_max, MARGIN, [
        "The object farthest away goes in the {to} bin.",
        "Put the one that is furthest from the robot in the {to} bin.",
        "Whatever is most distant goes {to}."]),
    "next_to_bin": ("spatial", g_next_to_bin, m_bin, MARGIN, [
        "Whatever is next to the {ref} bin goes {to}.",
        "The object closest to the {ref} bin goes in the {to} bin.",
        "Put the item sitting nearest the {ref} bin into the {to} bin."]),
    "leftmost": ("ordinal", g_leftmost, m_y_max, MARGIN, [
        "The leftmost object goes in the {to} bin.",
        "Put the object furthest to the left in the {to} bin."]),
    "rightmost": ("ordinal", g_rightmost, m_y_min, MARGIN, [
        "The rightmost object goes in the {to} bin.",
        "Put the object furthest to the right in the {to} bin."]),
    "two_rightmost": ("ordinal", g_two_rightmost, m_y_min2, MARGIN, [
        "The two rightmost objects go in the {to} bin.",
        "Put the two objects furthest right into the {to} bin."]),
    "two_leftmost": ("ordinal", g_two_leftmost, m_y_max2, MARGIN, [
        "The two leftmost objects go in the {to} bin.",
        "Put the two objects furthest left into the {to} bin."]),
    "biggest": ("ordinal", g_biggest, m_s_max, SIZE_MARGIN, [
        "The biggest one goes in the {to} bin.",
        "Put the largest object in the {to} bin."]),
    "smallest": ("ordinal", g_smallest, m_s_min, SIZE_MARGIN, [
        "The smallest one goes in the {to} bin.",
        "Put the tiniest object in the {to} bin."]),
    "nearest_bin": ("relational", g_nearest_bin, m_nearest_bin, MARGIN, [
        "Put each object in the bin nearest to it.",
        "Every object goes into whichever bin is closest to it."]),
    "odd_one_out_color": ("relational", g_odd_color, m_odd, 0.5, [
        "The odd one out goes in the {to} bin (the odd one is the object whose colour no other object shares).",
        "Put the object with a colour nobody else has in the {to} bin."]),
}
# (rule, count): 27 spatial, 27 ordinal, 26 relational = 80
PLAN = [("closest_to_robot", 9), ("farthest_from_robot", 9), ("next_to_bin", 9),
        ("leftmost", 6), ("rightmost", 6), ("two_rightmost", 4), ("two_leftmost", 4),
        ("biggest", 4), ("smallest", 3),
        ("nearest_bin", 13), ("odd_one_out_color", 13)]


def apply_rule(rule, scene, p):
    return RULES[rule][1](scene["objects"], p)


def ambiguous(rule, objs, p):
    return RULES[rule][2](objs, p) < RULES[rule][3]


# ---------------------------------------------------------------- scene generation
def _make_scene(rng, n):
    objs = []
    while len(objs) < n:
        x, y = round(rng.uniform(0.28, 0.40), 3), round(rng.uniform(-0.19, 0.19), 3)
        if all(math.hypot(x - o["x"], y - o["y"]) >= 0.06 for o in objs):
            objs.append({"x": x, "y": y})
    return {"objects": [{"id": f"obj_{i}", "color": rng.choice(COLORS), "shape": rng.choice(SHAPES),
                         "x": o["x"], "y": o["y"], "size": round(rng.uniform(0.032, 0.062), 3)}
                        for i, o in enumerate(objs)],
            "bins": {k: dict(v) for k, v in BINS.items()}}


def _note(rule, scene, p, gold):
    o = {x["id"]: x for x in scene["objects"]}
    if rule == "nearest_bin":
        return "; ".join(f"{i}: y={o[i]['y']} (x={o[i]['x']}) nearest {b}" for i, b in gold.items())
    ids = ", ".join(gold)
    return {"closest_to_robot": lambda: f"{ids} has the smallest x ({o[ids]['x']})",
            "farthest_from_robot": lambda: f"{ids} has the largest x ({o[ids]['x']})",
            "next_to_bin": lambda: f"{ids} is {_dist(o[ids], p['ref']):.3f} m from the {p['ref']} bin, the smallest",
            "leftmost": lambda: f"{ids} has the largest y ({o[ids]['y']}), +y is the robot's left",
            "rightmost": lambda: f"{ids} has the smallest y ({o[ids]['y']}), -y is the robot's right",
            "two_rightmost": lambda: f"{ids} have the two smallest y (-y is the robot's right)",
            "two_leftmost": lambda: f"{ids} have the two largest y (+y is the robot's left)",
            "biggest": lambda: f"{ids} has the largest size ({o[ids]['size']})",
            "smallest": lambda: f"{ids} has the smallest size ({o[ids]['size']})",
            "odd_one_out_color": lambda: f"{ids} is the only {o[ids]['color']} object"}[rule]()


def generate():
    rng = random.Random(SEED)
    recs, k = [], 0
    for rule, count in PLAN:
        fam, _, _, _, templates = RULES[rule]
        for j in range(count):
            to = BIN_NAMES[j % 3]
            ref = BIN_NAMES[(j // 3 + (j % 3) + 1) % 3]   # ref != to
            p = {"to": to, "ref": ref}
            need = (["to"] if rule != "nearest_bin" else []) + (["ref"] if rule == "next_to_bin" else [])
            while True:
                scene = _make_scene(rng, rng.choice((3, 4, 5)))
                if rule == "two_rightmost" or rule == "two_leftmost":
                    if len(scene["objects"]) < 3:
                        continue
                if not ambiguous(rule, scene["objects"], p):
                    break
            gold = apply_rule(rule, scene, p)
            k += 1
            recs.append({"id": f"ov_{k:03d}", "family": fam, "rule": rule,
                         "sentence": templates[j % len(templates)].format(**p),
                         "scene": scene, "gold": gold, "gold_note": _note(rule, scene, p, gold),
                         "params": {q: p[q] for q in need}})
    return recs


def dumps(recs):
    return "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in recs)


def load():
    return [json.loads(l) for l in OUT.read_text().splitlines() if l.strip()]


# ---------------------------------------------------------------- evaluation
# instruction.resolve() NEVER returns None: when it parses nothing it hands back FIXED_MAPPING
# (red->left, yellow->center, blue->right), a blind guess. "Abstain" here means "parsed nothing",
# so we resolve with fallback={}. StubBackend calls resolve() with the default fallback and would
# therefore guess; NoFallbackStub is the same stub minus that guess.
class NoFallbackStub(StubBackend):
    def choose(self, world, instruction, options):
        from brain_lang import classical_choice
        try:
            c = classical_choice(options, world, brain_mapping(instruction))
        except Exception:
            return None
        return None if c is None else (c, 1.0)


def evaluate(backend, recs, verbose=False):
    stats = {}
    for r in recs:
        world, st = {"frame": FRAME, **r["scene"]}, stats.setdefault(r["family"], dict(n=0, ok=0, bad=0, skip=0, abst=0, rok=0, rbad=0, fok=0, fbad=0))
        mapping = brain_mapping(r["sentence"])
        shipped = resolve(r["sentence"])   # what the resolver alone would do in production
        objs = {o["id"]: o for o in world["objects"]}
        for oid, gbin in r["gold"].items():
            options = [f"pick_place({oid}, {b})" for b in BIN_NAMES] + [f"skip({oid})"]
            llm = backend.choose(world, r["sentence"], options)
            d = decide(options, world, mapping, llm=llm)
            chosen = None if d["index"] is None else options[d["index"]]
            st["n"] += 1
            rb = resolve_bin(mapping, objs[oid]["color"], objs[oid]["shape"])
            sb = resolve_bin(shipped, objs[oid]["color"], objs[oid]["shape"])
            st["fok" if sb == gbin else "fbad"] += 1
            if rb is None:
                st["abst"] += 1
            else:
                st["rok" if rb == gbin else "rbad"] += 1
            if chosen is None or chosen.startswith("skip"):
                st["skip"] += 1
                res = "skip"
            elif chosen == f"pick_place({oid}, {gbin})":
                st["ok"] += 1
                res = "ok"
            else:
                st["bad"] += 1
                res = "WRONG"
            if verbose:
                print(f"{r['id']} {r['rule']:<20} {oid} gold={gbin:<6} chose={chosen} [{res}] "
                      f"resolver={rb} src={d['source']}")
    return stats


def sweep(backend, recs, thresholds=(0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.01)):
    """Query the model ONCE per decision, then replay those answers at every threshold.

    The 0.95 override threshold is a guess until something measures it, and Oliver's review makes
    the point that a VLM's confidence is uncalibrated. Re-running the whole set per threshold would
    cost N model passes on someone else's GPU; the model's answer does not depend on the threshold,
    so one pass is enough and the rest is arithmetic.

    Read the table by EV, not accuracy. 1.01 is "never override" (pure resolver) and 0.0 is "always
    override when confident enough to answer at all"; if neither end wins, the middle is real.
    """
    import brain_lang
    cached = []          # (options, world, mapping, oid, gold_bin, llm_answer)
    for r in recs:
        world = {"frame": FRAME, **r["scene"]}
        mapping = brain_mapping(r["sentence"])
        for oid, gbin in r["gold"].items():
            options = [f"pick_place({oid}, {b})" for b in BIN_NAMES] + [f"skip({oid})"]
            cached.append((options, world, mapping, oid, gbin,
                           backend.choose(world, r["sentence"], options)))

    original = brain_lang.LLM_OVERRIDE_CONF
    print(f"\n{'threshold':>10}{'ok':>6}{'wrong':>7}{'skip':>6}{'points':>8}{'EV%':>8}{'llm-used':>10}")
    rows = []
    try:
        for t in thresholds:
            brain_lang.LLM_OVERRIDE_CONF = t
            ok = bad = skip = used = 0
            for options, world, mapping, oid, gbin, llm in cached:
                d = decide(options, world, mapping, llm=llm)
                if d["source"] == "llm":
                    used += 1
                chosen = None if d["index"] is None else options[d["index"]]
                if chosen is None or chosen.startswith("skip"):
                    skip += 1
                elif chosen == f"pick_place({oid}, {gbin})":
                    ok += 1
                else:
                    bad += 1
            pts = 2 * ok - bad
            n = len(cached)
            rows.append((t, pts))
            label = f"{t:.2f}" + ("  (never)" if t > 1 else "  (always)" if t == 0 else "")
            print(f"{label:>10}{ok:>6}{bad:>7}{skip:>6}{pts:>8}{100 * pts / (2 * n):>7.1f}%{used:>10}")
    finally:
        brain_lang.LLM_OVERRIDE_CONF = original     # never leave the module mutated

    best_t, best_pts = max(rows, key=lambda kv: kv[1])
    shipped = dict(rows).get(0.95)
    print(f"\nbest threshold by points: {best_t} ({best_pts} pts); shipped 0.95 scores {shipped} pts")
    if shipped is not None and best_pts > shipped:
        print("  -> 0.95 is NOT optimal on this set. Worth a look, but do not tune the threshold")
        print("     on the set you then report: pick on dev, report on something else.")
    return rows


def report(stats, backend_name):
    tot = {k: sum(s[k] for s in stats.values()) for k in next(iter(stats.values()))}
    print(f"\n{'family':<11}{'objs':>5}{'ok':>5}{'wrong':>6}{'skip':>6}{'points':>8}{'EV%':>7}{'abstain%':>10}")
    rows = list(stats.items()) + [("OVERALL", tot)]
    for name, s in rows:
        pts = 2 * s["ok"] - s["bad"]
        print(f"{name:<11}{s['n']:>5}{s['ok']:>5}{s['bad']:>6}{s['skip']:>6}{pts:>8}"
              f"{100 * pts / (2 * s['n']):>6.1f}%{100 * s['abst'] / s['n']:>9.1f}%")
    print("\n" + "=" * 70)
    print(f"RESOLVER ABSTENTION RATE ON THIS SET: {100 * tot['abst'] / tot['n']:.1f}% "
          f"({tot['abst']}/{tot['n']} gold objects)")
    if tot["n"] - tot["abst"]:
        print(f"  resolver answered {tot['n'] - tot['abst']} anyway: {tot['rok']} right, {tot['rbad']} wrong "
              f"(a stray 'goes in the X bin' read as a default key, not understanding)")
    print(f"Shipped resolver alone (blind FIXED_MAPPING when it parses nothing): {tot['fok']} right, "
          f"{tot['fbad']} wrong -> {2 * tot['fok'] - tot['fbad']} points ({100 * (2 * tot['fok'] - tot['fbad']) / (2 * tot['n']):.1f}%)")
    print("=" * 70)
    if backend_name == "stub":
        print("NOTE: ~100% abstention and ~0 points is the CORRECT baseline for the stub (no model), not a bug.")


# ---------------------------------------------------------------- self-check
def selfcheck():
    a, b = dumps(generate()), dumps(generate())
    assert a == b, "regeneration not deterministic"
    if OUT.exists():
        assert OUT.read_text() == a, f"{OUT.name} on disk differs from regeneration (run --regenerate)"
    recs = [json.loads(l) for l in a.splitlines()]
    fams = {}
    for r in recs:
        fams[r["family"]] = fams.get(r["family"], 0) + 1
        objs = r["scene"]["objects"]
        assert 3 <= len(objs) <= 5
        assert r["gold"] == apply_rule(r["rule"], r["scene"], r["params"]), r["id"]
        assert r["gold"], r["id"]
        assert set(r["gold"]) <= {o["id"] for o in objs}
        assert all(v in BIN_NAMES for v in r["gold"].values()), r["id"]
        assert not ambiguous(r["rule"], objs, r["params"]), f"{r['id']} tie/ambiguity"
        for i, o in enumerate(objs):
            assert 0.28 <= o["x"] <= 0.40 and -0.19 <= o["y"] <= 0.19, r["id"]
            assert o["color"] in COLORS and o["shape"] in SHAPES
            for q in objs[i + 1:]:
                assert math.hypot(o["x"] - q["x"], o["y"] - q["y"]) >= 0.06 - 1e-9, r["id"]
        assert r["scene"]["bins"] == BINS
        if r["rule"] == "leftmost":
            assert r["gold"] == {max(objs, key=lambda o: o["y"])["id"]: r["params"]["to"]}
        if r["rule"] == "closest_to_robot":
            assert r["gold"] == {min(objs, key=lambda o: o["x"])["id"]: r["params"]["to"]}
        if r["rule"] == "odd_one_out_color":
            cs = [o["color"] for o in objs]
            assert sum(cs.count(c) == 1 for c in cs) == 1
    # convention sanity on a hand-built scene
    s = {"objects": [{"id": "a", "x": 0.40, "y": 0.18, "size": 0.05}, {"id": "b", "x": 0.30, "y": -0.15, "size": 0.04},
                     {"id": "c", "x": 0.34, "y": 0.01, "size": 0.045}]}
    assert apply_rule("leftmost", s, {"to": "left"}) == {"a": "left"}      # largest y
    assert apply_rule("rightmost", s, {"to": "left"}) == {"b": "left"}     # smallest y
    assert apply_rule("closest_to_robot", s, {"to": "right"}) == {"b": "right"}   # smallest x
    assert apply_rule("farthest_from_robot", s, {"to": "right"}) == {"a": "right"}   # largest x
    assert apply_rule("two_leftmost", s, {"to": "left"}) == {"a": "left", "c": "left"}   # two largest y
    assert apply_rule("two_rightmost", s, {"to": "left"}) == {"b": "left", "c": "left"}  # two smallest y
    assert apply_rule("nearest_bin", s, {}) == {"a": "left", "b": "right", "c": "center"}
    assert apply_rule("biggest", s, {"to": "left"}) == {"a": "left"}
    assert len(recs) == 80 and set(fams) == {"spatial", "ordinal", "relational"}, fams
    print(f"selfcheck OK: {len(recs)} records, families {fams}, deterministic, gold==rule(scene), no ties")


def main():
    ap = argparse.ArgumentParser(description="Open-vocabulary DEV eval (spatial/ordinal/relational); "
                                 "gold is computed from the scene.")
    ap.add_argument("--backend", choices=("stub", "llamacpp_http"), default="stub")
    ap.add_argument("--regenerate", action="store_true", help=f"rewrite {OUT.name}")
    ap.add_argument("--limit", type=int, help="only the first N records")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    ap.add_argument("--sweep", action="store_true",
                    help="query the model once per decision, replay at many override thresholds")
    a = ap.parse_args()
    if a.selfcheck:
        return selfcheck()
    if a.regenerate or not OUT.exists():
        OUT.write_text(dumps(generate()))
        print(f"wrote {OUT}")
    recs = load()[: a.limit]
    if not recs:
        raise SystemExit('no records selected (--limit must be >= 1)')
    backend = NoFallbackStub() if a.backend == "stub" else LlamaCppHTTPBackend()
    if a.sweep:
        return sweep(backend, recs)
    report(evaluate(backend, recs, a.verbose), a.backend)


if __name__ == "__main__":
    main()
