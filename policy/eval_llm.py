"""One-command language evaluation of the brain (brain_lang) against the shipped resolver.

  python3 -I policy/eval_llm.py --backend stub            # regression guard: must equal resolver
  python3 -I policy/eval_llm.py --backend llamacpp_http   # real LLM
DEV material only (training corpus + dev lists in test_instruction.py). Never reads sealed sets.
Scoring: per object +2 correct / -1 wrong / 0 skip; guess only pays when p > 1/3.
Gold = resolver's own mapping, compared by BEHAVIOUR over all 9 (colour, shape) types, never dict ==.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
for _p in (HERE, ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from instruction import COLORS, SHAPES, resolve  # noqa: E402

BINS = ("left", "center", "right")
TYPES = [(c, s) for c in COLORS for s in SHAPES]
DEFAULT_CORPORA = [os.path.join(ROOT, "docs", "instruction-training-corpus.json"),
                   os.path.join(HERE, "test_instruction.py")]
DEV_LISTS = ("ADVERSARIAL", "BIN_REFERENCES")  # each: list of (sentence, expected_mapping)


def resolve_bin(mapping, color, shape=None):
    """Same precedence as probe_families.py:11 -- a bare colour outranks a bare shape."""
    for key in ((color, shape), color, shape, "default"):
        if key in mapping:
            return mapping[key]
    return None


def behaviour(mapping):
    """The 9-type behaviour signature. Compare THIS, never the dicts."""
    return tuple(resolve_bin(mapping, c, s) for c, s in TYPES)


def same_behaviour(a, b):
    return behaviour(a) == behaviour(b)


def make_scene(sentence):
    """Deterministic 3-5 object scene seeded from the sentence; a spread of distinct types first."""
    rng = random.Random(int(hashlib.sha256(sentence.encode()).hexdigest()[:16], 16))
    n = rng.randint(3, 5)
    types = rng.sample(TYPES, n)  # distinct types -> spread
    objs = [{"id": f"obj_{i}", "color": c, "shape": s} for i, (c, s) in enumerate(types)]
    return {"objects": objs, "bins": list(BINS)}


def options_for(obj_id):
    return [f"pick_place({obj_id}, {b})" for b in BINS] + [f"skip({obj_id})"]


def bin_of(option):
    """'pick_place(obj_0, left)' -> 'left'; skip -> None."""
    if option.startswith("pick_place("):
        return option[:-1].rsplit(",", 1)[1].strip()
    return None


def choose_bin(brain, llm, world, obj, mapping, instruction):
    """Run the brain for one object; returns chosen bin or None (skip). Any failure = skip."""
    opts = options_for(obj["id"])
    w = dict(world, target=obj["id"], instruction=instruction)
    # decide() takes the backend's (index, confidence) answer, not the backend itself
    d = brain.decide(opts, w, mapping, llm.choose(w, instruction, opts))
    i = d.get("index") if isinstance(d, dict) else None
    return bin_of(opts[i]) if isinstance(i, int) and 0 <= i < len(opts) else None


def probe_behaviour(brain, llm, sentence, mapping):
    """What the brain does for each of the 9 types (one-object world each)."""
    out = []
    for k, (c, s) in enumerate(TYPES):
        obj = {"id": "obj_0", "color": c, "shape": s}
        world = {"objects": [obj], "bins": list(BINS)}
        out.append(choose_bin(brain, llm, world, obj, mapping, sentence))
    return tuple(out)


def points(chosen, gold):
    """+2 correct, -1 wrong, 0 skip. If the resolver abstains (gold None) any bin guess is wrong."""
    if chosen is None:
        return 0, "skipped"
    return (2, "correct") if chosen == gold else (-1, "wrong")


def eval_sentence(brain, llm, sentence):
    mapping = resolve(sentence)
    world = make_scene(sentence)
    tally = {"correct": 0, "wrong": 0, "skipped": 0, "pts": 0, "max": 0}
    for obj in world["objects"]:
        gold = resolve_bin(mapping, obj["color"], obj["shape"])
        p, kind = points(choose_bin(brain, llm, world, obj, mapping, sentence), gold)
        tally[kind] += 1
        tally["pts"] += p
        tally["max"] += 2 if gold is not None else 0
    chosen = probe_behaviour(brain, llm, sentence, mapping)
    return tally, chosen, behaviour(mapping)


def load_sentences(paths):
    """Returns ([sentence], [notes]). DEV files only; a missing path is reported, not fatal."""
    sents, notes = [], []
    for path in paths:
        if not os.path.exists(path):
            notes.append(f"MISSING corpus, skipped: {path}")
            continue
        before = len(sents)
        if path.endswith(".py"):
            import importlib.util
            spec = importlib.util.spec_from_file_location("_dev_" + os.path.basename(path)[:-3], path)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            for name in DEV_LISTS:
                sents += [row[0] for row in getattr(mod, name, [])]
        else:
            with open(path) as f:
                data = json.load(f)
            items = data["items"] if isinstance(data, dict) else data
            sents += [it["sentence"] if isinstance(it, dict) else it for it in items]
        notes.append(f"{len(sents) - before:4d} sentences  {os.path.relpath(path, ROOT)}")
    seen, uniq = set(), []
    for s in sents:
        if s not in seen:
            seen.add(s)
            uniq.append(s)
    return uniq, notes


def import_brain():
    try:
        try:
            import brain_lang as bl
        except ModuleNotFoundError as e:
            if e.name != "brain_lang":
                raise
            from policy import brain_lang as bl
        for n in ("build_prompt", "gbnf_for", "parse_response", "classical_choice", "decide",
                  "StubBackend", "LlamaCppHTTPBackend"):
            getattr(bl, n)
        return bl
    except (ImportError, AttributeError) as e:
        print(f"eval_llm: cannot import policy/brain_lang.py ({e}). Not written yet?", file=sys.stderr)
        sys.exit(2)


def run(args):
    brain = import_brain()
    llm = brain.StubBackend() if args.backend == "stub" else brain.LlamaCppHTTPBackend()
    sents, notes = load_sentences(args.corpus or DEFAULT_CORPORA)
    tot = {"correct": 0, "wrong": 0, "skipped": 0, "pts": 0, "max": 0}
    diffs = []
    for s in sents:
        t, chosen, gold = eval_sentence(brain, llm, s)
        for k in tot:
            tot[k] += t[k]
        if chosen != gold:
            diffs.append((s, chosen, gold))
    n = len(sents)
    ev = 100.0 * tot["pts"] / tot["max"] if tot["max"] else 0.0
    print(f"eval_llm  backend={args.backend}  sentences={n}  (dev corpora only)")
    for line in notes:
        print("  " + line)
    print("-" * 56)
    print(f"objects  correct {tot['correct']:5d} | wrong {tot['wrong']:5d} | skipped {tot['skipped']:5d}")
    print(f"points   {tot['pts']:5d} of {tot['max']} achievable   EV = {ev:.1f}% of max")
    print(f"behaviour differs from resolver (9-type): {len(diffs)}/{n} sentences")
    ok = not diffs
    if args.backend == "stub":
        print(f"STUB == RESOLVER regression guard: {'PASS' if ok else 'FAIL'} ({len(diffs)} differing)")
    if args.verbose:
        for s, ch, gold in diffs[: args.limit]:
            print(f"  DIFF {s!r}\n       chosen={ch}\n       resolver={gold}")
    return 0 if (ok or args.backend != "stub") else 1


def _selfcheck():
    s = "Red goes left, blue goes center, everything else right."
    assert make_scene(s) == make_scene(s), "scene not deterministic"
    assert 3 <= len(make_scene(s)["objects"]) <= 5
    a = {"red": "left", "blue": "center", "default": "right"}
    b = dict(a, yellow="right")
    assert a != b and same_behaviour(a, b), "9-type scorer must see a and b as equal"
    assert not same_behaviour(a, dict(a, yellow="left"))
    assert points("left", "right") == (-1, "wrong")
    assert points("left", "left") == (2, "correct") and points(None, "left") == (0, "skipped")
    assert resolve_bin({"red": "left", "cube": "right"}, "red", "cube") == "left"  # colour outranks shape
    assert bin_of("pick_place(obj_3, center)") == "center" and bin_of("skip(obj_3)") is None
    print("selfcheck OK")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--backend", choices=("stub", "llamacpp_http"), default="stub")
    ap.add_argument("--corpus", action="append", metavar="PATH", help="repeatable; default = local dev corpora")
    ap.add_argument("--limit", type=int, default=10, help="max disagreements shown with --verbose")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--selfcheck", action="store_true")
    a = ap.parse_args()
    if a.selfcheck:
        _selfcheck()
        sys.exit(0)
    sys.exit(run(a))
