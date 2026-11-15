#!/usr/bin/env python3
"""Generate docs/instruction-heldout-corpus.json: held-out style instructions with ground truth.

Why: the organisers' oracle parser is exact string equality on the 288 training sentences (oracle.py:196-218).
The finals use an unseen sentence (x1.5). This corpus is OUR OWN guess at unseen phrasings, so a matcher can be
tested beyond 288/288. It is NOT the organisers' held-out set (we have not seen it and must not look for it).

Run (stdlib only, no repo needed):  python3 docs/gen_instruction_heldout.py
Optional check against the training templates:  python3 docs/gen_instruction_heldout.py --templates path/to/tasks.py

Ground truth is always colour -> bin key in {"left","center","right"} (scene.resolve_bin's keys), even when the
sentence says "centre" or "middle".
Out of scope on purpose: negation (organisers: none), shape-keyed sentences are a separate list.
"""
import argparse
import itertools
import json
import random
import re
from pathlib import Path

COLORS = ("red", "yellow", "blue")
BINS = ("left", "center", "right")
SEED = 20261114

# --- surface forms -------------------------------------------------------------------------------------------
POS = {  # position words, default style (American "center", as in the training set)
    "left": "left", "center": "center", "right": "right"}
POS_UK = {"left": "left", "center": "centre", "right": "right"}
POS_MID = {"left": "left", "center": "middle", "right": "right"}
POS_CENTRAL = {"left": "left", "center": "central", "right": "right"}
POS_HAND = {"left": "left-hand", "center": "center", "right": "right-hand"}
POS_SIDE = {"left": "on the left", "center": "in the middle", "right": "on the right"}
POS_OF = {"left": "leftmost", "center": "middle", "right": "rightmost"}


def cap(s):
    return s[0].upper() + s[1:]


# A pattern: (category id, category name, fn(c, b, P) -> sentence). c = 3 colours in sentence order,
# b = the 3 bin keys matching them, P = position-word dict for this pattern.
def p_verb(verb):
    def f(c, b, P):
        return (f"{cap(verb)} the {c[0]} objects in the {P[b[0]]} bin, the {c[1]} objects in the {P[b[1]]} bin "
                f"and the {c[2]} objects in the {P[b[2]]} bin.")
    return f


def p_noun(noun):
    def f(c, b, P):
        return (f"Put all {c[0]} {noun} in the {P[b[0]]} bin, all {c[1]} {noun} in the {P[b[1]]} bin, "
                f"and all {c[2]} {noun} in the {P[b[2]]} bin.")
    return f


def p_container(word):
    def f(c, b, P):
        return (f"Put the {c[0]} objects in the {P[b[0]]} {word}, the {c[1]} objects in the {P[b[1]]} {word} "
                f"and the {c[2]} objects in the {P[b[2]]} {word}.")
    return f


def p_bin_first(c, b, P):
    return (f"In the {P[b[0]]} bin goes everything {c[0]}, in the {P[b[1]]} bin everything {c[1]}, "
            f"and in the {P[b[2]]} bin everything {c[2]}.")


def p_bin_first2(c, b, P):
    return (f"The {P[b[0]]} bin takes the {c[0]} objects. The {P[b[1]]} bin takes the {c[1]} objects. "
            f"The {P[b[2]]} bin takes the {c[2]} objects.")


def p_should(c, b, P):
    return (f"{cap(c[0])} objects should go in the {P[b[0]]} bin, {c[1]} objects should go in the {P[b[1]]} bin, "
            f"and {c[2]} objects should go in the {P[b[2]]} bin.")


def p_respective(c, b, P):
    return (f"Put the {c[0]}, {c[1]} and {c[2]} objects in the {P[b[0]]}, {P[b[1]]} and {P[b[2]]} bins, "
            f"respectively.")


def p_respective2(c, b, P):
    return f"{cap(c[0])}, {c[1]}, {c[2]}: {P[b[0]]}, {P[b[1]]}, {P[b[2]]} (in that order)."


def p_ones(c, b, P):
    return (f"Take the {c[0]} ones to the {P[b[0]]} bin, then the {c[1]} ones to the {P[b[1]]} bin, "
            f"and finally the {c[2]} ones to the {P[b[2]]} bin.")


def _one(word):
    return {"left": "the bin on the left", "center": f"the {word} one", "right": "the bin on the right"}


def p_one_on(word):
    def f(c, b, P):
        m = _one(word)
        return (f"{cap(m[b[0]])} is for {c[0]}, {m[b[1]]} is for {c[1]}, and {m[b[2]]} is for {c[2]}.")
    return f


def p_side(c, b, P):
    return (f"Sort by colour: {c[0]} {P[b[0]]}, {c[1]} {P[b[1]]}, {c[2]} {P[b[2]]}.")


def p_your(c, b, P):
    return (f"I want the {c[0]} things in the {P[b[0]]} bin, the {c[1]} things in the {P[b[1]]} bin, "
            f"and the {c[2]} things in the {P[b[2]]} bin.")


def p_please(c, b, P):
    return (f"Please put the {c[0]} objects into the {P[b[0]]} bin, the {c[1]} objects into the {P[b[1]]} bin, "
            f"and the {c[2]} objects into the {P[b[2]]} bin.")


def p_each(c, b, P):
    return (f"Each {c[0]} object goes to the {P[b[0]]} bin, each {c[1]} object goes to the {P[b[1]]} bin, "
            f"each {c[2]} object goes to the {P[b[2]]} bin.")


def p_multi(c, b, P):
    return (f"First, the {c[0]} objects: {P[b[0]]} bin. Second, the {c[1]} objects: {P[b[1]]} bin. "
            f"Third, the {c[2]} objects: {P[b[2]]} bin.")


def p_multi2(c, b, P):
    return (f"Listen carefully. {cap(c[0])} goes in the {P[b[0]]} bin. {cap(c[1])} goes in the {P[b[1]]} bin. "
            f"{cap(c[2])} goes in the {P[b[2]]} bin.")


def p_two_then_rest(c, b, P):  # third by elimination
    return (f"Put the {c[0]} objects in the {P[b[0]]} bin and the {c[1]} objects in the {P[b[1]]} bin. "
            f"The {c[2]} ones go in the remaining bin.")


def p_two_rest(c, b, P):
    return (f"{cap(c[0])} to the {P[b[0]]}, {c[1]} to the {P[b[1]]}; whatever is {c[2]} goes in the last bin.")


def p_two_other(c, b, P):
    return (f"The {c[0]} objects belong in the {P[b[0]]} bin, the {c[1]} objects belong in the {P[b[1]]} bin, "
            f"and the {c[2]} ones belong in the other bin.")


CATS = {
    1: "verb paraphrase",
    2: "object noun paraphrase",
    3: "bin synonym (box, container, ...)",
    4: "bin-first word order",
    5: "relative position phrasing (the one on the left)",
    6: "modal / polite / first person",
    7: "'respectively' and list forms",
    8: "central / middle synonyms for center",
    9: "terse and punctuation-light",
    10: "multi-sentence",
    11: "British 'centre'",
    12: "third pair by elimination (remaining / other / last bin)",
    13: "casing, spacing, whitespace",
}

# (category, pattern fn, position dict)
PATTERNS = (
    [(1, p_verb(v), POS) for v in ("stack", "load", "transfer", "deposit", "send", "carry", "toss", "sort")]
    + [(2, p_noun(n), POS) for n in ("blocks", "pieces", "items", "shapes")]
    + [(3, p_container(w), POS) for w in ("box", "container", "bucket", "tray")]
    + [(4, p_bin_first, POS), (4, p_bin_first2, POS)]
    + [(5, p_one_on("middle"), POS), (5, p_one_on("center"), POS)]
    + [(6, p_should, POS), (6, p_please, POS), (6, p_your, POS), (6, p_each, POS)]
    + [(7, p_respective, POS), (7, p_respective2, POS)]
    + [(8, p_verb("put"), POS_MID), (8, p_verb("put"), POS_CENTRAL), (8, p_bin_first2, POS_MID)]
    + [(9, p_side, POS), (9, p_ones, POS)]
    + [(10, p_multi, POS), (10, p_multi2, POS)]
    + [(11, p_verb("put"), POS_UK), (11, p_bin_first2, POS_UK), (11, p_respective, POS_UK),
       (11, p_container("box"), POS_UK), (11, p_noun("blocks"), POS_UK), (11, p_side, POS_UK)]
    + [(12, p_two_then_rest, POS), (12, p_two_rest, POS), (12, p_two_other, POS)]
)

# Category 13 transforms a base sentence (no new wording; tests normalisation).
TRANSFORMS = (
    ("UPPER", str.upper),
    ("lower", str.lower),
    ("no final period", lambda s: s.rstrip(".")),
    ("double spaces", lambda s: s.replace(" ", "  ", 3)),
    ("leading/trailing space", lambda s: "  " + s + "  "),
    ("newline between clauses", lambda s: s.replace(", ", ",\n", 1)),
)

# Shape-keyed list (organisers: "may also test shape-based instructions"). shape -> bin.
SHAPES = ("cube", "cylinder", "triangle")
SHAPE_PLURAL = {"cube": "cubes", "cylinder": "cylinders", "triangle": "triangles"}


def shape_sentences(rng):
    out = []
    forms = [
        lambda s, b, P: f"Put the {SHAPE_PLURAL[s[0]]} in the {P[b[0]]} bin, the {SHAPE_PLURAL[s[1]]} in the {P[b[1]]} bin "
                        f"and the {SHAPE_PLURAL[s[2]]} in the {P[b[2]]} bin.",
        lambda s, b, P: f"{cap(SHAPE_PLURAL[s[0]])} go left-to-right as listed: {P[b[0]]}, then {SHAPE_PLURAL[s[1]]} "
                        f"to the {P[b[1]]}, {SHAPE_PLURAL[s[2]]} to the {P[b[2]]}.",
        lambda s, b, P: f"The {P[b[0]]} bin is for {SHAPE_PLURAL[s[0]]}, the {P[b[1]]} bin for {SHAPE_PLURAL[s[1]]}, "
                        f"the {P[b[2]]} bin for {SHAPE_PLURAL[s[2]]}.",
    ]
    # forms[1] is awkward on purpose? No: drop it, keep corpus natural.
    forms = [forms[0], forms[2]]
    for fi, fn in enumerate(forms):
        for P in (POS, POS_UK):
            for perm in itertools.permutations(BINS):
                shapes = list(SHAPES)
                rng.shuffle(shapes)
                out.append({"text": fn(shapes, list(perm), P), "mapping_by_shape": dict(zip(shapes, perm))})
    return out


def build():
    rng = random.Random(SEED)
    items = []
    perms = list(itertools.permutations(BINS))
    for cat, fn, P in PATTERNS:
        # 6 sentences per pattern: each bin permutation once, colour order varied.
        corders = list(itertools.permutations(COLORS))
        rng.shuffle(corders)
        for i, bperm in enumerate(perms):
            c = list(corders[i])
            b = list(bperm)
            text = fn(c, b, P)
            items.append({"category": cat, "text": text, "mapping": dict(zip(c, b))})
    # category 13: transform a rotating subset of base sentences
    base = [it for it in items if it["category"] in (1, 6)]
    rng.shuffle(base)
    for i, (name, tf) in enumerate(TRANSFORMS):
        for it in base[i * 3:(i + 1) * 3]:
            items.append({"category": 13, "text": tf(it["text"]), "mapping": dict(it["mapping"]), "transform": name})
    return items


def check(items, templates_path=None):
    problems = []
    seen = set()
    for it in items:
        m = it["mapping"]
        if sorted(m) != sorted(COLORS) or sorted(m.values()) != sorted(BINS):
            problems.append(("not a bijection", it["text"]))
        if it["text"] in seen:
            problems.append(("duplicate", it["text"]))
        seen.add(it["text"])
    if templates_path:
        src = Path(templates_path).read_text()
        block = src[src.index("TEMPLATES = ("):]
        block = block[:block.index("\n)")]
        tpls = re.findall(r'"((?:[^"\\]|\\.)*)"', block)
        train = set()
        for t in tpls:
            if "{" not in t:
                continue
            for cs in itertools.permutations(COLORS):
                for bs in itertools.permutations(BINS):
                    w = {f"c{i}": c for i, c in enumerate(cs)}
                    w.update({f"C{i}": c.capitalize() for i, c in enumerate(cs)})
                    w.update({f"b{i}": b for i, b in enumerate(bs)})
                    train.add(t.format(**w))
        for it in items:
            if it["text"] in train:
                problems.append(("equals a training sentence", it["text"]))
        print(f"checked against {len(train)} training sentences")
    return problems


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--templates")
    ap.add_argument("--out", default=str(Path(__file__).with_name("instruction-heldout-corpus.json")))
    a = ap.parse_args()
    items = build()
    rng = random.Random(SEED + 1)
    shape_items = shape_sentences(rng)
    problems = check(items, a.templates)
    for p in problems:
        print("PROBLEM:", *p)
    by = {}
    for it in items:
        by[it["category"]] = by.get(it["category"], 0) + 1
    doc = {
        "note": "Our own guess at unseen phrasings. Not the organisers' held-out set. Ground truth keys: left/center/right.",
        "categories": {str(k): v for k, v in CATS.items()},
        "counts": {str(k): by[k] for k in sorted(by)},
        "n_colour": len(items), "n_shape": len(shape_items),
        "colour_sentences": items,
        "shape_sentences": shape_items,
    }
    Path(a.out).write_text(json.dumps(doc, indent=1, ensure_ascii=False))
    print(f"wrote {a.out}: {len(items)} colour + {len(shape_items)} shape sentences; per category {doc['counts']}")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
