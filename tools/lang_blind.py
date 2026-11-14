#!/usr/bin/env python3
"""Generate and score a deterministic, repository-owned language corpus.

The generated JSONL is self describing: every row contains its seed, split,
template id, text and gold mapping.  The held-out templates are disjoint from
the development templates.  Scoring follows the task reward for each of the
nine colour/shape object types: +2 correct, -1 wrong bin, 0 abstain.

Examples:
    ../v16/bin/python tools/lang_blind.py --compare finals_r21 ab_langzip --pretty
    ../v16/bin/python tools/lang_blind.py -n 3000 --seed 17 --write /tmp/lang.jsonl
    ../v16/bin/python tools/lang_blind.py --corpus /tmp/lang.jsonl --overlay finals_r21
"""
from __future__ import annotations

import argparse
import itertools
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from aaltonomous.baselines.classical.config import load_config  # noqa: E402
from aaltonomous.baselines.classical.policy import ClassicalPolicy  # noqa: E402
from aaltonomous.interface.tasks import BINS, COLORS, SHAPES, TEMPLATES  # noqa: E402
from aaltonomous.scene.scene import resolve_bin  # noqa: E402

DEFAULT_COUNT = 3000
DEFAULT_SEED = 20261114
OBJECTS = tuple(itertools.product(COLORS, SHAPES))

# Every entry maps a surface form to the internal ontology.  Development and
# held-out words are separate so the split tests both structure and vocabulary.
WORDS = {
    "dev": {
        "red": ("red", "orange"), "yellow": ("yellow", "golden"), "blue": ("blue", "purple"),
        "cube": ("cube", "square block"), "cylinder": ("cylinder", "can"),
        "triangle": ("triangle", "prism"),
        "left": ("left", "leftmost"), "center": ("center", "middle"), "right": ("right", "rightmost"),
    },
    "heldout": {
        "red": ("tangerine", "oange"), "yellow": ("lemon", "yelo"), "blue": ("violet", "pruple"),
        "cube": ("cuboid", "cuubes"), "cylinder": ("tube", "cylindrs"),
        "triangle": ("wedge", "triagnle"),
        "left": ("lefthand", "leeft"), "center": ("central", "cente"), "right": ("righthand", "rigth"),
    },
}

POLITE = {
    "dev": (("", ""), ("Please, ", ""), ("Kindly ", ", please.")),
    "heldout": (("", ""), ("Could you please ", "?"), ("When you can, ", ". Thanks.")),
}

# Renderer names, not parser phrases.  Held-out renderers are never selected
# for development rows, which prevents sentence-template leakage across splits.
STYLES = {
    "dev": {
        "direct": ("direct_put", "direct_bins_for"),
        "zip": ("zip_go_respectively", "zip_put_order"),
        "directional": ("direction_ltr",),
        "colon": ("colon_bin_first", "colon_key_first"),
        "conditional": ("if_put",),
        "ordinal": ("ordinal_count",),
        "ambiguous": ("bad_zip_mismatch", "bad_colon_missing", "bad_cond_source",
                      "bad_zip_orientation", "bad_cond_bin_object"),
    },
    "heldout": {
        "direct": ("direct_move", "direct_use"),
        "zip": ("zip_belongs_respectively", "zip_bins_receive"),
        "directional": ("direction_reading",),
        "colon": ("colon_in_bin", "colon_multisentence"),
        "conditional": ("if_belongs", "if_send"),
        "ordinal": ("ordinal_number", "ordinal_starting"),
        "ambiguous": ("bad_zip_unknown", "bad_direction", "bad_ordinal",
                      "bad_direction_relation", "bad_modifier"),
    },
}
CATEGORIES = tuple(next(iter(STYLES.values())))


def _join(items: list[str]) -> str:
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def _surface(rng: random.Random, split: str, canonical: str) -> str:
    return rng.choice(WORDS[split][canonical])


def _keys(rng: random.Random, split: str, mode: str) -> tuple[list[str], list[str]]:
    if mode == "color":
        order = list(COLORS)
        rng.shuffle(order)
        return [_surface(rng, split, k) for k in order], order
    if mode == "shape":
        order = list(SHAPES)
        rng.shuffle(order)
        return [_surface(rng, split, k) for k in order], order
    colors, shapes = list(COLORS), list(SHAPES)
    rng.shuffle(colors)
    rng.shuffle(shapes)
    shown = [f"{_surface(rng, split, c)} {_surface(rng, split, s)}" for c, s in zip(colors, shapes)]
    return shown, [f"{c}|{s}" for c, s in zip(colors, shapes)]


def _bins(rng: random.Random, split: str) -> tuple[list[str], list[str]]:
    order = list(BINS)
    rng.shuffle(order)
    return [_surface(rng, split, b) for b in order], order


def _ordinal(bin_name: str, side: str, numeric: bool) -> str:
    walk = BINS if side == "left" else tuple(reversed(BINS))
    i = walk.index(bin_name)
    return (("1st", "2nd", "3rd") if numeric else ("first", "second", "third"))[i]


def _render(style: str, keys: list[str], bins: list[str], canonical_bins: list[str], mode: str,
            rng: random.Random) -> str:
    pairs = list(zip(keys, bins))
    if style == "direct_put":
        return "Put " + ", ".join(f"{k} in the {b} bin" for k, b in pairs) + "."
    if style == "direct_bins_for":
        return ", ".join(f"The {b} bin is for {k}" for k, b in pairs) + "."
    if style == "direct_move":
        return "Move " + ", then ".join(f"{k} to {b}" for k, b in pairs) + "."
    if style == "direct_use":
        return "; ".join(f"Use {b} for {k}" for k, b in pairs) + "."
    if style == "zip_go_respectively":
        return f"{_join(keys)} go to {_join(bins)}, respectively."
    if style == "zip_put_order":
        return f"Put {_join(keys)} into {_join(bins)}, in that order."
    if style == "zip_belongs_respectively":
        return f"{_join(keys)} belong in {_join(bins)}, respectively."
    if style == "zip_bins_receive":
        return f"{_join(bins)} receive {_join(keys)}, respectively."
    if style in ("direction_ltr", "direction_reading"):
        ordered = [keys[canonical_bins.index(b)] for b in BINS]
        head = "Left to right" if style == "direction_ltr" else "Reading from left to right"
        return f"{head}: {_join(ordered)}."
    if style == "colon_bin_first":
        return "; ".join(f"{b}: {k}" for k, b in pairs) + "."
    if style == "colon_key_first":
        return "; ".join(f"{k}: {b}" for k, b in pairs) + "."
    if style == "colon_in_bin":
        return "; ".join(f"In the {b} bin: {k}" for k, b in pairs) + "."
    if style == "colon_multisentence":
        return ". ".join(f"{b.title()} bin: {k}" for k, b in pairs) + "."
    if style == "if_put":
        return " ".join(f"If an object is {'a ' if mode != 'color' else ''}{k}, put it in the {b} bin."
                        for k, b in pairs)
    if style == "if_belongs":
        return " ".join(f"If it is {'a ' if mode != 'color' else ''}{k}, it belongs in {b}." for k, b in pairs)
    if style == "if_send":
        return " ".join(f"If you see {'a ' + k if mode != 'color' else k + ' object'}, send it to the {b} bin."
                        for k, b in pairs)
    if style in ("ordinal_count", "ordinal_number", "ordinal_starting"):
        side = rng.choice(("left", "right"))
        numeric = style != "ordinal_count"
        clauses = [f"{k} goes in the {_ordinal(b, side, numeric)} bin" for k, b in zip(keys, canonical_bins)]
        if style == "ordinal_count":
            head = f"Count the bins from the {side}."
        elif style == "ordinal_number":
            head = f"Number the bins starting from the {side}."
        else:
            head = f"Starting at the {side},"
        return head + " " + "; ".join(clauses) + "."
    if style == "bad_zip_mismatch":
        return f"{_join(keys)} go to {_join(bins[:2])}, respectively."
    if style == "bad_colon_missing":
        return f"{bins[0]}: {keys[0]}. {bins[1]}: {keys[1]}. {bins[2]}:"
    if style == "bad_cond_source":
        return f"If it is {keys[0]}, move it away from the {bins[0]} bin."
    if style == "bad_zip_unknown":
        return f"{keys[0]}, banana and {keys[1]} go to {bins[0]} and {bins[1]}, respectively."
    if style == "bad_direction":
        return f"Left to right: {keys[0]} stays put, {keys[1]} remains on the table, {keys[2]} goes nowhere."
    if style == "bad_ordinal":
        return f"Count the bins from the left. {keys[0]} avoids the first bin; {keys[1]} avoids the second bin."
    if style == "bad_zip_orientation":
        return f"{_join(bins)} go to {_join(keys)}, respectively."
    if style == "bad_cond_bin_object":
        return f"If it is {keys[0]}, move the {bins[0]} bin."
    if style == "bad_direction_relation":
        return f"Left to right: {keys[0]} goes to {keys[1]}, {keys[2]}."
    if style == "bad_modifier":
        return f"{bins[0]}: {keys[0]}. {bins[1]}: {keys[1]}. Actually reverse that."
    raise ValueError(style)


def generate(count: int = DEFAULT_COUNT, seed: int = DEFAULT_SEED) -> list[dict]:
    """Return exactly ``count`` deterministic generated cases."""
    if count < 0:
        raise ValueError("count must be >= 0")
    rng = random.Random(seed)
    rows = []
    splits = ("dev", "heldout", "heldout")
    modes = ("color", "shape", "both")
    for i in range(count):
        category = CATEGORIES[i % len(CATEGORIES)]
        split = splits[(i // len(CATEGORIES)) % len(splits)]
        mode = modes[(i // (len(CATEGORIES) * len(splits))) % len(modes)]
        style = rng.choice(STYLES[split][category])
        shown_keys, gold_keys = _keys(rng, split, mode)
        shown_bins, canonical_bins = _bins(rng, split)
        text = _render(style, shown_keys, shown_bins, canonical_bins, mode, rng)
        prefix, suffix = rng.choice(POLITE[split])
        if prefix or suffix:
            text = prefix + text.rstrip(".") + suffix
        rows.append({
            "case": i, "seed": seed, "split": split, "category": category,
            "template": style, "mode": mode, "text": text,
            "gold": {} if category == "ambiguous" else dict(zip(gold_keys, canonical_bins)),
        })
    return rows


def organiser_cases() -> list[dict]:
    """The eight published templates over all 36 colour/bin permutations."""
    rows = []
    for ti, template in enumerate(TEMPLATES):
        for colors in itertools.permutations(COLORS):
            for bins in itertools.permutations(BINS):
                words = {f"c{i}": c for i, c in enumerate(colors)}
                words.update({f"C{i}": c.title() for i, c in enumerate(colors)})
                words.update({f"b{i}": b for i, b in enumerate(bins)})
                rows.append({
                    "case": len(rows), "seed": None, "split": "organiser", "category": "organiser",
                    "template": f"organiser_{ti + 1}", "mode": "color", "text": template.format(**words),
                    "gold": dict(zip(colors, bins)),
                })
    return rows


def _decode(mapping: dict) -> dict:
    return {(tuple(k.split("|", 1)) if "|" in k else k): v for k, v in mapping.items()}


class _PolicyStub:
    _cap_text = ClassicalPolicy._cap_text

    def __init__(self, overlay: str):
        self.cfg = load_config(overlay=overlay)
        self.messages: list[str] = []
        self.log = self.messages.append

    def mapping(self, text: str) -> dict:
        return ClassicalPolicy._instruction_mapping(self, {"instruction": text})


def score(rows: list[dict], overlay: str) -> dict:
    policy = _PolicyStub(overlay)
    totals = defaultdict(lambda: {"cases": 0, "exact": 0, "wrong_bin": 0, "abstain": 0,
                                  "points": 0, "objects": 0})
    examples = []
    for row in rows:
        gold = _decode(row["gold"])
        pred = policy.mapping(row["text"])
        expected = [resolve_bin(gold, c, s) for c, s in OBJECTS]
        actual = [resolve_bin(pred, c, s) for c, s in OBJECTS]
        wrong = any(a is not None and a != g for g, a in zip(expected, actual))
        exact = expected == actual
        # Case outcomes are exclusive: any confident wrong bin outranks a partial abstain; otherwise a non-exact
        # result is an abstention. Gold-abstain cases parsed as no mapping are exact, not failures.
        abstain = not exact and not wrong
        points = sum(2 if g is not None and a == g else -1 if a is not None and a != g else 0
                     for g, a in zip(expected, actual))
        for key in dict.fromkeys(("all", row["category"], row["split"], row["template"])):
            t = totals[key]
            t["cases"] += 1
            t["exact"] += exact
            t["wrong_bin"] += wrong
            t["abstain"] += abstain
            t["points"] += points
            t["objects"] += len(OBJECTS)
        if not exact and len(examples) < 12:
            examples.append({"case": row["case"], "category": row["category"], "text": row["text"],
                             "gold": row["gold"], "pred": {str(k): v for k, v in pred.items()}})

    def finish(t: dict) -> dict:
        n = t["cases"] or 1
        return {"cases": t["cases"], "exact": t["exact"], "exact_rate": t["exact"] / n,
                "wrong_bin": t["wrong_bin"], "wrong_bin_rate": t["wrong_bin"] / n,
                "abstain": t["abstain"], "abstain_rate": t["abstain"] / n,
                "points_per_object": t["points"] / (t["objects"] or 1)}

    return {"overlay": overlay, "metrics": finish(totals["all"]),
            "groups": {k: finish(v) for k, v in sorted(totals.items()) if k != "all"},
            "examples": examples, "parser_messages": policy.messages[:5]}


def read_jsonl(paths: list[str]) -> list[dict]:
    rows = []
    for name in paths:
        with open(name, encoding="utf-8") as f:
            for line_no, line in enumerate(f, 1):
                if line.strip():
                    row = json.loads(line)
                    missing = {"text", "gold"} - set(row)
                    if missing:
                        raise ValueError(f"{name}:{line_no}: missing {sorted(missing)}")
                    row.setdefault("case", len(rows))
                    row.setdefault("seed", None)
                    row.setdefault("split", "external")
                    row.setdefault("category", "external")
                    row.setdefault("template", "external")
                    rows.append(row)
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("-n", "--count", type=int, default=DEFAULT_COUNT)
    ap.add_argument("--seed", type=int, default=DEFAULT_SEED)
    ap.add_argument("--split", choices=("all", "dev", "heldout"), default="all")
    ap.add_argument("--corpus", action="append", default=[], help="score JSONL instead of generating cases")
    ap.add_argument("--write", help="write the selected generated/read corpus as JSONL")
    ap.add_argument("--overlay", action="append", help="overlay to score (repeatable)")
    ap.add_argument("--compare", nargs=2, metavar=("BEFORE", "AFTER"))
    ap.add_argument("--organiser", action="store_true", help="score only the 8 x 36 published-template cases")
    ap.add_argument("--pretty", action="store_true")
    args = ap.parse_args(argv)

    rows = organiser_cases() if args.organiser else (read_jsonl(args.corpus) if args.corpus else generate(args.count, args.seed))
    if args.split != "all":
        rows = [r for r in rows if r["split"] == args.split]
    if args.write:
        with open(args.write, "w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, sort_keys=True) + "\n")
    overlays = list(args.compare or args.overlay or ["ab_langzip"])
    report = {"seed": args.seed, "cases": len(rows), "split": args.split,
              "source": "organiser" if args.organiser else ("jsonl" if args.corpus else "generated"),
              "results": [score(rows, overlay) for overlay in overlays]}
    print(json.dumps(report, indent=2 if args.pretty else None, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
