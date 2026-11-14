"""Pre-resolver normaliser: maps near-miss tokens onto the resolver's closed vocabulary.

Token-wise and conservative: a token already in the resolver's vocabulary is never changed, a token
near two different canonical meanings is left alone (abstain costs 0, a wrong key costs -1), and
synonyms are a small table of ordinary English words (no sentence from any held-out set).
Does not touch policy/instruction.py (frozen); it only reads its vocabulary tables.
"""
from __future__ import annotations

import functools
import re

from . import instruction as _ins

# Closed vocabulary, read from the resolver: word -> canonical meaning.
VOCAB: dict[str, str] = {**_ins._BIN_WORDS, **_ins._COLOR_WORDS, **_ins._SHAPE_WORDS}
# Words that must never be rewritten at all: the resolver's own vocabulary plus its real-English guard
# list (words that merely sit near a target, e.g. light/night vs right, fellow vs yellow).
PROTECTED: frozenset[str] = frozenset(_ins._KNOWN) | _ins._NEAR_REAL | frozenset({
    "cute", "cure", "cures", "cub", "cubs", "cubed", "cubby", "cue", "cut", "cup", "cups", "cab", "lube",
    "rube", "read", "reed", "lift", "lent", "leaf", "lead", "rig", "rid", "ride", "wedge", "wedgy",
    "violent", "violin", "violets", "people", "purpose", "orang", "golfer", "holden", "olden",
})
# Words a typo may be corrected TO. Deliberately a subset of VOCAB: short or common words (red, mid,
# port, gold, round, disc...) have too many real-word neighbours, so they are matched exactly only.
# ponytail: "red" is not fuzzy-matched (read/bed/rod/reed collide); add if a guard list covers it.
# Real finals object colours are purple / yellow / orange: the resolver aliases purple/violet -> blue and
# orange -> red, so their typos are corrected to those words (VOCAB below gives them their alias meaning).
VOCAB.update({"purple": "blue", "violet": "blue", "orange": "red"})
FUZZY_TARGETS: tuple[str, ...] = (
    "left", "right", "center", "centre", "middle", "yellow", "blue", "purple", "violet", "orange", "golden",
    "cube", "cubes", "cylinder", "cylinders", "triangle", "triangles",
    "prism", "prisms", "wedge", "wedges",
)
# Ordinary synonyms -> a canonical word. NB "blocks" / "box" are absent on purpose (see instruction.py:
# they are generic nouns for objects/bins) and "can" is a modal verb, so only plural "cans" is listed.
SYNONYMS: dict[str, str] = {
    # cube
    "die": "cube", "dice": "cube", "brick": "cube", "bricks": "cubes",
    # cylinder
    "cans": "cylinders", "tin": "cylinder", "tins": "cylinders", "drum": "cylinder", "drums": "cylinders",
    "barrel": "cylinder", "barrels": "cylinders", "canister": "cylinder", "canisters": "cylinders",
    "pillar": "cylinder", "pillars": "cylinders",
    # triangle (triangular prism)
    "pyramid": "triangle", "pyramids": "triangles", "tetrahedron": "triangle", "tetrahedra": "triangles",
    # containers -> bin(s)
    "receptacle": "bin", "receptacles": "bins", "bucket": "bin", "buckets": "bins",
    "basket": "bin", "baskets": "bins", "pail": "bin", "pails": "bins", "vessel": "bin", "vessels": "bins",
    "compartment": "bin", "compartments": "bins", "carton": "bin", "cartons": "bins",
}
_WORD = re.compile(r"[A-Za-z]+")


def _dist(a: str, b: str) -> int:
    """Edit distance: Levenshtein plus adjacent transposition (optimal string alignment)."""
    d = [[i + j if not (i and j) else 0 for j in range(len(b) + 1)] for i in range(len(a) + 1)]
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1, d[i - 1][j - 1] + (a[i - 1] != b[j - 1]))
            if i > 1 and j > 1 and a[i - 1] == b[j - 2] and a[i - 2] == b[j - 1]:
                d[i][j] = min(d[i][j], d[i - 2][j - 2] + 1)
    return d[-1][-1]


def _collapse(w: str) -> str:
    return re.sub(r"(.)\1+", r"\1", w)


def _skeleton(w: str) -> str:
    return w[0] + re.sub(r"[aeiou]", "", w[1:])


def _limit(n: int) -> int:
    return 0 if n < 3 else 1 if n <= 5 else 2


# ponytail: collapsed forms precomputed once; a target not in this table (e.g. a selfcheck injection) is collapsed on the fly.
_COLLAPSED: dict[str, str] = {t: _collapse(t) for t in FUZZY_TARGETS}


@functools.lru_cache(maxsize=4096)
def _fuzzy(tok: str) -> str | None:
    """Unique canonical target for a near-miss token, else None (unknown or ambiguous)."""
    lim = _limit(len(tok))
    hits: dict[str, int] = {}
    for t in FUZZY_TARGETS:
        ct = _COLLAPSED.get(t) or _collapse(t)
        forms = {t, ct}
        # first letter is almost never the typo, and requiring it keeps real words (below/yellow) out
        # edit distance >= length difference: skip forms that cannot be within lim (a 12k-char token costs nothing)
        d = min((_dist(tok, f) for f in forms if abs(len(f) - len(tok)) <= lim), default=9) if tok[0] == t[0] else 9
        if len(tok) >= 3 and tok in (_skeleton(t), _skeleton(ct)) and tok != t:
            d = 0 if d > lim else d  # vowel-dropped form (cntr, rght, lft, ylw)
        if d <= lim and lim:
            hits[t] = d
    if len({VOCAB[t] for t in hits}) != 1:
        return None  # nothing near, or two different meanings: abstain
    best = min(hits.values())
    return max((t for t, d in hits.items() if d == best), key=lambda t: t[-1] == tok[-1])


def _fix(m: re.Match[str]) -> str:
    raw = m.group(0)
    tok = raw.lower()
    if tok in VOCAB or tok in PROTECTED:
        return raw
    new = SYNONYMS.get(tok) or _fuzzy(tok)
    if new is None:
        return raw
    return new.capitalize() if raw[0].isupper() else new


# Colour-named bins ("the yellow box", "the orange basket"): bins are only left / centre / right, so a colour-named
# bin is never a position, and the resolver reads its colour word as an OBJECT colour (red -> yellow box became
# {red, yellow, blue: left}). A clause that sends objects to a colour-named bin is dropped: those objects stay on
# the counter (0) instead of a wrong bin (-1). Only a clear destination counts ("to/in/into ... the yellow box", or a
# clause with two colour words), so "put the blue box in the left bin" (box = an object) is kept.
_C = r"(?:red|orange|yellow|gold(?:en)?|blue|purple|violet|green|pink|white|black)"
_B = r"(?:bins?|box(?:es)?|baskets?|containers?|buckets?|crates?|tubs?|trays?|receptacles?)"
_CB = re.compile(rf"\b{_C}(?:\s+colou?red)?\s+{_B}\b", re.I)
_TO_CB = re.compile(rf"\b(?:in|into|to|inside|onto)\s+(?:the\s+|a\s+)?{_C}(?:\s+colou?red)?\s+{_B}\b", re.I)
_COLOUR = re.compile(rf"\b{_C}\b", re.I)
_SPLIT = re.compile(r"\s*(?:[;,.:]|\band\b|\bwhile\b|\bbut\b)\s*", re.I)


# A position word names the bin even when it is colour-named ("the orange box on the left": the real bins ARE
# orange), so such a clause is kept for the resolver.
_POS = re.compile(r"\b(?:left|right|centre|center|middle|leftmost|rightmost)\b", re.I)


def _colour_bin_clause(clause: str) -> bool:
    if not _CB.search(clause) or _POS.search(clause):
        return False
    return bool(_TO_CB.search(clause)) or len({c.lower() for c in _COLOUR.findall(clause)}) >= 2


def drop_colour_bin_clauses(sentence: str) -> str:
    """Remove every clause that sends objects to a colour-named bin; unchanged when there is none."""
    parts = [p for p in _SPLIT.split(sentence) if p]
    kept = [p for p in parts if not _colour_bin_clause(p)]
    if len(kept) == len(parts) or not kept:   # nothing left = the resolver's FIXED fallback: never worse than raw
        return sentence
    return "; ".join(kept)


# Ordinal bins counted from one side ("the third bin (left to right)", "second box from the right"): the resolver
# would read the side word as the bin. Rewritten to the position; a bare ordinal without a side is left alone.
_ORD = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2, "last": 2}
_ORD_RE = re.compile(
    rf"\b(first|1st|second|2nd|third|3rd|last)(?:\s+(?:one|{_B}))?\s*"
    r"(?:\(\s*)?(?:counting\s+|when\s+counting\s+)?(?:(?:from\s+(?:the\s+)?|starting\s+(?:from|at)\s+the\s+)(left|right)"
    r"|(left|right)\s+to\s+(?:the\s+)?(?:left|right))(?:\s+side)?(?:\s*\))?", re.I)
_SIDE = ("left", "middle", "right")


def _ord_sub(m: re.Match[str]) -> str:
    i, side = _ORD[m.group(1).lower()], (m.group(2) or m.group(3)).lower()
    return f"{_SIDE[i] if side == 'left' else _SIDE[2 - i]} bin"


def rewrite_ordinal_bins(sentence: str) -> str:
    return _ORD_RE.sub(_ord_sub, sentence)


# Colour names for the real finals objects (purple / yellow / orange) the resolver does not know, incl. Finnish and
# Czech (any inflection, with or without diacritics). Applied before the token pass; "amber" is ORANGE here (the
# resolver alone reads it as yellow) because the real objects have no amber-yellow, only yellow and orange.
_COLOUR_NAMES = [
    (re.compile(r"\b(?:lavender|mauve|plum|amethyst|violet(?:t|i)\w*|purppur\w*|liila\w*|fialov\w*)", re.I), "purple"),
    (re.compile(r"\b(?:tangerine|apricot|peach|amber|oranssi\w*|oran[žz]ov\w*)", re.I), "orange"),
    (re.compile(r"\b(?:lemon|canary|keltai\w*|[žz]lut\w*)", re.I), "yellow"),
]


def name_colours(sentence: str) -> str:
    for rx, word in _COLOUR_NAMES:
        sentence = rx.sub(lambda m: word.capitalize() if m.group(0)[0].isupper() else word, sentence)
    return sentence


def normalize(sentence: str) -> str:
    """Map colour names and near-miss tokens onto the resolver's vocabulary, then drop colour-named-bin clauses."""
    if sentence is None:
        return ""
    return drop_colour_bin_clauses(rewrite_ordinal_bins(_WORD.sub(_fix, name_colours(sentence))))


def _selfcheck() -> None:
    cases = {"lfet": "left", "rgiht": "right", "rigth": "right", "rght": "right", "cente": "center",
             "cntr": "center", "midlle": "middle", "centur": "center", "yelo": "yellow", "ylw": "yellow",
             "cuubes": "cubes", "cylindrs": "cylinders", "triagnle": "triangle", "prissms": "prisms",
             "cubbes": "cubes", "cans": "cylinders", "tubes": "tubes", "wedges": "wedges",
             "receptacle": "bin"}
    for bad, good in cases.items():
        got = normalize(bad)
        assert got == good or VOCAB.get(got) == VOCAB[good] == "center", (bad, got, good)  # center/centre are one meaning
    s = "All cans should go into the right receptacle, blocks into the left receptacle, and wedges into the middle receptacle."
    assert normalize(s) == "All cylinders should go into the right bin, blocks into the left bin, and wedges into the middle bin."
    # idempotence
    for t in (s, "put the yelo cuubes in teh lfet bin", "Rgiht, cntr; ylw!"):
        assert normalize(normalize(t)) == normalize(t), t
    # correct sentences come back byte-identical
    ok = ["Put all red cubes in the left bin, yellow cylinders in the center bin, and blue triangles in the right bin.",
          "Everything goes to the centre bin except the blue ones -- those go right.",
          "  Red => left;  yellow -> middle,\tblue -> rightmost\n", "That is the right answer, not the light one.",
          "Sort the blocks: cube-shaped to the left box, round ones to the right."]
    for t in ok:
        assert normalize(t) == t, (t, normalize(t))
    # ambiguity abstains. No two real targets with different meanings share a first letter and sit one edit
    # apart, so inject one: "lefz" is 1 edit from left (left) and from the fake "lefy" (red).
    global FUZZY_TARGETS
    saved = FUZZY_TARGETS
    assert normalize("lefz") == "left"
    FUZZY_TARGETS, VOCAB["lefy"] = saved + ("lefy",), "red"
    _fuzzy.cache_clear()   # the lookup is cached; the injected target changes its answer
    try:
        assert normalize("lefz") == "lefz"
    finally:
        FUZZY_TARGETS = saved
        del VOCAB["lefy"]
        _fuzzy.cache_clear()
    # real-English neighbours and "right" meaning correct are not mangled; nothing non-near-miss becomes a bin word
    assert normalize("blur light fight night") == "blur light fight night"
    assert normalize("it is right, so read the rite and write") == "it is right, so read the rite and write"
    assert normalize("Is that correct? Yes, all right.") == "Is that correct? Yes, all right."
    for w in ("correct", "write", "rite", "ride", "ligt", "lift", "bring", "rug", "mud", "side", "port"):
        assert normalize(w) in (w,), (w, normalize(w))
    print("normalize selfcheck OK")


if __name__ == "__main__":  # python -m aaltonomous.baselines.classical.normalize
    _selfcheck()
