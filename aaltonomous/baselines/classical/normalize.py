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
# Ordinary synonyms -> a canonical word. NB "blocks" is absent on purpose (instruction.py: a generic noun for all three
# solids; only "square block(s)" / "round block(s)" are rewritten, see _block_phrases). "box" is a bin word too, so it is
# decided by context (_box_objects). "can" / "roll" are a modal and a verb, so the singular is decided by context
# (_noun_shapes). Rod/pipe/roof/ridge can describe scene fixtures, so they need positive placement syntax too
# (_ambiguous_shape_objects) instead of being rewritten token-by-token.
SYNONYMS: dict[str, str] = {
    # cube
    "die": "cube", "dice": "cube", "brick": "cube", "bricks": "cubes",
    # cylinder
    "cans": "cylinders", "tin": "cylinder", "tins": "cylinders", "drum": "cylinder", "drums": "cylinders",
    "barrel": "cylinder", "barrels": "cylinders", "canister": "cylinder", "canisters": "cylinders",
    "pillar": "cylinder", "pillars": "cylinders", "column": "cylinder", "columns": "cylinders",
    "rolls": "cylinders",
    # triangle (triangular prism)
    "pyramid": "triangle", "pyramids": "triangles", "tetrahedron": "triangle", "tetrahedra": "triangles",
    "prismatic": "triangle",
    # cube (cuboid is the resolver's word for the same solid)
    "hexahedron": "cube", "hexahedra": "cubes", "hexahedrons": "cubes", "cuboids": "cubes",
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
    if tok in VOCAB or (tok in PROTECTED and tok not in SYNONYMS):   # an explicit synonym beats the guard list ("rods")
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


def _bin_for(word: str, side: str) -> str:
    i = _ORD[word.lower()]
    return f"{_SIDE[i] if side == 'left' else _SIDE[2 - i]} bin"


def _ord_sub(m: re.Match[str]) -> str:
    return _bin_for(m.group(1), (m.group(2) or m.group(3)).lower())


# A bare ordinal ("..., yellow cylinders in the second, and blue triangles in the first"): after a place word and
# followed by a clause end. "the first red object" is not a bin and is left alone.
_BARE_ORD = re.compile(
    r"(\b(?:the|in|into|to|onto|inside|on)\s+)(first|1st|second|2nd|third|3rd|last)\b"
    rf"(?:\s+(?:one|{_B}))?(?=\s*(?:[,;.:!?)]|\band\b|\bor\b|\bwhile\b|\bbut\b|$))", re.I)


def rewrite_ordinal_bins(sentence: str) -> str:
    """Anchored ordinal bins -> position words. Bare ordinals in the same sentence take the same anchor direction.
    Anchors pointing both ways: from the left the sentence is left to the resolver; from the right only the anchored
    ones are rewritten."""
    sides = [(m.group(2) or m.group(3)).lower() for m in _ORD_RE.finditer(sentence)]
    if not sides:
        return sentence
    out = _ORD_RE.sub(_ord_sub, sentence)
    if len(set(sides)) > 1:
        return sentence if sides[0] == "left" else out
    return _BARE_ORD.sub(lambda m: m.group(1) + _bin_for(m.group(2), sides[0]), out)


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


_REGARDLESS = re.compile(r"\bwhatever\s+their\s+colou?r\b|\bregardless\s+of\s+(?:their\s+)?colou?r\b"
                         r"|\bno\s+matter\s+(?:the|their)\s+colou?r\b", re.I)
# Finnish / Czech words the resolver cannot read. A word already in a table (VOCAB, SYNONYMS, PROTECTED) is never
# rewritten, so an English word cannot change here.
_FOREIGN = tuple((re.compile(p, re.I), w) for p, w in (
    (r"\bkuutio\w*|\bkostk\w*", "cube"),
    (r"\bsylinter\w*|\bv[áa]lc\w*|\bv[áa]lec\b", "cylinder"),
    (r"\bkolmi\w*|\bprism\w*|\bhranol\w*", "triangle"),
    (r"\bvasemma\w*|\bvasen\w*|\bvlev\w*", "left"),
    (r"\boikea\w*|\bvprav\w*", "right"),
    (r"\bkeski\w*|\bkeske\w*|\bdoprost\w*|\buprost\w*|\bst[řr]ed\w*", "middle"),
    (r"\blaatikk\w*|\blokero\w*|\bkrabic\w*|\bp[řr]ihr[áa]d\w*|\bkoš\w*|\bkos\b", "bin"),
))


def name_foreign(sentence: str) -> str:
    def sub(word: str):
        def f(m: re.Match[str]) -> str:
            tok = m.group(0)
            if tok.lower() in VOCAB or tok.lower() in PROTECTED or tok.lower() in SYNONYMS:
                return tok
            return word.capitalize() if tok[0].isupper() else word
        return f
    for rx, word in _FOREIGN:
        sentence = rx.sub(sub(word), sentence)
    return sentence


# Ambiguous scene nouns are shapes only inside an actionable object placement. In particular, relational descriptions
# ("the red roof is above the left bin", "the red rod points to the left bin") must remain unknown so strict mode cannot
# manufacture a sortable compound key from them.
_AMBIG_SHAPE = {
    "rod": "cylinder", "rods": "cylinders", "pipe": "cylinder", "pipes": "cylinders",
    "roof": "triangle", "roofs": "triangles", "ridge": "triangle", "ridges": "triangles",
}
_AMBIG_SHAPE_RE = re.compile(r"\b(?:" + "|".join(_AMBIG_SHAPE) + r")\b", re.I)
_PLACE_VERBS = frozenset("put move place drop send sort carry take".split())
_FLOW_VERBS = frozenset("go goes belong belongs".split())
_PLACE_PREPS = frozenset("in into to onto inside".split())


def _clause_words(sentence: str, start: int, end: int) -> tuple[list[str], list[str]]:
    """Words before/after one token inside its punctuation-delimited clause."""
    lo = max((sentence.rfind(c, 0, start) for c in ".;:!?\n"), default=-1) + 1
    hi = min((p for c in ".;:!?\n" if (p := sentence.find(c, end)) >= 0), default=len(sentence))
    return _WORD.findall(sentence[lo:start].lower()), _WORD.findall(sentence[end:hi].lower())


def _positive_placement(sentence: str, start: int, end: int) -> bool:
    """True only when this noun participates in a clear object -> positional-bin command."""
    before, after = _clause_words(sentence, start, end)
    positions = set(_ins._BIN_WORDS)
    if not positions.intersection(after):
        return False
    # "red rods (should) go left" / "yellow roofs belong in the right bin"
    if _FLOW_VERBS.intersection(after[:3]):
        return True
    # "put/move the red rod into the left bin". Requiring a destination preposition after the noun keeps
    # descriptive uses such as "the rod points to the left bin" out: "points" is not a placement verb.
    return bool(_PLACE_VERBS.intersection(before) and _PLACE_PREPS.intersection(after))


def _ambiguous_shape_objects(sentence: str) -> str:
    def sub(m: re.Match[str]) -> str:
        if not _positive_placement(m.string, m.start(), m.end()):
            return m.group(0)
        new = _AMBIG_SHAPE[m.group(0).lower()]
        return new.capitalize() if m.group(0)[0].isupper() else new
    return _AMBIG_SHAPE_RE.sub(sub, sentence)


# Singular "can" / "roll" is a cylinder only as a noun: after a determiner or at a clause start, with an optional colour
# word ("the red can", "Red roll goes left"). "you can put" and "Blue ones roll to the left" stay untouched.
_NOUN_SHAPE = re.compile(
    r"(^|[.;,:!?]\s*|\b(?:the|a|an|each|every|this|that|these|those|one)\s+)"
    r"((?:(?:red|orange|yellow|purple|violet|blue|lilac|green|pink|white|black|gold(?:en)?)\s+)?)(can|roll)\b", re.I)


def _noun_shapes(sentence: str) -> str:
    def sub(m: re.Match[str]) -> str:
        # A bare clause-leading "Can" is a modal/question, not a cylinder. Clause-leading colour+noun ("Red roll")
        # remains valid because group 2 is non-empty.
        if m.group(3).lower() == "can" and not m.group(2) and not m.group(1).strip():
            return m.group(0)
        return m.group(1) + m.group(2) + ("Cylinder" if m.group(3)[0].isupper() else "cylinder")
    return _NOUN_SHAPE.sub(sub, sentence)


# "square block" / "round block" name the cube / cylinder. A bare "block" stays generic (all three solids).
_SHAPE_BLOCK = re.compile(r"\b(square|round)\s+blocks?\b", re.I)
_BLOCK_TO = {"square": "cube", "round": "cylinder"}


def _block_phrases(sentence: str) -> str:
    def sub(m: re.Match[str]) -> str:
        new = _BLOCK_TO[m.group(1).lower()] + ("s" if m.group(0).lower().endswith("s") else "")
        return new.capitalize() if m.group(1)[0].isupper() else new
    return _SHAPE_BLOCK.sub(sub, sentence)


# "box" as an OBJECT (a cube) versus a BIN. Rewriting is opt-in: the noun needs a colour (or plural object form) and a
# positive placement whose destination is a separate positional bin. This keeps both copular bin-description orders
# ("the box is orange", "the left one is the orange box") and box A/B/C as bins.
_BOX = re.compile(r"\b(boxes|box)\b", re.I)
_OBJECT_COLOURS = frozenset((*_ins._COLOR_WORDS, "orange", "purple", "violet", "lilac"))


def _is_object_box(m: re.Match[str]) -> bool:
    before, after = _clause_words(m.string, m.start(), m.end())
    if after[:1] and after[0] in ("a", "b", "c"):
        return False
    object_noun = m.group(0).lower() == "boxes" or bool(_OBJECT_COLOURS.intersection(before[-3:]))
    return object_noun and _positive_placement(m.string, m.start(), m.end())


def _box_objects(sentence: str) -> str:
    def sub(m: re.Match[str]) -> str:
        if not _is_object_box(m):
            return m.group(0)
        new = "cubes" if m.group(0).lower() == "boxes" else "cube"
        return new.capitalize() if m.group(0)[0].isupper() else new
    return _BOX.sub(sub, sentence)


_LIFT_LR = re.compile(r"\(\s*lift\s+to\s+(right|left)\s*\)", re.I)   # "first bin (lift to right)": a left/right typo


def normalize(sentence: str) -> str:
    """Map colour names and near-miss tokens onto the resolver's vocabulary, then drop colour-named-bin clauses."""
    if sentence is None:
        return ""
    sentence = _LIFT_LR.sub(lambda m: f"(left to {m.group(1)})", sentence)
    sentence = _REGARDLESS.sub(lambda m: "Regardless of colour" if m.group(0)[0].isupper() else "regardless of colour", sentence)
    sentence = _ambiguous_shape_objects(_noun_shapes(_block_phrases(name_foreign(name_colours(sentence)))))
    return _box_objects(drop_colour_bin_clauses(rewrite_ordinal_bins(_WORD.sub(_fix, sentence))))


# Strict mode (instruction.strict): the mapping never guesses, so this filters the permissive resolver's output.
# Every check only turns a key into an abstain (None) or drops it; abstain scores 0, a wrong bin scores -1.
#   (a) the resolver parsed nothing, or a swap re-maps the whole sentence -> abstain on all (never the FIXED fallback);
#   (b) a clause the resolver cannot read reliably (negation, condition, table placement, relation, quantity) abstains
#       every key it names, and "default" is dropped;
#   (c) a key whose colour/shape word is nowhere in the text is dropped (no inference by elimination); a colour-named
#       bin ("the purple box") is a bin, not an object colour, so it names no key. "default" needs a catch-all phrase;
#   (d) a key keeps its bin only if every clause that names it states that same bin. A bin-less clause borrows the bin
#       of the next one in two shapes only: a list ("Rules: red: left") and a bare key joined by "and" ("Red and yellow
#       go left"). The resolver's completion and clause-inheritance guesses name a bin no clause gives.
#   Beyond the spec (found by the stress/novel residuals): (e) a bare colour key whose clause names a shape the mapping
#       does not bind to that colour ("orange prism" under "red") is an over-generalisation; (f) a named object the
#       resolver did not bind blocks the fall-through to "default"; "lift to right" in a bin list is "left to right".
_CLAUSE_SEP = re.compile(r"(\s*(?:[;,.:!?\n]|\band\b|\bwhile\b|\bbut\b)\s*)", re.I)
_TRIGGERS = tuple(re.compile(p, re.I) for p in (
    r"\b(?:not|never|no|none|nor|neither|nothing|without|except|excluding|other\s+than|apart\s+from)\b|n't\b",
    r"\b(?:if|unless|otherwise|in\s+case|when\s+there)\b",
    r"\b(?:on\s+the\s+table|leave|in\s+front\s+of|behind|beside|next\s+to|near|between|on\s+top\s+of|outside)\b",
    r"\b(?:closest|nearest|farthest|furthest|biggest|largest|smallest|tallest|left\s+of|right\s+of)\b",
    r"^\s*(?:can|could|would|should|may|might|will|do|does|did|is|are)\s+(?:the\s+)?"
    r"(?:red|orange|yellow|purple|violet|blue|cube|cylinder|triangle|rod|pipe|roof|ridge)\b",
    r"\b(?:back\s+to)\b|\bfrom\s+the\s+(?:\w+\s+){0,2}(?:bins?|box(?:es)?|containers?|slots?|trays?|crates?|baskets?)\b",
    r"\b(?:one|two|three|four|five|\d+)\s+of\b|\b(?:only|just)\s+(?:one|two|three)\b|\ba\s+single\b|\bhalf\b"
    r"|\ball\s+but\b|\bfirst\s+(?:two|three)\b|\b(?:exactly|at\s+(?:most|least)|a\s+few|some\s+of|several)\b",
))
_SWAP = re.compile(r"\b(?:swap|switch|exchange|instead)\b", re.I)
_BINCOL = re.compile(r"\b(?:red|orange|yellow|gold(?:en)?|blue|purple|violet|lilac|magenta|green|pink|white|black)"
                     r"\s+(?:bins?|box(?:es)?|containers?|baskets?|crates?|tubs?|trays?|receptacles?|buckets?)\b", re.I)
_CATCH_ALL = re.compile(r"\b(?:everything\s+else|anything\s+else|the\s+rest|remaining|all\s+(?:the\s+)?others?|"
                        r"all\s+other|any\s+others?)\b(?!\s+(?:\w+\s+)?(?:colou?rs?|shapes?|cubes?|cylinders?|triangles?|"
                        r"prisms?|red|yellow|blue|purple|violet|orange|lilac|gold\w*|cube|cylinder|triangle)\b)", re.I)
_BLOCKS = re.compile(r"\bblocks?\b", re.I)
_AMBIG_RELATION = re.compile(r"\b(?:above|over|with|points?\s+to)\b", re.I)
_AMBIG_NOUN_IN = re.compile(r"\b(?:rods?|pipes?|roofs?|ridges?)\b", re.I)
_MULTI_DEST = re.compile(
    r"\bboth\s+(?:the\s+)?(?:left|right|center|centre|middle)\s+(?:bins?|box(?:es)?|containers?|baskets?|"
    r"crates?|tubs?|trays?|receptacles?|buckets?)\b[^.;!?\n]{0,80}\band\s+(?:the\s+)?"
    r"(?:left|right|center|centre|middle)\s+(?:bins?|box(?:es)?|containers?|baskets?|crates?|tubs?|trays?|"
    r"receptacles?|buckets?)\b", re.I)


def _strict_keys(text: str) -> tuple[set[str], set[str]]:
    """(colours, shapes) the resolver reads as OBJECTS in text (purple/violet -> blue, orange -> red); bins are not objects."""
    toks = _ins._tokens(_ins._colour_aliases(_BINCOL.sub(" bin ", text).lower()))
    return ({_ins._COLOR_WORDS[t] for t in toks if t in _ins._COLOR_WORDS},
            {_ins._SHAPE_WORDS[t] for t in toks if t in _ins._SHAPE_WORDS})


_FILLER = frozenset("and or the all every ones one too also go goes should must be placed put".split())


def _bins_in(text: str, ord_ok: bool = False) -> set[str]:
    """Bins a piece states: position words, and ordinals the resolver itself would read as bins (ord_ok: the sentence
    enumerates bins by ordinal, or this piece holds a lone ordinal bound to a bin noun)."""
    toks = _ins._tokens(text)
    bins = {_ins._BIN_WORDS[t] for t in toks if t in _ins._BIN_WORDS}
    if ord_ok or _ins._lone_ordinal(toks):
        bins |= {_ins._ORDINAL_BINS[t] for t in toks if t in _ins._ORDINAL_BINS and t not in ("a", "b", "c")}
    return bins


def _nonkey_words(text: str) -> int:
    """Words that are neither a colour, a shape, a bin word nor filler: a bare key clause ("Red", "lilac ones too") has 0."""
    return sum(1 for t in _ins._tokens(text) if t not in _ins._COLOR_WORDS and t not in _ins._SHAPE_WORDS
               and t not in _ins._BIN_WORDS and t not in _FILLER)


def _triggered(clause: str) -> bool:
    return (any(rx.search(clause) for rx in _TRIGGERS)
            or bool(_AMBIG_NOUN_IN.search(clause) and _AMBIG_RELATION.search(clause)))


def _cross_unbound(k, out: dict, named) -> bool:
    """A bare colour key whose clause also names a shape the mapping does not bind to that colour (or the reverse) is an
    over-generalisation: "orange prism" under "red" would also send orange cubes. Shape-only keys cover their colours."""
    for cs, _, _ in named:
        cols, shps = cs
        if isinstance(k, tuple):
            continue
        if k in _ins.COLORS:
            if any((k, s) not in out and s not in out for s in shps):
                return True
        else:
            if any((c, k) not in out and c not in out for c in cols):
                return True
    return False


def _names(k, cols: set[str], shps: set[str]) -> bool:
    """True if the key k (colour, shape or (colour, shape)) is named by a text with these colours and shapes."""
    if isinstance(k, tuple):
        return k[0] in cols and k[1] in shps
    return k in (cols if k in _ins.COLORS else shps)


# instruction.structured: small, deterministic grammars for parallel lists.  This is separate from the
# permissive resolver and the conservative strict filter: a recognised structure returns either a complete,
# conflict-free mapping or a truthy whole-instruction abstain.  None means "not this grammar" and preserves the
# old strict path byte-for-byte when the flag is off or the sentence is an ordinary clause instruction.
_STRUCT_MARKER = re.compile(r"\b(?:respectively|in\s+that\s+order)\b", re.I)
_STRUCT_POLITE_HEAD = re.compile(r"^\s*(?:(?:please|kindly)\s*,?\s*|could\s+you\s+please\s+|when\s+you\s+can\s*,?\s*)", re.I)
_STRUCT_POLITE_TAIL = re.compile(r"(?:\s*,?\s*please|[.!?]?\s*thanks)\s*[.!?]*\s*$", re.I)
_STRUCT_BAD = re.compile(
    r"\b(?:not|never|no|none|unless|except|excluding|without|swap|switch|exchange|otherwise|either|neither|both)\b|n't\b",
    re.I,
)
_STRUCT_ASSIGN_BAD = re.compile(
    r"\b(?:avoid|avoids|away|beside|near|outside|out|stay|stays|remain|remains|table|counter|nowhere|"
    r"forbid|forbids|forbidden|leave|leaves|skip|skips|from)\b", re.I,
)
_STRUCT_ZIP = tuple((re.compile(p, re.I), order) for p, order in (
    (r"^(?:put|place|move|send|sort|drop)\s+(?P<a>.+?)\s+(?:in|into|to|onto)\s+(?P<b>.+)$", "keys_bins"),
    (r"^(?P<a>.+?)\s+(?:go|goes|belong|belongs)(?:\s+(?:in|into|to|onto))?\s+(?P<b>.+)$", "keys_bins"),
    (r"^(?P<a>.+?)\s+(?:receive|receives|take|takes|hold|holds|are\s+for)\s+(?P<b>.+)$", "bins_keys"),
))
_STRUCT_DIRECTION = re.compile(
    r"^(?:(?:reading|listed|ordered)\s+)?(?:from\s+)?(left|right)(?:most|hand)?\s*(?:-|to)\s*"
    r"(?:the\s+)?(left|right)(?:most|hand)?\s*:\s*(.+)$", re.I,
)
_STRUCT_ORD_HEAD = re.compile(
    r"^(?:(?:(?:count|number)\s+the\s+(?:bins|boxes|containers)\s+"
    r"(?:(?:starting|beginning)\s+)?from)|(?:(?:starting|beginning)\s+at)|(?:counting\s+from))\s+"
    r"(?:the\s+)?(left|right)(?:most|hand)?\s*[.,:]?\s*(.+)$", re.I,
)
_STRUCT_ORD_BIN = re.compile(r"\b(first|1st|second|2nd|third|3rd)\s+(?:bin|box|container|one)\b", re.I)
_STRUCT_PLACE = re.compile(r"\b(?:put|place|send|move|drop|sort|go|goes|belong|belongs|use|take)\b", re.I)
_STRUCT_HEADER = re.compile(r"^(?:rules?|sorting|assignments?|destinations?|as\s+follows)\s*:?$", re.I)
_STRUCT_FILLER = frozenset(
    "a an the all every each object objects item items piece pieces thing things one ones block blocks bin bins box "
    "boxes container containers slot slots tray trays crate crates basket baskets receptacle receptacles in into to "
    "onto inside for is are be should must it you see then goes go belong belongs receive receives take takes hold "
    "holds put place send move drop sort use coloured colored and".split()
) | frozenset(("first", "1st", "second", "2nd", "third", "3rd", "1", "2", "3", "st", "nd", "rd"))


def _struct_clean(text: str) -> str:
    text = str(text).strip()
    while True:
        new = _STRUCT_POLITE_HEAD.sub("", text)
        if new == text:
            break
        text = new
    return _STRUCT_POLITE_TAIL.sub("", text).strip(" \t,.;!?")


def _struct_keys(text: str):
    """Ordered semantic keys, [] for no keys, or None when colour/shape mentions were only partly consumed."""
    low = _ins._colour_aliases(_BINCOL.sub(" bin ", text).lower())
    ms = _ins._mentions(_ins._tokens(low), False)
    mentions = [(i, k, v) for i, k, v in ms if k != "bin"]
    keys, kind = _ins._keys_from(ms)
    out = [k for k, _ in keys]
    ncol = sum(k == "color" for _, k, _ in mentions)
    nshp = sum(k == "shape" for _, k, _ in mentions)
    if ncol and nshp:
        if kind != "compound" or len(out) != ncol or len(out) != nshp:
            return None
    elif len(out) != len(mentions):
        return None
    return out


def _struct_bins(text: str) -> list[str]:
    return [v for _, k, v in _ins._mentions(_ins._tokens(text), False) if k == "bin"]


def _struct_unknown(text: str) -> set[str]:
    low = _ins._colour_aliases(text.lower())
    known = set(_ins._COLOR_WORDS) | set(_ins._SHAPE_WORDS) | set(_ins._BIN_WORDS) | set(_ins._BIN_NOUNS)
    return {t for t in _ins._tokens(low) if t not in known and t not in _STRUCT_FILLER}


def _struct_slots(text: str) -> int:
    """Number of comma/and separated list members, including unknown members."""
    # Treat an Oxford comma as one separator, but retain every other empty member so a dangling/duplicate
    # separator cannot turn an incomplete list into an apparently complete one.
    text = re.sub(r",\s*and\b", " and ", text, flags=re.I)
    parts = re.split(r"\s*(?:,|/|\band\b)\s*", text, flags=re.I)
    return -1 if any(not part.strip() for part in parts) else len(parts)


def _struct_merge(rules) -> dict | None:
    out = {}
    for key, bin_name in rules:
        if key in out and out[key] != bin_name:
            return None
        out[key] = bin_name
    return out or None


def _struct_pair(a: str, b: str):
    """Return rules for one pure key:bin or bin:key pair, [] if it is not a pair, None if malformed."""
    ak, bk = _struct_keys(a), _struct_keys(b)
    ab, bb = _struct_bins(a), _struct_bins(b)
    if (ak is None or bk is None or _STRUCT_ASSIGN_BAD.search(a) or _STRUCT_ASSIGN_BAD.search(b)
            or _struct_unknown(a) or _struct_unknown(b)):
        return None
    if ak and not ab and not bk and len(bb) == 1:
        return [(k, bb[0]) for k in ak]
    if bk and not bb and not ak and len(ab) == 1:
        return [(k, ab[0]) for k in bk]
    return []


def _struct_zip(text: str):
    marker = _STRUCT_MARKER.search(text)
    if not marker:
        return None
    if (text[marker.end():].strip(" \t,.;!?") or _STRUCT_BAD.search(text)
            or _STRUCT_ASSIGN_BAD.search(text[:marker.start()])):
        return {"default": None}
    body = text[:marker.start()].strip(" \t,.;!?")
    for rx, order in _STRUCT_ZIP:
        m = rx.fullmatch(body)
        if not m:
            continue
        a, b = m.group("a").strip(" ,"), m.group("b").strip(" ,")
        ak, bk = _struct_keys(a), _struct_keys(b)
        ab, bb = _struct_bins(a), _struct_bins(b)
        if ak is None or bk is None:
            return {"default": None}
        if order == "keys_bins" and ak and not ab and not bk and bb:
            keys, bins = ak, bb
        elif order == "bins_keys" and bk and not bb and not ak and ab:
            keys, bins = bk, ab
        else:
            return {"default": None}
        key_text, bin_text = (a, b) if keys is ak else (b, a)
        if (_struct_unknown(key_text) or _struct_unknown(bin_text)
                or len(keys) != _struct_slots(key_text) or len(bins) != _struct_slots(bin_text)
                or len(keys) != len(bins) or len(keys) < 2 or len(set(keys)) != len(keys)):
            return {"default": None}
        merged = _struct_merge(zip(keys, bins))
        return merged if merged is not None else {"default": None}
    return {"default": None}


def _struct_directional(text: str):
    m = _STRUCT_DIRECTION.fullmatch(text)
    if not m:
        return None
    if (m.group(1).lower() == m.group(2).lower() or _STRUCT_BAD.search(text)
            or _STRUCT_ASSIGN_BAD.search(m.group(3)) or _struct_unknown(m.group(3))):
        return {"default": None}
    keys = _struct_keys(m.group(3))
    if (keys is None or len(keys) != 3 or len(set(keys)) != 3 or _struct_bins(m.group(3))
            or _struct_slots(m.group(3)) != 3 or _STRUCT_PLACE.search(m.group(3))):
        return {"default": None}
    bins = _ins.BINS if m.group(1).lower() == "left" else tuple(reversed(_ins.BINS))
    return dict(zip(keys, bins))


def _struct_colons(text: str):
    if ":" not in text:
        return None
    segments = [s.strip() for s in re.split(r"[.;!?\n]+", text) if s.strip()]
    rules, candidates, malformed = [], 0, False
    for segment in segments:
        if ":" not in segment:
            if not _STRUCT_HEADER.fullmatch(segment):
                malformed = True
            continue
        left, right = segment.split(":", 1)
        pair = _struct_pair(left, right)
        if pair:
            candidates += 1
            rules.extend(pair)
        elif (pair is None or _struct_keys(left) or _struct_keys(right) or _struct_bins(left)
              or _struct_bins(right) or not _STRUCT_HEADER.fullmatch(segment)):
            malformed = True
    if not candidates:
        # A single prose introducer still belongs to the legacy resolver. Multiple colons, or a lone semantic
        # member opposite an empty side, are recognisably structured but incomplete and must fail closed.
        semantic_half = any(
            bool(_struct_keys(side) or _struct_bins(side)) and not other.strip()
            for segment in segments if ":" in segment
            for side, other in (tuple(reversed(segment.split(":", 1))), segment.split(":", 1))
        )
        return {"default": None} if malformed and (text.count(":") > 1 or semantic_half) else None
    if malformed or _STRUCT_BAD.search(text):
        return {"default": None}
    merged = _struct_merge(rules)
    return merged if merged is not None else {"default": None}


def _struct_ordinals(text: str):
    m = _STRUCT_ORD_HEAD.fullmatch(text)
    if not m:
        return None
    if _STRUCT_BAD.search(text):
        return {"default": None}
    side, body = m.group(1).lower(), m.group(2)
    walk = _ins.BINS if side == "left" else tuple(reversed(_ins.BINS))
    index = {"first": 0, "1st": 0, "second": 1, "2nd": 1, "third": 2, "3rd": 2}
    rules = []
    for clause in [s.strip() for s in re.split(r"[.;!?\n]+", body) if s.strip()]:
        keys, ords = _struct_keys(clause), _STRUCT_ORD_BIN.findall(clause)
        # A positional word here competes with the ordinal direction and is therefore ambiguous.
        if (keys is None or len(keys) != 1 or len(ords) != 1 or _struct_bins(clause)
                or _STRUCT_ASSIGN_BAD.search(clause) or _struct_unknown(clause)):
            return {"default": None}
        rules.append((keys[0], walk[index[ords[0].lower()]]))
    merged = _struct_merge(rules)
    return merged if merged is not None else {"default": None}


def _struct_positionals(text: str):
    """Verb-less positional rows separated by semicolons: "left bin, red; center bin, yellow"."""
    segments = [s.strip() for s in re.split(r"[.;!?\n]+", text) if s.strip()]
    rules, candidates, malformed = [], 0, False
    for segment in segments:
        if "," not in segment or _STRUCT_PLACE.search(segment):
            if not _STRUCT_HEADER.fullmatch(segment):
                malformed = True
            continue
        left, right = segment.split(",", 1)
        pair = _struct_pair(left, right)
        if pair:
            candidates += 1
            rules.extend(pair)
        else:
            malformed = True
    if not candidates:
        return None
    if malformed or _STRUCT_BAD.search(text):
        return {"default": None}
    merged = _struct_merge(rules)
    return merged if merged is not None else {"default": None}


def _struct_object_action(action: str) -> bool:
    """The condition's object, not its destination bin, must be the thing acted on."""
    tokens = _ins._tokens(_ins._colour_aliases(action.lower()))
    if any(token in ("and", "or") for token in tokens):
        return False
    mentions = _ins._mentions(tokens, False)
    bin_positions = [i for i, kind, _ in mentions if kind == "bin"]
    if len(bin_positions) != 1:
        return False
    destination = bin_positions[0]
    object_positions = {i for i, kind, _ in mentions if kind != "bin"}
    object_positions.update(i for i, token in enumerate(tokens)
                            if token in ("it", "object", "objects", "item", "items", "piece", "pieces",
                                         "thing", "things", "one", "ones"))
    imperative = {i for i, token in enumerate(tokens)
                  if token in ("put", "place", "send", "move", "drop", "sort", "take", "use")}
    flow = {i for i, token in enumerate(tokens) if token in ("go", "goes", "belong", "belongs")}
    def bridge_ok(start: int) -> bool:
        bridge = tokens[start + 1:destination]
        return not any(token in ("and", "or") for token in bridge)

    return (any(verb < obj < destination and bridge_ok(obj)
                for verb in imperative for obj in object_positions)
            or any(obj < verb < destination and bridge_ok(verb)
                   for verb in flow for obj in object_positions))


def _struct_conditionals(text: str):
    if not re.search(r"\bif\b", text, re.I):
        return None
    if _STRUCT_BAD.search(text):
        return {"default": None}
    segments = [s.strip() for s in re.split(r"[.;!?\n]+", text) if s.strip()]
    rules = []
    for segment in segments:
        m = re.fullmatch(r"if\s+(.+?)(?:\s*,\s*|\s+then\s+)(.+)", segment, re.I)
        if not m or re.search(r"\b(?:or|and)\b", m.group(1), re.I):
            return {"default": None}
        cond, action = m.group(1), m.group(2)
        keys, action_keys, bins = _struct_keys(cond), _struct_keys(action), _struct_bins(action)
        if keys is None or action_keys is None or len(keys) != 1 or _struct_bins(cond) or len(bins) != 1:
            return {"default": None}
        if (_STRUCT_ASSIGN_BAD.search(cond) or _STRUCT_ASSIGN_BAD.search(action)
                or _struct_unknown(cond) or _struct_unknown(action)
                or not _struct_object_action(action) or any(k != keys[0] for k in action_keys)):
            return {"default": None}
        rules.append((keys[0], bins[0]))
    merged = _struct_merge(rules)
    return merged if merged is not None else {"default": None}


def _structured(text: str):
    text = _struct_clean(text)
    if not text:
        return None
    for parser in (_struct_conditionals, _struct_zip, _struct_directional, _struct_ordinals,
                   _struct_colons, _struct_positionals):
        result = parser(text)
        if result is not None:
            return result
    return None


def structured_mapping(text: str):
    """Parse strict parallel-list forms; None defers to strict_mapping, a dict is final. Never raises."""
    try:
        return _structured("" if text is None else str(text))
    except Exception:
        return {"default": None}


def _pieces(text: str) -> list[tuple[tuple[set[str], set[str]], set[str], bool, bool]]:
    """Clauses (split at _CLAUSE_SEP) as (keys, bins, catch-all phrase, triggered). A bin-less clause right after a
    triggered one is part of that trigger ("between the yellow cylinder and the orange cube"). A clause with keys (or a
    catch-all) and no bin takes the bins of the next clause when that one has bins and no keys ("Rules: red: left")."""
    parts = _CLAUSE_SEP.split(text)
    raw = [(p, parts[i + 1] if i + 1 < len(parts) else "") for i, p in enumerate(parts[0::2]) if p and p.strip()]
    ord_ok = _ins._ordinals_ok(_ins._tokens(text))
    info, prev = [], False
    for p, _ in raw:
        ks, bs = _strict_keys(p), _bins_in(p, ord_ok)
        t = _triggered(p) or (prev and not bs)
        info.append([ks, bs, bool(_CATCH_ALL.search(p)), t, _nonkey_words(p)])
        prev = t
    out = []
    for i, (ks, bs, c, t, nonkey) in enumerate(info):
        if (ks[0] or ks[1] or c) and not bs and not t and i + 1 < len(info):
            nk, nb, _, nt, _ = info[i + 1]
            colon_list = nb and not (nk[0] or nk[1])          # "Rules: red: left"
            and_joined = (nb and (nk[0] or nk[1]) and nonkey <= 2 and not nt   # "Red and yellow go in the left bin"
                          and raw[i][1].strip().lower() == "and")
            if not nt and (colon_list or and_joined):
                bs = nb
        out.append((ks, set() if t else bs, c, t))
    return out


def _strict(text, mapping: dict) -> dict:
    text = "" if text is None else str(text)
    if not text.strip():
        return dict(mapping)
    if not mapping or _SWAP.search(text):   # (a)
        return {"default": None}
    out = dict(mapping)
    # One object cannot be placed in two bins simultaneously. The base resolver chooses the first; strict mode marks
    # every key in that sentence-level command as an abstain instead.
    conflicts = []
    for m in _MULTI_DEST.finditer(text):
        lo = max((text.rfind(c, 0, m.start()) for c in ".;!?\n"), default=-1) + 1
        hi = min((p for c in ".;!?\n" if (p := text.find(c, m.end())) >= 0), default=len(text))
        conflicts.append(_strict_keys(text[lo:hi]))
    for cols, shps in conflicts:
        for k in (*cols, *shps):
            out[k] = None
        for k in list(out):
            if isinstance(k, tuple) and (k[0] in cols or k[1] in shps):
                out[k] = None
    pieces = _pieces(text)
    for (cols, shps), _, _, t in pieces:   # (b)
        if not t:
            continue
        for k in (*cols, *shps):            # also keys the resolver never bound: None stops the lookup at this key
            out[k] = None
        for k in list(out):
            if isinstance(k, tuple) and (k[0] in cols or k[1] in shps):
                out[k] = None
    if any(t for *_, t in pieces):
        out.pop("default", None)
    cols_all, shps_all = _strict_keys(text)
    if _BLOCKS.search(text):                # "blocks" is the cube only beside another shape: the resolver decides that
        shps_all = shps_all | {"cube"}
    if "default" in out:                    # a named object the resolver did not bind must not fall through to "default"
        tuple_cols = {k[0] for k in out if isinstance(k, tuple)}
        tuple_shps = {k[1] for k in out if isinstance(k, tuple)}
        for k in (cols_all - set(out) - tuple_cols) | (shps_all - set(out) - tuple_shps):
            out[k] = None
    catch = bool(_CATCH_ALL.search(text))
    for k in list(out):
        if out[k] is None:
            continue
        if k == "default":                  # (c) needs the catch-all phrase; (d) its piece states the bin
            if not catch:
                del out[k]
            elif out[k] not in set().union(*(b for _, b, c, _ in pieces if c)):
                out[k] = None
            continue
        if not _names(k, cols_all, shps_all):   # (c)
            del out[k]
            continue
        named = [(cs, b, t) for cs, b, _, t in pieces if _names(k, *cs)]
        if any(t for _, _, t in named):          # a triggered piece names it: the bin is not stated plainly
            out[k] = None
            continue
        if _cross_unbound(k, out, named):        # (e) "orange prism" under a bare colour key
            out[k] = None
            continue
        bs = [b for _, b, _ in named if b]     # (d) pieces with no bin defer to the ones that state one
        ok = bool(bs) and all(len(b) > 1 or out[k] in b for b in bs) and any(out[k] in b for b in bs)
        if not ok:
            out[k] = None
    return out or {"default": None}


def strict_mapping(text: str, mapping: dict) -> dict:
    """resolve(text, fallback={}) filtered to abstain rather than guess (instruction.strict). Never raises."""
    try:
        return _strict(text, mapping)
    except Exception:
        return {"default": None}


# ---------------------------------------------------------------------------------------------------------------------
# instruction.structured, second pass (langfix2). Everything below runs ONLY behind `instruction.structured`
# (policy.py calls structured_normalize / structured_final instead of normalize / structured_mapping), so with the flag
# off nothing in this block is reachable and the older path is byte-identical. It reuses the tables above (name_colours,
# name_foreign, normalize) and only adds forms they lack. Every rule maps a phrase to a closed-vocabulary word or turns an
# ambiguity into an abstain (None); none returns a bin the sentence does not state.
#   amber    -> ambiguous (yellow or orange): becomes an unknown token, and the objects it could name are abstained.
#   can      -> cylinder after a colour word when "can" cannot be the modal verb ("for orange can,", "orange can in the...").
#   right in / right at the + position -> "right" is an intensifier, not a bin ("right in the middle" = middle).
#   Czech    -> extra direction / colour inflections (doleva, vpravo, uprostred; cerveny, modry...) and "a" = "and".
#   typos    -> a colour or "cubs" is repaired only when a shape word follows / a colour word precedes it.
#   rest     -> "for the remaining shapes, red and blue left..." maps only the complement of what was named before.
#   blocks   -> "yellow blocks X, the rest Y": cube-only vs colour-wide is ambiguous, only the cube cell is mapped.
# ---------------------------------------------------------------------------------------------------------------------
_S_COLOURS = r"(?:red|orange|yellow|purple|violet|blue|lilac|gold(?:en)?)"
_S_AMBER = re.compile(r"\bamber\b", re.I)
_S_AMBER_TOKEN = "xamber"          # not a colour word (normalize matches amber at a word start only); no fuzzy target starts with x
_S_CAN = re.compile(
    rf"\b({_S_COLOURS})(\s+)(can)\b(?=\s*(?:[.,;:!?]|$)|\s+(?:in|into|onto|inside|to|at|goes|belongs|is|are)\b)", re.I)
_S_INTENSIFIER = re.compile(
    r"(?<![A-Za-z])((?:the\s+)?)right(\s+)((?:in|at|on|by)\s+the\s+(?:very\s+|far\s+)?(?:left|right|center|centre|middle)\b)",
    re.I)
_S_CZ_REPL = tuple((re.compile(p, re.I), w) for p, w in (
    (r"\b(?:dolev|nalev|zlev)\w*|\bvlevo\b|\blev(?:[ýáéí]|ou|ého|ém|ému|ým)\w*", "left"),
    (r"\b(?:doprav|naprav|zprav)\w*|\bvpravo\b|\bprav(?:[ýáéí]|ou|ého|ém|ému|ým)\w*", "right"),
    (r"\b(?:do|u|na|v)?prost[řr]ed\w*|\bst[řr]edn\w*|\bst[řr]ed\w*", "middle"),
    (r"\b[čc]erven\w*", "red"),
    (r"\bmodr\w*", "blue"),
))
_S_CZ_DETECT = re.compile(r"\b(?:[žz]lut|fialov|oran[žz]ov|[čc]erven|modr|dolev|doprav|nalev|naprav|vlevo|vpravo|"
                          r"prost[řr]ed|st[řr]ed|kostk|v[áa]lc|hranol|krabic|ko[šs]\b)", re.I)
_S_CZ_AND = re.compile(r"(?<![A-Za-zÀ-ɏ])a(?![A-Za-zÀ-ɏ])")
_S_COLOUR_TARGETS = ("yellow", "purple", "violet", "orange", "golden")
_S_WORDS = re.compile(r"[A-Za-z]+")


def _s_colour_typo(word: str) -> str | None:
    """Unique colour a misspelt word stands for, else None. Same first letter; 1 edit, or 2 edits with the same last
    letter; 5+ letters ("blue" and "red" are too short to repair safely)."""
    w = word.lower()
    if len(w) < 5 or w in VOCAB or w in _ins._KNOWN:
        return None
    hits = {t for t in _S_COLOUR_TARGETS if t[0] == w[0] and abs(len(t) - len(w)) <= 2
            and ((d := _dist(w, t)) <= 1 or (d == 2 and t[-1] == w[-1]))}
    return hits.pop() if len(hits) == 1 else None


def _s_typos(sentence: str) -> str:
    """After normalize(): repair a colour typo that sits directly before a shape word, and "cubs" after a colour."""
    ws = list(_S_WORDS.finditer(sentence))
    out, last = [], 0
    for i, m in enumerate(ws):
        w, new = m.group(0), None
        nxt = ws[i + 1] if i + 1 < len(ws) else None
        if nxt and sentence[m.end():nxt.start()].isspace() and nxt.group(0).lower() in _ins._SHAPE_WORDS:
            c = _s_colour_typo(w) if w.lower() in PROTECTED or w.lower() not in _ins._KNOWN else None
            new = c
        prev = ws[i - 1] if i else None
        if (w.lower() == "cubs" and prev and sentence[prev.end():m.start()].isspace()
                and (prev.group(0).lower() in _ins._COLOR_WORDS or prev.group(0).lower() in VOCAB and VOCAB[prev.group(0).lower()] in _ins.COLORS
                     or ((_s_colour_typo(prev.group(0)) or "") in _S_COLOUR_TARGETS))):
            new = "cubes"
        if new:
            out.append(sentence[last:m.start()] + (new.capitalize() if w[0].isupper() else new))
            last = m.end()
    return "".join(out) + sentence[last:]


def structured_normalize(text: str) -> str:
    """normalize() plus the structured-pass rewrites above. Never raises; on an error falls back to normalize()."""
    if text is None:
        return ""
    try:
        s = str(text)
        s = _S_AMBER.sub(_S_AMBER_TOKEN, s)
        s = _S_INTENSIFIER.sub(lambda m: m.group(0) if m.group(1) else m.group(3), s)
        s = _S_CAN.sub(lambda m: m.group(1) + m.group(2) + ("Cylinder" if m.group(3)[0].isupper() else "cylinder"), s)
        if _S_CZ_DETECT.search(s):
            for rx, w in _S_CZ_REPL:
                s = rx.sub(lambda m, w=w: w.capitalize() if m.group(0)[0].isupper() else w, s)
            s = _S_CZ_AND.sub("and", s)
        return _s_typos(normalize(s))
    except Exception:
        return normalize(text)


_S_ALL_OBJECTS = tuple((c, sh) for c in _ins.COLORS for sh in _ins.SHAPES)
_S_REST = re.compile(
    r"\b(?:for\s+)?(?:the\s+|all\s+(?:the\s+)?)?(?:(?:remaining|other|leftover)\s+(?:shapes?|objects?|items?|pieces?|"
    r"colou?rs?|ones|things)|rest(?:\s+of\s+(?:the\s+)?(?:shapes?|objects?|items?|pieces?))?|everything\s+else|"
    r"anything\s+else|others)\b\s*[,:]?", re.I)


def _s_core(text: str):
    parsed = structured_mapping(text)
    return parsed if parsed is not None else strict_mapping(text, _ins.resolve(text, fallback={}))


def _s_lookup(mapping: dict, colour: str, shape: str):
    """resolve_bin without "default": (present, bin)."""
    for k in ((colour, shape), colour, shape):
        if k in mapping:
            return True, mapping[k]
    return False, None


def _s_complement(text: str):
    """"<named part>. For the remaining shapes, <rules>": rules apply only to the objects the first part did not name.
    None if the sentence has no such split or either half is not understood (the unchanged path then decides)."""
    ms = list(_S_REST.finditer(text))
    if len(ms) != 1:
        return None
    m = ms[0]
    first, second = text[:m.start()], text[m.end():]
    k1, k2 = _strict_keys(first), _strict_keys(second)
    if not (k1[0] or k1[1]) or not (k2[0] or k2[1]) or not _bins_in(first) or not _bins_in(second):
        return None                         # a plain "the rest go left" is the catch-all path, not this one
    m1, m2 = _s_core(first.rstrip(" ,;:.-")), _s_core(second.lstrip(" ,;:.-"))
    m1 = {k: v for k, v in m1.items() if k != "default"}
    m2 = {k: v for k, v in m2.items() if k != "default"}
    if not any(v for v in m1.values()) or not any(v for v in m2.values()):
        return None                         # a half is not understood: leave the sentence to the unchanged path
    out = dict(m1)
    for c, sh in _S_ALL_OBJECTS:
        if _s_lookup(m1, c, sh)[0]:
            continue                        # already named by the first part
        out[(c, sh)] = _s_lookup(m2, c, sh)[1]
    out["default"] = None
    return out


_S_COLOUR_BLOCKS = re.compile(rf"\b({_S_COLOURS})\s+blocks\b", re.I)


def _s_block_scope(text: str, mapping: dict) -> dict:
    """"yellow blocks center, the rest left": whether "blocks" means cubes or every yellow solid is unstated. Map only the
    cube cell (the reading both agree on) and abstain on the other shapes of that colour; the catch-all stays."""
    if "default" not in mapping or mapping["default"] is None:
        return mapping
    out = dict(mapping)
    for m in _S_COLOUR_BLOCKS.finditer(text):
        word = m.group(1).lower()
        colour = _ins._COLOR_WORDS.get(word) or VOCAB.get(word)
        if colour in _ins.COLORS and out.get(colour) and (colour, "cube") not in out:
            out[(colour, "cube")] = out[colour]
            out[colour] = None
    if out.get("cube") is None and not any(_ins._SHAPE_WORDS.get(t) == "cube" for t in _ins._tokens(text.lower())):
        out.pop("cube", None)               # strict put it there only because of the word "blocks"
    return out


def _s_amber(text: str, mapping: dict) -> dict:
    """amber is yellow or orange: abstain every object it could name. Colours the text names explicitly keep their bin."""
    explicit = _strict_keys(text)[0]
    out = dict(mapping)
    for c in ("yellow", "red"):
        if c not in explicit:
            out[c] = None
    out["default"] = None
    return out


def structured_final(text: str) -> dict:
    """The instruction.structured mapping for already structured_normalize()d text. Never raises."""
    try:
        out = _s_complement(text)
        if out is None:
            out = _s_block_scope(text, _s_core(text))
        if _S_AMBER_TOKEN in text.lower():
            out = _s_amber(text, out)
        return out
    except Exception:
        return {"default": None}


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
