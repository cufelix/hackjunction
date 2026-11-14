"""Pre-resolver normaliser: maps near-miss tokens onto the resolver's closed vocabulary.

Token-wise and conservative: a token already in the resolver's vocabulary is never changed, a token
near two different canonical meanings is left alone (abstain costs 0, a wrong key costs -1), and
synonyms are a small table of ordinary English words (no sentence from any held-out set).
Does not touch policy/instruction.py (frozen); it only reads its vocabulary tables.
"""
from __future__ import annotations

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


def _fuzzy(tok: str) -> str | None:
    """Unique canonical target for a near-miss token, else None (unknown or ambiguous)."""
    lim = _limit(len(tok))
    hits: dict[str, int] = {}
    for t in FUZZY_TARGETS:
        forms = {t, _collapse(t)}
        # first letter is almost never the typo, and requiring it keeps real words (below/yellow) out
        d = min((_dist(tok, f) for f in forms), default=9) if tok[0] == t[0] else 9
        if len(tok) >= 3 and tok in (_skeleton(t), _skeleton(_collapse(t))) and tok != t:
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


_LIFT_LR = re.compile(r"\(\s*lift\s+to\s+(right|left)\s*\)", re.I)   # "first bin (lift to right)": a left/right typo


def normalize(sentence: str) -> str:
    """Map colour names and near-miss tokens onto the resolver's vocabulary, then drop colour-named-bin clauses."""
    sentence = _LIFT_LR.sub(lambda m: f"(left to {m.group(1)})", sentence)
    sentence = _REGARDLESS.sub(lambda m: "Regardless of colour" if m.group(0)[0].isupper() else "regardless of colour", sentence)
    sentence = name_foreign(name_colours(sentence))
    return drop_colour_bin_clauses(rewrite_ordinal_bins(_WORD.sub(_fix, sentence)))


# Strict mode (instruction.strict): the mapping never guesses. instruction.py is FROZEN, so this filters its output.
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
    return any(rx.search(clause) for rx in _TRIGGERS)


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
    try:
        assert normalize("lefz") == "lefz"
    finally:
        FUZZY_TARGETS = saved
        del VOCAB["lefy"]
    # real-English neighbours and "right" meaning correct are not mangled; nothing non-near-miss becomes a bin word
    assert normalize("blur light fight night") == "blur light fight night"
    assert normalize("it is right, so read the rite and write") == "it is right, so read the rite and write"
    assert normalize("Is that correct? Yes, all right.") == "Is that correct? Yes, all right."
    for w in ("correct", "write", "rite", "ride", "ligt", "lift", "bring", "rug", "mud", "side", "port"):
        assert normalize(w) in (w,), (w, normalize(w))
    print("normalize selfcheck OK")


if __name__ == "__main__":  # python -m aaltonomous.baselines.classical.normalize
    _selfcheck()
