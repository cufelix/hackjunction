"""Instruction -> bin mapping, for the sort_language episodes that carry 25 of the 30 final points.

Why this file exists
--------------------
The handed-out classical baseline never reads the instruction at all
(``baselines/classical/policy.py``: ``self.mapping = info.get("mapping") or dict(FIXED_MAPPING)``,
and the reset info a submitted policy receives carries only ``{"instruction": ...}``).  The
organisers' own oracle "parser" (``sim/oracle.py::parse_instruction``) is a dict lookup over the
288 training sentences and returns ``None`` on anything else, after which the oracle falls back to
the privileged mapping that a Docker submission does not get.  So neither reference solution can
score a held-out phrasing, and the finals weight those at 1.5x.

What the scorer accepts
-----------------------
``scene.py::resolve_bin(mapping, color, shape)`` tries, most specific first:
``(color, shape)``, ``color``, ``shape``, ``"default"``.  So a mapping may key on colour, on
shape, on both, and may carry a catch-all.  All four are produced here.

Why it never abstains
---------------------
sort_language scores +2 for the right bin, -1 for the wrong one, 0 for leaving the object on the
counter.  A uniformly random bijection is worth (1/3)(+2) + (2/3)(-1) = 0 per object, exactly what
abstaining is worth, so a guess is never worse in expectation than refusing -- and a parse that is
merely *probably* right is strictly better.  ``resolve`` therefore always returns a usable mapping
and falls back to FIXED_MAPPING rather than to None.

Pure stdlib on purpose: it runs in the policy container, in a unit test, and on a laptop with no
simulator.
"""
from __future__ import annotations

import itertools
import re

COLORS = ("red", "yellow", "blue")
SHAPES = ("cube", "cylinder", "triangle")
BINS = ("left", "center", "right")
FIXED_MAPPING = {"red": "left", "yellow": "center", "blue": "right"}

# Bin words. "centre" is the British spelling and the single most likely held-out trap; the rest are
# the ordinary ways a person names one of three positions in a row.
_BIN_WORDS = {
    "left": "left", "leftmost": "left", "leftmost-": "left", "lefthand": "left", "port": "left",
    "right": "right", "rightmost": "right", "righthand": "right", "starboard": "right",
    "center": "center", "centre": "center", "central": "center", "centermost": "center",
    "middle": "center", "mid": "center", "middlemost": "center", "between": "center",
    "inbetween": "center",
}
# Consulted ONLY when the sentence names fewer than two positional bins, so that an ordinal used for
# something else ("the first red object") cannot hijack a sentence that already says left/right.
_ORDINAL_BINS = {
    "first": "left", "1": "left", "a": "left",
    "second": "center", "2": "center", "b": "center",
    "third": "right", "3": "right", "c": "right",
}
# Colour words, including the adjective forms a paraphrase may use.
_COLOR_WORDS = {
    "red": "red", "reds": "red", "reddish": "red", "crimson": "red", "scarlet": "red",
    "yellow": "yellow", "yellows": "yellow", "yellowish": "yellow", "gold": "yellow", "golden": "yellow",
    "amber": "yellow",
    "blue": "blue", "blues": "blue", "bluish": "blue", "azure": "blue", "navy": "blue",
}
# Shape words. The brief's own names plus the adjective and plural forms.
_SHAPE_WORDS = {
    # NB: "box"/"boxes" are deliberately NOT here. The challenge calls the three bins boxes
    # ("three identical orange boxes"), so reading them as the cube shape injects phantom shape
    # mentions into every sentence that names a bin that way, which silently wrecks the pairing.
    "cube": "cube", "cubes": "cube", "cubic": "cube",
    "square": "cube", "squares": "cube", "cuboid": "cube", "block": "cube", "blocks": "cube",
    "cylinder": "cylinder", "cylinders": "cylinder", "cylindrical": "cylinder",
    "circle": "cylinder", "circles": "cylinder", "circular": "cylinder", "round": "cylinder",
    "disc": "cylinder", "disk": "cylinder", "tube": "cylinder", "tubes": "cylinder",
    "triangle": "triangle", "triangles": "triangle", "triangular": "triangle", "prism": "triangle",
    "prisms": "triangle", "wedge": "triangle", "wedges": "triangle", "tri": "triangle",
}
# "everything else" -> the "default" key resolve_bin() falls back to.
_DEFAULT_PATTERNS = (
    r"\beverything\s+else\b", r"\banything\s+else\b", r"\ball\s+(?:the\s+)?others?\b",
    r"\bany\s+others?\b", r"\bthe\s+rest\b", r"\bthe\s+remaining\b", r"\bremaining\s+ones?\b",
    r"\ball\s+other\b", r"\bevery(?:thing)?\s+other\b", r"\bwhatever(?:'s|\s+is)?\s+left\b",
    r"\bthe\s+others\b",
)
_TOKEN_RE = re.compile(r"[a-z]+|\d+")


def _tokens(text):
    """Lowercased word tokens. Hyphens join ('left-hand' -> 'lefthand') so the bin table catches them."""
    return _TOKEN_RE.findall(text.lower().replace("-", "").replace("_", ""))


def _mentions(toks):
    """(index, kind, value) for every colour, shape and bin word, in order.

    Bins are read in two tiers. Positional words (left / centre / right and their synonyms) are
    always trusted. Ordinals and numbers ("the first bin", "bin 2", "bin C") are consulted only if
    the sentence names fewer than two positional bins, and then only next to the word "bin" or
    "box", so that "the first red object" cannot be mistaken for the left bin.
    """
    out = []
    for i, t in enumerate(toks):
        if t in _BIN_WORDS:
            out.append((i, "bin", _BIN_WORDS[t]))
        if t in _COLOR_WORDS:
            out.append((i, "color", _COLOR_WORDS[t]))
        if t in _SHAPE_WORDS:
            out.append((i, "shape", _SHAPE_WORDS[t]))

    if len({v for _, k, v in out if k == "bin"}) < 2:
        container = {"bin", "bins", "box", "boxes", "one", "ones", "slot", "slots",
                     "container", "containers", "crate", "crates", "tray", "trays"}
        found = [(i, _ORDINAL_BINS[t]) for i, t in enumerate(toks) if t in _ORDINAL_BINS]
        anchored = any(
            toks[j] in container
            for i, _ in found for j in (i - 1, i + 1) if 0 <= j < len(toks)
        )
        # An enumeration names at least two of them and anchors at least one to the noun; one
        # ordinal on its own is ordinary English, not a bin name.
        if anchored and len({v for _, v in found}) >= 2:
            out.extend((i, "bin", v) for i, v in found)
            out.sort()
    return out


def _keys_from(mentions, compound_window=2):
    """Decide what the sentence keys on, and return [(key, token_index), ...].

    A colour and a shape within `compound_window` tokens of each other ("red cubes") is one
    compound key. A sentence that names >=2 distinct colours keys on colour; otherwise, if it names
    >=2 distinct shapes, it keys on shape. That ordering matters because the organisers said the
    finals lead with colour, and because "the red objects" must not be read as a shape mention.
    """
    colors = [(i, v) for i, k, v in mentions if k == "color"]
    shapes = [(i, v) for i, k, v in mentions if k == "shape"]

    compounds = []
    used_c, used_s = set(), set()
    for ci, cv in colors:
        near = [(abs(ci - si), si, sv) for si, sv in shapes if abs(ci - si) <= compound_window]
        if near:
            _, si, sv = min(near)
            if si not in used_s:
                compounds.append(((cv, sv), ci))
                used_c.add(ci)
                used_s.add(si)
    # Every colour mention paired with a shape, and at least two pairs -> the sentence is compound.
    if len(compounds) >= 2 and len(used_c) == len(colors):
        return compounds

    if len({v for _, v in colors}) >= 2:
        return [(v, i) for i, v in colors]
    if len({v for _, v in shapes}) >= 2:
        return [(v, i) for i, v in shapes]
    # One key only (e.g. "put the red ones in the left bin, everything else in the right bin").
    return [(v, i) for i, v in colors] or [(v, i) for i, v in shapes]


def _assign(keys, bins_):
    """Min total token-distance bijection between distinct keys and distinct bins.

    English puts a colour and its bin in the same clause, so nearest-neighbour pairing reads every
    one of the 8 training templates and does not care about their word order -- template 5 names the
    bin first, template 2 names the colour first, and both come out the same. Solved exactly: with
    at most 3 bins the permutation count is 6.
    """
    key_order, bin_order = [], []
    for k, _ in keys:
        if k not in key_order:
            key_order.append(k)
    for b, _ in bins_:
        if b not in bin_order:
            bin_order.append(b)
    if not key_order or not bin_order:
        return {}

    def dist(k, b):
        return min(abs(ki - bi) for kk, ki in keys if kk == k for bb, bi in bins_ if bb == b)

    first_key = {k: min(i for kk, i in keys if kk == k) for k in key_order}
    first_bin = {b: min(i for bb, i in bins_ if bb == b) for b in bin_order}

    def inversions(ks, bs):
        """How many assigned pairs are listed out of order relative to each other."""
        n_ = len(ks)
        return sum(
            1
            for a in range(n_)
            for b in range(a + 1, n_)
            if (first_key[ks[a]] - first_key[ks[b]]) * (first_bin[bs[a]] - first_bin[bs[b]]) < 0
        )

    n = min(len(key_order), len(bin_order))
    best, best_cost = None, None

    # Assign the n closest-fitting keys; a sentence may name more keys than bins or the reverse.
    for ks in itertools.permutations(key_order, n):
        for bs in itertools.permutations(bin_order, n):
            cost = (sum(dist(k, b) for k, b in zip(ks, bs)), inversions(ks, bs))
            if best_cost is None or cost < best_cost:
                best, best_cost = dict(zip(ks, bs)), cost
    return best or {}


def resolve(text, fallback=None):
    """Instruction sentence -> mapping for scene.resolve_bin(). Never returns None.

    >>> resolve("Put the blue objects in the left bin, the red objects in the center bin "
    ...         "and the yellow objects in the right bin.") == {
    ...     "blue": "left", "red": "center", "yellow": "right"}
    True
    >>> resolve("The centre bin is for red, the left bin for blue, the right bin for yellow.") == {
    ...     "red": "center", "blue": "left", "yellow": "right"}
    True
    """
    if not text or not str(text).strip():
        return dict(FIXED_MAPPING if fallback is None else fallback)
    raw = str(text)
    toks = _tokens(raw)
    mentions = _mentions(toks)
    bins_ = [(v, i) for i, k, v in mentions if k == "bin"]
    keys = _keys_from(mentions)
    mapping = _assign(keys, bins_) if keys and bins_ else {}

    # "everything else" -> the catch-all key, pointed at the bin nearest that phrase.
    low = raw.lower()
    for pat in _DEFAULT_PATTERNS:
        m = re.search(pat, low)
        if not m:
            continue
        at = len(_TOKEN_RE.findall(low[: m.start()]))
        claimed = set(mapping.values())
        free = [(abs(i - at), b) for b, i in bins_ if b not in claimed] or [(abs(i - at), b) for b, i in bins_]
        if free:
            mapping["default"] = min(free)[1]
        break

    # Complete the bijection rather than drop a key. Every episode seen so far maps all three
    # colours onto all three bins, and a key with no bin leaves the policy nowhere to put that
    # object (worth 0), whereas a guess from the leftovers is worth (1/3)(+2)+(2/3)(-1) = 0 at worst
    # and +2 when right.
    if mapping and "default" not in mapping:
        missing = [k for k, _ in keys if k not in mapping]
        seen = set()
        missing = [k for k in missing if not (k in seen or seen.add(k))]
        free = [b for b in BINS if b not in set(mapping.values())]
        for k, b in zip(missing, free):
            mapping[k] = b

    if not mapping:
        return dict(FIXED_MAPPING if fallback is None else fallback)
    return mapping
