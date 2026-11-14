"""Instruction -> bin mapping, for the sort_language episodes that carry ~5/6 of the final points.

Why this file exists
--------------------
The handed-out classical baseline never reads the instruction at all
(``baselines/classical/policy.py:73``: ``self.mapping = info.get("mapping") or dict(FIXED_MAPPING)``,
and the reset info a submitted policy receives carries only ``{"instruction": ...}``).  The
organisers' own oracle "parser" (``sim/oracle.py::parse_instruction``) is a dict lookup over the
288 training sentences and returns ``None`` on anything else, after which the oracle falls back to
``privileged_mapping()`` -- the true mapping read out of the simulator, which
``RULES_AND_SCORING.md:89`` forbids a submission from using.  So neither reference solution can
score a held-out phrasing.

What the scorer accepts
-----------------------
``scene.py::resolve_bin(mapping, color, shape)`` tries, most specific first:
``(color, shape)``, ``color``, ``shape``, ``"default"``.  All four are produced here.  Note it is a
plain dict lookup, so the mapping is a **function, not a bijection**: two colours may share a bin.

The model: clauses, not distance
--------------------------------
An earlier version paired every key with its nearest bin token and solved one global bijection.
An adversarial review broke that four ways, all silently: two colours sharing one bin got
reassigned, a trailing recap ("...and the red ones first") reshuffled everything because the metric
took the minimum over all mentions of a key, incidental colour words stole the keys from a
shape-keyed sentence, and "everything ... except X" lost its catch-all.

English binds a key to its bin **inside a clause**, so this version splits on clause boundaries
first and resolves each clause on its own.  One bin in a clause claims every key in that clause,
which is what makes many-to-one work.  Distance is only used to pair inside a clause that names
several bins.

Why it never abstains
---------------------
sort_language scores +2 for the right bin, -1 for the wrong one, 0 for leaving the object on the
counter (``scene.yaml:80``).  A uniformly random bijection is worth (1/3)(+2) + (2/3)(-1) = 0 per
object, exactly what abstaining is worth, so a guess is never worse in expectation -- and a parse
that is merely probably right is strictly better.  ``resolve`` always returns a usable mapping.

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
    "left": "left", "leftmost": "left", "lefthand": "left", "port": "left",
    "right": "right", "rightmost": "right", "righthand": "right", "starboard": "right",
    "center": "center", "centre": "center", "central": "center", "centermost": "center",
    "centremost": "center", "middle": "center", "mid": "center", "middlemost": "center",
    "between": "center", "inbetween": "center",
}
# Colour words, including the adjective forms a paraphrase may use.
_COLOR_WORDS = {
    "red": "red", "reds": "red", "reddish": "red", "crimson": "red", "scarlet": "red",
    "yellow": "yellow", "yellows": "yellow", "yellowish": "yellow", "gold": "yellow",
    "golden": "yellow", "amber": "yellow",
    "blue": "blue", "blues": "blue", "bluish": "blue", "azure": "blue", "navy": "blue",
    # Exotic / hyphenated ("cherry-red" tokenises to "cherryred": hyphens are joined). The spaced
    # forms "cherry red" / "sky blue" already work because the trailing word is the colour.
    "ruby": "red", "cherryred": "red", "sapphire": "blue", "skyblue": "blue", "mustard": "yellow",
}
# Shape words.
#
# NB: "box"/"boxes" and "block"/"blocks" are deliberately absent. This challenge calls the three
# bins boxes ("three identical orange boxes"), and "blocks" is the natural generic noun for all
# three solids ("put all yellow blocks in the left bin"). Reading either as the cube shape turns
# every colour-keyed sentence into a compound-keyed one. Both cost real points before they were
# caught -- "box" by a hand-written probe, "blocks" by a held-out corpus.
_SHAPE_WORDS = {
    "cube": "cube", "cubes": "cube", "cubic": "cube",
    "square": "cube", "squares": "cube", "cuboid": "cube",
    "cylinder": "cylinder", "cylinders": "cylinder", "cylindrical": "cylinder",
    "circle": "cylinder", "circles": "cylinder", "circular": "cylinder", "round": "cylinder",
    "disc": "cylinder", "disk": "cylinder", "tube": "cylinder", "tubes": "cylinder",
    "triangle": "triangle", "triangles": "triangle", "triangular": "triangle",
    "prism": "triangle", "prisms": "triangle", "wedge": "triangle", "wedges": "triangle",
}
# Consulted ONLY when the sentence names fewer than two positional bins, so an ordinal used for
# something else ("the first red object") cannot hijack a sentence that already says left/right.
_ORDINAL_BINS = {
    "first": "left", "1": "left", "a": "left",
    "second": "center", "2": "center", "b": "center",
    "third": "right", "3": "right", "c": "right",
}
_BIN_NOUNS = {"bin", "bins", "box", "boxes", "one", "ones", "slot", "slots",
              "container", "containers", "crate", "crates", "tray", "trays"}

# "the remaining bin" / "the other box" / "the last one" names a bin by ELIMINATION: whichever bin
# the sentence has not already used. It is a bin reference, so it gets a placeholder here and is
# resolved against the leftovers. Reading it as the "everything else" catch-all is the opposite
# meaning, one word apart, and cost an entire corpus category before it was caught.
_ELIM = "__elim__"
_ELIM_WORDS = {"remaining", "other", "last", "final", "leftover", "unused", "spare"}

# A catch-all over the colours the sentence does not name -> resolve_bin's "default" key. Each
# pattern refuses a following bin noun, which is what separates it from an elimination phrase.
_NOT_A_BIN = r"(?!\s*(?:bin|box|one|ones|container|crate|tray|slot)\b)"
_DEFAULT_PATTERNS = tuple(p + _NOT_A_BIN for p in (
    r"\beverything\s+else\b", r"\banything\s+else\b", r"\ball\s+(?:the\s+)?others?\b",
    r"\bany\s+others?\b", r"\bthe\s+rest\b", r"\bthe\s+remaining\b", r"\bremaining\s+ones?\b",
    r"\ball\s+other\b", r"\bevery(?:thing)?\s+other\b", r"\bwhatever(?:'s|\s+is)?\s+left\b",
    r"\bthe\s+others\b",
))
# A clause that says "everything"/"all of them" and names a bin, with no key of its own, is also a
# catch-all: "Everything goes in the center bin, except red, which goes right."
_BLANKET = re.compile(r"\b(?:everything|anything|every\s+object|all\s+(?:the\s+)?(?:objects|items|"
                      r"pieces|shapes|of\s+them)|"
                      # A catch-all NOUN PHRASE: "all remaining pieces", "the other colour", "whatever
                      # remains". The head noun is what separates it from "the remaining BIN", which
                      # stays an elimination bin reference.
                      r"(?:remaining|other|leftover|unnamed|last|final)\s+(?:colou?rs?|shapes?|objects?|pieces?|items?)|"
                      r"whatever\s+remains|everything\s+(?:that\s+)?remains)\b")

_KW = "|".join(sorted([*_COLOR_WORDS, *_SHAPE_WORDS], key=len, reverse=True))
# "all but X" / "everything except the cubes": X is EXCLUDED, everything else is the catch-all.
_EXC = r"(?:but|except(?:\s+for)?|excluding|apart\s+from|other\s+than|save\s+for|with\s+the\s+exception\s+of)"
_EXC_KEYS = r"((?:\s*(?:the|and|or|,)?\s*\b(?:" + _KW + r")\b)+)"
_ALL_BUT = re.compile(r"\b(?:all|everything|anything|everyone|every|each)(?:\s+(?:the\s+)?(?:colou?rs?|shapes?|objects?|items?|pieces?|one))?\s+"
                      + _EXC + r"\s+" + _EXC_KEYS)
# Prefix form, at a sentence start: "Excluding the cylinders, everything goes left."
_EXC_FIRST = re.compile(r"(^|[.;!?]\s*)" + _EXC + r"\s+" + _EXC_KEYS + r"\s*,?")
# "everything else, i.e. the yellow stuff": the appositive colour/shape IS the key, not a blanket.
_APPOS = re.compile(r"\b(?:everything\s+else|anything\s+else|the\s+rest|(?:all\s+)?the\s+others?)\s*[,(]?\s*"
                    r"(?:i\.\s?e\.?|that\s+is|namely|meaning|which\s+is|which\s+means|i\s+mean)[,:]?\s*"
                    r"(?:the\s+)?(?=(?:" + _KW + r")\b)")
# A bin named by OTHER bins: "between the left and right ones" is the centre, though it contains
# the words left and right (and the clause splitter would cut it at "and").
_BIN_W = r"(?:left|right|center|centre|middle)(?:most|hand)?"
_BIN_N = r"(?:\s+(?:bins?|ones?|box(?:es)?))?"
_BETWEEN = re.compile(r"\bbetween\s+(?:the\s+)?" + _BIN_W + _BIN_N + r"\s+and\s+(?:the\s+)?" + _BIN_W + _BIN_N)
# "after the left one" / "before the right one" / "next to the left one": a neighbour of an end bin
# is the centre; after/before the centre is an end. Anything else is left alone.
_NEIGHBOUR = re.compile(r"\b(after|before|next\s+to|beside|adjacent\s+to)\s+(?:the\s+)?(left|right|center|centre|middle)"
                        r"(?:most|hand)?" + _BIN_N + r"\b")
_NEIGHBOUR_BIN = {"after": {"left": "middle", "center": "right", "centre": "right", "middle": "right"},
                  "before": {"right": "middle", "center": "left", "centre": "left", "middle": "left"},
                  "next": {"left": "middle", "right": "middle"}}
# Phrases whose "left" is NOT the left bin: rewrite them into the elimination vocabulary first.
_LEFT_NOUN = re.compile(r"\b(whatever|what|(?:the|any)\s+(?:colou?rs?|shapes?|objects?|pieces?))"
                        r"(?:\s+(?:colou?rs?|shapes?|objects?|pieces?|items?))?(?:'s|\s+(?:is|are))\s+left"
                        r"(?:\s+over)?\b")
# Not-yet-used, said negatively: "whatever colour has not been mentioned", "the bin nobody has used".
_UNUSED_KEY = re.compile(r"\bwhatever\s+(?:colou?rs?|shapes?|objects?|pieces?)\s+(?:has|have|is|are)(?:\s+not|n't)\s+(?:been\s+)?\w+")
_UNUSED_BIN = re.compile(r"\b(bin|box|slot|one)\s+(?:that\s+)?(?:(?:nobody|no\s+one)\s+(?:has\s+)?\w+|"
                         r"(?:is|has)(?:\s+not|n't)\s+(?:been\s+)?\w+)")
_LEFT_BIN = re.compile(r"\b(bin|box|slot|one)\s+(?:that(?:'s|\s+is)\s+|which\s+is\s+)?"
                       r"(?:left|remains|remaining)(?:\s+over)?\b(?!\s+of\b)")  # "left of centre" is a direction

# Clause boundaries. Punctuation, plus the words that introduce an exception or an aside: those
# carry a key that belongs to a DIFFERENT bin than the clause before them.
_SEG_RE = re.compile(r"[,;:.!?\n]+|\b(?:and|then|also|plus|except|excluding|but|regardless|"
                     r"whereas|while|apart\s+from|other\s+than|rather\s+than)\b")
_TOKEN_RE = re.compile(r"[a-z]+|\d+")
# "Don't put red in the left bin" / "red doesn't go centre" / "no red in the right bin" / "but not
# the blue ones": a clause with a negator, a key and a bin names a bin to AVOID, not to use.
_NEG = re.compile(r"\b(?:don't|dont|do\s+not|doesn't|doesnt|does\s+not|never|shouldn't|should\s+not|"
                  r"mustn't|must\s+not|not|no)\b")
_NEG_SPLIT = re.compile(r"([.;!?]|,?\s*\bbut\b)")
# "Swap red and blue: red right, blue left." -- the targets are spelled out, the swap is decoration.
_SWAP = re.compile(r"\bswap\s+(?:the\s+)?(?:" + _KW + r")\s+(?:ones\s+)?(?:and|with)\s+(?:the\s+)?(?:"
                   + _KW + r")(?:\s+ones)?\s*[:,]")
# "..., and so do the blue ones": inherits the bin of the clause before it.
_SO_DO = re.compile(r"\band\s+so\s+(?:do|does|should|would)\s+(?:the\s+)?((?:" + _KW + r")\b(?:\s+ones?)?)")
# "Left to right: red, yellow, blue." is the bins-first parallel list in disguise.
# "leave the red ones", "don't move the blue ones", "let the yellow ones stay": the counter's 0 IS the answer.
_LK = r"\s+(?:the\s+|all\s+)?(?:" + _KW + r")\b(?:\s+(?:ones|objects|items|pieces))?"
# "Let red go left" / "Leave red in the left bin" are PLACEMENTS: leave/let only mean leave-alone when
# a stay-phrase (or the end of the clause) follows the key. ignore / don't move|touch need no test.
_STAY = (r"(?=\s*(?:where\s+(?:it|they)\s+(?:is|are)|alone|be\b|as\s+(?:it\s+|they\s+)?(?:is|are)|in\s+place|put\b|"
         r"on\s+the\s+(?:counter|table)|untouched|out\s+of\s+it|stays?\b|[.,;!]|$))")
_LEAVE = re.compile(r"\b(?:(?:ignore|(?:don'?t|do\s+not|never)\s+(?:move|touch))" + _LK + r"|(?:leave|let)" + _LK + _STAY + r")")
# "Bin A is the left bin, bin B is the right one" / "the left bin is box A": the sentence names its own
# labels. Extracted first, substituted into the placements, and the definition clause removed.
_LABEL = r"(?:bin|box|container)\s+([abc123])\b"
_DEF_BIN = r"(left|right|center|centre|middle)(?:most|hand)?(?:\s+(?:bin|box|container|one))?"
_DEFS = (re.compile(r"\b" + _LABEL + r"\s+(?:is|=)\s+(?:the\s+)?" + _DEF_BIN + r"\b"),
         re.compile(r"\b(?:the\s+)?(?:(?:bin|box|container)\s+(?:on|at)\s+the\s+)?" + _DEF_BIN
                    + r"\s+(?:is|=)\s+" + _LABEL, 0))
_L2R = re.compile(r"\bleft\s*(?:-|to|->)\s*right\s*[:,-]?\s*")
_R2L = re.compile(r"\bright\s*(?:-|to|->)\s*left\s*[:,-]?\s*")


def _tokens(text):
    """Lowercased word tokens. Hyphens join ('left-hand' -> 'lefthand') so the bin table catches them."""
    return _TOKEN_RE.findall(text.lower().replace("-", "").replace("_", ""))


def _ordinals_ok(toks):
    """True when the sentence enumerates bins by ordinal or letter ("the first bin, the second...").

    Requires at least two distinct ones and at least one sitting next to the noun, so a lone
    ordinal ("the first red object") can never be read as a bin.
    """
    found = [i for i, t in enumerate(toks) if t in _ORDINAL_BINS]
    if len({_ORDINAL_BINS[toks[i]] for i in found}) < 2:
        return False
    return any(toks[j] in _BIN_NOUNS for i in found for j in (i - 1, i + 1) if 0 <= j < len(toks))


def _mentions(toks, allow_ordinals):
    """(index, kind, value) for every colour, shape and bin word in one clause, in order."""
    out = []
    for i, t in enumerate(toks):
        # "left of the middle one": the word after "<bin> of" is a landmark, not a bin being named.
        k = i - 1 - (i >= 1 and toks[i - 1] == "the")  # step back over "the"
        landmark = k >= 1 and toks[k] == "of" and (toks[k - 1] in _BIN_WORDS or toks[k - 1] in _BIN_NOUNS)
        if t in _BIN_WORDS and not landmark:
            out.append((i, "bin", _BIN_WORDS[t]))
        if t in _COLOR_WORDS:
            out.append((i, "color", _COLOR_WORDS[t]))
        if t in _SHAPE_WORDS:
            out.append((i, "shape", _SHAPE_WORDS[t]))
        if t in _ELIM_WORDS and i + 1 < len(toks) and toks[i + 1] in _BIN_NOUNS:
            out.append((i, "bin", _ELIM))

    if allow_ordinals:
        out.extend((i, "bin", _ORDINAL_BINS[t]) for i, t in enumerate(toks) if t in _ORDINAL_BINS)
    out.sort()
    return out


def _keys_from(mentions, compound_window=2):
    """Decide what this clause keys on: [(key, token_index), ...].

    A colour and a shape within `compound_window` tokens ("red cubes") can be one compound key. A
    clause naming any colour keys on colour, otherwise on shape -- that order matters because the
    organisers said the finals lead with colour, and because "the red objects" must not be read as
    a shape mention.
    """
    colors = [(i, v) for i, k, v in mentions if k == "color"]
    shapes = [(i, v) for i, k, v in mentions if k == "shape"]

    compounds, used_c, used_s = [], set(), set()
    for ci, cv in colors:
        near = sorted((abs(ci - si), si, sv) for si, sv in shapes if abs(ci - si) <= compound_window)
        for _, si, sv in near:
            if si not in used_s:
                compounds.append(((cv, sv), ci))
                used_c.add(ci)
                used_s.add(si)
                break
    if compounds and len(used_c) == len(colors) and len(colors) >= 1:
        return compounds, "compound"
    if colors:
        return [(v, i) for i, v in colors], "color"
    if shapes:
        return [(v, i) for i, v in shapes], "shape"
    return [], None


def _assign(keys, bins_):
    """Min total token-distance pairing between the distinct keys and distinct bins of one clause.

    Solved exactly: with at most 3 bins the permutation count is 6. Ties break towards mention
    order, because English lists a key and its bin in the same order ("blue and red go in the left
    and center bins respectively").
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
        return sum(1 for a in range(len(ks)) for b in range(a + 1, len(ks))
                   if (first_key[ks[a]] - first_key[ks[b]]) * (first_bin[bs[a]] - first_bin[bs[b]]) < 0)

    n = min(len(key_order), len(bin_order))
    best, best_cost = None, None
    for ks in itertools.permutations(key_order, n):
        for bs in itertools.permutations(bin_order, n):
            cost = (sum(dist(k, b) for k, b in zip(ks, bs)), inversions(ks, bs))
            if best_cost is None or cost < best_cost:
                best, best_cost = dict(zip(ks, bs)), cost
    return best or {}


def _clauses(low, ord_ok):
    """Split into clauses, but only where BOTH sides carry a key->bin binding.

    Splitting on every comma destroys an enumeration: "the red, yellow and blue objects in the
    left, center and right bins, respectively" has a comma inside each list, and each piece on its
    own is meaningless. So the pieces are re-joined until a piece can stand as a binding AND
    everything after it can too. A clause is a binding when it names a bin and either a key or a
    blanket word ("everything").
    """
    pieces = [p for p in _SEG_RE.split(low) if p and p.strip()]

    def counts(text_):
        ms = _mentions(_tokens(text_), ord_ok)
        bins_ = {v for _, k, v in ms if k == "bin"}
        keys = {k for k, _ in _keys_from(ms)[0]}
        return keys, bins_

    def binding(text_):
        keys, bins_ = counts(text_)
        return bool(bins_) and (bool(keys) or bool(_BLANKET.search(text_)))

    out, buf = [], []
    for n, piece in enumerate(pieces):
        buf.append(piece)
        here, rest = " ".join(buf), " ".join(pieces[n + 1:])
        if not (binding(here) and rest):
            continue
        k_rest, b_rest = counts(rest)
        # A clause closes when the rest binds too -- or when the rest names keys but NO bin
        # ("...triangles go center, | regardless of color: red, yellow or blue"): that fragment
        # has to stand alone, or its keys inherit this clause's bin.
        if not (binding(rest) or k_rest):
            continue
        # A clause holding more keys than bins is ambiguous: it is either many-to-one ("red and
        # yellow go in the left bin") or a list cut short ("the red, yellow and blue objects in
        # the left, | center and right bins"). Counting decides it: if what remains has MORE bins
        # than keys, those spare bins belong to the keys sitting here, so keep reading.
        k_here, b_here = counts(here)
        # (A rest that is a keyless catch-all owns its bin: "cubes and cylinders go left; any leftover
        # objects go right" must not hand the spare bin to the cylinders.)
        # Only the NEXT piece decides whether a blanket owns the spare bin; a blanket further on
        # ("...left and | right bins, everything else in the centre") does not.
        if len(k_here) > len(b_here) and len(b_rest) > len(k_rest) and not _BLANKET.search(pieces[n + 1]):
            continue
        # Mirror case, bins listed first: "the left, | center and right bins take red, yellow and
        # blue" -- spare keys in the rest belong to the bins sitting here.
        if len(b_here) > len(k_here) and len(k_rest) > len(b_rest):
            continue
        out.append(here)
        buf = []
    if buf:
        tail = " ".join(buf)
        k_tail, _ = counts(tail)
        # A keyless fragment ("respectively") belongs to the clause before it. A fragment that
        # names keys but no bin must stay separate, or its keys would inherit that clause's bin:
        # "...triangles go center, regardless of color: red, yellow or blue" must not put red in
        # the center. Left alone it becomes a deferred key and the kind filter drops it.
        #
        # But a CATCH-ALL tail is keyless BY DESIGN -- "everything that remains on the right"
        # names a bin and deliberately no key, and that is the whole point of it. Gluing it on
        # handed the previous clause two bins and one key, so `_assign` paired the key with the
        # nearer bin and the catch-all's bin was silently dropped: every unnamed shape ended up
        # with no bin at all. So only glue a tail that cannot stand on its own.
        if out and not k_tail and not binding(tail):
            out[-1] += " " + tail
        else:
            out.append(tail)
    return out or [low]


def _aliases(low):
    """Apply label definitions ("bin A is the left bin") to the placements; no definitions -> unchanged."""
    found = []
    for n, rx in enumerate(_DEFS):
        for m in rx.finditer(low):
            lab, b = (m.group(1), m.group(2)) if n == 0 else (m.group(2), m.group(1))
            found.append((m.span(), lab, _BIN_WORDS[b]))
    labs = [(lab, b) for _, lab, b in found]
    # Self-contradictory (a label twice with different bins, or two labels on one bin): keep the defaults.
    if not found or len(set(labs)) != len({l for l, _ in labs}) or len({b for _, b in labs}) != len(set(labs)):
        return low
    table = dict(labs)
    # Partial definition: undefined labels take the still-unused bins in label order, not the global default.
    order = "abc" if any(l in "abc" for l in table) else "123"
    free = [b for b in BINS if b not in table.values()]
    table.update(zip([l for l in order if l not in table], free))
    for (a, z), _, _ in sorted(found, reverse=True):
        low = low[:a] + " " + low[z:]
    low = re.sub(r"(?:[\s,;]|\band\b)+(?=[.!?])", "", low)  # leftovers of a removed definition list
    return re.sub(r"\b(?:bin|box|container)\s+([abc123])\b", lambda m: f"the {table[m.group(1)]} bin" if m.group(1) in table else m.group(0), low)


# A keyless catch-all tail that says the leftovers STAY ("leave everything else on the counter",
# "don't move anything else"). Cut out before any other machinery sees it: it names no key, so it must
# neither create a default bin nor feed the negation/elimination rules that would re-derive a named key.
_REST_STAYS = re.compile(
    r"(?:,|\band\b|\.)?\s*(?:please\s+)?(?:"
    r"(?<!n't )(?<!not )(?<!never )(?:leave|keep|let)\s+(?:everything\s+else|anything\s+else|all\s+else|the\s+rest)\s+"
    r"(?:on\s+the\s+(?:counter|table)|where\s+(?:it|they)\s+(?:is|are)|alone|be|as\s+is|untouched|in\s+place)"
    r"|(?:everything\s+else|the\s+rest)\s+stays?\s+"
    r"(?:on\s+the\s+(?:counter|table)|where\s+(?:it|they)\s+(?:is|are)|put|in\s+place|untouched)"
    r"|(?:do\s*n[o']?t|do\s+not|never)\s+(?:move|touch)\s+(?:anything\s+else|everything\s+else|the\s+rest)"
    r")(?=\s*(?:[.,;!]|$))")  # must end the clause: "don't move anything else to the left" is not a tail


def resolve(text, fallback=None):
    """Instruction sentence -> mapping for scene.resolve_bin(). Never returns None.

    >>> resolve("Put the blue objects in the left bin, the red objects in the center bin "
    ...         "and the yellow objects in the right bin.") == {
    ...     "blue": "left", "red": "center", "yellow": "right"}
    True
    >>> resolve("The centre bin is for red, the left bin for blue, the right bin for yellow.") == {
    ...     "red": "center", "blue": "left", "yellow": "right"}
    True
    >>> resolve("Red and yellow go in the left bin, blue goes in the right bin.") == {
    ...     "red": "left", "yellow": "left", "blue": "right"}
    True
    """
    if not text or not str(text).strip():
        return dict(FIXED_MAPPING if fallback is None else fallback)
    low = _aliases(str(text).lower())
    # Structural pre-pass (see the patterns above). Order matters: appositive first, so the key
    # word survives; all-but second, so the excluded key never reaches clause parsing.
    rest_stays = bool(_REST_STAYS.search(low))
    low = _REST_STAYS.sub(" ", low)
    stays = rest_stays or bool(_LEAVE.search(low) or _NEG.search(low))
    low = _LEAVE.sub(" ", low)  # a key told to stay put is never mentioned again, so nothing completes it
    low = _APPOS.sub("the ", low)
    low = _BETWEEN.sub("middle", low)  # "the bin between the left and right ones" -> the centre bin
    low = _NEIGHBOUR.sub(lambda m: _NEIGHBOUR_BIN.get(m.group(1).split()[0] if m.group(1) in ("after", "before", "next to") else "next",
                                                       {}).get(m.group(2), m.group(0)), low)
    low = _LEFT_NOUN.sub("the remaining objects", low)
    low = _LEFT_BIN.sub(r"remaining \1", low)
    low = _UNUSED_KEY.sub("the remaining objects", low)
    low = _UNUSED_BIN.sub(r"remaining \1", low)
    excluded = []

    def _exclude(m):
        ks = [_COLOR_WORDS.get(t) or _SHAPE_WORDS[t] for t in _TOKEN_RE.findall(m.group(1)) if t in _COLOR_WORDS or t in _SHAPE_WORDS]
        cs = [k for k in ks if k in COLORS]
        excluded.extend(cs or ks)
        return "all remaining objects "
    low = _ALL_BUT.sub(_exclude, low)
    low = _EXC_FIRST.sub(lambda m: m.group(1) + ("" if _exclude(m) else ""), low)
    low = re.sub(r"\botherwise\b", "everything else", low)  # "..., otherwise use the right bin" -> catch-all
    low = _SWAP.sub("", low)
    low = _L2R.sub("the left, center and right bins take ", low)
    low = _R2L.sub("the right, center and left bins take ", low)

    def _inherit(m):  # "red goes left, and so do the blue ones" -> "..., and the blue ones go in the left bin"
        prev = [_BIN_WORDS[t] for t in _tokens(low[: m.start()]) if t in _BIN_WORDS]
        return f"and the {m.group(1)} go in the {prev[-1]} bin" if prev else m.group(0)
    low = _SO_DO.sub(_inherit, low)

    # A descriptive opening sentence ("There are red, yellow and blue objects on the table.") names
    # colours but no bin, and would glue them onto the clause after it. Drop it. Ordinals and digits
    # are kept: "First bin takes red." is a bin sentence.
    head, _, tail = low.partition(". ")
    htoks = _tokens(head)
    if tail.strip() and not any(k == "bin" for _, k, _ in _mentions(htoks, False)) \
            and not any(t in _ORDINAL_BINS and t not in ("a", "b", "c") for t in htoks):
        low = tail

    # Negated clauses are cut out and resolved by elimination after the positive ones (see below).
    negs, parts, last = {}, _NEG_SPLIT.split(low), []
    for n in range(0, len(parts), 2):
        ms = _mentions(_tokens(parts[n]), False)
        bs = [v for _, k, v in ms if k == "bin" and v != _ELIM]
        # "..., but not the blue ones": no bin of its own, so it negates the previous clause's bin.
        if not bs and _NEG.search(parts[n]) and last:
            bs = last[-1:]
        last = bs or last
        if not (_NEG.search(parts[n]) and bs and any(k != "bin" for _, k, _ in ms)):
            continue
        if not _keys_from(ms)[0]:
            continue
        for key, _ in _keys_from(ms)[0]:
            negs.setdefault(key, set()).add(bs[0])
        parts[n] = ""
    low = "".join(parts)

    # Both of these are decided over the WHOLE sentence, because a clause sees too little.
    all_toks = _tokens(low)
    whole = _mentions(all_toks, allow_ordinals=False)
    positional = len({v for _, k, v in whole if k == "bin" and v != _ELIM})
    allow_ord = positional < 2 and _ordinals_ok(all_toks)
    if allow_ord:
        whole = _mentions(all_toks, allow_ordinals=True)

    mapping, deferred, kinds, mentioned = {}, [], [], []
    for seg in _clauses(low, allow_ord):
        toks = _tokens(seg)
        ms = _mentions(toks, allow_ord)
        bins_ = [(v, i) for i, k, v in ms if k == "bin"]
        keys, kind = _keys_from(ms)
        if kind:
            kinds.append(kind)
        mentioned.extend(k for k, _ in keys)

        if not bins_:
            deferred.extend(k for k, _ in keys)
            continue
        if not keys:
            # A clause naming a bin and "everything", with no key of its own, is the catch-all.
            if _BLANKET.search(seg):
                mapping.setdefault("default", bins_[0][0])
            continue
        if len({b for b, _ in bins_}) == 1:
            # One bin in the clause claims every key in it. This is what makes a many-to-one
            # mapping possible ("red and yellow go in the left bin"), which resolve_bin allows.
            for k, _ in keys:
                mapping.setdefault(k, bins_[0][0])
        else:
            for k, v in _assign(keys, bins_).items():
                mapping.setdefault(k, v)

    # A shape that appears in EVERY compound key is a filler noun, not a discriminator:
    # "the red cubes go left, the blue cubes go right" keys on colour. Keeping the tuple would
    # leave every non-cube object unmapped.
    comp = [k for k in mapping if isinstance(k, tuple)]
    if comp and len({k[1] for k in comp}) == 1 and len(comp) >= 2:
        mapping = {(k[0] if isinstance(k, tuple) else k): v for k, v in mapping.items()}

    # "the remaining bin" -> whichever bin the sentence did not name.
    if _ELIM in mapping.values():
        free = [b for b in BINS if b not in set(mapping.values())]
        if len(free) == 1:
            mapping = {k: (free[0] if v == _ELIM else v) for k, v in mapping.items()}
        else:  # ambiguous: drop the key rather than send an object to a guessed bin
            mapping = {k: v for k, v in mapping.items() if v != _ELIM}

    # Negation by elimination: drop the named bin, take the bin nobody claimed, else the key's usual
    # bin (FIXED_MAPPING) if still allowed, else the first allowed bin -- left, center, right order,
    # arbitrary but deterministic; any choice is >= 1/2 right against 1/3 blind.
    for k, no in negs.items():
        if k in mapping:
            continue
        taken = set(mapping.values())
        allowed = [b for b in BINS if b not in no]
        pool = [b for b in allowed if b not in taken] or allowed
        usual = FIXED_MAPPING.get(k[0] if isinstance(k, tuple) else k)
        mapping[k] = usual if usual in pool else pool[0]

    # An explicit catch-all phrase, pointed at the bin nearest it.
    if "default" not in mapping:
        for pat in _DEFAULT_PATTERNS:
            m = re.search(pat, low)
            if not m:
                continue
            at = len(_TOKEN_RE.findall(low[: m.start()]))
            all_bins = [(v, i) for i, k, v in whole if k == "bin" and v != _ELIM]
            claimed = set(mapping.values())
            free = ([(abs(i - at), b) for b, i in all_bins if b not in claimed]
                    or [(abs(i - at), b) for b, i in all_bins])
            if free:
                mapping["default"] = min(free)[1]
            break

    # A key named in a clause with no bin gets a leftover bin rather than nothing: an unmapped
    # colour leaves the policy nowhere to put that object (worth 0), where a guess from the
    # leftovers is worth 0 at worst and +2 when right. Only when its kind matches the keys that
    # DID get a bin, so an incidental "red, yellow or blue" cannot displace shape keys.
    if mapping and "default" not in mapping:
        want = kinds[0] if kinds else None
        pool = COLORS if want == "color" else SHAPES if want == "shape" else ()
        seen = set()
        missing = [k for k in deferred
                   if k in pool and k not in mapping and not (k in seen or seen.add(k))]
        free = [b for b in BINS if b not in set(mapping.values())]
        for k, b in zip(missing, free):
            mapping[k] = b

    # "all but X": X is not the catch-all's bin. Guess it from the leftovers (+0.5 expected) rather
    # than let it fall through to "default" (-1 certain).
    if "default" in mapping:
        for k in excluded:
            free = [b for b in BINS if b not in set(mapping.values())]
            if k not in mapping and free:
                mapping[k] = free[0]

    # COMPLETION: a key the sentence named that still has no bin is worth 0 (object stays on the
    # counter), so giving it one is worth 3p - 1 >= -1 and +2 when right. This is NOT the "bijection
    # repair" deleted earlier, which MOVED keys that already had a bin (+2 -> -1, a 3-point loss, and
    # it broke scored many-to-one). Completion only touches keys currently scoring 0 and can never
    # damage a correct one -- do not remove it by analogy. Order: bins still free, in left/center/right
    # order, against the keys in sentence order (one key + one free bin = elimination); then a colour's
    # FIXED_MAPPING bin. A "leave it alone" key is cut out of the sentence up front (_LEAVE) so it keeps its 0, and a "default" already covers the rest.
    if mapping and "default" not in mapping:
        kind_of = lambda k: 0 if isinstance(k, tuple) else 1 if k in COLORS else 2  # noqa: E731
        if comp and len({k[1] for k in comp}) == 1 and len(comp) >= 2:
            mentioned = [k[0] if isinstance(k, tuple) else k for k in mentioned]
        kept = {kind_of(k) for k in mapping}
        todo = [k for k in dict.fromkeys(mentioned) if k not in mapping and kind_of(k) in kept]
        free = [b for b in BINS if b not in set(mapping.values())]
        for i, k in enumerate(todo):
            b = free[i] if i < len(free) else FIXED_MAPPING.get(k)
            if b:
                mapping[k] = b

    # Two colours on two different bins, one bin free, nothing told to stay (or negated: a "but not X"
    # bin is a guess already): the third colour is
    # implied and has no bin (0). Same 0 / -1 / +2 bet as completion; two colours sharing one bin
    # leave two bins free, so a many-to-one sentence can never reach this.
    if (len(mapping) == 2 and all(k in COLORS for k in mapping) and len(set(mapping.values())) == 2
            and not stays):
        free = [b for b in BINS if b not in mapping.values()]
        mapping[next(c for c in COLORS if c not in mapping)] = free[0]

    if not mapping:
        return dict(FIXED_MAPPING if fallback is None else fallback)
    return mapping
