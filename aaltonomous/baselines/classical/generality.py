"""instruction.generality: closed-vocabulary last-resort parser for phrasings the stricter layers abstain on.

Runs ONLY when the structured/strict stack has placed nothing (every value None), on the ORIGINAL text. It understands a
small grammar completely or returns None (= keep the abstain):

  * rule clauses "<keys> <bin>" / "<bin>: <keys>" (any word order, comma / "and" / sentence separated);
  * "everything / all objects <bin>" (-> "default") and "everything except K <bin>" (K -> abstain, default -> bin);
  * "the rest / everything else <bin>" (-> "default");
  * foreign words (fi cs de sv es + Mandarin) mapped onto the same classes.

Every word must be classified (key, bin, bin noun, filler, quantifier, except, conjunction). An unknown word, a negation,
a count, a conditional, an ordinal, an unsafe verb, "block(s)", "amber", two different bins for one key, or two different
catch-alls -> None. A key with two different bins is abstained for that key only. Pure stdlib, never raises.
"""
from __future__ import annotations

import re
import unicodedata

LEFT, CENTER, RIGHT = "left", "center", "right"

_COL = {
    "red": "red", "reds": "red", "reddish": "red", "crimson": "red", "scarlet": "red", "ruby": "red", "orange": "red",
    "oranges": "red", "orangish": "red", "orangeish": "red", "yellow": "yellow", "yellows": "yellow", "yellowish": "yellow",
    "gold": "yellow", "golden": "yellow", "lemon": "yellow", "mustard": "yellow",
    "blue": "blue", "blues": "blue", "bluish": "blue", "azure": "blue", "navy": "blue", "sapphire": "blue",
    "purple": "blue", "purples": "blue", "violet": "blue", "violets": "blue", "lilac": "blue", "purplish": "blue",
}
_SHP = {
    "cube": "cube", "cubes": "cube", "cubic": "cube", "cuboid": "cube", "cuboids": "cube", "square": "cube",
    "squares": "cube", "dice": "cube",
    "cylinder": "cylinder", "cylinders": "cylinder", "cylindrical": "cylinder", "round": "cylinder",
    "circular": "cylinder", "circle": "cylinder", "circles": "cylinder", "disc": "cylinder", "disk": "cylinder",
    "tube": "cylinder", "tubes": "cylinder",
    "triangle": "triangle", "triangles": "triangle", "triangular": "triangle", "prism": "triangle",
    "prisms": "triangle", "wedge": "triangle", "wedges": "triangle", "pyramid": "triangle", "pyramids": "triangle",
    "pointy": "triangle", "pointed": "triangle", "threesided": "triangle",
}
_BIN = {
    "left": LEFT, "leftmost": LEFT, "lefthand": LEFT, "right": RIGHT, "rightmost": RIGHT, "righthand": RIGHT,
    "center": CENTER, "centre": CENTER, "central": CENTER, "middle": CENTER, "centermost": CENTER,
    "centremost": CENTER, "mid": CENTER,
}
_BINNOUN = frozenset("bin bins box boxes container containers tray trays basket baskets crate crates tub tubs "
                     "bucket buckets compartment compartments slot slots".split())
_ALL = frozenset("everything all every each entire".split())
_EXCEPT = frozenset("except excluding but".split())
_REST = ("all else", "everything else", "anything else", "the rest", "all others", "all the others", "all other", "the others",
         "the remaining", "remaining", "the other ones", "all remaining", "any others", "others", "the rest of them")
_AND = frozenset(("and", "then", "plus", "also"))
_FILL = frozenset("""put place move drop sort send go goes belong belongs is are be should must will gets get receives
receive takes take for the a an of in into to onto on at inside by toward towards side one ones objects object items item
things thing pieces piece shapes shape stuff colour color colours colors coloured colored please kindly now just only
them it they those these that this hand far very most can you could would respectively i want need let lets let's
within your then next collect gather bring carry transfer relocate deposit store shift lay set arrange""".split())
_FILL = _FILL - {"can"}  # "can" is a noun for a cylinder and a modal verb: ambiguous, so unknown
_IGNORE_PHRASES = (("no", "matter", "the", "colour"), ("no", "matter", "the", "color"), ("no", "matter", "what"),
                   ("regardless",), ("irrespective",))

# ------------------------------------------------------------------------------------------ foreign languages
_X_ALL = "alles allt todo todos todas vsechno vse vsechny kaikki kaikkea alla allting".split()
_X_EXCEPT = "ausser außer utom excepto salvo kromě krome kromĕ paitsi".split()
_X_FILL = ("sakerna saker sachen dinge cosas veci asiat kommt kommen geht gehen lo va jde jdou gar menee menevat "
           "esineet kappaleet foremal foremalen").split()
_X_FILL += "dejte dej polozte presunte lozte dat pridej nastav".split()
_X_COL = {"modrych": "blue", "modrou": "blue", "cervenych": "red", "zlutych": "yellow", "fialovych": "blue",
          "oranzovych": "red", "roten": "red", "blauen": "blue", "gelben": "yellow", "violetten": "blue"}
_X_BN = frozenset("caja cajas schachtel schachteln lada ladan ladon rasia rasiat rasian krabice krabici krabicky".split())
_X_ABORT = frozenset("bloque bloques block blocks blocke blocken blok bloky blocket klotz klotze kloss klossar palikka "
                     "palikat".split())
_X_BINW = {"mittersta": CENTER, "mellersta": CENTER, "mittlere": CENTER}
_X_REST = ("alles andere", "allt annat", "todo lo demas", "todo lo demás", "všechno ostatní", "vsechno ostatni",
           "kaikki muut", "kaikki muu", "ostatní", "ostatni", "muut")

_ZH = (  # longest match first
    ("所有东西", "all"), ("所有物体", "all"), ("所有物品", "all"), ("所有", "all"), ("全部", "all"), ("一切", "all"),
    ("其他的", "rest"), ("其他", "rest"), ("其余的", "rest"), ("其余", "rest"),
    ("除了", "except"), ("除", "except"),
    ("不要", "NEG"), ("别", "NEG"), ("不", "NEG"), ("没有", "NEG"),
    ("立方体", "cube"), ("正方体", "cube"), ("方块", "cube"), ("圆柱体", "cylinder"), ("圆柱", "cylinder"),
    ("三角形", "triangle"), ("三棱柱", "triangle"), ("棱柱", "triangle"), ("三角", "triangle"),
    ("红色", "red"), ("红", "red"), ("橙色", "red"), ("橙", "red"), ("黄色", "yellow"), ("黄", "yellow"),
    ("蓝色", "blue"), ("蓝", "blue"), ("紫色", "blue"), ("紫", "blue"),
    ("左边", "left"), ("左侧", "left"), ("左", "left"), ("右边", "right"), ("右侧", "right"), ("右", "right"),
    ("中间", "center"), ("中央", "center"), ("正中", "center"),
    ("箱子", "bin"), ("盒子", "bin"), ("容器", "bin"), ("篮子", "bin"), ("箱", "bin"), ("盒", "bin"),
    ("放进", "put"), ("放到", "put"), ("放入", "put"), ("放在", "put"), ("放", "put"), ("把", "put"), ("将", "put"),
    ("的", "the"), ("里", "the"), ("里面", "the"), ("都", "all"), ("个", "the"), ("请", "please"), ("东西", "the"), ("物体", "the"), ("物品", "the"),
)
_ZH_PUNCT = {"，": ",", "。": ".", "；": ";", "：": ":", "！": "!", "？": "?", "、": ","}

_TYPO = {"rd": "red", "blu": "blue", "bleu": "blue", "bue": "blue", "ylw": "yellow", "lft": "left", "rgt": "right",
         "cntr": "center", "ctr": "center", "cente": "center", "centr": "center", "midle": "middle",
         "cubs": "cubes", "cilinder": "cylinder", "cilinders": "cylinders", "triangls": "triangles",
         "cubez": "cubes"}
_LEX = {**{w: ("COL", c) for w, c in _COL.items()}, **{w: ("SHP", s) for w, s in _SHP.items()},
        **{w: ("BIN", b) for w, b in _BIN.items()}, **{w: ("BN", None) for w in _BINNOUN},
        **{w: ("ALL", None) for w in _ALL}, **{w: ("EXC", None) for w in _EXCEPT},
        **{w: ("AND", None) for w in _AND}, **{w: ("FILL", None) for w in _FILL}}
_FUZZ_TARGETS = ("yellow orange purple violet cube cubes cylinder cylinders triangle triangles prism prisms left right "
                 "center centre middle leftmost rightmost central container containers basket baskets").split()
_NO_FUZZ = frozenset(_FILL) | {"orange"}


def _fold(s: str) -> str:
    s = unicodedata.normalize("NFD", s.casefold().replace("ß", "ss"))
    return unicodedata.normalize("NFC", "".join(c for c in s if not unicodedata.combining(c)))


def _lev1(a: str, b: str) -> bool:
    """True if a, b differ by one substitution / insertion / deletion / adjacent swap."""
    if a == b:
        return True
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        d = [i for i in range(len(a)) if a[i] != b[i]]
        return len(d) == 1 or (len(d) == 2 and d[1] == d[0] + 1 and a[d[0]] == b[d[1]] and a[d[1]] == b[d[0]])
    if len(a) > len(b):
        a, b = b, a
    i = 0
    while i < len(a) and a[i] == b[i]:
        i += 1
    return a[i:] == b[i + 1:]


def _repair(w: str):
    if w in _TYPO:
        return _TYPO[w]
    if len(w) < 5:
        return None
    c = {t for t in _FUZZ_TARGETS if _lev1(w, t) and t[0] == w[0]}
    vals = {_LEX[t] for t in c}
    return next(iter(c)) if len(vals) == 1 else None


def _classify(word: str, foreign: bool):
    if word in _X_ABORT:
        return None
    if word in _X_BN:
        return ("BN", None)
    if word in _X_COL:
        return ("COL", _X_COL[word])
    lex = _LEX.get(word)
    if lex and not (foreign and word == "a"):
        return lex
    if foreign and word == "a":
        return ("AND", None)
    if word in _X_ALL:
        return ("ALL", None)
    if word in _X_EXCEPT:
        return ("EXC", None)
    if word in _X_FILL:
        return ("FILL", None)
    if word in _X_BINW:
        return ("BIN", _X_BINW[word])
    if foreign:
        from .normalize import _ML_LEXICON  # {folded word: (kind, value)}
        kind, value = _ML_LEXICON.get(word, (None, None))
        if kind == "color":
            return ("COL", {"purple": "blue", "orange": "red"}.get(value, value))
        if kind == "shape":
            return ("SHP", value)
        if kind == "direction":
            return ("BIN", value)
        if kind == "bin":
            return ("BN", None)
        if kind in ("and", "ambig_and"):
            return ("AND", None)
        if kind in ("verb", "filler"):
            return ("FILL", None)
    fixed = _repair(word)
    if fixed and fixed in _LEX:
        return _LEX[fixed]
    return None


_NEG_WORDS = frozenset(("not", "dont", "don't", "never", "no", "none", "nothing", "nowhere", "avoid", "without", "leave",
                        "ignore", "skip", "unless", "instead", "swap", "switch", "exchange", "otherwise", "either",
                        "neither", "both", "nicht", "kein", "keine", "inte", "ingen", "nedavej", "nedavejte", "ala",
                        "alä", "pongas", "ponga", "evita", "nevkladej", "nikdy", "nie", "nunca", "aldrig", "nooit"))


def _zh_translate(text: str):
    """Mandarin text -> English-ish token string for the same grammar; None when a CJK char is unknown."""
    out, i, n = [], 0, len(text)
    while i < n:
        ch = text[i]
        if ch in _ZH_PUNCT:
            out.append(_ZH_PUNCT[ch]); i += 1; continue
        if not ("一" <= ch <= "鿿"):
            out.append(ch); i += 1; continue
        for zh, en in _ZH:
            if text.startswith(zh, i):
                out.append(" NEGATION " if en == "NEG" else f" {en} "); i += len(zh); break
        else:
            return None
    return "".join(out)


def _split_clauses(text: str):
    """Words with separators kept as tokens: ',' ';' '.' ':' -> sep."""
    return re.findall(r"[^\W\d_]+(?:['’][^\W\d_]+)?|\d+|[,;:.!?\n]", text)


def general_mapping(text):
    """Original instruction text -> mapping for resolve_bin, or None (not understood / unsafe: keep the abstain)."""
    try:
        parsed = _parse(text)
        if parsed is None:
            return None
        rules, excepts = parsed
        if any(k == ("all",) for k, _ in rules) and (len(rules) > 1 or excepts) and not excepts:
            return None                      # a blanket rule beside keyed rules is contradictory, not "everything else"
        return _merge(rules, excepts)
    except Exception:
        return None


_UNSAFE = re.compile(
    r"\b(?:hit|hits|punch|kick|smash|knock|break|crush|destroy|stab|shoot|hurt|attack|bash|fire|burn|ignite|explode|"
    r"bomb|poison|weapon|knife|kill)\b|"
    r"\b(?:at|on|toward|towards|onto|to|for|near|closest\s+to|nearest\s+to|beside)\s+(?:the\s+|a\s+|that\s+)?"
    r"(?:person|people|human|humans|man|woman|child|kid|someone|somebody|operator|me|face|head|"
    r"organisers?|judge|judges|audience|spectators?)\b|\bover\s+the\s+(?:bin|box|table|counter)\b|\bknock\w*\s+over\b",
    re.I)
# spatial relations / self-corrections / injections: the bin is not stated plainly
_AMBIGUOUS = re.compile(
    r"\b(?:left|right|centre|center|middle)\s+of\b|\b(?:closest|nearest|furthest|farthest)\b|\b(?:ignore|forget|disregard|scratch|never\s*mind|undo|cancel)\s+(?:that|this|it|the\s+"
    r"(?:above|previous|last)|what|your|all|everything)\b", re.I)
_COUNT = re.compile(
    r"\b(?:one|two|three|four|five|six|first\s+(?:two|three)|only\s+(?:one|two|three)|both|\d+)\s+"
    r"(?:(?:red|yellow|blue|orange|purple|violet|cube|cylinder|triangle|prism|round|square|small|big|large)\s+){0,2}"
    r"(?:objects?|items?|pieces?|things?|cubes?|cylinders?|triangles?|prisms?|ones?)\b", re.I)
_COUNT_OK = re.compile(r"\b(?:other|remaining|rest|left|right|centre|center|middle)\s+(?:two|three)\b|\bthe\s+other\s+"
                       r"(?:two|three)\b", re.I)


def veto(text) -> str | None:
    """Reason to abstain on EVERY object, whatever the other layers decided; None = no hazard. Never raises."""
    try:
        t = str(text or "")
        if _UNSAFE.search(t):
            return "unsafe-verb-or-target"
        if _AMBIGUOUS.search(t):
            return "relation-or-retraction"
        if _COUNT.search(t) and not _COUNT_OK.search(t):
            return "count-or-instance"
        parsed = _parse(t)
        if parsed is not None:
            rules, excepts = parsed
            if any(k == ("all",) for k, _ in rules) and not excepts and len(rules) > 1:
                return "blanket-vs-keyed-contradiction"
        return None
    except Exception:
        return None


def _parse(text):
    if not isinstance(text, str) or not text.strip() or len(text) > 600:
        return None
    foreign = False
    if re.search(r"[一-鿿]", text):
        text = _zh_translate(text)
        if text is None:
            return None
        foreign = False
    low = _fold(text)
    low = re.sub(r"(?<=\w)-(?=ish\b)", "", low)                 # orange-ish
    low = re.sub(r"three[- ]sided", "threesided", low)
    low = re.sub(r"(\w)-(?=\w)", r"\1", low) if not re.search(r"->|=>", low) else low
    low = re.sub(r"->|=>|→|=", " : ", low)
    for ph in ("no matter the colour", "no matter the color", "no matter what", "regardless of the colour",
               "regardless of the color", "regardless"):
        low = low.replace(ph, " ")
    for ph in _REST + tuple(_fold(p) for p in _X_REST):
        low = re.sub(r"(?<!\w)" + re.escape(ph) + r"(?!\w)", " REST ", low)
    low = re.sub(r"\betc\b\.?", " ", low)
    toks = _split_clauses(low)
    if not toks:
        return None
    from .normalize import _ML_SIGNALS        # a word only a foreign table knows
    foreign = any(t in _ML_SIGNALS or t in _X_ALL or t in _X_EXCEPT for t in toks)
    items = []   # (kind, value) with SEP markers
    for t in toks:
        if t in ",;:.!?\n":
            items.append(("SEP", t))
        elif t == "REST":
            items.append(("REST", None))
        elif t.isdigit():
            return None
        elif t in _NEG_WORDS or t == "negation" or t.endswith("n't"):
            return None
        else:
            c = _classify(t, foreign)
            if c is None:
                return None
            items.append(c)
    for i, (k, v) in enumerate(items):          # "but" is an exception marker only right after everything / all
        if k == "EXC" and toks[i] == "but":
            j = i - 1
            while j >= 0 and items[j][0] == "FILL":
                j -= 1
            if not (j >= 0 and items[j][0] == "ALL"):
                items[i] = ("AND", None)
    # orange as the colour of the bins: "orange left bin" / "orange box" -> filler (words after it up to the bin noun)
    for i, (k, v) in enumerate(items):
        if k == "COL" and toks[i] in ("orange", "oranges"):
            j = i + 1
            while j < len(items) and items[j][0] in ("BIN", "FILL"):
                j += 1
            if j < len(items) and items[j][0] == "BN" and not any(x[0] == "BN" for x in items[i + 1:j]):
                items[i] = ("FILL", None)
    # split into sentences (. ! ? newline), then comma/semicolon/colon/and-separated segments
    sentences, cur = [], []
    for it in items:
        if it[0] == "SEP" and it[1] in ".!?\n":
            sentences.append(cur); cur = []
        else:
            cur.append(it)
    sentences.append(cur)
    rules = []   # (keysets, bin) ; keys = ("all",) | ("rest",) | frozenset-ish tuple (cols, shps)
    excepts = []
    for sent in sentences:
        r = _sentence(sent)
        if r is None:
            return None
        rules += r[0]
        excepts += r[1]
    if not rules and not excepts:
        return None
    return rules, excepts


def _segments(sent):
    """Split a sentence at , ; : and conjunctions when both sides will hold a bin; return [[items]]."""
    segs, cur = [], []
    for it in sent:
        if it[0] == "SEP":
            segs.append((cur, it[1])); cur = []
        else:
            cur.append(it)
    segs.append((cur, ""))
    return segs


def _units(sent):
    """Comma pieces -> rule units. A colon glues its two sides ('left: blue'); 'and' splits when each side has a bin."""
    pieces, buf = [], []
    for seg, sep in _segments(sent):
        buf += seg
        if sep == ":":
            continue
        pieces.append(buf); buf = []
    if buf:
        pieces.append(buf)
    units = []
    for p in pieces:
        part, parts = [], []
        for it in p:
            if it[0] == "AND":
                parts.append(part); part = []
            else:
                part.append(it)
        parts.append(part)
        # re-join and-parts that do not each carry a bin ("red and yellow go left")
        merged = []
        for q in parts:
            merged.append(q)
        units += _join_and(merged)
    return [u for u in units if any(k != "FILL" for k, _ in u)]


def _join_and(parts):
    out = []
    pend = []
    for q in parts:
        has_bin = any(k == "BIN" for k, _ in q)
        has_key = any(k in ("COL", "SHP", "REST", "ALL") for k, _ in q)
        if not has_bin and has_key and not any(k == "EXC" for k, _ in q):
            pend += q                       # "red and ..." keys wait for the next part's bin
            continue
        if has_bin and not has_key and pend:
            out.append(pend + q); pend = []
            continue
        out.append(pend + q if (has_bin and pend) else q)
        pend = []
    if pend:
        out.append(pend)
    return out


def _sentence(sent):
    units = _units(sent)
    # a keys-only unit directly before a bin-only unit (or vice versa, "left bin, blue") shares it
    fixed = []
    i = 0
    while i < len(units):
        u = units[i]
        hb = any(k == "BIN" for k, _ in u)
        hk = any(k in ("COL", "SHP", "REST", "ALL") for k, _ in u)
        if hk and not hb and i + 1 < len(units):
            n = units[i + 1]
            if any(k == "BIN" for k, _ in n) and not any(k in ("COL", "SHP", "REST", "ALL") for k, _ in n):
                fixed.append(u + n); i += 2; continue
        fixed.append(u); i += 1
    rules, excepts = [], []
    for u in fixed:
        bins = {v for k, v in u if k == "BIN"}
        if len(bins) != 1 or sum(k == "BIN" for k, _ in u) > 2:
            return None
        b = next(iter(bins))
        # except part
        kinds = [k for k, _ in u]
        if "EXC" in kinds:
            e = kinds.index("EXC")
            before, after = u[:e], u[e + 1:]
            if any(k in ("COL", "SHP", "REST") for k, _ in before):
                return None
            if not any(k == "ALL" for k, _ in before):          # "except blue, everything goes left"
                ai = next((i for i, (k, _) in enumerate(after) if k == "ALL"), None)
                if ai is None:
                    return None
                after = after[:ai]
            else:
                after = [x for x in after if x[0] != "ALL"]
            ek = _keys(after)
            if ek is None or ek in (("all",), ("rest",)):
                return None
            excepts.append(ek)
            rules.append((("all",), b))
            continue
        keys = _keys(u)
        if keys is None:
            return None
        rules.append((keys, b))
    return rules, excepts


def _keys(u):
    cols = {v for k, v in u if k == "COL"}
    shps = {v for k, v in u if k == "SHP"}
    rest = any(k == "REST" for k, _ in u)
    allq = any(k == "ALL" for k, _ in u)
    if rest:
        return ("rest",) if not (cols or shps) else None
    if not cols and not shps:
        return ("all",) if allq else None
    if len(cols) > 1 and len(shps) > 1:
        return None
    if len(cols) > 1 and shps:           # a colour adjective comes BEFORE its shape noun ("red and blue cubes")
        ci = [i for i, (k, _) in enumerate(u) if k == "COL"]
        si = [i for i, (k, _) in enumerate(u) if k == "SHP"]
        if max(ci) > min(si):
            return None
    return (frozenset(cols), frozenset(shps))


def _cells(keys):
    cols, shps = keys
    if cols and shps:
        return [(c, s) for c in cols for s in shps]
    return list(cols or shps)


def _merge(rules, excepts):
    out, weak = {}, {}
    default = None
    for keys, b in rules:
        if keys == ("rest",) or keys == ("all",):
            if default is not None and default != b:
                return None
            default = b
            continue
        for cell in _cells(keys):
            if cell in out and out[cell] != b:
                out[cell] = None            # two bins for one key: abstain that key only
            elif cell not in out:
                out[cell] = b
    for ek in excepts:
        for cell in _cells(ek):
            if cell not in out:
                out[cell] = None
    if default is not None:
        out["default"] = default
    return out or None
