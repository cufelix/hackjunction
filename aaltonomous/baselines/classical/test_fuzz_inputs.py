"""Seeded fuzz tests: instruction normaliser, resolver, policy mapping, and the server's scene/obs parse path.

Run this file alone (never the whole suite):
    PYTHONPATH=. python3 -m aaltonomous.baselines.classical.test_fuzz_inputs

Stdlib only (random, unittest). No network, no sim, no episode data is read.
Text fuzz: ~5000 instructions (fixed edge cases + seeded generated ones).
Scene fuzz: random observation / reset messages through deploy.policy_server.handle (the call serve() wraps).
"""
from __future__ import annotations

import random
import re
import string
import sys
import time
import unittest
from functools import lru_cache

import numpy as np

from . import instruction as _ins
from .normalize import FUZZY_TARGETS, SYNONYMS, VOCAB, normalize

SEED = 20261114
FLAGS_ON = "ab_robust_input"   # instruction.max_chars 2000 + safety.nan_guard (the fixes for findings 1 and 2)
N_TOTAL = 5000
# Per-call bound for resolve / normalize on the production path (inputs cut by instruction.max_chars first). It is NOT the
# act loop or the 1 s reset watchdog (policy.py:~70, safety.py:69): those are unaffected by a 200 ms resolver call. 200 ms
# leaves room for the frozen resolver's ~50-95 ms worst case on 2k-char random noise on a loaded shared box.
SLOW_MS = 200.0
LEGAL_VALUES = {"left", "center", "right", None}
# colour aliases (normalize._COLOUR_NAMES) and ordinal words are legitimate vocabulary, not nonsense
ALIASES = {"lavender", "mauve", "plum", "amethyst", "violet", "tangerine", "apricot", "peach", "amber", "lemon",
           "canary", "first", "second", "third", "last", "1st", "2nd", "3rd"}
KEYWORDS = frozenset(VOCAB) | frozenset(SYNONYMS) | frozenset(FUZZY_TARGETS) | ALIASES
_TOKEN = re.compile(r"[A-Za-z]+")

COLOURS = sorted(set(_ins._COLOR_WORDS) | {"purple", "orange", "amber", "lavender", "lemon", "green", "pink", "white", "black"})
SHAPES = sorted(_ins._SHAPE_WORDS)
BINS = sorted(_ins._BIN_WORDS)
POS = ["left", "right", "center", "centre", "middle", "leftmost", "rightmost", "first", "third", "last"]
FILLER = ("put the objects go to in into a an and then all every each except only but also please sort place "
          "bin bins box").split()
NONSENSE = ("lorem ipsum dolor sit amet consectetur adipiscing elit sed do eiusmod tempor cloud seven window "
            "table paper garden river quickly maybe banana zebra pencil").split()
UNICODE_FRAGS = ["​", "‍", "﻿", "⁠", "é", "́", "‮", "\U0001F600", "\U0001F7E5",
                 "ش", "中", "ő", "ž", "ß"]
META_FRAGS = ["((((a+)+)+$", "[[[", "\\", "%s", "{0}", "'", "\"", "<b>", "</b>", "$1", "\\1", ".*", "(?P<x>",
              "\x00", "\x1b[31m", "\r", "\t", "|", "^$", "--", ";", "%", "{", "}", "(", ")", "[", "]"]

EDGE = [
    "", " ", "   \t\n\r  ", " 　",
    "\U0001F600\U0001F389 red \U0001F7E5 -> left ✅", "\U0001F534\U0001F7E1\U0001F535 go left",
    "\U0001F468‍\U0001F469‍\U0001F467 red",
    "ضع المكعبات الحمراء في الصندوق الأيسر",
    "الأحمر يسار",
    "红色的方块放在左边的箱子里", "把黄色圆柱放到中间",
    "Aseta punaiset kuutiot vasempaan laatikkoon ja siniset oikeaan.",
    "Lähetä keltaiset kuutiot keskelle, siniset oikealle",
    "Dej červené kostky do levého koše a modré do pravého.",
    "Žluté válce vlož do prostředního boxu, modré kostky vpravo.",
    "re​d objects go to the le‍ft bin", "​‌‍⁠﻿", "﻿left", "‮right‬",
    "\x00\x01\x02 red\x1b[31m left\x7f", "red\x00left", "line1\nred\rleft\r\n",
    "((((a+)+)+$", "[[[", "\\", "\\\\\\", "%s %s %n", "{0} {1} {}", "${HOME} $(rm -rf /)", "' OR 1=1 --",
    "'; DROP TABLE bins; --", "<script>alert(1)</script> red left", "<b>red</b> <i>left</i>",
    "(?P<x>red)(?P=x)", ".* .+ ^$ \\1 \\b", "red|left|(", "[a-z]{1000}",
    "red left. red right.", "red left, red left, red right", "all red objects go left and also right and center",
    "everything else except everything else", "all but all", "not not not red", "except except except",
    "1", "42 3.14 -7 1e309 NaN inf", "0 0 0", "١٢٣", "９９",
    "left right center bin", "bin bin bin", "left", "right", "center", "middle", "box box basket",
    "a" * 12000, "left" * 3000, "red" + "x" * 12000, "a" * 20000 + "!", "(" * 10000, "a " * 20000,
    "a" * 30 + "!", "red" + "a" * 10000 + "left", "x" * 5000,
    "Put the red objects in the left bin. " * 1352,   # ~50k chars, repeated sentence
]


def _typo(rng, w):
    if len(w) < 2:
        return w
    i = rng.randrange(len(w))
    op = rng.randrange(4)
    if op == 0:
        return w[:i] + w[i + 1:]
    if op == 1:
        return w[:i] + w[i] + w[i:]
    if op == 2 and i + 1 < len(w):
        return w[:i] + w[i + 1] + w[i] + w[i + 2:]
    return w[:i] + rng.choice(string.ascii_lowercase) + w[i + 1:]


def _word(rng):
    r = rng.random()
    if r < .22:
        return rng.choice(COLOURS)
    if r < .36:
        return rng.choice(SHAPES)
    if r < .50:
        return rng.choice(BINS)
    if r < .57:
        return rng.choice(POS)
    if r < .67:
        return _typo(rng, rng.choice(COLOURS + SHAPES + BINS + POS))
    if r < .82:
        return rng.choice(FILLER)
    if r < .90:
        return "".join(rng.choice(string.ascii_letters) for _ in range(rng.randint(1, 14)))
    if r < .95:
        return rng.choice(UNICODE_FRAGS)
    return rng.choice(META_FRAGS)


def _sentence(rng):
    seps = [" ", " ", " ", ", ", ". ", "; ", " and ", " but ", "\n", "  "]
    n = rng.randint(1, 25)
    out = []
    for _ in range(n):
        out.append(_word(rng))
        out.append(rng.choice(seps))
    return "".join(out[:-1])


def _long_random(rng, n=5000):
    alphabet = string.ascii_letters + string.digits + " .,;:!?-()[]{}\\%$'\"" + "".join(UNICODE_FRAGS)
    return "".join(rng.choice(alphabet) for _ in range(n))


def _colour_bin(rng_unused=None):
    templates = ["Put the {c} objects in the {b} bin.", "{b} is for {c}."]
    bins = ["left", "center", "centre", "middle", "right", "leftmost", "rightmost", "first bin", "bin on the left",
            "the box on the right"]
    return [t.format(c=c, b=b) for c in COLOURS for b in bins for t in templates]


def _nonsense(rng):
    words = [rng.choice(NONSENSE) if rng.random() < .6 else
             "".join(rng.choice(string.ascii_lowercase) for _ in range(rng.randint(2, 9)))
             for _ in range(rng.randint(1, 12))]
    return " ".join(words)


def build_inputs(seed=SEED, total=N_TOTAL):
    """Fixed edge cases + colour x bin sweep + seeded generated sentences, ~total items (list of str or None)."""
    rng = random.Random(seed)
    fixed = [None] + EDGE + _colour_bin()
    gen = [_long_random(rng) for _ in range(20)] + [_nonsense(rng) for _ in range(300)]
    while len(fixed) + len(gen) < total:
        gen.append(_sentence(rng))
    return fixed + gen


def _timed(fn, arg):
    t0 = time.perf_counter()
    try:
        out = fn(arg)
        err = None
    except Exception as e:   # noqa: BLE001 - we want every escape
        out, err = None, e
    return out, err, (time.perf_counter() - t0) * 1e3


@lru_cache(maxsize=1)
def run_text_fuzz():
    from .config import load_config
    from .policy import ClassicalPolicy
    from .instruction import resolve

    logged = []
    pol = ClassicalPolicy(config=load_config(overlay=FLAGS_ON), log=lambda m: logged.append(m))
    inputs = build_inputs()
    st = {"N": len(inputs), "crashes": [], "policy_raise": [], "policy_swallowed": [], "illegal": [],
          "non_idem": [], "slow": [], "max_ms": {"normalize": (0, ""), "resolve": (0, ""), "policy": (0, "")},
          "nonsense_total": 0, "nonsense_nonempty": 0, "nonsense_examples": [],
          "fixed_fallback_nonempty_on_nonsense": 0}

    def note_time(kind, ms, s):
        if ms > st["max_ms"][kind][0]:
            st["max_ms"][kind] = (round(ms, 2), _show(s))
        if ms > SLOW_MS:
            st["slow"].append((kind, round(ms, 1), _show(s)))

    for s in inputs:
        label = _show(s)
        if s is None:   # normalize is typed str; the policy guards None before calling it
            n_out, n_err, n_ms = _timed(normalize, s)
            if n_err is not None:
                st["crashes"].append(("normalize", label, f"{type(n_err).__name__}: {n_err}"))
            _, r_err, _ = _timed(resolve, s)
            if r_err is not None:
                st["crashes"].append(("resolve", label, f"{type(r_err).__name__}: {r_err}"))
            logged.clear()
            p_out, p_err, p_ms = _timed(lambda x: pol._instruction_mapping({"instruction": x}), s)
            if p_err is not None:
                st["policy_raise"].append(("policy", label, repr(p_err)))
            if _swallowed(logged):
                st["policy_swallowed"].append((label, _swallowed(logged)[-1]))
            continue

        # production path: the cap runs before the normaliser and the resolver (policy._instruction_mapping)
        n_out, n_err, n_ms = _timed(lambda x: normalize(pol._cap_text(x)), s)
        note_time("normalize", n_ms, s)
        if n_err is not None:
            st["crashes"].append(("normalize", label, f"{type(n_err).__name__}: {n_err}"))
        else:
            n2, n2_err, _ = _timed(normalize, n_out)
            if n2_err is not None or n2 != n_out:
                st["non_idem"].append((label, _show(n_out), _show(n2) if n2 is not None else repr(n2_err)))

        # production path: the instruction.max_chars cap runs before the resolver (policy._instruction_mapping)
        r_out, r_err, r_ms = _timed(lambda x: resolve(pol._cap_text(x)), s)
        note_time("resolve", r_ms, s)
        if r_err is not None:
            st["crashes"].append(("resolve", label, f"{type(r_err).__name__}: {r_err}"))
        else:
            bad = [v for v in r_out.values() if v not in LEGAL_VALUES]
            if bad:
                st["illegal"].append(("resolve", label, bad[:3]))

        logged.clear()
        p_out, p_err, p_ms = _timed(lambda x: pol._instruction_mapping({"instruction": x}), s)
        note_time("policy", p_ms, s)
        if p_err is not None:
            st["policy_raise"].append(("policy", label, f"{type(p_err).__name__}: {p_err}"))
        if _swallowed(logged):
            st["policy_swallowed"].append((label, _swallowed(logged)[-1]))
        elif p_out is not None:
            bad = [v for v in p_out.values() if v not in LEGAL_VALUES]
            if bad:
                st["illegal"].append(("policy", label, bad[:3]))

        # "never a wrong bin from nonsense": the text AFTER the normaliser has no colour / shape / bin word. A typo the
        # normaliser repairs ("riht centrre" -> "right center") is not nonsense, so classify n_out, not the raw s.
        if n_out is not None and not (set(t.lower() for t in _TOKEN.findall(n_out)) & KEYWORDS):
            st["nonsense_total"] += 1
            empty_fb, e2, _ = _timed(lambda x: resolve(x, fallback={}), n_out)
            if e2 is None and empty_fb:
                st["nonsense_nonempty"] += 1
                if len(st["nonsense_examples"]) < 5:
                    st["nonsense_examples"].append((label, empty_fb))
            prod, e3, _ = _timed(lambda x: resolve(x), n_out)
            if e3 is None and prod and prod == {"red": "left", "yellow": "center", "blue": "right"}:
                st["fixed_fallback_nonempty_on_nonsense"] += 1
    return st


def _swallowed(logged):
    """Log lines that mean an exception was caught (the '[instruction] truncated' info line is not one)."""
    return [m for m in logged if m.startswith("instruction resolve failed")]


def _show(s, n=70):
    if s is None:
        return "None"
    r = repr(s)
    return r if len(r) <= n else r[:n] + f"...(len {len(s)})"


# ---------------------------------------------------------------- scene / obs fuzz
def _rng_np(rng):
    return np.random.default_rng(rng.randrange(2 ** 31))


def _arr(rng):
    k = rng.randrange(13)
    g = _rng_np(rng)
    if k == 0:
        return None
    if k == 1:
        return "not-an-array"
    if k == 2:
        return [float("nan")] * 14
    if k == 3:
        return np.full(14, np.nan)
    if k == 4:
        return np.array([])
    if k == 5:
        return 0
    if k == 6:
        return {"q": 1}
    if k == 7:
        return np.array(["a"] * 14)
    if k == 8:
        return np.full(14, np.inf)
    if k == 9:
        return list(range(rng.randint(0, 200000)))            # huge array
    if k == 10:
        return g.normal(0, 3, size=(rng.randint(1, 4), 14))   # wrong rank / chunk shape
    if k == 11:
        return np.zeros((14, 14, 14))
    return g.normal(0, 1.2, 14)


def _img(rng, depth):
    k = rng.randrange(8)
    g = _rng_np(rng)
    if k == 0:
        return None
    if k == 1:
        return "img"
    if k == 2:
        return np.zeros((0, 0, 3), np.uint8)
    if k == 3:
        return np.full((4, 4), np.nan, np.float32)
    if k == 4:
        return np.full((480, 640), -1.0, np.float32) if depth else np.zeros((480, 640, 3), np.uint8)
    if k == 5:
        return g.integers(0, 255, (rng.randint(1, 40), rng.randint(1, 40), 3), dtype=np.uint8)
    if k == 6:   # realistic-size frame, plausible values
        if depth:
            return g.uniform(0.2, 2.0, (480, 640)).astype(np.float32)
        return g.integers(0, 255, (480, 640, 3), dtype=np.uint8)
    return np.full((480, 640), np.inf, np.float32) if depth else g.integers(0, 255, (480, 640, 3), np.uint8)


def _obs(rng):
    keys = {"head_rgb": lambda: _img(rng, False), "head_depth": lambda: _img(rng, True),
            "left_wrist_rgb": lambda: _img(rng, False), "right_wrist_rgb": lambda: _img(rng, False),
            "arm_q": lambda: _arr(rng), "arm_dq": lambda: _arr(rng), "hand_q": lambda: _arr(rng),
            "t": lambda: rng.choice([0.0, 1.0, float("nan"), float("inf"), -5.0, "x", None, 1e308, 3]),
            "instruction": lambda: rng.choice([None, "", "red left", 7, b"bytes", "x" * 3000])}
    obs = {k: f() for k, f in keys.items() if rng.random() < 0.8}
    if rng.random() < 0.1:
        obs["junk_" + str(rng.randrange(9))] = [1, 2, {"a": None}]
    return obs


@lru_cache(maxsize=1)
def run_scene_fuzz(n_act=300, n_reset=40):
    from aaltonomous.deploy import protocol as P
    from aaltonomous.deploy.policy_server import handle
    from .config import load_config
    from .policy_server import ClassicalServerPolicy

    rng = random.Random(SEED + 1)
    pol = ClassicalServerPolicy(time_limit=120, config=load_config(overlay=FLAGS_ON))
    st = {"act": 0, "reset": 0, "caught": {}, "escaped": [], "unpackable": [], "nonfinite_targets": 0,
          "max_ms": 0.0, "max_case": "", "slow_gt_5s": 0}

    def serve_like(msg, kind):
        """Mirror policy_server.serve().conn: handle() inside `except Exception` (logged), then ws.send(pack(out))."""
        t0 = time.perf_counter()
        try:
            out = handle(pol, msg)
        except Exception as e:   # serve() catches exactly this and replies with an error
            st["caught"][type(e).__name__] = st["caught"].get(type(e).__name__, 0) + 1
            st.setdefault("msgs", {}).setdefault(f"{type(e).__name__}: {str(e)[:90]}", kind)
            out = {"type": "error", "message": f"{type(e).__name__}: {e}"}
        except BaseException as e:   # not caught by serve(): a real FAIL
            st["escaped"].append((kind, repr(e)))
            raise
        ms = (time.perf_counter() - t0) * 1e3
        if ms > st["max_ms"]:
            st["max_ms"], st["max_case"] = round(ms, 1), kind
        if ms > 5000:
            st["slow_gt_5s"] += 1
        try:
            P.pack(out)   # ws.send(P.pack(out)) sits OUTSIDE serve()'s try: a failure here drops the connection
        except Exception as e:
            st["unpackable"].append((kind, f"{type(e).__name__}: {e}"))
        if isinstance(out, dict) and out.get("type") == "action":
            a = out["action"]
            for key in ("arm_q_target", "hand_q_target"):
                if not np.all(np.isfinite(np.asarray(a.get(key), float))):
                    st["nonfinite_targets"] += 1
        return out

    for _ in range(n_reset):
        instr = rng.choice([None, "", "   ", "red left", 5, b"x", "y" * 50000, "\x00\x1b", "((((a+)+)+$"])
        task = rng.choice([None, "sort_fixed", "insert_shapes", "zzz", 3, ["x"]])
        serve_like({"type": "reset", "instruction": instr, "task": task, "seed": rng.choice([None, 1, -1])}, "reset")
        st["reset"] += 1
    for bad in ["not a dict", {"type": "nope"}, {"type": "act"}, {"type": "act", "obs": None}, {"type": "act", "obs": "x"}]:
        serve_like(bad, "malformed")
    for _ in range(n_act):
        serve_like({"type": "act", "obs": _obs(rng)}, "act")
        st["act"] += 1
    return st


# ---------------------------------------------------------------- tests
class TextFuzzTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.st = run_text_fuzz()

    def test_normalize_and_resolve_never_raise(self):
        self.assertEqual(self.st["crashes"], [], self.st["crashes"][:10])

    def test_policy_mapping_never_raises(self):
        self.assertEqual(self.st["policy_raise"], [], self.st["policy_raise"][:10])
        self.assertEqual(self.st["policy_swallowed"], [], self.st["policy_swallowed"][:10])

    def test_mapping_values_legal(self):
        self.assertEqual(self.st["illegal"], [], self.st["illegal"][:10])

    def test_idempotent(self):
        self.assertEqual(self.st["non_idem"], [], self.st["non_idem"][:10])

    def test_each_call_under_200ms(self):
        self.assertEqual(self.st["slow"], [], self.st["slow"][:10])

    def test_nonsense_never_gives_a_bin_without_fallback(self):
        self.assertEqual(self.st["nonsense_nonempty"], 0, self.st["nonsense_examples"])


class SceneFuzzTests(unittest.TestCase):
    def test_act_and_reset_never_escape_serve_loop(self):
        st = run_scene_fuzz()
        self.assertEqual(st["escaped"], [], st["escaped"][:5])
        self.assertEqual(st["unpackable"], [], st["unpackable"][:5])

    def test_nan_and_inf_joints_never_reach_the_planner(self):   # finding 2
        st = run_scene_fuzz()
        self.assertEqual([m for m in st.get("msgs", {}) if "NaN" in m], [], "planner raised on a NaN joint")
        self.assertEqual(st["nonfinite_targets"], 0, "a non-finite target was sent to the runner")


class FlagOffTests(unittest.TestCase):
    """Old behaviour with the flags off. Only cheap cases here: the 50k-char instruction takes 19 s with the cap off."""
    def test_long_instruction_is_not_capped_when_off(self):
        from .config import load_config
        from .instruction import resolve
        from .policy import ClassicalPolicy
        logged = []
        pol = ClassicalPolicy(config=load_config(overlay="cand_r17"), log=logged.append)
        text = "Put the red objects in the left bin. " * 60   # ~2.3k chars: resolver cost is milliseconds
        self.assertEqual(pol._instruction_mapping({"instruction": text}), resolve(text) or None)
        self.assertFalse([m for m in logged if "truncated" in m])
        self.assertFalse(load_config(overlay="cand_r17")["safety"]["nan_guard"])


def report():
    t = run_text_fuzz()
    s = run_scene_fuzz()
    mx = t["max_ms"]
    print(f"COUNTS: {t['N']} inputs, crashes {len(t['crashes'])}, slow (>{SLOW_MS:.0f}ms) {len(t['slow'])} "
          f"(max normalize {mx['normalize'][0]} ms, resolve {mx['resolve'][0]} ms, policy {mx['policy'][0]} ms), "
          f"non-idempotent {len(t['non_idem'])}, nonsense-nonempty (fallback={{}}) {t['nonsense_nonempty']} "
          f"of {t['nonsense_total']}; policy swallowed {len(t['policy_swallowed'])}; "
          f"production-fallback FIXED on nonsense {t['fixed_fallback_nonempty_on_nonsense']} (expected)")
    print("REPROS:")
    for kind, label, err in t["crashes"][:10]:
        print(f"  {kind}({label}) -> {err}")
    for label, msg in t["policy_swallowed"][:3]:
        print(f"  policy swallowed on {label}: {msg}")
    for kind, ms, label in t["slow"][:5]:
        print(f"  slow {kind} {ms} ms on {label}")
    for label, a, b in t["non_idem"][:5]:
        print(f"  non-idempotent: {label}\n     once={a}\n     twice={b}")
    for label, mp in t["nonsense_examples"]:
        print(f"  nonsense gives mapping: {label} -> {mp}")
    for m, kind in list(s.get("msgs", {}).items())[:6]:
        print(f"  caught ({kind}): {m}")
    print(f"SCENE FUZZ: act {s['act']}, reset {s['reset']}, caught-by-serve {s['caught']}, "
          f"escaped {len(s['escaped'])}, unpackable {len(s['unpackable'])}, "
          f"non-finite action targets {s['nonfinite_targets']}, max {s['max_ms']} ms ({s['max_case']})")
    for kind, err in s["escaped"][:3] + s["unpackable"][:3]:
        print(f"  {kind}: {err}")
    return t, s


if __name__ == "__main__":
    report()
    unittest.main(argv=[sys.argv[0]], exit=False, verbosity=1)
