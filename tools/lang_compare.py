#!/usr/bin/env python3
"""Per-object (+2/-1/0) comparison of two overlays on external paraphrase corpora (jsonl with gold per object)."""
import json, sys, itertools
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.lang_generality import Stub  # noqa: E402
from aaltonomous.interface.tasks import COLORS, SHAPES  # noqa: E402
from aaltonomous.scene.scene import resolve_bin  # noqa: E402
SHP = {"prism": "triangle", "triangle": "triangle", "cube": "cube", "cylinder": "cylinder"}
CAN = {"purple": "blue", "violet": "blue", "orange": "red", "blue": "blue", "red": "red", "yellow": "yellow"}


def gold_of(r):
    out = {}
    if "gold" in r:
        for k, v in r["gold"].items():
            if isinstance(k, str) and " " in k:
                c, s = k.split(); out[(CAN[c], SHP[s])] = v if v in ("left", "center", "right") else None
        if out:
            return out
    m = r.get("mapping") or r.get("gold") or {}
    for c, s in itertools.product(COLORS, SHAPES):
        out[(c, s)] = None
    for k, v in m.items():
        k = CAN.get(k, k)
        if k in COLORS and v in ("left", "center", "right"):
            for s in SHAPES:
                out[(k, s)] = v
    return out


def main(corpora, a, b):
    A, B = Stub(a), Stub(b)
    for f in corpora:
        rows = [json.loads(l) for l in open(f)]
        res = {}
        diffs = []
        for name, st in ((a, A), (b, B)):
            pts = wrong = cor = n = 0
            for r in rows:
                g = gold_of(r); pred = st.mapping(r["text"])
                for (c, s), gv in g.items():
                    p = resolve_bin(pred, c, s); p = p if p in ("left", "center", "right") else None
                    n += 1
                    if p is None: continue
                    if p == gv: pts += 2; cor += 1
                    else: pts -= 1; wrong += 1
            res[name] = (pts, cor, wrong, n)
        for r in rows:
            g = gold_of(r); pa, pb = A.mapping(r["text"]), B.mapping(r["text"])
            for (c, s), gv in g.items():
                x, y = resolve_bin(pa, c, s), resolve_bin(pb, c, s)
                if x != y:
                    diffs.append((r["text"], f"{c}/{s}", gv, x, y)); 
        print(f"{Path(f).name}: " + "  ".join(f"{k}: pts={v[0]} correct={v[1]} wrong={v[2]} /{v[3]}" for k, v in res.items()))
        seen = set()
        for t, o, gv, x, y in diffs:
            if (t) in seen: continue
            if y is not None and y != gv or (x is not None and y is None):
                seen.add(t); print(f"   CHANGE {t!r} {o} gold={gv} before={x} after={y}")
if __name__ == "__main__":
    main(sys.argv[3:], sys.argv[1], sys.argv[2])
