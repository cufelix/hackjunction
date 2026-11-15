#!/usr/bin/env python3
"""Score the instruction-generality bank (aaltonomous/baselines/classical/instruction_bank.py) offline.

    python tools/lang_generality.py [--overlay finals_r25] [--fails] [--sim]
Full reset path (_episode_mapping, task=sort_language) so the insert guard counts.  Per object: +2 correct,
-1 wrong bin, 0 abstain.  pass = every object's result is acceptable; wrong = some object got a wrong bin.
"""
import argparse, itertools, sys
from collections import defaultdict
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from aaltonomous.baselines.classical.config import load_config  # noqa: E402
from aaltonomous.baselines.classical.policy import ClassicalPolicy  # noqa: E402
from aaltonomous.baselines.classical.instruction_bank import BANK as BANK1, BANK2  # noqa: E402
from aaltonomous.interface.tasks import COLORS, SHAPES  # noqa: E402
sys.path.insert(0, str(ROOT / "tools"))
from lang_bank3 import BANK3  # noqa: E402
from aaltonomous.scene.scene import resolve_bin  # noqa: E402

OBJECTS = tuple(itertools.product(COLORS, SHAPES))


class Stub:
    _cap_text = ClassicalPolicy._cap_text
    _instruction_mapping = ClassicalPolicy._instruction_mapping
    _insert_guard_mapping = ClassicalPolicy._insert_guard_mapping
    _episode_mapping = ClassicalPolicy._episode_mapping
    _log_fixed_conflict = ClassicalPolicy._log_fixed_conflict
    _instruction_fixed = ClassicalPolicy._instruction_fixed

    def __init__(self, overlay, sim=False):
        self.cfg = load_config(overlay=overlay)
        if sim:
            from aaltonomous.baselines.classical.config import deep_merge  # may not exist
        self.fixed_mapping = {}
        self.messages = []
        self.log = self.messages.append

    def mapping(self, text):
        return self._episode_mapping({"task": "sort_language", "instruction": text})[0]


def expect(gold, c, s):
    v = resolve_bin(gold, c, s)
    if isinstance(v, str) and v.endswith("?"):
        return v[:-1], True
    return v, False


def judge(gold, pred):
    wrong, miss, pts, full = [], [], 0, True
    for c, s in OBJECTS:
        e, opt = expect(gold, c, s)
        a = resolve_bin(pred, c, s)
        a = a if a in ("left", "center", "right") else None
        if a is None:
            if e is not None and not opt:
                miss.append(f"{c}/{s}:{e}->abstain")
            full &= e is None
        elif e is None or a != e:
            wrong.append(f"{c}/{s}:{e}->{a}"); full = False
        else:
            pts += 2
        if a is not None and a == e:
            pts += 0
    return wrong, miss, full


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overlay", default="finals_r25")
    ap.add_argument("--fails", action="store_true")
    ap.add_argument("--bank", default="1", choices=["1", "2", "3", "all"])
    a = ap.parse_args()
    BANK = {'1': BANK1, '2': BANK2, '3': BANK3, 'all': BANK1 + BANK2 + BANK3}[a.bank]
    st = Stub(a.overlay)
    cats = defaultdict(lambda: [0, 0, 0, 0])   # n, pass, wrong, complete
    fails = []
    for cat, text, gold in BANK:
        pred = st.mapping(text)
        wrong, miss, full = judge(gold, pred)
        row = cats[cat]; row[0] += 1
        ok = not wrong and not miss
        row[1] += ok; row[2] += bool(wrong); row[3] += full
        if not ok:
            fails.append((cat, text, wrong, miss, {str(k): v for k, v in pred.items()}))
    tot = [sum(r[i] for r in cats.values()) for i in range(4)]
    print(f"overlay={a.overlay}  {'category':10s} n pass wrong(cases) complete")
    for k, r in cats.items():
        print(f"  {k:10s} {r[0]:3d} {r[1]:3d} {r[2]:3d} {r[3]:3d}")
    print(f"  {'TOTAL':10s} {tot[0]:3d} {tot[1]:3d} {tot[2]:3d} {tot[3]:3d}   pass={tot[1]/tot[0]:.1%}")
    if a.fails:
        for cat, text, wrong, miss, pred in fails:
            print(f"[{cat}] {'WRONG' if wrong else 'miss '} {text!r}\n      wrong={wrong} miss={miss}\n      pred={pred}")


if __name__ == "__main__":
    main()
