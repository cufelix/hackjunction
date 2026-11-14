#!/usr/bin/env python3
"""Offline report for the perception `[shape_diag]` log lines (patch docs/no-blob-shape.patch, perception._log_diag).

Usage:
    python3 shape_diag_report.py LOG [LOG ...] [--expect triangle] [--min-points 40]
    some_cmd | python3 shape_diag_report.py -          # read stdin

Line format parsed (perception.py _log_diag):
    [shape_diag] <colour> src=<depth|plane> n_top=<int> r_cloud=<f|None> r_blob=<f|None> path=<...>
                 label=<cube|cylinder|triangle|None> conf=<f> near_bin=<True|False> dz_top=<f|None> [rej=<why>]

Prints: counts by path x colour x label, n_top distribution (overall and per path), r_cloud / r_blob stats per label,
and, with --expect, the accuracy against a known true shape (e.g. t_sign: every object is a triangle).
Stdlib only. Lines with rej=... (rejected detections) are reported separately and excluded from the main tables.
"""
import argparse
import re
import statistics as st
import sys
from collections import Counter, defaultdict

LINE = re.compile(r"\[shape_diag\]\s+(?P<colour>\S+)\s+(?P<kv>.*)$")
KV = re.compile(r"(\w+)=(\S+)")


def num(x):
    if x in (None, "None"):
        return None
    try:
        return float(x)
    except ValueError:
        return None


def parse(lines):
    out = []
    for ln in lines:
        m = LINE.search(ln)
        if not m:
            continue
        kv = dict(KV.findall(m.group("kv")))
        out.append({
            "colour": m.group("colour"), "src": kv.get("src"), "n_top": int(num(kv.get("n_top")) or 0),
            "r_cloud": num(kv.get("r_cloud")), "r_blob": num(kv.get("r_blob")), "path": kv.get("path"),
            "label": kv.get("label"), "conf": num(kv.get("conf")), "near_bin": kv.get("near_bin") == "True",
            "dz_top": num(kv.get("dz_top")), "rej": kv.get("rej")})
    return out


def dist(xs):
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return "n=0"
    q = lambda p: xs[min(len(xs) - 1, int(p * (len(xs) - 1) + 0.5))]  # noqa: E731
    return (f"n={len(xs)} min={xs[0]:.3g} p10={q(0.1):.3g} med={st.median(xs):.3g} "
            f"p90={q(0.9):.3g} max={xs[-1]:.3g}")


def hist_ntop(xs, edges=(0, 10, 20, 40, 80, 160, 320, 10 ** 9)):
    c = Counter()
    for x in xs:
        for lo, hi in zip(edges, edges[1:]):
            if lo <= x < hi:
                c[f"{lo}-{hi - 1 if hi < 10 ** 9 else '+'}"] += 1
                break
    return "  ".join(f"[{k}]:{v}" for k, v in c.items())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--expect", choices=("cube", "cylinder", "triangle"),
                    help="true shape of every logged object (t_sign: triangle)")
    ap.add_argument("--min-points", type=int, default=40, help="shape_id.SHAPE_MIN_POINTS")
    a = ap.parse_args()
    lines = []
    for p in a.logs:
        lines += (sys.stdin if p == "-" else open(p, errors="replace")).read().splitlines()
    rows = parse(lines)
    acc = [r for r in rows if not r["rej"]]
    rej = [r for r in rows if r["rej"]]
    print(f"[shape_diag] lines: {len(rows)} (accepted {len(acc)}, rejected {len(rej)})")
    if not acc:
        return
    print("\n== path x colour x label (accepted detections)")
    for (p, c, l), n in sorted(Counter((r["path"], r["colour"], r["label"]) for r in acc).items()):
        print(f"  {p:22s} {c:7s} {str(l):9s} {n}")
    print("\n== path totals")
    for p, n in Counter(r["path"] for r in acc).most_common():
        print(f"  {p:22s} {n}  ({n / len(acc):.0%})")
    print(f"\n== n_top  overall: {dist([r['n_top'] for r in acc])}")
    print(f"   histogram: {hist_ntop([r['n_top'] for r in acc])}")
    below = sum(r["n_top"] < a.min_points for r in acc)
    print(f"   below min-points {a.min_points}: {below}/{len(acc)} ({below / len(acc):.0%})")
    by_path = defaultdict(list)
    for r in acc:
        by_path[r["path"]].append(r["n_top"])
    for p, xs in by_path.items():
        print(f"   {p:22s} {dist(xs)}")
    print(f"   by src: " + ", ".join(f"{s}={n}" for s, n in Counter(r["src"] for r in acc).items()))
    print("\n== ratios per label (cube ~1.0, cylinder ~0.785, triangle ~0.5; thresholds 0.88 / 0.66)")
    for l in sorted({str(r["label"]) for r in acc}):
        rs = [r for r in acc if str(r["label"]) == l]
        print(f"  label={l:9s} r_cloud {dist([r['r_cloud'] for r in rs])}")
        print(f"  {'':15s} r_blob  {dist([r['r_blob'] for r in rs])}")
        print(f"  {'':15s} conf    {dist([r['conf'] for r in rs])}   dz_top {dist([r['dz_top'] for r in rs])}")
    nb = sum(r["near_bin"] for r in acc)
    print(f"\n== near_bin True: {nb}/{len(acc)}")
    if a.expect:
        ok = sum(r["label"] == a.expect for r in acc)
        ab = sum(r["label"] in (None, "None") for r in acc)
        wrong = len(acc) - ok - ab
        print(f"\n== vs expected '{a.expect}': correct {ok}, abstain {ab}, WRONG {wrong}  (n={len(acc)})")
        for l, n in Counter(r["label"] for r in acc if r["label"] not in (a.expect, None, "None")).items():
            print(f"   wrong as {l}: {n}")
    if rej:
        print("\n== rejected detections by reason: " + ", ".join(f"{k}={v}" for k, v in Counter(r["rej"] for r in rej).items()))


if __name__ == "__main__":
    main()
