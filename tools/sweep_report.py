#!/usr/bin/env python3
"""Report for runs/sweep/<ts>/: success per phi x arm vs baseline (n, Wilson 95% CI), plan_s mean/p95, fallback rate, wall_s.
    python3 tools/sweep_report.py runs/sweep/<ts> [more dirs]
Success = outcome == 'correct'. Fallback = (yaw path ran but a stock-hint plan won, or nothing) / (episodes where yaw path ran).
Pure stdlib. Reads every <arm>_<cell>.json in the dir (partial cells included; n shows it)."""
import argparse, glob, json, math, os, statistics as S


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n; d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d; h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def p95(v):
    v = sorted(v)
    return v[min(len(v) - 1, math.ceil(0.95 * len(v)) - 1)] if v else float("nan")


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "*.json"))):
        try:
            rows += json.load(open(f))
        except Exception as e:
            print(f"# skip {f}: {e}")
    return rows


def key(cell):
    return -1.0 if cell == "base" else float(cell[3:])


def table(rows):
    g = {}
    for r in rows:
        g.setdefault((r["arm"], r["cell"]), []).append(r)
    out = []
    for (arm, cell), v in sorted(g.items(), key=lambda kv: (kv[0][0], key(kv[0][1]))):
        n = len(v); k = sum(r["outcome"] == "correct" for r in v); lo, hi = wilson(k, n)
        yaw = [r for r in v if r["yaw_calls"]]
        fb = sum(r["yaw_none"] > 0 or not r["yaw_used"] for r in yaw)
        ps = [r["plan_s"][0] for r in v if r["plan_s"]]          # first plan call per episode
        base = g.get((arm, "base"))
        bk = sum(r["outcome"] == "correct" for r in base) if base else None
        out.append(dict(arm=arm, cell=cell, n=n, k=k, rate=k / n, lo=lo, hi=hi,
                        vs_base=None if bk is None or cell == "base" else k - bk * n / len(base),
                        yaw_ran=len(yaw), fb=fb, plan_mean=S.mean(ps) if ps else float("nan"), plan_p95=p95(ps),
                        wall_mean=S.mean(r["wall_s"] for r in v), side_off=sum(r["plan_side"] != arm for r in v),
                        base_sane=(cell != "base") or all(r["yaw_calls"] == 0 for r in v)))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("dir", nargs="+")
    a = ap.parse_args()
    rows = [r for d in a.dir for r in load(d)]
    if not rows:
        print("no rows"); return
    print("| arm | cell | n | success | rate | Wilson95 | vs base (poses) | fallback | plan_s mean/p95 | wall_s | other arm used |")
    print("|" + "---|" * 11)
    for t in table(rows):
        vb = "-" if t["vs_base"] is None else f"{t['vs_base']:+.1f}"
        fb = f"{t['fb']}/{t['yaw_ran']}" + (f" ({100*t['fb']/t['yaw_ran']:.0f}%)" if t["yaw_ran"] else "")
        warn = "" if t["base_sane"] else "  BASELINE yaw_calls>0 (flag leak)!"
        print(f"| {t['arm']} | {t['cell']} | {t['n']} | {t['k']}/{t['n']} | {100*t['rate']:.0f}% | "
              f"{100*t['lo']:.0f}-{100*t['hi']:.0f}% | {vb} | {fb} | {t['plan_mean']:.2f}/{t['plan_p95']:.2f} | "
              f"{t['wall_mean']:.0f} | {t['side_off']} |{warn}")
    errs = sorted(abs(r["apex_err_deg"]) for r in rows if r.get("apex_err_deg") is not None)
    if errs:
        print(f"\napex_yaw error vs truth mod 120: n={len(errs)} median {S.median(errs):.1f} p90 {errs[int(0.9*(len(errs)-1))]:.1f} deg")


if __name__ == "__main__":
    main()
