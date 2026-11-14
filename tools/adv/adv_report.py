#!/usr/bin/env python3
"""adv_report.py: per-episode JSON from run_adv.sh output and paired comparison between two runs.

    adv_report.py episodes <run_dir>            -> <run_dir>/episodes.json + table (correct/wrong/counter/off, clips, Hz, end)
    adv_report.py compare <run_A> <run_B>       paired by seed: per-seed score A, B, delta; mean, std, mean-std, wins/losses
    adv_report.py summary <run_dir> [...]       one line per run

Time (team rule: a new layer may add at most 2 % to the median episode time vs the base): episode time = SIM seconds
(sim_s, independent of box load / RTF). Reported per run: median, p95, mean time per PLACED object (sum episode time / sum
correct placements) and the placement count. `compare` prints "TIME FAIL" when run B (candidate) median time > 1.02 x run A (base).

Score used here: the evaluator's episode score, but an episode that ended in a safety stop (end_reason safe_stop / aborted /
crash, or safety mode safe_stop) counts as -6 (spec: worst case). Robust metric = mean - std over episodes.
A run_dir may be the run_adv.sh dir or the eval dir inside it (anything containing eval.json).
Stdlib only.
"""
import glob
import json
import os
import statistics as st
import sys

SAFETY_SCORE = -6
STOP_REASONS = ("safe_stop", "aborted", "crash", "policy_connection_lost")


def load(run):
    fs = sorted(glob.glob(os.path.join(run, "eval.json")) + glob.glob(os.path.join(run, "*", "eval.json")))
    if not fs:
        sys.exit(f"no eval.json under {run}")
    d = json.load(open(fs[-1]))
    meta = {}
    for m in (os.path.join(run, "meta.json"), os.path.join(os.path.dirname(fs[-1]), "meta.json")):
        if os.path.exists(m):
            meta = json.load(open(m))
    return d, meta


def episode(t):
    oc = [o["outcome"] for o in t.get("outcomes", [])]
    sf = t.get("safety") or {}
    stop = t.get("end_reason") in STOP_REASONS or sf.get("mode") == "safe_stop"
    return {"seed": t["seed"], "episode": t.get("episode"), "n_objects": t.get("n_objects"),
            "correct": oc.count("correct"), "wrong": oc.count("wrong"), "counter": oc.count("counter"), "off": oc.count("off"),
            "raw_score": t.get("score"), "safety_stop": bool(stop), "score": SAFETY_SCORE if stop else t.get("score"),
            "clips": sf.get("clipped_steps"), "command_hz": t.get("command_hz"), "policy_hz_rt": t.get("policy_hz_rt"),
            "end_reason": t.get("end_reason"), "sim_s": t.get("sim_s"), "rtf": t.get("rtf"),
            "adv": (t.get("dr_params") or {}).get("adv")}


def episodes(run):
    d, meta = load(run)
    eps = [episode(t) for t in d["trials"]]
    return eps, meta, d


def pct(x, q):
    x = sorted(x)
    if not x:
        return float("nan")
    k = (len(x) - 1) * q
    lo, hi = int(k), min(int(k) + 1, len(x) - 1)
    return x[lo] + (x[hi] - x[lo]) * (k - lo)


def tstats(eps):
    t = [e["sim_s"] for e in eps if e.get("sim_s") is not None]
    placed = sum(e["correct"] for e in eps)
    return {"t_median": round(pct(t, 0.5), 2), "t_p95": round(pct(t, 0.95), 2), "t_total": round(sum(t), 1),
            "placements": placed, "t_per_placed": round(sum(t) / placed, 2) if placed else None}


def stats(eps):
    s = [e["score"] for e in eps]
    m = st.mean(s) if s else float("nan")
    sd = st.pstdev(s) if len(s) > 1 else 0.0
    return {"n": len(s), "mean": round(m, 3), "std": round(sd, 3), "mean_minus_std": round(m - sd, 3),
            "safety_stops": sum(e["safety_stop"] for e in eps), "correct": sum(e["correct"] for e in eps),
            "wrong": sum(e["wrong"] for e in eps), "counter": sum(e["counter"] for e in eps), "off": sum(e["off"] for e in eps),
            "objects": sum(e["n_objects"] or 0 for e in eps), **tstats(eps)}


def cmd_episodes(run):
    eps, meta, _ = episodes(run)
    out = {"meta": meta, "stats": stats(eps), "episodes": eps}
    json.dump(out, open(os.path.join(run, "episodes.json"), "w"), indent=1)
    print(f"{'seed':>5} {'score':>5} {'corr':>4} {'wrong':>5} {'ctr':>3} {'off':>3} {'clips':>5} {'hz':>5} {'rt_hz':>6} end")
    for e in eps:
        print(f"{e['seed']:5d} {e['score']:+5d} {e['correct']:4d} {e['wrong']:5d} {e['counter']:3d} {e['off']:3d} "
              f"{e['clips']!s:>5} {e['command_hz']!s:>5} {e['policy_hz_rt']!s:>6} {e['end_reason']}{' (STOP=-6)' if e['safety_stop'] else ''}")
    s = out["stats"]
    print(f"n={s['n']} mean={s['mean']:+.2f} std={s['std']:.2f} mean-std={s['mean_minus_std']:+.2f} stops={s['safety_stops']} "
          f"correct/wrong/counter/off={s['correct']}/{s['wrong']}/{s['counter']}/{s['off']} of {s['objects']}  -> {run}/episodes.json")
    print(f"time (sim s): median {s['t_median']} p95 {s['t_p95']}  per placed object {s['t_per_placed']} ({s['placements']} placements)")


def cmd_compare(a, b):
    ea, ma, _ = episodes(a)
    eb, mb, _ = episodes(b)
    A, B = {e["seed"]: e for e in ea}, {e["seed"]: e for e in eb}
    seeds = sorted(set(A) & set(B))
    if not seeds:
        sys.exit("no common seeds")
    lab = lambda m, r: f"{m.get('profile', '?')}/{m.get('overlay', '?')}/{m.get('sha', '?')}" if m else os.path.basename(r.rstrip('/'))
    print(f"A = {lab(ma, a)}   B = {lab(mb, b)}   paired on {len(seeds)} seeds")
    print(f"{'seed':>5} {'A':>4} {'B':>4} {'B-A':>4}  A:c/w/ctr/off  B:c/w/ctr/off")
    d = []
    for s in seeds:
        x, y = A[s], B[s]
        d.append(y["score"] - x["score"])
        print(f"{s:5d} {x['score']:+4d} {y['score']:+4d} {y['score'] - x['score']:+4d}  "
              f"{x['correct']}/{x['wrong']}/{x['counter']}/{x['off']}{'*' if x['safety_stop'] else ' '}          "
              f"{y['correct']}/{y['wrong']}/{y['counter']}/{y['off']}{'*' if y['safety_stop'] else ''}")
    sa, sb = stats([A[s] for s in seeds]), stats([B[s] for s in seeds])
    for nm, s in (("A", sa), ("B", sb)):
        print(f"{nm}: mean {s['mean']:+.2f} std {s['std']:.2f} mean-std {s['mean_minus_std']:+.2f} stops {s['safety_stops']}  "
              f"| time sim s: median {s['t_median']} p95 {s['t_p95']} per placed {s['t_per_placed']} ({s['placements']} placed)")
    ma_, mb_ = sa["t_median"], sb["t_median"]
    dt = 100.0 * (mb_ - ma_) / ma_ if ma_ else float("nan")
    print(f"time delta B-A: median {mb_ - ma_:+.2f} s ({dt:+.1f} %)  p95 {sb['t_p95'] - sa['t_p95']:+.2f} s  per placed "
          f"{(sb['t_per_placed'] or 0) - (sa['t_per_placed'] or 0):+.2f} s" + ("   *** TIME FAIL (median > +2 %) ***" if dt > 2.0 else "   time ok (<= +2 %)"))
    se = st.pstdev(d) / (len(d) ** 0.5) if len(d) > 1 else float("nan")
    print(f"paired delta B-A: mean {st.mean(d):+.2f} (se {se:.2f})  wins {sum(x > 0 for x in d)} losses {sum(x < 0 for x in d)} "
          f"ties {sum(x == 0 for x in d)}  d(mean-std) {sb['mean_minus_std'] - sa['mean_minus_std']:+.2f}   (* = safety stop, counted -6)")


def main(argv):
    if len(argv) >= 3 and argv[1] == "episodes":
        cmd_episodes(argv[2])
    elif len(argv) == 4 and argv[1] == "compare":
        cmd_compare(argv[2], argv[3])
    elif len(argv) >= 3 and argv[1] == "summary":
        for r in argv[2:]:
            eps, m, _ = episodes(r)
            s = stats(eps)
            print(f"{m.get('profile', '?'):13} {m.get('overlay', '?'):18} {m.get('sha', '?'):9} n={s['n']} mean={s['mean']:+.2f} "
                  f"std={s['std']:.2f} m-s={s['mean_minus_std']:+.2f} stops={s['safety_stops']} c/w/ctr/off={s['correct']}/{s['wrong']}/{s['counter']}/{s['off']} "
                  f"t_med={s['t_median']} t_p95={s['t_p95']} t/placed={s['t_per_placed']}")
    else:
        print(__doc__)
        sys.exit(2)


if __name__ == "__main__":
    main(sys.argv)
