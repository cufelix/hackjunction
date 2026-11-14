#!/usr/bin/env python3
"""Offline held-object evaluator (T5): stall vs finger_sum vs softgrip, from classical policy logs. Stdlib only.

  python3 tools/held_offline_eval.py LOG [LOG ...] [--results RESULT ...] [--step 0.01] [--per-event]
  python3 tools/held_offline_eval.py --self-check

LOG      policy_server.log (or a run's container.log = docker logs of the same process). Read: `reset task=.. seed=..`
         (episode start), `[grasp_diag]` (one per pick, close + lift samples), `[held_abort]` (close or post-lift
         abort, no numbers), `[soft_baseline]` (present only when held.soft_detector ran).
RESULT   per-trial outcomes: the sim evaluate eval.json (`{"trials": [...]}`), a JSON list, or JSONL. Each trial needs
         `seed` and `outcomes` [{color | name, outcome}]: correct/wrong = object in a bin, counter = still on counter.
         Without RESULT the verdicts are computed but the TP/FP/FN/TN table is not.

Ground truth (per close, rule stated because the logs do not carry object position):
  object in a bin  -> the LAST close of that colour in the episode is held=yes; earlier closes are unknown (excluded).
  object on counter -> every close of that colour is held=no (a close that carried and dropped the object is a known
                       blind spot; it shows up as a FP).
  several objects of one colour -> only "all on counter" is used; otherwise unknown.
Censored closes: an aborted close (`[held_abort] ... close sample empty`) means the detector read it empty at the run's
threshold, so it is a "no" at any threshold >= that run threshold and excluded below it.
"""
import argparse
import json
import re
import sys
from collections import Counter, defaultdict

STALL_THR = 0.010                      # HeldDetector.held(): stall branch, hard-coded there
IN_BIN, ON_COUNTER = {"correct", "wrong"}, {"counter"}
DETECTORS = ("stall", "finger_sum", "softgrip")
RESET_RE = re.compile(r"reset task=\S+ seed=(\S+)")
DIAG_RE = re.compile(r"\[grasp_diag\] t=([\d.]+) (\S+) (\S+) target=\S+ held=\S+ (.*)")
ABORT_RE = re.compile(r"\[held_abort\] t=([\d.]+) (\S+) (\S+) target=\S+ (close|post-lift) sample empty")
SOFT_RE = re.compile(r"\[soft_baseline\] ")
SEG_SPLIT = re.compile(r"\s(?=(?:close|lift|carry_max)@\d+:)")
FIELD_RE = re.compile(r"(\w+)=(\[[^\]]*\]|[^\s\[]+)")


def _value(text):
    if text.startswith("["):
        inner = text[1:-1].strip()
        return [float(x) for x in inner.split(",")] if inner else []
    try:
        return float(text)
    except ValueError:
        return text


def parse_log(text, name):
    """Episodes of one log file: [{seed, soft, close: [event], lift_aborts}]. A close event is a diag or abort row."""
    eps, cur = [], None
    for line in text.splitlines():
        m = RESET_RE.search(line)
        if m:
            cur = {"file": name, "seed": m.group(1), "soft": False, "close": [], "lift_aborts": 0}
            eps.append(cur)
            continue
        if cur is None:                 # log started mid-episode: seed unknown
            cur = {"file": name, "seed": None, "soft": False, "close": [], "lift_aborts": 0}
            eps.append(cur)
        if SOFT_RE.search(line):
            cur["soft"] = True
            continue
        m = DIAG_RE.search(line)
        if m:
            t, color, side, rest = m.groups()
            segs = {}
            for seg in SEG_SPLIT.split(rest):
                head, _, body = seg.partition(": ")
                segs[head.split("@")[0]] = {k: _value(v) for k, v in FIELD_RE.findall(body)}
            vals = segs.get("close", {})
            cur["close"].append({"kind": "diag", "t": float(t), "color": color, "side": side, "vals": vals})
            continue
        m = ABORT_RE.search(line)
        if m:
            t, color, side, sample = m.groups()
            if sample == "close":
                cur["close"].append({"kind": "abort", "t": float(t), "color": color, "side": side, "vals": {}})
            else:
                cur["lift_aborts"] += 1  # post-lift empty: the close had passed, no close numbers logged
    return eps


def applies(det, ep):
    return ep["soft"] if det == "softgrip" else (det == "stall" or not ep["soft"])


def verdicts(ev, ep):
    """Detector -> bool for one close event, from the logged fields only. Missing field -> key absent."""
    out = {}
    if ev["kind"] == "abort":          # empty at the run's threshold (same data as the close check)
        out["softgrip" if ep["soft"] else "finger_sum"] = False
        return out
    vals = ev["vals"]
    if "track_err_max" in vals:
        err = vals["track_err_max"]
    elif "stall_vs_closed" in vals and "cmd_vs_closed" in vals:   # track_err = |cmd - q| = |stall_vs - cmd_vs|
        err = max(abs(a - b) for a, b in zip(vals["stall_vs_closed"], vals["cmd_vs_closed"]))
    else:
        err = None
    if err is not None:
        out["stall"] = err >= STALL_THR
    if "finger_sum" in vals and "threshold" in vals:
        if ep["soft"]:                  # soft mode logs the excess (score - baseline ratio x travel), margin as threshold
            out["softgrip"] = vals["finger_sum"] > vals["threshold"]
        else:
            out["finger_sum"] = vals["finger_sum"] >= vals["threshold"]
    return out


def load_results(paths):
    trials = {}
    for p in paths:
        txt = open(p, encoding="utf-8").read().strip()
        try:
            obj = json.loads(txt)
        except json.JSONDecodeError:
            obj = [json.loads(x) for x in txt.splitlines() if x.strip()]
        rows = obj.get("trials", [obj]) if isinstance(obj, dict) else obj
        for tr in rows:
            if isinstance(tr, dict) and "seed" in tr:
                trials[str(tr["seed"])] = tr
    return trials


def labels(trial, colours):
    per = defaultdict(list)
    for o in trial.get("outcomes", []):
        col = o.get("color") or next((tok for tok in re.split(r"[^a-z]+", str(o.get("name", "")).lower())
                                      if tok in colours), None)
        per[col].append(o.get("outcome"))
    lab = {}
    for col, outs in per.items():
        if col is None:
            continue
        if len(outs) == 1:
            lab[col] = "in_bin" if outs[0] in IN_BIN else "counter" if outs[0] in ON_COUNTER else None
        else:
            lab[col] = "counter" if all(o in ON_COUNTER for o in outs) else None
    return lab


def attach_truth(eps, trials):
    for ep in eps:
        colours = {ev["color"] for ev in ep["close"]}
        trial = trials.get(str(ep["seed"]))
        lab = labels(trial, colours) if trial else {}
        by_col = defaultdict(list)
        for ev in ep["close"]:
            ev["gt"] = None
            by_col[ev["color"]].append(ev)
        for col, evs in by_col.items():
            evs.sort(key=lambda e: e["t"])
            if lab.get(col) == "in_bin":
                evs[-1]["gt"] = True
            elif lab.get(col) == "counter":
                for ev in evs:
                    ev["gt"] = False


def confusion(rows):
    tp = sum(1 for v, g in rows if v and g)
    fp = sum(1 for v, g in rows if v and not g)
    fn = sum(1 for v, g in rows if not v and g)
    tn = sum(1 for v, g in rows if not v and not g)
    return tp, fp, fn, tn


def fmt_row(name, rows, extra=""):
    tp, fp, fn, tn = confusion(rows)
    acc = (tp + tn) / len(rows) if rows else float("nan")
    return f"{name:<12}{len(rows):>5}{tp:>5}{fp:>5}{fn:>5}{tn:>5}{acc:>7.2f}  {extra}"


def sweep(eps, det, step):
    """Threshold sweep on the logged score. Censored aborts count as 'no' only at thresholds >= the run threshold."""
    op = (lambda v, t: v > t) if det == "softgrip" else (lambda v, t: v >= t)
    vals, cens, run_thr = [], [], None
    for ep in eps:
        if not applies(det, ep) or det == "stall":
            continue
        for ev in ep["close"]:
            if ev["gt"] is None:
                continue
            if ev["kind"] == "abort":
                cens.append(ev["gt"])
            elif "finger_sum" in ev["vals"]:
                vals.append((ev["vals"]["finger_sum"], ev["gt"]))
                run_thr = ev["vals"]["threshold"]
    if not vals and not cens:
        return []
    top = max([v for v, _ in vals] + [0.0])
    out = []
    for i in range(int(top / step) + 2):
        t = round(i * step, 6)
        rows = [(op(v, t), g) for v, g in vals]
        if run_thr is not None and t >= run_thr:
            rows += [(False, g) for g in cens]
            excluded = 0
        else:
            excluded = len(cens)
        out.append((t, rows, excluded))
    return out


def report(eps, step, per_event, out=print):
    out("== episodes ==")
    for f in sorted({ep["file"] for ep in eps}):
        mine = [ep for ep in eps if ep["file"] == f]
        n_soft = sum(ep["soft"] for ep in mine)
        n_close = sum(len(ep["close"]) for ep in mine)
        n_abort = sum(ev["kind"] == "abort" for ep in mine for ev in ep["close"])
        n_lift = sum(ep["lift_aborts"] for ep in mine)
        n_gt = sum(ev["gt"] is not None for ep in mine for ev in ep["close"])
        out(f"{f}: episodes={len(mine)} soft={n_soft} closes={n_close} close_aborts={n_abort} "
            f"post_lift_aborts={n_lift} with_ground_truth={n_gt}")
    out("\n== reconstructable from the logged fields ==")
    missing = {
        "stall": "track_err_max, or stall_vs_closed + cmd_vs_closed, on a grasp_diag close line. Not logged on "
                 "[held_abort] closes: _grasp_diag returns before the diag line when the close aborts",
        "finger_sum": "finger_sum= and threshold= on a grasp_diag close line of a grasp.held_detector=finger_sum "
                      "run. Not logged on [held_abort] closes (no score in that line) or in stall runs",
        "softgrip": "excess finger_sum= and threshold= on a grasp_diag close line of a soft run. [soft_baseline] "
                    "gives the ratios only. Not logged on [held_abort] closes or in non-soft runs",
    }
    for det in DETECTORS:
        n_app = sum(1 for ep in eps if applies(det, ep) for ev in ep["close"] if ev["kind"] == "diag")
        n_ok = sum(1 for ep in eps if applies(det, ep) for ev in ep["close"]
                   if ev["kind"] == "diag" and det in verdicts(ev, ep))
        state = "yes" if n_ok and n_ok == n_app else ("partly" if n_ok else "no")
        out(f"  {det:<11} {state:<7} ({n_ok}/{n_app} diag closes)"
            + ("" if state == "yes" else f"  missing: {missing[det]}"))
    out("\n== verdict vs ground truth (close sample, run threshold) ==")
    out(f"{'detector':<12}{'n':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'acc':>7}  excluded")
    for det in DETECTORS:
        rows, skipped = [], Counter()
        for ep in eps:
            if not applies(det, ep):
                continue
            for ev in ep["close"]:
                v = verdicts(ev, ep).get(det)
                if v is None:
                    skipped["no field"] += 1
                elif ev["gt"] is None:
                    skipped["no ground truth"] += 1
                else:
                    rows.append((v, ev["gt"]))
        out(fmt_row(det, rows, ", ".join(f"{k}={n}" for k, n in skipped.items())))
    if per_event:
        out("\n== per close ==")
        for ep in eps:
            for ev in ep["close"]:
                v = verdicts(ev, ep)
                out(f"{ep['file']} seed={ep['seed']} t={ev['t']:.1f} {ev['color']} {ev['side']} {ev['kind']} "
                    f"gt={ev['gt']} " + " ".join(f"{k}={'yes' if x else 'no'}" for k, x in sorted(v.items())))
    for det in ("finger_sum", "softgrip"):
        grid = sweep(eps, det, step)
        if not grid:
            continue
        unit = "excess" if det == "softgrip" else "finger_sum"
        out(f"\n== {det} threshold sweep ({unit} >= thr; censored aborts only at thr >= run thr) ==")
        out(f"{'thr':>7}{'n':>5}{'TP':>5}{'FP':>5}{'FN':>5}{'TN':>5}{'acc':>7}  censored_excluded")
        prev = None
        for t, rows, excl in grid:       # one line per distinct confusion (the sweep is dense)
            if rows and confusion(rows) != prev:
                out(fmt_row(f"{t:.3f}", rows, str(excl)))
                prev = confusion(rows)


def self_check():
    stall_log = "\n".join([
        "2026-11-14 10:00:00,000 INFO reset task=sort_fixed seed=3 instruction='x'",
        "2026-11-14 10:00:01,000 INFO [grasp_diag] t=12.3 red left target=red_bin held=yes close@120: "
        "track_err_max=0.250 stall_vs_closed=[0.1, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0] cmd_vs_closed=[0.0, 0.0, 0, 0, 0, 0, 0] "
        "lift@150: track_err_max=0.001 stall_vs_closed=[0.0] cmd_vs_closed=[0.0]",
        "2026-11-14 10:00:02,000 INFO [grasp_diag] t=20.0 blue left target=blue_bin held=no close@120: "
        "stall_vs_closed=[0.05, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] cmd_vs_closed=[0.045, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0] "
        "lift@150: track_err_max=0.0 stall_vs_closed=[0.0] cmd_vs_closed=[0.0]",
    ])
    fs_log = "\n".join([
        "2026-11-14 11:00:00,000 INFO reset task=sort_fixed seed=4 instruction='x'",
        "2026-11-14 11:00:01,000 INFO [grasp_diag] t=5.0 red left target=red_bin held=yes close@120: "
        "finger_sum=0.230 index_sum=0.230 middle_sum=0.010 threshold=0.045 sample_held=yes "
        "stall_vs_closed=[0.0] cmd_vs_closed=[0.0] lift@150: finger_sum=0.2 index_sum=0.2 middle_sum=0 "
        "threshold=0.045 sample_held=yes stall_vs_closed=[0.0] cmd_vs_closed=[0.0]",
        "2026-11-14 11:00:02,000 INFO [held_abort] t=9.0 blue left target=blue_bin close sample empty: carry skipped, "
        "object remains",
    ])
    soft_log = "\n".join([
        "2026-11-14 12:00:00,000 INFO reset task=sort_fixed seed=5 instruction='x'",
        "2026-11-14 12:00:00,500 INFO [soft_baseline] n=12 ok={'left': True, 'right': True} empty-close error/travel",
        "2026-11-14 12:00:01,000 INFO [grasp_diag] t=5.0 red left target=red_bin held=yes close@120: "
        "finger_sum=1.750 index_sum=1.7 middle_sum=0.1 threshold=0.120 sample_held=yes "
        "stall_vs_closed=[0.0] cmd_vs_closed=[0.0] lift@150: finger_sum=1.6 index_sum=1.6 middle_sum=0 "
        "threshold=0.080 sample_held=yes stall_vs_closed=[0.0] cmd_vs_closed=[0.0]",
        "2026-11-14 12:00:02,000 INFO [grasp_diag] t=9.0 blue left target=blue_bin held=no close@120: "
        "finger_sum=0.000 index_sum=0.0 middle_sum=0.0 threshold=0.120 sample_held=no "
        "stall_vs_closed=[0.0] cmd_vs_closed=[0.0] lift@150: finger_sum=0.0 index_sum=0.0 middle_sum=0 "
        "threshold=0.080 sample_held=no stall_vs_closed=[0.0] cmd_vs_closed=[0.0]",
    ])
    eps = parse_log(stall_log, "stall") + parse_log(fs_log, "fs") + parse_log(soft_log, "soft")
    trials = {"3": {"seed": 3, "outcomes": [{"color": "red", "outcome": "correct"},
                                            {"color": "blue", "outcome": "counter"}]},
              "4": {"seed": 4, "outcomes": [{"color": "red", "outcome": "wrong"},
                                            {"name": "blue_cube_1", "outcome": "counter"}]},
              "5": {"seed": 5, "outcomes": [{"name": "red_cube_0", "outcome": "correct"},
                                            {"color": "blue", "outcome": "counter"}]}}
    attach_truth(eps, trials)
    byfile = {ep["file"]: ep for ep in eps}
    s, f, z = byfile["stall"]["close"], byfile["fs"]["close"], byfile["soft"]["close"]
    assert verdicts(s[0], byfile["stall"]) == {"stall": True}, verdicts(s[0], byfile["stall"])   # from track_err_max
    assert verdicts(s[1], byfile["stall"]) == {"stall": False}, verdicts(s[1], byfile["stall"])  # from the vectors
    assert verdicts(f[0], byfile["fs"])["finger_sum"] is True
    assert verdicts(f[1], byfile["fs"])["finger_sum"] is False
    assert verdicts(z[0], byfile["soft"])["softgrip"] is True
    assert verdicts(z[1], byfile["soft"])["softgrip"] is False
    assert [e["gt"] for e in s] == [True, False] and [e["gt"] for e in f] == [True, False]
    assert [e["gt"] for e in z] == [True, False]
    assert not applies("finger_sum", byfile["soft"]) and applies("softgrip", byfile["soft"])
    lines = []
    report(eps, 0.005, False, out=lines.append)
    grid = sweep(eps, "finger_sum", 0.005)
    at_run = next(rows for t, rows, _ in grid if abs(t - 0.045) < 1e-9)
    assert confusion(at_run) == (1, 0, 0, 1), confusion(at_run)   # red TP, blue censored abort TN
    print("\n".join(lines))
    print("SELF_CHECK: OK (asserts: stall from track_err and from vectors, finger_sum, softgrip, ground truth by colour)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="*", help="policy_server.log / container.log files")
    ap.add_argument("--results", nargs="*", default=[], help="eval.json / per-trial JSON with outcomes")
    ap.add_argument("--step", type=float, default=0.01, help="threshold sweep step (finger_sum / excess units)")
    ap.add_argument("--per-event", action="store_true", help="also print one line per close")
    ap.add_argument("--self-check", action="store_true", help="run the built-in hand-written-log check and exit")
    a = ap.parse_args(argv)
    if a.self_check:
        self_check()
        return 0
    if not a.logs:
        ap.error("give at least one log file (or --self-check)")
    eps = []
    for p in a.logs:
        with open(p, encoding="utf-8", errors="replace") as fh:
            eps += parse_log(fh.read(), p)
    attach_truth(eps, load_results(a.results) if a.results else {})
    report(eps, a.step, a.per_event)
    return 0


if __name__ == "__main__":
    sys.exit(main())
