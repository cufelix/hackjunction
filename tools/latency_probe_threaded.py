#!/usr/bin/env python3
"""act() / look / plan latency on mock episodes, in production mode (threaded look, safety shell per overlay).

Production = ClassicalServerPolicy (policy_server.py:39): ClassicalPolicy(threaded=True, time_limit=120).
Modes:
    default          timing run: act() wall ms; look (_look) and plan (_plan_pick) wall ms; thread of each.
    --trace          attribution run: wraps perception.* / planner.plan* / _plan_pick and records every call that
                     runs on the act() thread (main thread) with the policy.py call site under _act. Its timings
                     are inflated by the stack capture: use it for attribution only, not for the table.
    --safety-shell   overrides policy.safety_shell=true (not the overlay file).
    --gil            act() paced at 30 Hz on the main thread with a static mock frame; reports achieved Hz and the
                     worst gap between act() starts. --busy N starts N busy-loop CPU processes for the run.

    PYTHONPATH=. python tools/latency_probe_threaded.py --overlays finals_r19 --seeds 1 2 3 4 5
"""
from __future__ import annotations

import argparse
import inspect
import os
import re
import signal
import subprocess
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

EPISODE_WALL_CAP_S = 120.0
STEP_CAP = 6000
BUDGET_MS = 33.0
ACT_FUNCS = {"act", "_act", "_act_shell"}
CALL_PAT = re.compile(r"^(detect|calibrat|pitch|fit|estimate|plan|plan_to_counter|go_to|_plan_pick)")
THIS = os.path.abspath(__file__)


class EpisodeTimeout(BaseException):
    pass


def _on_alarm(signum, frame):
    raise EpisodeTimeout(f"episode wall cap {EPISODE_WALL_CAP_S:g} s")


def _pct(xs, q):
    if not xs:
        return float("nan")
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def act_site(stack):
    """'policy.py:L in _act -> callee' for the last frame named act/_act/_act_shell, skipping this probe's wrappers."""
    for k in range(len(stack) - 1, -1, -1):
        f = stack[k]
        if f.name in ACT_FUNCS and "classical" in f.filename:
            callee = next((g for g in stack[k + 1:] if os.path.abspath(g.filename) != THIS), None)
            cname = callee.name if callee else "?"
            return f"{os.path.basename(f.filename)}:{f.lineno} {f.name} -> {cname}"
    return "NOT-UNDER-ACT"


def make_wrap(fn, label, events, main_id, trace):
    def w(*a, **kw):
        on_main = threading.get_ident() == main_id
        site = act_site(traceback.extract_stack()) if (trace and on_main) else None
        s = time.perf_counter()
        try:
            return fn(*a, **kw)
        finally:
            events.append((label, (time.perf_counter() - s) * 1000.0, on_main, site))
    return w


def build(overlay, time_limit, sync, overrides):
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    ov = {"brain.backend": "stub", "brain.find": False, **overrides}
    cfg = load_config(overlay=overlay, overrides=ov)
    # Production constructor args (policy_server.py:39): threaded=True, require_confirm=False, default safety.
    pol = ClassicalPolicy(config=cfg, threaded=not sync, require_confirm=False, time_limit=time_limit, log=None)
    return cfg, pol


def wait_look(pol):  # production reset(): let a running look thread finish before the next episode
    t0 = time.monotonic()
    while pol._job is not None and pol._job["result"] is None and time.monotonic() - t0 < 30.0:
        time.sleep(0.005)


def instrument(pol, main_id, trace, events):
    """Timing (look/plan) always; call-level wrappers on perception and planner when trace."""
    orig_look, orig_plan = pol._look, pol._plan_pick
    pol._look = make_wrap(orig_look, "look", events, main_id, False)
    pol._plan_pick = make_wrap(orig_plan, "plan", events, main_id, False)
    if trace:
        for obj, prefix in ((pol.perception, "perception"), (pol.planner, "planner")):
            for name in dir(obj):
                if name.startswith("__") or not CALL_PAT.match(name):
                    continue
                attr = getattr(obj, name, None)
                if inspect.ismethod(attr):
                    setattr(obj, name, make_wrap(attr, f"{prefix}.{name}", events, main_id, True))
        pol._plan_pick = make_wrap(orig_plan, "plan", events, main_id, True)


def run_episode(overlay, seed, time_limit, sync, overrides, trace, rec):
    from aaltonomous.baselines.classical.mock_world import make_mock_env
    cfg, pol = build(overlay, time_limit, sync, overrides)
    env = make_mock_env(seed=seed, cfg=cfg)
    env.time_limit = time_limit
    main_id = threading.get_ident()
    instrument(pol, main_id, trace, rec["events"])
    rec["mode"] = {"threaded": pol.threaded, "safety_shell": pol._shell}
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    k, info = 0, {}
    while True:
        s = time.perf_counter()
        in_flight = bool(pol.threaded and pol._job is not None and pol._job["result"] is None)
        a = pol.act(obs)
        rec["act"].append(((time.perf_counter() - s) * 1000.0, in_flight))
        obs, info = env.step(a)
        k += 1
        if info["episode_over"] or k >= STEP_CAP:
            break
    wait_look(pol)
    rec["end"] = info.get("end_reason", "step_cap")
    rec["correct"] = env.backend.outcomes(pol.mapping).count("correct")
    rec["n"] = len(env.layout)


def summarise(label, xs):
    vals = [v for v, *_ in xs]
    if not vals:
        return f"{label:<5} n=0"
    over = 100.0 * sum(v > BUDGET_MS for v in vals) / len(vals)
    return (f"{label:<5} n={len(vals):<6} p50={_pct(vals, 50):9.2f}  p99={_pct(vals, 99):9.2f}  "
            f"max={max(vals):9.2f}  %>{BUDGET_MS:g}ms={over:6.2f}")


def timing_main(args, overrides):
    for ov in args.overlays:
        acts, looks, plans = [], [], []
        off_main = Counter()
        flight_acts = 0
        for seed in args.seeds:
            rec = {"act": [], "events": [], "end": None}
            signal.setitimer(signal.ITIMER_REAL, EPISODE_WALL_CAP_S)
            try:
                run_episode(ov, seed, args.time_limit, args.sync, overrides, False, rec)
                status = f"end={rec['end']} correct={rec['correct']}/{rec['n']}"
            except BaseException as e:  # noqa: BLE001
                status = f"{type(e).__name__}: {e}"
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            print(f"[{ov} seed={seed}] mode={rec.get('mode')} act={len(rec['act'])} "
                  f"look={sum(e[0] == 'look' for e in rec['events'])} "
                  f"plan={sum(e[0] == 'plan' for e in rec['events'])} {status}", flush=True)
            acts += rec["act"]
            for lab, ms, on_main, _ in rec["events"]:
                (looks if lab == "look" else plans).append((ms, on_main))
                off_main[lab] += (not on_main)
            flight_acts += sum(f for _, f in rec["act"])
        tag = " ".join(f"{k}" for k in [ov])
        print(f"== {tag} act: " + summarise("act", acts), flush=True)
        print(f"   {'':<{len(tag)}}  " + summarise("look", looks), flush=True)
        print(f"   {'':<{len(tag)}}  " + summarise("plan", plans), flush=True)
        print(f"   off-act-thread looks={off_main['look']}/{len(looks)} plans={off_main['plan']}/{len(plans)}; "
              f"acts with a look in flight={flight_acts}/{len(acts)}", flush=True)


def trace_main(args, overrides):
    for ov in args.overlays:
        events = []
        sites = Counter()
        for seed in args.seeds:
            rec = {"act": [], "events": [], "end": None}
            signal.setitimer(signal.ITIMER_REAL, EPISODE_WALL_CAP_S)
            try:
                run_episode(ov, seed, args.time_limit, args.sync, overrides, True, rec)
            except BaseException as e:  # noqa: BLE001
                print(f"[{ov} seed={seed}] {type(e).__name__}: {e}", flush=True)
            finally:
                signal.setitimer(signal.ITIMER_REAL, 0)
            events += rec["events"]
        by = Counter()
        total = Counter()
        ms_of = {}
        for lab, ms, on_main, site in events:
            total[lab] += 1
            if on_main:
                by[lab] += 1
                sites[(lab, site)] += 1
                ms_of.setdefault((lab, site), []).append(ms)
        print(f"== trace {ov}: calls={dict(total)} on act thread={dict(by)}", flush=True)
        for (lab, site), n in sites.most_common():
            v = ms_of[(lab, site)]
            print(f"   {n:4d} x {lab:<28} at {site}  total={sum(v):.1f} ms max={max(v):.1f} ms", flush=True)


def gil_run(overlay, seconds, busy, overrides):
    import numpy as np  # noqa: F401
    from aaltonomous.baselines.classical.mock_world import make_mock_env
    procs = [subprocess.Popen([sys.executable, "-c", "while 1: pass"]) for _ in range(busy)]
    try:
        cfg, pol = build(overlay, 120.0, False, overrides)
        env = make_mock_env(seed=1, cfg=cfg)
        obs = env.reset()
        pol.reset({"instruction": obs["instruction"]})
        starts, durs, looks, main_id = [], [], [], threading.get_ident()
        ev = []
        instrument(pol, main_id, False, ev)
        t0 = time.perf_counter()
        nxt = t0
        i = 0
        while time.perf_counter() - t0 < seconds:
            now = time.perf_counter()
            if now < nxt:
                time.sleep(nxt - now)
            nxt += 1.0 / 30.0
            o = dict(obs, t=i / 30.0)
            s = time.perf_counter()
            starts.append(s)
            pol.act(o)
            durs.append((time.perf_counter() - s) * 1000.0)
            i += 1
        wall = time.perf_counter() - t0
        gaps = [(b - a) * 1000.0 for a, b in zip(starts, starts[1:])]
        lk = [ms for lab, ms, *_ in ev if lab == "look"]
        wait_look(pol)
    finally:
        for p in procs:
            p.kill()
            p.wait()
    print(f"[gil busy={busy}] {overlay}: acts={len(starts)} achieved={len(starts) / wall:.1f} Hz "
          f"(target 30) worst_gap={max(gaps):.1f} ms p99_gap={_pct(gaps, 99):.1f} ms "
          f"act p99={_pct(durs, 99):.2f} ms max={max(durs):.2f} ms  looks_done={len(lk)} "
          f"look_p50={_pct(lk, 50):.0f} ms", flush=True)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--overlays", nargs="+", default=["finals_r19", "rc1"])
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4, 5])
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--sync", action="store_true", help="look inline (not production); comparison only")
    ap.add_argument("--safety-shell", action="store_true", help="override policy.safety_shell=true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--gil", action="store_true")
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--busy", type=int, default=0)
    args = ap.parse_args(argv)
    overrides = {"policy.safety_shell": True} if args.safety_shell else {}
    signal.signal(signal.SIGALRM, _on_alarm)
    print(f"mode: {'SYNC (NOT production)' if args.sync else 'threaded (production)'}; "
          f"overrides={overrides}; time_limit={args.time_limit}s; seeds={args.seeds}; "
          f"loadavg(os)={os.getloadavg()}", flush=True)
    if args.gil:
        for ov in args.overlays:
            gil_run(ov, args.seconds, args.busy, overrides)
    elif args.trace:
        trace_main(args, overrides)
    else:
        timing_main(args, overrides)


if __name__ == "__main__":
    main()
