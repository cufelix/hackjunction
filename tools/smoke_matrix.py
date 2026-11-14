#!/usr/bin/env python3
"""Smoke matrix: every classical overlay x mock episodes (seeds) + synthetic faults, on the render/grasp mock world.

Run from the repo root:  PYTHONPATH=. python3 tools/smoke_matrix.py [--out smoke_matrix.md] [--workers 2]
    --time-limit SIM_S   episode cap in sim seconds (the mock uses a virtual clock; wall cost ~ 0.1x sim)
    --seeds 0 1 2 3 4    clean episodes per overlay
    --overlays a b       subset (default: every overlays/*.yaml)

Per overlay: load_config(overlay=name) (load error recorded, episodes skipped), a ClassicalPolicy on
make_mock_env, then the clean seeds plus three variants on seed 0: depth blackout (head_depth zeros for 30 act
frames), missing bin (the first bin's projected footprint zeroed in head_rgb + head_depth), stale frame (the obs
from frame 60 fed for 30 act frames). Exceptions are counted whether they escape act() or are swallowed by the
safety shell (policy._act is wrapped to record them). Backends that would contact a server (brain.backend
llamacpp, brain.find) are forced to stub / find off. No network is touched.
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import itertools
import os
import signal
import sys
import time
import traceback
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
REPO = Path(__file__).resolve().parents[1]
OVERLAY_DIR = REPO / "aaltonomous" / "baselines" / "classical" / "overlays"
P99_LIMIT_MS = 33.0
FAULT_START, FAULT_LEN = 5, 20
STEP_CAP = 6000  # hard guard against a policy that never ends the episode
EPISODE_WALL_CAP_S = 30.0  # wall-clock cap per episode: a stuck or pathologically slow act() is a failure, not a hang


class EpisodeTimeout(BaseException):  # BaseException: the policy's own `except Exception` must not swallow it
    pass


def _on_alarm(signum, frame):
    raise EpisodeTimeout(f"episode wall cap {EPISODE_WALL_CAP_S:g} s")
VARIANTS = ("blackout", "missing_bin", "stale")


def _innermost_repo_frame(tb):
    frames = [f for f in traceback.extract_tb(tb) if str(REPO) in f.filename and f.name != "_on_alarm"]
    f = frames[-1] if frames else traceback.extract_tb(tb)[-1]
    return f"{os.path.relpath(f.filename, REPO)}:{f.lineno} in {f.name}"


def _missing_bin_mask(env, shape):
    """(v0, v1, u0, u1) pixel box of the first bin's inner volume, or None if it projects off-image."""
    import numpy as np
    sc, rend = env.scene, env.backend.renderer
    lo, hi = sc.bin_inner_box(sc.bin_names[0])
    pts = [sc.mat_to_pelvis(np.array(c, float)) for c in itertools.product(*zip(lo, hi))]
    px = rend._project(np.array(pts))
    px = px[px[:, 2] > 0]
    if len(px) == 0:
        return None
    H, W = shape
    u0, u1 = int(max(0, np.floor(px[:, 0].min()))), int(min(W, np.ceil(px[:, 0].max())))
    v0, v1 = int(max(0, np.floor(px[:, 1].min()))), int(min(H, np.ceil(px[:, 1].max())))
    return (v0, v1, u0, u1) if u1 > u0 and v1 > v0 else None


def _run_episode(cfg, seed, variant, time_limit, rec):
    """One mock episode under a wall-clock cap. Returns the episode dict; rec collects latencies and exceptions."""
    signal.signal(signal.SIGALRM, _on_alarm)
    signal.setitimer(signal.ITIMER_REAL, EPISODE_WALL_CAP_S)
    try:
        return _run_episode_body(cfg, seed, variant, time_limit, rec)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def _run_episode_body(cfg, seed, variant, time_limit, rec):
    import numpy as np
    from aaltonomous.baselines.classical.mock_world import make_mock_env
    from aaltonomous.baselines.classical.policy import ClassicalPolicy

    env = make_mock_env(seed=seed, cfg=cfg)
    env.time_limit = time_limit
    pol = ClassicalPolicy(config=cfg, log=None, safety=env.filter.cfg)
    orig_act = pol._act

    def recording_act(obs):  # sees exceptions the safety shell would otherwise swallow
        try:
            return orig_act(obs)
        except Exception as e:  # noqa: BLE001 - recorded, then re-raised for the shell's own handling
            rec["exc"].append((f"{type(e).__name__}: {e}", _innermost_repo_frame(e.__traceback__), variant, seed))
            raise

    pol._act = recording_act
    obs = env.reset()
    pol.reset({"instruction": obs["instruction"]})
    mask = _missing_bin_mask(env, obs["head_depth"].shape) if variant == "missing_bin" else None
    rec["masked_px"] = 0 if mask is None else (mask[1] - mask[0]) * (mask[3] - mask[2])
    frozen, k, info, end = None, 0, {}, None
    while True:
        if variant == "stale" and k == FAULT_START:
            frozen = obs
        o = obs
        in_fault = FAULT_START <= k < FAULT_START + FAULT_LEN
        if variant == "blackout" and in_fault:
            o = dict(obs, head_depth=np.zeros_like(obs["head_depth"]))
        elif variant == "missing_bin" and mask is not None:
            v0, v1, u0, u1 = mask
            rgb, dep = obs["head_rgb"].copy(), obs["head_depth"].copy()
            rgb[v0:v1, u0:u1], dep[v0:v1, u0:u1] = 0, 0
            o = dict(obs, head_rgb=rgb, head_depth=dep)
        elif variant == "stale" and in_fault:
            o = frozen
        s = time.perf_counter()
        try:
            a = pol.act(o)
        finally:
            rec["lat"].append((time.perf_counter() - s) * 1000.0)
        if isinstance(a, dict) and "arm_q_target" in a:
            rec["trace"].append(tuple(round(float(x), 4) for x in a["arm_q_target"][-1]))
        obs, info = env.step(a)
        k += 1
        if info["episode_over"]:
            end = info["end_reason"]
            break
        if k >= STEP_CAP:
            end = "step_cap"
            break
    n = len(env.layout)
    correct = env.backend.outcomes(pol.mapping).count("correct")
    stats = dict(getattr(pol, "_shell_stats", {}) or {})
    rec["overrun"] = rec.get("overrun", 0) + stats.get("overrun", 0)
    return {"seed": seed, "variant": variant, "correct": correct, "n": n, "end": end, "t": round(info.get("t", 0.0), 1),
            "trace": rec["trace"]}


def run_overlay(name, seeds, time_limit, variants=VARIANTS):
    """Full row for one overlay. Never raises: every failure becomes a field of the returned dict."""
    row = {"overlay": name, "load_ok": False, "load_error": None, "episodes": [], "lat": [], "exc": [],
           "notes": [], "masked_px": None, "overrun": 0}
    try:
        from aaltonomous.baselines.classical.config import load_config
        cfg = load_config(overlay=name)
        row["load_ok"] = True
        brain = cfg.get("brain") or {}
        ov = {}
        if brain.get("backend") != "stub":
            ov["brain.backend"] = "stub"
        if brain.get("find"):
            ov["brain.find"] = False
        if ov:
            cfg = load_config(overlay=name, overrides=ov)
            row["notes"].append(f"forced {', '.join(f'{k}={v}' for k, v in ov.items())} (no server contact)")
    except Exception as e:  # noqa: BLE001
        row["load_error"] = f"{type(e).__name__}: {str(e).splitlines()[0][:160]}"
        return row
    jobs = [(s, "clean") for s in seeds] + [(0, v) for v in variants]
    for seed, variant in jobs:
        rec = {"lat": [], "exc": [], "trace": []}
        try:
            ep = _run_episode(cfg, seed, None if variant == "clean" else variant, time_limit, rec)
            ep["variant"] = variant
        except BaseException as e:  # noqa: BLE001 - setup, step or wall-cap failure (outside act)
            rec["exc"].append((f"{type(e).__name__}: {e}", _innermost_repo_frame(e.__traceback__), variant, seed))
            ep = {"seed": seed, "variant": variant, "correct": None, "n": None,
                  "end": "timeout" if isinstance(e, EpisodeTimeout) else "crashed", "t": None}
        row["episodes"].append(ep)
        ep.setdefault("trace", rec["trace"])
        row["lat"] += rec["lat"]
        row["exc"] += rec["exc"]
        row["overrun"] += rec.get("overrun", 0)
        if rec.get("masked_px") is not None and variant == "missing_bin":
            row["masked_px"] = rec["masked_px"]
    clean0 = next((e for e in row["episodes"] if e["variant"] == "clean" and e["seed"] == 0), None)
    for ep in row["episodes"]:
        if ep["variant"] in variants and clean0 is not None and ep.get("trace") is not None:
            a, b = clean0["trace"], ep["trace"]
            k = next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), None)
            if k is None and len(a) == len(b):
                row["notes"].append(f"{ep['variant']}: no action change")
            else:
                row["notes"].append(f"{ep['variant']}: action diverges from clean at act #{k if k is not None else min(len(a), len(b))}")
    for ep in row["episodes"]:
        ep.pop("trace", None)
    return row


def _pct(xs, q):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(round(q / 100 * (len(xs) - 1))))]


def summarise(row):
    lat = row["lat"]
    eps = [e for e in row["episodes"] if e["variant"] == "clean" and e["n"]]
    score = (f"{sum(e['correct'] for e in eps)}/{sum(e['n'] for e in eps)} clean" if eps else "-")
    vs = [f"{e['variant']} {e['correct']}/{e['n']}" for e in row["episodes"] if e["variant"] != "clean" and e["n"]]
    row["p50"] = _pct(lat, 50)
    row["n_slow"] = sum(1 for x in lat if x >= P99_LIMIT_MS)
    row["p99"] = _pct(lat, 99)
    row["score"] = score + (("; " + ", ".join(vs)) if vs else "")
    ends = sorted({e["end"] for e in row["episodes"] if e["end"]})
    row["ends"] = ends
    first = row["exc"][0] if row["exc"] else None
    row["exc_n"] = len(row["exc"])
    row["exc_head"] = first[0] if first else None
    row["exc_loc"] = f"{first[1]} (variant {first[2]}, seed {first[3]})" if first else None
    row["is_fail"] = (not row["load_ok"]) or row["exc_n"] > 0 or (row["p99"] is not None and row["p99"] >= P99_LIMIT_MS)
    return row


def _cell(s):
    return str(s).replace("|", "/").replace("\n", " ")


def render_md(rows, args, wall_s):
    n = len(rows)
    n_load = sum(1 for r in rows if not r["load_ok"])
    n_exc = sum(1 for r in rows if r["exc_n"] > 0)
    n_lat = sum(1 for r in rows if r["p99"] is not None and r["p99"] >= P99_LIMIT_MS)
    order = sorted(rows, key=lambda r: (0 if not r["load_ok"] else 1 if r["exc_n"] else 2 if r["p99"] and r["p99"] >= P99_LIMIT_MS else 3, r["overlay"]))
    out = [f"# Smoke matrix (mock world)", "",
           f"Run: {time.strftime('%Y-%m-%d %H:%M')}; {n} overlays; seeds {args.seeds}; time_limit {args.time_limit} sim s; "
           f"variants {', '.join(VARIANTS)} (seed 0, {FAULT_LEN} act frames from frame {FAULT_START}); "
           f"workers {args.workers}; wall {wall_s:.0f} s.", "",
           f"SUMMARY: {n} overlays, {n_load} load errors, {n_exc} with exceptions, {n_lat} with p99 >= {P99_LIMIT_MS:g} ms", "",
           "Requirement: 0 exceptions and p99 < 33 ms. Score = mock_world outcome rule (correct placements), not a task score; "
           "mock_world reports no score of its own.", "",
           "| overlay | load | exc | p50 ms | p99 ms | score | notes |", "|---|---|---|---|---|---|---|"]
    for r in order:
        load = "ok" if r["load_ok"] else "ERR"
        exc = str(r["exc_n"])
        p50 = "-" if r["p50"] is None else f"{r['p50']:.1f}"
        p99 = "-" if r["p99"] is None else f"{r['p99']:.1f}" + (" FAIL" if r["p99"] >= P99_LIMIT_MS else "")
        notes = []
        if not r["load_ok"]:
            notes.append(f"load: {r['load_error']}")
            score = "SKIP (load error)"
        else:
            score = r["score"]
            notes += r["notes"]
            if r["ends"]:
                notes.append("ends: " + ",".join(r["ends"]))
            if r["exc_n"]:
                notes.append(f"exc: {r['exc_head']} @ {r['exc_loc']}")
            if r.get("n_slow"):
                notes.append(f"{r['n_slow']}/{len(r['lat'])} act calls >= 33 ms")
            if r["overrun"]:
                notes.append(f"{r['overrun']} safety-shell overruns")
            if r["masked_px"] is not None and r["masked_px"] == 0:
                notes.append("missing_bin: bin off-image, nothing masked")
        out.append(f"| {_cell(r['overlay'])} | {load} | {exc} | {p50} | {p99} | {_cell(score)} | {_cell('; '.join(notes) or '-')} |")
    out += ["", "Notes: p50/p99 pool every act() call of the overlay's episodes (clean + variants). A p99 over ~1000 calls "
            "is dominated by the few look/calibration calls, so it is a worst-tail measure, not a steady rate."]
    return "\n".join(out) + "\n", n_exc, n_load, n_lat


def run_matrix(names=None, seeds=(0, 1, 2, 3, 4), time_limit=30.0, workers=2, variants=VARIANTS):
    if names is None:
        names = sorted(p.stem for p in OVERLAY_DIR.glob("*.yaml"))
    t0 = time.time()
    rows = []
    with cf.ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(run_overlay, n, list(seeds), time_limit, variants): n for n in names}
        for fut in cf.as_completed(futs):
            n = futs[fut]
            try:
                rows.append(summarise(fut.result()))
            except Exception as e:  # worker died (e.g. OOM): record as an exception on that overlay
                rows.append(summarise({"overlay": n, "load_ok": True, "load_error": None, "episodes": [], "lat": [],
                                       "exc": [(f"worker: {type(e).__name__}: {e}", "pool", "-", "-")], "notes": [],
                                       "masked_px": None, "overrun": 0}))
            print(f"[{len(rows)}/{len(names)}] {n} done ({time.time() - t0:.0f} s)", file=sys.stderr, flush=True)
    return rows, time.time() - t0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlays", nargs="*", default=None)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--time-limit", type=float, default=30.0)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--no-variants", action="store_true")
    ap.add_argument("--out", default=None, help="markdown output path (default: stdout only)")
    args = ap.parse_args(argv)
    args.seeds = tuple(args.seeds)
    rows, wall = run_matrix(args.overlays, args.seeds, args.time_limit, args.workers,
                            () if args.no_variants else VARIANTS)
    md, n_exc, n_load, n_lat = render_md(rows, args, wall)
    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(md)
    print(md)
    return rows


if __name__ == "__main__":
    main()
