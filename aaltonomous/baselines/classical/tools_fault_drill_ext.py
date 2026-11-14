"""Extra release-candidate drills on top of tools_fault_drill: offline cold start, websocket drop + reconnect,
release gate (submission flags == rc flags).  Every episode drill reports: no exception, act() latency p50/p99/max
(>= 27 Hz needs p99 < 37 ms), objects in a wrong bin (must be 0).

    python -m aaltonomous.baselines.classical.tools_fault_drill_ext --overlay rc1 [--gate-rc rc1]
"""
from __future__ import annotations

import argparse
import json
import socket
import subprocess
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

P99_LIMIT_MS = 37.0   # 27 Hz


@dataclass
class Drill:
    name: str
    overlay: str
    no_exception: bool = True
    p50: float = float("nan")
    p99: float = float("inf")
    max: float = float("inf")
    wrong: int = -1
    ok: bool = True        # drill-specific condition (zero socket attempts, reconnect worked, ...)
    detail: str = ""

    @property
    def passed(self):
        return self.no_exception and self.ok and self.p99 < P99_LIMIT_MS and self.wrong == 0


def _stats(d, lat):
    if len(lat):
        d.p50, d.p99, d.max = (float(np.percentile(lat, 50)), float(np.percentile(lat, 99)), float(np.max(lat)))
    return d


def _episode(env, pol, act, lat, ticks=None, on_tick=None):
    """Step the mock env with act(obs) until the episode ends (or `ticks` steps); lat gets act() ms. Returns wrong count
    at the end (objects whose bin is wrong under the policy's mapping)."""
    obs, n = env.reset(), 0
    pol.reset({"instruction": obs["instruction"], "time_limit": 60.0})
    while ticks is None or n < ticks:
        if on_tick:
            on_tick(n)
        t0 = time.perf_counter_ns()
        action = act(obs)
        lat.append((time.perf_counter_ns() - t0) / 1e6)
        obs, info = env.step(action)
        n += 1
        if info["episode_over"]:
            break
    return env.backend.outcomes(pol.mapping).count("wrong"), n


def _build(overlay, seed=0):
    from .config import load_config
    from .mock_world import make_mock_env
    from .policy import ClassicalPolicy
    cfg = load_config(overlay=overlay)
    env = make_mock_env(seed=seed, cfg=cfg, realtime=False)
    return cfg, env, ClassicalPolicy(config=cfg, safety=env.filter.cfg, threaded=True, time_limit=60.0, log=None)


# ---------------------------------------------------------------- offline cold start
def _offline_child(overlay, ticks=120):
    """Runs in a fresh interpreter: patch sockets FIRST, then import, build, reset, first episode."""
    attempts = []

    def deny(*a, **k):
        attempts.append(repr(a[1:2] or a)[:80])
        raise OSError("offline drill: network blocked")
    socket.socket.connect = deny
    socket.socket.connect_ex = deny
    socket.create_connection = deny
    socket.getaddrinfo = deny
    out = {"exc": None, "lat": []}
    try:
        from . import tools_fault_drill_ext as m   # import happens AFTER the patch
        cfg, env, pol = m._build(overlay)
        out["wrong"], out["ticks"] = m._episode(env, pol, pol.act, out["lat"], ticks=ticks)
        env.close()
    except BaseException as e:
        out["exc"] = f"{type(e).__name__}: {e}"
    out["attempts"] = attempts
    print("DRILLJSON" + json.dumps(out))


def drill_offline(overlay, ticks=120):
    d = Drill("offline_cold_start", overlay)
    code = (f"import sys; from aaltonomous.baselines.classical.tools_fault_drill_ext import _offline_child as c; "
            f"c({overlay!r}, {ticks})")
    r = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, timeout=300)
    line = next((x for x in r.stdout.splitlines() if x.startswith("DRILLJSON")), None)
    if line is None:
        d.no_exception, d.detail = False, f"child died rc={r.returncode}: {r.stderr[-300:]}"
        return d
    out = json.loads(line[len("DRILLJSON"):])
    d.no_exception, d.wrong = out["exc"] is None, out.get("wrong", -1)
    d.ok = not out["attempts"]
    d.detail = f"socket_attempts={len(out['attempts'])} {out['attempts'][:3]} ticks={out.get('ticks')} {out['exc'] or ''}"
    return _stats(d, out["lat"])


# ---------------------------------------------------------------- websocket drop + reconnect
def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def drill_ws_drop(overlay, drop_at=40):
    """Serve the policy with deploy/policy_server.py on localhost; kill the TCP connection mid-episode.
    Contract (runner.run_episode): act() on the dead link must raise promptly (runner ends the episode as
    policy_connection_lost), PolicyClient.reconnect() restores service, the server survives, and the NEXT episode
    (reset + full run) completes with 0 wrong. Also reports whether the interrupted episode could simply resume."""
    from aaltonomous.deploy import policy_server as ps
    from aaltonomous.deploy.runner import PolicyClient
    d = Drill("ws_drop_reconnect", overlay)
    cfg, env, pol = _build(overlay)

    class Adapter(ps.Policy):
        name, action_horizon = "classical", None

        def reset(self, instruction, task=None, seed=None):
            pol.reset({"instruction": instruction, "time_limit": 60.0})

        def act(self, obs):
            return pol.act(obs)

    port, ready, lat = _free_port(), threading.Event(), []
    threading.Thread(target=ps.serve, args=(Adapter(), port, "127.0.0.1", ready), daemon=True).start()
    client = None
    try:
        ready.wait(10)
        client = PolicyClient(f"ws://127.0.0.1:{port}", startup_timeout=10, recv_timeout=5)
        obs, info, notes = env.reset(), {}, []
        client.reset("sort_fixed", obs["instruction"], 0)
        pol.reset({"instruction": obs["instruction"], "time_limit": 60.0})   # local copy for the outcome mapping
        for n in range(drop_at):
            t0 = time.perf_counter_ns()
            action = client.act(obs)
            lat.append((time.perf_counter_ns() - t0) / 1e6)
            obs, info = env.step(action)
        client.ws.socket.shutdown(socket.SHUT_RDWR)       # the network drop
        t0 = time.monotonic()
        try:
            client.act(obs)
            d.ok = False
            notes.append("act() on a dead link did not raise")
        except Exception as e:
            notes.append(f"dead link raised {type(e).__name__} after {1e3 * (time.monotonic() - t0):.0f} ms")
        t0 = time.monotonic()
        client.reconnect(timeout=10)
        notes.append(f"reconnect {1e3 * (time.monotonic() - t0):.0f} ms")
        # resume the SAME episode on the new connection (server kept the policy state)
        resumed = 0
        while not info["episode_over"] and resumed < 60:
            t0 = time.perf_counter_ns()
            obs, info = env.step(client.act(obs))
            lat.append((time.perf_counter_ns() - t0) / 1e6)
            resumed += 1
        notes.append(f"resumed {resumed} ticks, end={info.get('end_reason')}")
        # next episode, as the runner does after a lost connection
        d.wrong, n2 = _episode(env, pol, client.act, lat, ticks=400,
                               on_tick=lambda n: client.reset("sort_fixed", env._obs()["instruction"], 1) if n == 0 else None)
        notes.append(f"episode 2: {n2} ticks")
        d.detail = "; ".join(notes)
    except Exception as e:
        d.no_exception, d.detail = False, f"{type(e).__name__}: {e}"
    finally:
        if client is not None:
            client.close()
        env.close()
        srv = getattr(ready, "server", None)
        if srv is not None:
            srv.shutdown()
    return _stats(d, lat)


# ---------------------------------------------------------------- full-episode baseline + release gate
def drill_episode(overlay, ticks=None):
    d = Drill("full_episode", overlay)
    lat = []
    try:
        cfg, env, pol = _build(overlay)
        d.wrong, n = _episode(env, pol, pol.act, lat, ticks=ticks)
        env.close()
        d.detail = f"ticks={n}"
    except Exception as e:
        d.no_exception, d.detail = False, f"{type(e).__name__}: {e}"
    return _stats(d, lat)


def release_gate(rc, base="submission"):
    """Flag values of overlay `base` vs `rc` over every config.FLAGS key. Returns [(key, base, rc)] of differences."""
    from .config import FLAGS, _get, load_config
    a, b = load_config(overlay=base), load_config(overlay=rc)
    return [(k, _get(a, k), _get(b, k)) for k in FLAGS if _get(a, k) != _get(b, k)]


def print_drills(rows):
    h = ("overlay", "drill", "exc", "p50", "p99", "max", "wrong", "ok", "result", "detail")
    data = [h] + [(r.overlay, r.name, "OK" if r.no_exception else "FAIL", f"{r.p50:.1f}", f"{r.p99:.1f}", f"{r.max:.1f}",
                   r.wrong, "OK" if r.ok else "FAIL", "PASS" if r.passed else "FAIL", r.detail) for r in rows]
    w = [max(len(str(x[i])) for x in data) for i in range(len(h))]
    for x in data:
        print("  ".join(str(v).ljust(w[i]) for i, v in enumerate(x)))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--overlay", action="append", help="overlay to drill (repeat); default: none")
    ap.add_argument("--gate-rc", help="release gate: flags of overlay `submission` must equal this rc overlay's")
    ap.add_argument("--skip-ws", action="store_true")
    a = ap.parse_args(argv)
    rows = []
    for ov in a.overlay or ["none"]:
        rows += [drill_episode(ov), drill_offline(ov)] + ([] if a.skip_ws else [drill_ws_drop(ov)])
    print_drills(rows)
    bad = not all(r.passed for r in rows)
    if a.gate_rc:
        diff = release_gate(a.gate_rc)
        print(f"\nrelease gate submission == {a.gate_rc}: {'PASS' if not diff else 'FAIL'}")
        for k, x, y in diff:
            print(f"  {k}: submission={x!r} {a.gate_rc}={y!r}")
        bad |= bool(diff)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
