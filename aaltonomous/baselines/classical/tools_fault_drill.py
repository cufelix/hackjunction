"""Robustness fault drill for the classical policy server, over the real websocket + msgpack protocol.

Each episode gets a fresh ClassicalServerPolicy served in-process on a free 127.0.0.1 port (the same class the
Docker image runs, overlay picked via AALTO_OVERLAY). A client built on aaltonomous.deploy.protocol drives it
through a mock_world episode. Faults are injected into the observations the client sends; every returned
action is audited (finite, joint limits, per-tick velocity).

Per cell (fault x overlay):
  crashes    = server-side tracebacks (logs, threads), error replies, or unexpected client failures
  completed  = episode ended by the policy's "done" or by the time limit
  act Hz     = 1 / server-side act() time (compute rate), median and p5 over all act requests
  p99 act ms = 99th percentile of server-side act() time

    PYTHONPATH=. python -m aaltonomous.baselines.classical.tools_fault_drill --episodes 2 --report out.md
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import logging
import math
import os
import socket
import sys
import threading
import time
from dataclasses import dataclass, field

import numpy as np
from websockets.sync.client import connect

from aaltonomous.deploy import protocol as P
from aaltonomous.deploy.policy_server import Policy, serve
from aaltonomous.interface.safety import JointSpec

from .config import load_config
from .mock_world import Renderer, make_mock_env
from .policy import HZ
from .policy_server import ClassicalServerPolicy

FAULTS = ("instruction", "depth_blackout", "missing_bin", "ws_drop", "latency_jitter", "stale_frame")
OVERLAYS = ("submission", "rc1", "cand_r18")
MIN_MEDIAN_HZ = 27.0
INJECT_AT_S = 1.0           # virtual seconds before timed faults start
BLACKOUT_CALLS = 15         # act requests sent with depth blacked out
STALE_CALLS = 20            # act requests that repeat one frame
JITTER_S = (0.050, 0.150)   # client-side delay before every act request
RECV_TIMEOUT_S = 30.0
COMPLETE_REASONS = ("done", "time_limit")


class ActionAudit:
    """Audit every target the server returns: finite, inside joint limits, bounded per-tick step."""

    def __init__(self, cfg, obs):
        self.cfg, self.spec = cfg, JointSpec()
        self.arm = np.asarray(obs["arm_q"], float).copy()
        self.hand = np.asarray(obs["hand_q"], float).copy()
        self.sane = True

    def check(self, action):
        try:
            arm = np.asarray(action["arm_q_target"], float)
            hand = np.asarray(action["hand_q_target"], float)
        except (KeyError, TypeError, ValueError):
            self.sane = False
            return
        arm = arm[None] if arm.ndim == 1 else arm
        hand = hand[None] if hand.ndim == 1 else hand
        if arm.ndim != 2 or arm.shape[1] != 14 or hand.shape != arm.shape or not (np.isfinite(arm).all() and np.isfinite(hand).all()):
            self.sane = False
            return
        in_limits = (np.all((arm >= self.spec.arm_lower) & (arm <= self.spec.arm_upper)) and
                     np.all((hand >= self.spec.hand_lower) & (hand <= self.spec.hand_upper)))
        da = np.abs(np.diff(np.vstack([self.arm, arm]), axis=0)).max()
        dh = np.abs(np.diff(np.vstack([self.hand, hand]), axis=0)).max()
        vel_ok = (da <= self.cfg["motion"]["arm_max_vel"] / HZ + 1e-8 and
                  dh <= self.cfg["motion"]["hand_max_vel"] / HZ + 1e-8)
        self.sane = bool(self.sane and in_limits and vel_ok)
        self.arm, self.hand = arm[-1].copy(), hand[-1].copy()

    def hold(self):
        return {"arm_q_target": self.arm[None].copy(), "hand_q_target": self.hand[None].copy()}


def _without_bin(env, name):
    """Renderer for the head camera with one bin deleted from the scene (same trick as tools_fault_drill ref)."""
    scene = copy.deepcopy(env.scene)
    del scene.cfg["bins"]["centers"][name]
    return Renderer(scene, env.backend.renderer.cal)


class Injector:
    """Turns the clean observation into the one the server receives, per act request."""

    def __init__(self, fault, variant, env):
        self.fault, self.variant, self.env = fault, variant, env
        self.calls = 0
        self.blackout_from = None
        self.stale = None
        self.stale_left = 0
        self.bin_renderer = _without_bin(env, variant) if fault == "missing_bin" else None
        self.bin_cache = {}
        self.injected = 0          # act requests whose observation was actually altered

    def reset_instruction(self, instruction):
        """None means: leave the key out of the reset message."""
        if self.fault != "instruction":
            return instruction
        return "" if self.variant == "empty" else None

    def _missing_bin_frame(self):
        b = self.env.backend
        key = tuple(np.round(np.concatenate([o["pos"] for o in b.objects]), 4)) if b.objects else ()
        if self.bin_cache.get("key") != key:
            frame = self.bin_renderer.render(b.objects, np.random.default_rng(0), b.depth_noise_mm, b.hole_frac)
            self.bin_cache = {"key": key, "frame": frame}
        return self.bin_cache["frame"]

    def obs_for_act(self, obs):
        bad = dict(obs)
        self.calls += 1
        t = getattr(self.env, "t", 0.0)
        if self.fault == "instruction":
            if self.variant == "empty":
                bad["instruction"] = ""
            else:
                bad.pop("instruction", None)
            self.injected += 1
        elif self.fault == "missing_bin":
            bad["head_rgb"], bad["head_depth"] = self._missing_bin_frame()
            self.injected += 1
        elif self.fault == "depth_blackout":
            if self.blackout_from is None and t >= INJECT_AT_S:
                self.blackout_from = self.calls
            if self.blackout_from is not None and self.calls - self.blackout_from < BLACKOUT_CALLS:
                shape = obs["head_depth"].shape
                bad["head_depth"] = (np.zeros(shape, np.uint16) if self.variant == "zero"
                                     else np.full(shape, np.nan, np.float32))
                self.injected += 1
        elif self.fault == "stale_frame":
            if self.stale is None and t >= INJECT_AT_S:
                self.stale = (obs["head_rgb"].copy(), obs["head_depth"].copy())
                self.stale_left = STALE_CALLS
            if self.stale_left > 0:
                bad["head_rgb"], bad["head_depth"] = self.stale
                self.stale_left -= 1
                self.injected += 1
        return bad


class _TimedPolicy(Policy):
    """Times the real policy's act() on the server side; nothing else changes."""

    def __init__(self, inner, sink):
        self.inner, self.sink = inner, sink
        self.name, self.action_horizon = inner.name, inner.action_horizon

    def reset(self, instruction, task=None, seed=None):
        self.inner.reset(instruction, task=task, seed=seed)

    def act(self, obs):
        t0 = time.perf_counter()
        try:
            return self.inner.act(obs)
        finally:
            self.sink.append((time.perf_counter() - t0) * 1000.0)


class _ErrorLog(logging.Handler):
    """Server-side tracebacks: ERROR records, records carrying exc_info, or traceback text."""

    def __init__(self):
        super().__init__(level=logging.WARNING)
        self.hits = []

    def emit(self, record):
        text = record.getMessage()
        if record.levelno >= logging.ERROR or record.exc_info or "Traceback" in text:
            self.hits.append("\n".join(text.splitlines()[:8]))


@contextlib.contextmanager
def _capture_errors():
    handler = _ErrorLog()
    logger = logging.getLogger("aaltonomous")
    old_hook = threading.excepthook

    def hook(args):
        name = args.thread.name if args.thread else "?"
        handler.hits.append(f"thread {name}: {args.exc_type.__name__}: {args.exc_value}")

    logger.addHandler(handler)
    threading.excepthook = hook
    try:
        yield handler
    finally:
        logger.removeHandler(handler)
        threading.excepthook = old_hook


def _free_port():
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


class Server:
    """ClassicalServerPolicy behind serve() in a daemon thread, one per episode."""

    def __init__(self, overlay, time_limit):
        self.acts_ms = []
        prev = os.environ.get("AALTO_OVERLAY")
        os.environ["AALTO_OVERLAY"] = overlay
        try:
            inner = ClassicalServerPolicy(time_limit=time_limit)
        finally:
            if prev is None:
                os.environ.pop("AALTO_OVERLAY", None)
            else:
                os.environ["AALTO_OVERLAY"] = prev
        self.policy = _TimedPolicy(inner, self.acts_ms)
        self.port = _free_port()
        self.url = f"ws://127.0.0.1:{self.port}"
        self.thread = self.server = None

    def start(self):
        ready = threading.Event()
        self.thread = threading.Thread(target=serve, daemon=True, name="drill-policy-server",
                                       kwargs={"policy": self.policy, "port": self.port, "host": "127.0.0.1",
                                               "ready": ready})
        self.thread.start()
        if not ready.wait(10):
            raise RuntimeError("policy server did not start")
        t0 = time.monotonic()
        while getattr(ready, "server", None) is None and time.monotonic() - t0 < 5:  # set after ready.set()
            time.sleep(0.01)
        self.server = ready.server
        return self

    def stop(self):
        if self.server is not None:
            self.server.shutdown()
        if self.thread is not None:
            self.thread.join(5)


def _connect(url):
    ws = connect(url, max_size=None, compression=None, open_timeout=RECV_TIMEOUT_S, legacy=True)
    P.check_metadata(P.unpack(ws.recv(timeout=RECV_TIMEOUT_S)))
    return ws


@dataclass
class EpisodeResult:
    overlay: str
    fault: str
    episode: int
    crashes: int = 0
    completed: bool = False
    end_reason: str | None = None
    sane: bool = False
    act_ms: list = field(default_factory=list)
    crash_heads: list = field(default_factory=list)
    detail: str = ""


def run_episode(overlay, fault, episode, time_limit=10.0):
    cfg = load_config(overlay=overlay)
    env = make_mock_env(seed=episode, cfg=cfg, realtime=False)
    env.time_limit = time_limit
    variant = {"instruction": ("empty", "missing")[episode % 2],
               "depth_blackout": ("zero", "nan")[episode % 2]}.get(fault, "")
    if fault == "missing_bin":
        variant = env.scene.bin_names[episode % len(env.scene.bin_names)]
    res = EpisodeResult(overlay, fault, episode, detail=f"variant={variant or '-'}")
    rng = np.random.default_rng(episode)
    server = ws = audit = None
    error_replies = 0
    with _capture_errors() as errs:
        try:
            server = Server(overlay, time_limit).start()
            ws = _connect(server.url)
            obs = env.reset()
            inj = Injector(fault, variant, env)
            audit = ActionAudit(cfg, obs)
            reset = {"type": "reset", "task": "sort_fixed", "seed": episode}
            instr = inj.reset_instruction(obs["instruction"])
            if instr is not None:
                reset["instruction"] = instr
            ws.send(P.pack(reset))
            if P.unpack(ws.recv(timeout=RECV_TIMEOUT_S)).get("type") != "reset_ok":
                error_replies += 1
            info, ws_dropped = {}, False
            for _ in range(int(time_limit * HZ) + 10):
                if fault == "ws_drop" and not ws_dropped and getattr(env, "t", 0.0) >= time_limit / 2:
                    ws.close()
                    ws = _connect(server.url)   # reconnect; the episode continues on the same server
                    ws_dropped = True
                bad = inj.obs_for_act(obs)
                if fault == "latency_jitter":
                    time.sleep(rng.uniform(*JITTER_S))
                ws.send(P.pack({"type": "act", "obs": bad}))
                msg = P.unpack(ws.recv(timeout=RECV_TIMEOUT_S))
                if msg.get("type") == "action":
                    action = msg["action"]
                else:
                    error_replies += 1
                    res.detail += f" error={msg.get('message', msg)!s:.120}"
                    action = audit.hold()
                audit.check(action)
                obs, info = env.step(action)
                if info["episode_over"]:
                    break
            res.detail += f" injected={inj.injected}"
            events = getattr(server.policy.inner.pol, "events", None) or []
            if events:
                res.detail += f" last_event={events[-1][1]!r}"
            res.end_reason = info.get("end_reason")
            res.completed = res.end_reason in COMPLETE_REASONS
            if fault == "ws_drop":
                res.detail += f" reconnected={ws_dropped}"
        except Exception as exc:  # any unexpected client-side failure, e.g. the server went away
            res.detail += f" unexpected={type(exc).__name__}: {exc}"
            res.crashes += 1
        finally:
            with contextlib.suppress(Exception):
                if ws is not None:
                    ws.close()
            with contextlib.suppress(Exception):
                if server is not None:
                    server.stop()
            env.close()
    res.sane = audit is not None and audit.sane
    res.act_ms = list(server.acts_ms) if server is not None else []
    res.crash_heads = list(errs.hits)
    res.crashes += max(len(errs.hits), error_replies)
    return res


@dataclass
class Row:
    fault: str
    overlay: str
    crashes: int
    completed: int
    episodes: int
    median_hz: float
    p5_hz: float
    p99_ms: float
    sane: bool

    @property
    def passed(self):
        return (self.crashes == 0 and self.completed == self.episodes and self.sane and
                self.median_hz >= MIN_MEDIAN_HZ)


def _pct(values, q):
    return float(np.percentile(values, q)) if len(values) else math.nan


def summarise(results, overlays, faults):
    rows = []
    for fault in faults:
        for overlay in overlays:
            group = [r for r in results if r.fault == fault and r.overlay == overlay]
            if not group:
                continue
            ms = [m for r in group for m in r.act_ms if m > 0]
            hz = [1000.0 / m for m in ms]
            rows.append(Row(fault, overlay, sum(r.crashes for r in group), sum(r.completed for r in group),
                            len(group), _pct(hz, 50), _pct(hz, 5), _pct(ms, 99), all(r.sane for r in group)))
    return rows


def to_markdown(rows, time_limit, episodes):
    head = ["fault", "overlay", "crashes", "completed", "median Hz", "p5 Hz", "p99 act ms", "actions sane", "verdict"]
    lines = [f"Episodes per cell: {episodes}, virtual time limit {time_limit:g} s, mock_world, websocket + msgpack. "
             "Hz = 1 / server-side act() time. Pass: 0 crashes, all completed, median Hz >= 27, actions sane.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append(f"| {r.fault} | {r.overlay} | {r.crashes} | {r.completed}/{r.episodes} | {r.median_hz:.1f} | "
                     f"{r.p5_hz:.1f} | {r.p99_ms:.2f} | {'yes' if r.sane else 'NO'} | "
                     f"{'PASS' if r.passed else 'FAIL'} |")
    return "\n".join(lines) + "\n"


def run_drill(overlays=OVERLAYS, episodes=2, time_limit=10.0, faults=FAULTS):
    results = []
    for overlay in overlays:
        for fault in faults:
            for ep in range(episodes):
                r = run_episode(overlay, fault, ep, time_limit)
                results.append(r)
                print(f"[{overlay} {fault} ep{ep}] crashes={r.crashes} completed={r.completed} "
                      f"end={r.end_reason} acts={len(r.act_ms)} sane={r.sane}", flush=True)
    return results


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--overlay", action="append", help=f"overlay name (repeat); default {', '.join(OVERLAYS)}")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--time-limit", type=float, default=10.0, help="virtual seconds per episode")
    ap.add_argument("--fault", action="append", choices=FAULTS, help="subset of faults (repeat); default all")
    ap.add_argument("--report", help="also write the markdown table to this file")
    a = ap.parse_args(argv)
    overlays = a.overlay or list(OVERLAYS)
    faults = tuple(a.fault or FAULTS)
    results = run_drill(overlays, a.episodes, a.time_limit, faults)
    rows = summarise(results, overlays, faults)
    md = to_markdown(rows, a.time_limit, a.episodes)
    print("\n" + md)
    for r in results:
        if r.crashes:
            print(f"CRASH {r.overlay} {r.fault} ep{r.episode}: crashes={r.crashes}")
            for head in r.crash_heads[:3]:
                print("   " + head.replace("\n", "\n   "))
    if a.report:
        os.makedirs(os.path.dirname(os.path.abspath(a.report)), exist_ok=True)
        with open(a.report, "w") as f:
            f.write(md)
    return 0 if all(r.passed for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())
