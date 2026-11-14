#!/usr/bin/env python3
"""One reset + action websocket smoke, executed inside a network-none policy container."""
from __future__ import annotations

import json
import sys
import time

import numpy as np
from websockets.sync.client import connect

from aaltonomous.deploy import protocol as P


def validate_action(action):
    if not isinstance(action, dict):
        raise ValueError("action is not a dictionary")
    ticks = None
    for key in ("arm_q_target", "hand_q_target"):
        value = np.asarray(action.get(key), dtype=float)
        if value.ndim not in (1, 2) or value.shape[-1] != 14 or not np.isfinite(value).all():
            raise ValueError(f"{key} must be finite with shape (14,) or (k,14); got {value.shape}")
        count = value.shape[0] if value.ndim == 2 else 1
        if ticks is not None and count != ticks:
            raise ValueError("arm and hand action lengths differ")
        ticks = count
    return ticks


def rpc(ws, message):
    ws.send(P.pack(message))
    reply = P.unpack(ws.recv(timeout=120.0))
    if reply.get("type") == "error":
        raise RuntimeError(reply.get("message", "policy error"))
    return reply


def main():
    with connect("ws://127.0.0.1:8000", max_size=None, compression=None, open_timeout=5) as ws:
        metadata = P.check_metadata(P.unpack(ws.recv(timeout=5.0)))
        reset = rpc(ws, {"type": "reset", "task": "sort_fixed", "instruction": "", "seed": 0})
        if reset.get("type") != "reset_ok":
            raise RuntimeError(f"unexpected reset response: {reset!r}")
        obs = {
            "head_rgb": np.zeros((480, 640, 3), np.uint8),
            "head_depth": np.zeros((480, 640), np.uint16),
            "left_wrist_rgb": np.zeros((480, 640, 3), np.uint8),
            "right_wrist_rgb": np.zeros((480, 640, 3), np.uint8),
            "arm_q": np.zeros(14, np.float32),
            "arm_dq": np.zeros(14, np.float32),
            "hand_q": np.zeros(14, np.float32),
            "instruction": "",
            "t": 0.0,
        }
        started = time.monotonic()
        reply = rpc(ws, {"type": "act", "obs": obs})
        elapsed = time.monotonic() - started
        if reply.get("type") != "action":
            raise RuntimeError(f"unexpected action response: {reply!r}")
        ticks = validate_action(reply.get("action"))
        rate = ticks / max(elapsed, 1e-9)
        result = {
            "metadata": metadata,
            "first_action_latency_ms": round(1000.0 * elapsed, 3),
            "first_action_ticks": ticks,
            "policy_hz_wall": round(rate, 3),
            "min_policy_hz": 27.0,
            "pass": rate >= 27.0,
        }
        print(json.dumps(result, sort_keys=True))
        return 0 if result["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
