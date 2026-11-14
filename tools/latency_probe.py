#!/usr/bin/env python3
"""Offline smoke + timing of a policy image: cold start, act() latency, command rate on rendered mock episodes.

    python tools/latency_probe.py --image hackjunction/policy:<tag> --port 8010 --gpus none --episodes 2

Starts the image the way the organisers do (docker run -d, no env/args/volumes; the port is bound to 127.0.0.1 only),
measures `docker run` -> websocket metadata received (cold start, limit 120 s), then drives full mock episodes
(classical mock_world renderer, realtime 30 Hz, no sim, no GPU) through the server and reports act() latency
p50/p95/max and the command rate (ticks returned / wall time, sim gate >= 27 Hz). Always stops the container.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--image", required=True)
    ap.add_argument("--port", type=int, default=8010)
    ap.add_argument("--gpus", default="none", help='docker --gpus value, "none" = no GPU flag')
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    if a.port == 8000:
        ap.error("port 8000 is the sim's policy port; pick another")

    from aaltonomous.baselines.classical.mock_world import make_mock_env
    from aaltonomous.deploy.runner import PolicyClient
    from aaltonomous.interface.tasks import FIXED_MAPPING

    name = f"latprobe-{int(time.time())}"
    cmd = ["docker", "run", "-d", "--name", name, "-p", f"127.0.0.1:{a.port}:8000"]
    if a.gpus.lower() != "none":
        cmd += ["--gpus", a.gpus]
    t0 = time.monotonic()
    subprocess.run(cmd + [a.image], check=True, capture_output=True, text=True)
    out = {"image": a.image, "gpus": a.gpus}
    try:
        client = PolicyClient(f"ws://127.0.0.1:{a.port}", startup_timeout=120.0)
        out["cold_start_s"] = round(time.monotonic() - t0, 2)
        out["metadata"] = client.metadata
        lat, eps = [], []
        for k in range(a.episodes):
            env = make_mock_env(seed=a.seed + k, realtime=True)
            obs = env.reset()
            client.reset("sort_fixed", obs["instruction"], a.seed + k)
            ticks, w0 = 0, time.monotonic()
            while True:
                t = time.monotonic()
                act = client.act(obs)
                lat.append(time.monotonic() - t)
                arm = np.asarray(act["arm_q_target"])
                ticks += arm.shape[0] if arm.ndim == 2 else 1
                obs, info = env.step(act)
                if info["episode_over"]:
                    break
            wall = time.monotonic() - w0
            eps.append({"seed": a.seed + k, "end": info["end_reason"], "wall_s": round(wall, 1),
                        "command_hz": round(ticks / wall, 1),
                        "clips": info["safety_summary"]["clipped_steps"],
                        "outcomes": env.backend.outcomes(FIXED_MAPPING)})  # mock = sort_fixed
        client.close()
        ms = 1e3 * np.asarray(lat)
        out.update(n_acts=len(lat), act_ms_p50=round(float(np.percentile(ms, 50)), 2),
                   act_ms_p95=round(float(np.percentile(ms, 95)), 2), act_ms_max=round(float(ms.max()), 2),
                   episodes=eps)
    finally:
        logs = subprocess.run(["docker", "logs", "--tail", "400", name], capture_output=True, text=True)
        out["container_log_flags"] = [ln for ln in (logs.stdout + logs.stderr).splitlines()
                                      if "flags on" in ln or "loaded in" in ln]
        subprocess.run(["docker", "rm", "-f", name], capture_output=True)
    print(json.dumps(out, indent=2, default=str))
    ok = out.get("cold_start_s", 1e9) <= 120 and all(e["command_hz"] >= 27 and e["clips"] == 0
                                                     for e in out.get("episodes", [{"command_hz": 0, "clips": 1}]))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
