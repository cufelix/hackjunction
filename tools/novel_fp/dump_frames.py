"""Dump head RGB-D frames + joint states from stock sim episodes, for offline novel-object false-positive checks.

    cd <stock sim tree> && bash aaltonomous/cloud/aalto.sh py <this> --policy ws://localhost:8005 --out <dir> \
        [--eps ep0f,ep0l,...,ep7l] [--seconds 20] [--every 15]

Same episodes as the X8/A95 matrix: ep<k>f = sort_fixed seed k, ep<k>l = sort_language seed k (evaluator seeding:
env.rng = default_rng(seed), home_arms first). The policy drives the robot normally so the arms are where the real
first looks see them; every `--every` ticks the head RGB-D + arm_q/hand_q are kept. Episodes are cut at --seconds
(the first looks happen in the first ~10 s). Output: <out>/<ep>.npz (rgb uint8 [N,H,W,3], depth [N,H,W], arm_q,
hand_q, t) + <ep>.json (layout, instruction).
"""
from __future__ import annotations

import argparse
import json
import threading
from pathlib import Path

import numpy as np


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eps", default=",".join(f"ep{k}{t}" for k in range(8) for t in "fl"))
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--every", type=int, default=15)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)

    import os
    from aaltonomous.deploy.runner import PolicyClient, Recorder, run_episode
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.evaluate import apply_instance_env, home_arms
    from aaltonomous.sim.scene_client import SceneClient

    apply_instance_env()
    sc = SceneClient(a.host)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    client = PolicyClient(a.policy, 180.0)

    class FrameRecorder(Recorder):
        def __init__(self, path):
            super().__init__(path)
            self.n, self.keep = 0, {"rgb": [], "depth": [], "arm_q": [], "hand_q": [], "t": []}

        def add(self, row, obs):
            self.rows.append(row)
            if self.n % a.every == 0 and obs.get("head_rgb") is not None and obs.get("head_depth") is not None:
                for k, src in (("rgb", "head_rgb"), ("depth", "head_depth"), ("arm_q", "arm_q"), ("hand_q", "hand_q"),
                               ("t", "t")):
                    self.keep[k].append(np.asarray(obs[src]).copy())
            self.n += 1

    for ep in (e for e in a.eps.split(",") if e):
        k, task = int(ep[2]), {"f": "sort_fixed", "l": "sort_language"}[ep[3]]
        env = AaltoEnv(backend="sim", task=task, seed=k, time_limit=a.seconds, realtime=True, img_server_ip=a.host,
                       head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")), cameras=True)
        try:
            home_arms(env, sc)
            env.rng = np.random.default_rng(k)
            rec = FrameRecorder(out / ep)
            try:
                summary, _ = run_episode(env, client, 0, k, rec, threading.Event(), task)
            except Exception as e:  # a cut episode is fine: we only want the frames
                summary = {"end_reason": repr(e)}
            np.savez_compressed(out / f"{ep}.npz", **{kk: np.stack(v) for kk, v in rec.keep.items() if v})
            (out / f"{ep}.json").write_text(json.dumps({"ep": ep, "task": task, "seed": k, "layout": env.layout,
                                                        "instruction": env.instruction,
                                                        "end": summary.get("end_reason"),
                                                        "frames": len(rec.keep["t"])}, default=str))
            print(f"{ep}: {len(rec.keep['t'])} frames, layout {len(env.layout)} objects, end={summary.get('end_reason')}",
                  flush=True)
        finally:
            env.close()
    client.close()


if __name__ == "__main__":
    main()
