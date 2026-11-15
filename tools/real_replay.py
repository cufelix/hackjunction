#!/usr/bin/env python3
"""Acceptance test: replay recorded REAL episodes (LeRobot drop_3 frames + joint states) through a policy IMAGE over the
websocket, exactly as the organiser runs it (docker run -d --gpus all -p PORT:8000 IMAGE, no env, no args).

  drive   (needs docker, websockets, msgpack, pyarrow, cv2; run with the env that has pyarrow):
    python tools/real_replay.py drive IMAGE --data /data/work/drop3 --eps 0-11 --port 8261 --out OUT_DIR [--ticks 150]
  analyze (inside any policy image with pinocchio, tree mounted):
    python tools/real_replay.py analyze OUT_DIR

Frames: the extracted sample (frames 0-4, then every 10th); a tick uses the latest sampled frame <= tick (frozen between
samples). Joint states: observation.state[0:14] arm, [14:28] hand (as recorded). Checks per episode: first non-empty
action time since `docker run`, NaN/shape, joint limits, '[profile]' lines, pitch-cal + detection log lines, and that the
first second of commanded motion does not move the hand down.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

INSTR = "Sort the objects: red into left, yellow into center, blue into right."
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_states(data):
    import pyarrow.parquet as pq
    t = pq.read_table(f"{data}/data/chunk-000/file-000.parquet", columns=["observation.state", "episode_index", "frame_index"])
    st = np.asarray(t["observation.state"].to_pylist(), np.float32)
    return st, np.asarray(t["episode_index"].to_pylist()), np.asarray(t["frame_index"].to_pylist())


def frame_for(data, ep, tick):
    import cv2
    fr = Path(data) / "frames"
    k = tick if tick < 5 else (tick // 10) * 10
    while k > 0 and not (fr / f"e{ep:02d}_{k:04d}.png").exists():
        k -= 1
    rgb = cv2.cvtColor(cv2.imread(str(fr / f"e{ep:02d}_{k:04d}.png")), cv2.COLOR_BGR2RGB)
    depth = np.load(fr / f"e{ep:02d}_{k:04d}.npy").astype(np.uint16)
    return rgb, depth


def drive(a):
    from aaltonomous.deploy.runner import Container, PolicyClient
    from aaltonomous.interface.safety import JointSpec
    spec = JointSpec()
    st, epi, fi = load_states(a.data)
    lo, hi = a.eps.split("-") if "-" in a.eps else (a.eps, a.eps)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    for ep in range(int(lo), int(hi) + 1):
        m = epi == ep
        S = st[m][np.argsort(fi[m])]
        cm = Container(a.image, a.port, gpus="all")
        t0 = time.monotonic()
        cm.prepare().__enter__()
        try:
            cl = PolicyClient(f"ws://localhost:{a.port}", 180.0, alive=cm.alive)
            t_conn = time.monotonic() - t0
            cl.reset("sort_fixed", INSTR, ep)
            acts, t_first, errs, lat = [], None, [], []
            tick0 = time.monotonic()
            for tick in range(min(a.ticks, len(S))):
                rgb, depth = frame_for(a.data, ep, tick)
                s = S[tick]
                obs = {"head_rgb": rgb, "head_depth": depth, "left_wrist_rgb": np.zeros((480, 640, 3), np.uint8),
                       "right_wrist_rgb": np.zeros((480, 640, 3), np.uint8), "arm_q": s[:14].astype(np.float32),
                       "arm_dq": np.zeros(14, np.float32), "hand_q": s[14:28].astype(np.float32),
                       "instruction": INSTR, "t": tick / 30.0}
                while time.monotonic() - tick0 < tick / 30.0:
                    time.sleep(0.001)
                ts = time.monotonic()
                try:
                    r = cl.act(obs)
                except Exception as e:
                    errs.append(f"tick {tick}: {e}")
                    continue
                lat.append(time.monotonic() - ts)
                arm = np.asarray(r["arm_q_target"], float).reshape(-1, 14)
                hand = np.asarray(r["hand_q_target"], float).reshape(-1, 14)
                if not (np.isfinite(arm).all() and np.isfinite(hand).all()):
                    errs.append(f"tick {tick}: NaN/inf")
                if len(arm) and t_first is None:
                    t_first = time.monotonic() - t0
                acts.append((tick, arm, hand))
            cl.close()
            logs = cm.logs(tail=4000) if hasattr(cm, "logs") else ""
        finally:
            cm.__exit__()
        A = np.vstack([x[1] for x in acts if len(x[1])]) if any(len(x[1]) for x in acts) else np.zeros((0, 14))
        H = np.vstack([x[2] for x in acts if len(x[2])]) if len(A) else np.zeros((0, 14))
        viol = int(((A < spec.arm_lower - 1e-6) | (A > spec.arm_upper + 1e-6)).any(axis=1).sum()) if len(A) else 0
        hviol = int(((H < spec.hand_lower - 1e-6) | (H > spec.hand_upper + 1e-6)).any(axis=1).sum()) if len(H) else 0
        (out / f"e{ep:02d}.log").write_text(logs)
        pick = lambda *keys: [l.strip()[:200] for l in logs.splitlines() if any(k in l for k in keys)]
        rec = {"ep": ep, "connect_s": round(t_conn, 2), "first_action_s": None if t_first is None else round(t_first, 2),
               "first_action_after_connect_s": None if t_first is None else round(t_first - t_conn, 2),
               "p50_ms": round(1e3 * float(np.median(lat)), 1) if lat else None, "errors": errs[:3], "n_err": len(errs),
               "arm_limit_viol": viol, "hand_limit_viol": hviol, "rows": int(len(A)), "start_arm_q": S[0, :14].tolist(),
               "profile": pick("[profile]")[:3], "pitch": pick("pitch", "online_pitch")[:4],
               "detect": pick("detections", "execute ", "[retry", "skip")[:4], "exceptions": pick("Traceback", "Exception")[:3]}
        np.savez(out / f"e{ep:02d}.npz", arm=A, hand=H, start_arm=S[0, :14])
        (out / f"e{ep:02d}.json").write_text(json.dumps(rec, indent=1))
        print(json.dumps({k: rec[k] for k in ("ep", "first_action_s", "first_action_after_connect_s", "n_err", "arm_limit_viol",
                                               "hand_limit_viol", "rows")}), flush=True)
        rows.append(rec)


def analyze(a):
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.kinematics import ArmKinematics
    kin = ArmKinematics(load_config(profile="none"))
    for f in sorted(Path(a.out).glob("e*.npz")):
        d = np.load(f)
        A, q0 = d["arm"], d["start_arm"]
        if len(A) < 31:
            print(f.stem, "too few action rows", len(A))
            continue
        res = []
        for side in ("left", "right"):
            z0 = float(kin.grasp_pose(side, q0)[0][2])
            z = [float(kin.grasp_pose(side, A[i])[0][2]) for i in (0, 15, 30)]
            res.append((side, round(z0, 3), [round(x, 3) for x in z]))
        down = any(min(zz) < z0 - 0.01 for _, z0, zz in res)
        print(f.stem, "hand z (start, @0/0.5/1.0 s of motion)", res, "DOWN" if down else "ok")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("drive"); d.add_argument("image"); d.add_argument("--data", default="/data/work/drop3")
    d.add_argument("--eps", default="0-11"); d.add_argument("--port", type=int, default=8261)
    d.add_argument("--out", required=True); d.add_argument("--ticks", type=int, default=150)
    n = sp.add_parser("analyze"); n.add_argument("out")
    a = ap.parse_args()
    if a.cmd == "analyze":
        a.out = a.out
    drive(a) if a.cmd == "drive" else analyze(argparse.Namespace(out=a.out))
