"""Novel-object evaluation runner (OURS, not the official evaluator). Reuses aaltonomous.sim.evaluate as a library
(home_arms, settle, gate_ok, run_episode, ...) against a sim started from the novel tree (make_novel_tree.sh).

  aalto.sh py novel_eval.py --policy ws://localhost:8301 --arm A15 --scenes S1_ball_left,S2_mug_right --seeds 0,1,2 \
        --out /data/work/novel/runs/A15 [--container-name NAME] [--frames-dir DIR] [--time-limit 120]

Per episode: our layout + our instruction + our ground-truth bins (scenes.py), scored per object as
  correct  at rest in its GT bin (or a GT-None distractor still on the counter)
  wrong    in a wrong bin (or a GT-None distractor moved into a bin)
  left     GT bin given but object still on the counter / not at rest
  off      knocked off the counter
Output: <out>/<scene>_s<seed>.json (+ policy log slice) and results.jsonl.
"""
from __future__ import annotations

import argparse
import json
import os
import ssl  # noqa: F401  (before any DDS import)
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
REPO = Path(__file__).resolve().parents[3] if (HERE.parent.parent / "aaltonomous").exists() else None


def classify(sc, state, gt, rest_speed):
    """state: scene-service objects. -> list of per-object dicts."""
    out = []
    for o in state:
        if not o["active"]:
            continue
        want = gt.get(o["name"], "MISSING")
        p = sc.pelvis_to_mat(np.asarray(o["pos"], float))
        if not sc.on_counter(p):
            res, where = "off", "off"
        else:
            b = sc.bin_of(p) if o["speed"] <= rest_speed else None
            where = b or "counter"
            if want is None:
                res = "correct" if b is None else "wrong"
            elif b == want:
                res = "correct"
            elif b is None:
                res = "left"
            else:
                res = "wrong"
        out.append({"name": o["name"], "gt": want, "where": where, "outcome": res, "pos": o["pos"], "speed": o["speed"]})
    return out


def docker_lines(name):
    r = subprocess.run(["docker", "logs", name], capture_output=True, text=True)
    return (r.stdout + r.stderr).splitlines()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--policy", required=True)
    ap.add_argument("--arm", required=True)
    ap.add_argument("--scenes", default="all")
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--out", required=True)
    ap.add_argument("--container-name", default=None, help="docker container of the policy (for per-episode log slices)")
    ap.add_argument("--frames-dir", default=None, help="record demo-camera frames of every episode here")
    ap.add_argument("--time-limit", type=float, default=120.0)
    ap.add_argument("--settle", type=float, default=3.0)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--gate-min-hz", type=float, default=27.0)
    a = ap.parse_args()

    from aaltonomous.deploy.runner import PolicyClient, Recorder, run_episode
    from aaltonomous.interface import env as E
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.evaluate import apply_instance_env, gate_ok, home_arms, settle
    from aaltonomous.sim.scene_client import SceneClient
    from scenes import SCENES, build_layout

    apply_instance_env()
    scene_ids = list(SCENES) if a.scenes == "all" else a.scenes.split(",")
    seeds = [int(s) for s in a.seeds.split(",")]
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    sc = SceneClient(a.host)
    env = AaltoEnv(backend="sim", task="sort_language", seed=0, time_limit=a.time_limit, realtime=True,
                   img_server_ip=a.host, head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")),
                   cameras=True)
    rest = float(env.scene.cfg["success"]["at_rest_speed"])
    client = PolicyClient(a.policy, 120.0)
    stop = threading.Event()
    print("policy:", client.metadata, flush=True)
    orig_reset_scene = env.backend.reset_scene
    cur = {}

    def reset_scene(layout, instruction):
        orig_reset_scene(layout, instruction)
        # no waiting here: frames queue up on the rgbd socket while the client is idle and the policy would see an old one
    env.backend.reset_scene = reset_scene
    log_pos = len(docker_lines(a.container_name)) if a.container_name else 0
    try:
        for sid in scene_ids:
            for seed in seeds:
                layout, gt = build_layout(sid, seed)
                instr = SCENES[sid]["instruction"]
                E.make_instruction = lambda task, rng, _i=instr: (_i, None)
                E.sample_layout = lambda rng, _l=layout, **kw: _l
                home_err = home_arms(env, sc)
                env.rng = np.random.default_rng(seed)
                sc.reset(layout)                    # pre-place + settle: objects moved by the scene service show up in the
                time.sleep(4.0)                     # rendered frames only after a few sim steps; env.reset re-places the same layout
                fdir = None
                if a.frames_dir:
                    fdir = f"{a.frames_dir}/{a.arm}_{sid}_s{seed}"
                    log_pos0 = None
                rec = Recorder(out / f"{sid}_s{seed}_ep", images=bool(os.environ.get("NOVEL_DUMP")))
                t0 = time.monotonic()
                try:
                    if fdir:
                        sc.call(cmd="rec_start", dir=fdir)
                    summary, events = run_episode(env, client, 0, seed, rec, stop, "sort_language")
                except Exception as e:
                    traceback.print_exc()
                    summary, events = {"end_reason": "crash", "error": repr(e), "safety": env.filter.summary()}, []
                if os.environ.get("NOVEL_DUMP") and rec.imgs:
                    import cv2
                    for k in (0, len(rec.imgs) // 2, len(rec.imgs) - 1):
                        cv2.imwrite(str(out / f"{sid}_s{seed}_head{k}.jpg"), rec.imgs[k]["head_rgb"][..., ::-1])
                        np.save(out / f"{sid}_s{seed}_depth{k}.npy", rec.imgs[k]["head_depth"])
                rtf = summary.get("rtf") or 1.0
                settle(sc, a.settle / max(min(rtf, 1.0), 0.2), rest)
                nfr = None
                if fdir:
                    nfr = sc.call(cmd="rec_stop").get("frames")
                st = sc.state()["objects"]
                per = classify(env.scene, st, gt, rest)
                tgt = [p for p in per if p["gt"] is not None]
                dis = [p for p in per if p["gt"] is None]
                trial = {
                    "arm": a.arm, "scene": sid, "seed": seed, "instruction": instr, "layout": layout, "gt": gt,
                    "objects": per,
                    "target": {k: sum(p["outcome"] == k for p in tgt) for k in ("correct", "wrong", "left", "off")},
                    "distractors": {"kept": sum(p["outcome"] == "correct" for p in dis),
                                    "disturbed": sum(p["outcome"] != "correct" for p in dis)},
                    "n_target": len(tgt), "n_distractor": len(dis),
                    "all_correct": all(p["outcome"] == "correct" for p in per),
                    "end_reason": summary.get("end_reason"), "sim_s": summary.get("wall_s"),
                    "episode_wall_s": round(time.monotonic() - t0, 1), "command_hz": summary.get("command_hz"),
                    "policy_hz_rt": summary.get("policy_hz_rt"), "rtf": summary.get("rtf"),
                    "clipped_steps": (summary.get("safety") or {}).get("clipped_steps"),
                    "frames": nfr, "frames_dir": fdir, "home_err": round(home_err, 3),
                }
                trial["gate_ok"], trial["gate_why"] = gate_ok({**summary, "end_reason": summary.get("end_reason", "")},
                                                              0, a.gate_min_hz)
                if a.container_name:
                    lines = docker_lines(a.container_name)
                    (out / f"{sid}_s{seed}_policy.log").write_text("\n".join(lines[log_pos:]))
                    log_pos = len(lines)
                for f in out.glob(f"{sid}_s{seed}_ep.npz"):
                    f.unlink()                      # per-tick arrays are large and not needed here
                (out / f"{sid}_s{seed}.json").write_text(json.dumps(trial, indent=1, default=str))
                with open(out / "results.jsonl", "a") as fh:
                    fh.write(json.dumps(trial, default=str) + "\n")
                print(f"{a.arm} {sid} s{seed}: target {trial['target']} distractors {trial['distractors']} "
                      f"end={trial['end_reason']} sim_s={trial['sim_s']} wall={trial['episode_wall_s']}s "
                      f"hz={trial['command_hz']} rt_hz={trial['policy_hz_rt']} clips={trial['clipped_steps']} "
                      f"rtf={trial['rtf']}", flush=True)
    finally:
        client.close()
        env.close()


if __name__ == "__main__":
    main()
