"""Mirrored single-object grasp test for the classical policy: one object per episode, LEFT arm at pelvis (0.30, +y,
heading A) vs RIGHT arm; yellow object -> centre bin -> the arm on the object's side at the mirror pose (0.30, -y, -A). Prints outcome + held= per case.

    PYTHONPATH=$REPO python tools/lefthand_grasp_test.py --shapes cube --ys 0.10,0.18 --yaws 0,20,45,70 \
        --set '{"grasp.left_hand_fix": true}' --out runs/lh.json
(env `isaac`, sim running; same plumbing as aaltonomous/sim/grasp_test_cls.py)
"""
import argparse, json, os, re, tempfile, threading, time
from pathlib import Path
import numpy as np


def to_layout(arm, shape, y_pel, A_deg):
    return {"color": "yellow", "shape": shape, "x": round(0.45 + y_pel, 4),
            "y": round(0.30 - 0.14, 4), "yaw": float(-np.radians(A_deg))}


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="left,right"); ap.add_argument("--shapes", default="cube")
    ap.add_argument("--ys", default="0.10,0.18"); ap.add_argument("--yaws", default="0,20,45,70")
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--overlay", default="finals_r21"); ap.add_argument("--set", default=None)
    ap.add_argument("--time-limit", type=float, default=45.0); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--out", default="runs/lh.json")
    a = ap.parse_args(argv)
    import ssl  # noqa: F401
    from aaltonomous.sim import evaluate as ev
    from aaltonomous.deploy.runner import Recorder, run_episode
    from aaltonomous.interface import env as env_mod
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    ev.apply_instance_env()
    sc = SceneClient(a.host)
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=a.time_limit, realtime=True, img_server_ip=a.host,
                   head_rgbd=True, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    tmp, rows, stop = Path(tempfile.mkdtemp(prefix="lhg_")), [], threading.Event()
    ov = {"perception.hue_guard.enabled": False, "policy.grasp_diag_log": True, **(json.loads(a.set) if a.set else {})}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    i = 0
    try:
        for shape in a.shapes.split(","):
            for y in [float(v) for v in a.ys.split(",")]:
                for A in [float(v) for v in a.yaws.split(",")]:
                    for rep in range(a.reps):
                        for arm in a.arms.split(","):
                            s = 1 if arm == "left" else -1
                            lines = []
                            pol = ClassicalPolicy(config=load_config(profile="sim", overlay=a.overlay, overrides=ov),
                                                  safety=env.filter.cfg, threaded=True, time_limit=a.time_limit,
                                                  log=lines.append)
                            client = ev.LocalClient(pol)
                            try:
                                lay = [to_layout(arm, shape, s * y, s * A)]
                                env_mod.sample_layout = lambda rng, _l=lay, **kw: [dict(o) for o in _l]
                                ev.home_arms(env, sc)
                                t0 = time.monotonic()
                                summ, _ = run_episode(env, client, i, i, Recorder(tmp / f"{i:03d}"), stop, "sort_fixed")
                                dl = time.monotonic() + 20.0
                                while time.monotonic() < dl:
                                    job = pol._job
                                    if job is None or job.get("result") is not None:
                                        break
                                    time.sleep(0.05)
                                ev.settle(sc, 3.0, float(env.scene.cfg["success"]["at_rest_speed"]))
                                ev_ = sc.evaluate("sort_fixed", env.mapping)
                                out = ev_["outcomes"][0]["outcome"]
                                try:
                                    st_ = str(sc.state())[:400]
                                except Exception as e_:
                                    st_ = repr(e_)
                            finally:
                                client.close()
                            if not pol.plans:
                                print('POLICY LOG:', [str(l)[:220] for l in lines[-8:]], flush=True)
                            held = [m.group(1) for l in lines for m in [re.search(r"held=(yes|no)", str(l))] if m]
                            p0 = pol.plans[0] if pol.plans else None
                            rows.append({"arm": arm, "shape": shape, "y": s * y, "A": s * A, "outcome": out, "held": held,
                                         "plan_ok": bool(p0 and p0.ok), "side": getattr(p0, "side", None),
                                         "wall_s": round(time.monotonic() - t0, 1), "state": st_, "log": [str(l)[:300] for l in lines if "safety clip" not in str(l)]})
                            if out != "correct": print("LOG:", *[str(l)[:260] for l in lines[-6:]], "STATE " + st_, sep="\n  ", flush=True)
                            print(f"{arm:5s} {shape:8s} y={s*y:+.2f} A={s*A:+4.0f} -> {out:8s} held={held} side={rows[-1]['side']}", flush=True)
                            json.dump(rows, open(a.out, "w"), indent=1, default=str)
                            i += 1
    finally:
        env.close()
    for arm in ("left", "right"):
        r = [x for x in rows if x["arm"] == arm]
        print(arm, "correct", sum(x["outcome"] == "correct" for x in r), "/", len(r),
              "held=no", sum(h == "no" for x in r for h in x["held"][:1]))


if __name__ == "__main__":
    main()
