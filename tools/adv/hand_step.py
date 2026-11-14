#!/usr/bin/env python3
"""hand_step.py: Dex3 finger step response in the (adv) sim, to fit the real_soft profile to the real hand.

    source tools/adv/adv_env.sh; bash tools/adv/set_profile.sh real_soft
    cd $ADV_TREE && bash aaltonomous/cloud/aalto.sh py <tools>/hand_step.py [out.json] [--seed 1000] [--reps 3]

Run it on the adv sim when it is idle (no eval running on that instance). It applies the active profile (a scene
`randomize`, which sets hand_kp / hand_kd), homes the arms, then for each rep: holds `hand.open`, steps to
`hand.closed` (classical config, right hand mirrored) for 1.5 s, steps back for 1.5 s. Per joint with a step
> 0.2 rad it measures t10 / t63 / t90 (s after the command) and the final error at the end of the hold (= the
empty-close baseline: how far a soft finger stays from its target with nothing in the hand).

Target (Timi, real Dex3 at kp 1.5 / kd 0.2): 300-380 ms finger lag. Read `close.t90_median_s`; if it is outside,
change hand_kp / hand_kd in profiles/adv_real_soft.yaml (hand_kp lower = slower) and rerun. Compare with nominal.
Times are SIM seconds at the 30 Hz command rate (env.dt), independent of the box's real-time factor.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import yaml


def step_metrics(t, q, q0, q1, idx):
    """t (T,), q (T, N) after the command at t=0, q0 -> q1 commanded. -> per-joint dicts."""
    out = []
    for j in idx:
        d = q1[j] - q0[j]
        frac = (q[:, j] - q0[j]) / d
        ts = {}
        for name, f in (("t10", 0.1), ("t63", 0.632), ("t90", 0.9)):
            hit = np.nonzero(frac >= f)[0]
            ts[name] = float(t[hit[0]]) if len(hit) else None
        out.append({"joint": int(j), "step_rad": round(float(d), 3), **ts,
                    "final_err_rad": round(float(q1[j] - q[-1, j]), 4), "final_frac": round(float(frac[-1]), 3)})
    return out


def summarize(rows):
    s = {}
    for k in ("t10", "t63", "t90"):
        v = [r[k] for r in rows if r[k] is not None]
        s[f"{k}_median_s"] = round(float(np.median(v)), 3) if v else None
        s[f"{k}_missing"] = sum(r[k] is None for r in rows)
    s["final_err_median_rad"] = round(float(np.median([abs(r["final_err_rad"]) for r in rows])), 4) if rows else None
    return s


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("out", nargs="?", default="hand_step.json")
    ap.add_argument("--seed", type=int, default=1000)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--hold", type=float, default=1.5)
    ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args()

    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.sim.evaluate import apply_instance_env, home_arms
    from aaltonomous.sim.scene_client import SceneClient
    import os

    apply_instance_env()
    sc = SceneClient(a.host)
    params = sc.call(cmd="randomize", seed=a.seed).get("params") or {}
    applied = (params.get("applied") or {}).get("hand_gains")
    print(f"profile {params.get('name')}  hand gains applied: {applied}")
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=a.seed, time_limit=600, realtime=True, img_server_ip=a.host,
                   head_rgbd=False, head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")), cameras=False)
    try:
        home_arms(env, sc)
        spec = env.filter.spec
        cfg = yaml.safe_load((Path(__file__).resolve().parents[2] / "aaltonomous/baselines/classical/config.yaml").read_text())
        H = cfg["hand"]
        lo, hi = np.asarray(spec.hand_lower) + 0.03, np.asarray(spec.hand_upper) - 0.03
        mk = lambda left: np.clip(np.concatenate([np.asarray(left, float), -np.asarray(left, float)]), lo, hi)
        q_open, q_closed = mk(H["open"]), mk(H["closed"])
        arm = env.filter.arm_cmd.copy()
        n_hold = int(round(a.hold / env.dt))
        idx = [j for j in range(len(q_open)) if abs(q_closed[j] - q_open[j]) > 0.2]

        def run(target, n):
            qs = []
            for _ in range(n):
                env.step({"arm_q_target": arm, "hand_q_target": target})
                qs.append(env.backend.joints()[2].copy())
            return np.asarray(qs)

        close, opening = [], []
        run(q_open, n_hold)
        for _ in range(a.reps):
            t = (np.arange(n_hold) + 1) * env.dt
            close += step_metrics(t, run(q_closed, n_hold), q_open, q_closed, idx)
            opening += step_metrics(t, run(q_open, n_hold), q_closed, q_open, idx)
    finally:
        env.close()
    res = {"profile": params.get("name"), "seed": a.seed, "hand_gains": applied, "dt": env.dt,
           "close": summarize(close), "open": summarize(opening), "close_joints": close, "open_joints": opening,
           "target": "real Dex3 lag 300-380 ms (t90 of the close step)"}
    Path(a.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("profile", "hand_gains", "close", "open")}, indent=1))
    print(f"-> {a.out}")


if __name__ == "__main__":
    sys.exit(main())
