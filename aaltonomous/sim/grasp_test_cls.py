"""Classical-planner yaw sweep: one triangle per episode, cells = (arm, phi | baseline).

Fixes vs the recipe (docs/grasp-sweep-recipe.md @117a4cd):
  F1 `--phis ''` is accepted (baseline-only cell); the recipe crashed on float('').
  F2 `client`/`env` could be unbound in `finally`; the LocalClient is now closed per cell (recipe leaked a pool per cell).
  F3 race `pol._job is not None and pol._job.get(...)` (act thread sets _job=None between the two reads): local var + deadline.
  F4 first-look detection (shape/apex_yaw/conf) is logged (`det0`, `apex_err_deg` vs the true apex mod 120); recipe left it as TODO.
  F5 `--dry-cells` prints cells/poses without importing the sim stack; one process = one set of cells -> sweep_run.sh is resumable.
"""
import argparse, json, os, tempfile, threading, time
from pathlib import Path
import numpy as np

HZ = 30.0


def poses(arm):
    """(pelvis_y, apex_heading_deg) list, x_pelvis fixed 0.30. Right arm = mirror image (y, A) -> (-y, -A)."""
    base = [(y, A) for A in (0, 90, 270) for y in (0.05, 0.10, 0.15, 0.20)] + [(0.20, 180)]   # 13 poses
    s = 1 if arm == "left" else -1
    return [(s * y, (s * A) % 360) for y, A in base]


def to_layout(arm, y_pel, A_deg):
    """pelvis (0.30, y, apex heading A) -> mat layout. V scene.py:73-77, layout.py:45-50: x_mat=0.45+y, y_mat=x-0.14, psi=-A."""
    return {"color": "red" if arm == "left" else "blue", "shape": "triangle",
            "x": round(0.45 + y_pel, 4), "y": round(0.30 - 0.14, 4), "yaw": float(-np.radians(A_deg))}


def wrap120(d_deg):                      # signed difference mod 120 deg
    return (d_deg + 60) % 120 - 60


def cells_for(phis, baseline):
    ph = [p for p in str(phis).split(",") if p.strip()]
    return ([("base", None)] if baseline else []) + [(f"phi{float(p):g}", float(p)) for p in ph]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--arms", default="left,right"); ap.add_argument("--phis", default="60,90,105,120,135")
    ap.add_argument("--baseline", action="store_true", help="add the yaw_align=false cell (before the phi cells)")
    ap.add_argument("--time-limit", type=float, default=45.0)
    ap.add_argument("--set", default=None, help="extra overrides json"); ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--max-poses", type=int, default=99); ap.add_argument("--out", default="runs/sweep.json")
    ap.add_argument("--dry-cells", action="store_true", help="print the cells/poses and exit (no sim)")
    a = ap.parse_args(argv)
    cells = cells_for(a.phis, a.baseline)
    if a.dry_cells:
        for arm in a.arms.split(","):
            print(arm, [c for c, _ in cells], poses(arm)[: a.max_poses])
        return
    import ssl  # noqa: F401  before DDS (see sim/evaluate.py)
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
    tmp, rows, stop = Path(tempfile.mkdtemp(prefix="gtc_")), [], threading.Event()
    extra = json.loads(a.set) if a.set else {}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    try:
        for arm in a.arms.split(","):
            for cell, phi in cells:
                ov = {"perception.classify_shape": True, "perception.triangle_pose": True,
                      "grasp.yaw_align": phi is not None, **extra}
                if phi is not None: ov["grasp.yaw_phi_deg"] = phi
                pol = ClassicalPolicy(config=load_config(None, ov, None), safety=env.filter.cfg, threaded=True,
                                      time_limit=a.time_limit, log=None)
                st = {"t_plan": [], "yaw_calls": 0, "yaw_none": 0, "det0": None}
                op = pol.planner.plan
                def timed(*x, _op=op, **k):
                    if st["det0"] is None and x:
                        d = x[0]
                        ay = getattr(d, "apex_yaw", None)
                        st["det0"] = [getattr(d, "shape", None), None if ay is None else round(float(np.degrees(ay)), 1),
                                      round(float(getattr(d, "apex_conf", 0.0)), 2)]
                    t = time.perf_counter(); r = _op(*x, **k); st["t_plan"].append(time.perf_counter() - t); return r
                pol.planner.plan = timed
                oy = pol.planner._plan_side_yaw
                def wy(*x, _oy=oy, **k):
                    st["yaw_calls"] += 1; r = _oy(*x, **k); st["yaw_none"] += r is None; return r
                pol.planner._plan_side_yaw = wy
                client = ev.LocalClient(pol)
                try:
                    for i, (y, A) in enumerate(poses(arm)[: a.max_poses]):
                        lay = [to_layout(arm, y, A)]
                        env_mod.sample_layout = lambda rng, _l=lay, **kw: [dict(o) for o in _l]
                        st.update(t_plan=[], yaw_calls=0, yaw_none=0, det0=None)
                        ev.home_arms(env, sc)
                        t0 = time.monotonic()
                        summ, _ = run_episode(env, client, i, i, Recorder(tmp / f"{arm}_{cell}_{i:02d}"), stop, "sort_fixed")
                        dl = time.monotonic() + 20.0
                        while time.monotonic() < dl:                     # F3: no race on pol._job
                            job = pol._job
                            if job is None or job.get("result") is not None:
                                break
                            time.sleep(0.05)
                        wall = time.monotonic() - t0
                        ev.settle(sc, 3.0, float(env.scene.cfg["success"]["at_rest_speed"]))
                        out = sc.evaluate("sort_fixed", env.mapping)["outcomes"][0]["outcome"]
                        p0 = pol.plans[0] if pol.plans else None
                        d0 = st["det0"]
                        err = None if not d0 or d0[1] is None else round(float(wrap120(d0[1] - A)), 1)   # F4
                        rows.append({"arm": arm, "cell": cell, "phi": phi, "i": i, "y": y, "A": A, "outcome": out,
                                     "end": summ.get("end_reason"), "wall_s": round(wall, 1), "sim_ticks": summ.get("ticks"),
                                     "plan_ok": bool(p0 and p0.ok), "plan_side": getattr(p0, "side", None),
                                     "plan_reason": getattr(p0, "reason", None), "plan_s": [round(t, 2) for t in st["t_plan"]],
                                     "yaw_calls": st["yaw_calls"], "yaw_none": st["yaw_none"],
                                     "yaw_used": bool(p0 and p0.ok and str(p0.reason).startswith("yaw-aligned")),
                                     "det0": d0, "apex_err_deg": err,
                                     "n_det": len(pol.last_detections), "events": pol.events[-6:]})
                        print(f"{arm:5s} {cell:7s} #{i:02d} y={y:+.2f} A={A:3d} -> {out:8s} side={rows[-1]['plan_side']} "
                              f"yaw_used={rows[-1]['yaw_used']} plan_s={rows[-1]['plan_s']} wall={wall:.0f}s", flush=True)
                        json.dump(rows, open(a.out, "w"), indent=1, default=str)      # incremental: survives a crash
                finally:
                    client.close()
    finally:
        env.close()


if __name__ == "__main__":
    main()
