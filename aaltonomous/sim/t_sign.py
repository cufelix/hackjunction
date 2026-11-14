"""Sign-convention check (Felix): sim mesh yaw -> perception apex_yaw -> planner finger heading, 4 single-prism cases.

    python -m aaltonomous.sim.t_sign [--out runs/t_sign.json]        # env isaac, ONE sim running, nothing else using it

Fixes vs the recipe: dets[0] IndexError guarded; plan reason parsed and PASS/FAIL per criterion + overall verdict;
yaw_hint_angles() is compared with the plan's heading; results go to json; env/pol closed.
Criteria: (1) sim apex heading == -psi (3 deg); (2) apex_yaw == -psi mod 120 (10 deg), conf >= 0.5;
(3) plan.reason = 'yaw-aligned (X deg, err E)', X within 3 deg of the table value mod 360, E <= 12.
"""
import argparse, json, re, time
import numpy as np

CASES = [(0, 0.10, "left"), (-90, 0.10, "left"), (-20, 0.10, "left"), (0, -0.10, "right")]
w = lambda d: (d + 180) % 360 - 180          # noqa: E731
m120 = lambda d: (d + 60) % 120 - 60         # noqa: E731


def expected_heading(apex_deg, arm, phi=105.0, ref=25.0, max_dev=75.0):
    """Finger heading (deg) the planner should pick: best of apex + s*phi + k120 nearest to -s*ref (hand-computed rule)."""
    s = 1.0 if arm == "left" else -1.0
    c = sorted(((abs(w(apex_deg + s * phi + 120 * k + s * ref)), w(apex_deg + s * phi + 120 * k)) for k in range(3)))
    return [h for d, h in c if d <= max_dev]


def main(argv=None):
    ap = argparse.ArgumentParser(); ap.add_argument("--out", default="runs/t_sign.json"); ap.add_argument("--host", default="127.0.0.1")
    a = ap.parse_args(argv)
    import os, ssl  # noqa: F401
    from pathlib import Path
    from aaltonomous.sim.scene_client import SceneClient
    from aaltonomous.sim import evaluate as ev
    from aaltonomous.interface import env as env_mod
    from aaltonomous.interface.env import AaltoEnv
    from aaltonomous.baselines.classical.config import load_config
    from aaltonomous.baselines.classical.policy import ClassicalPolicy
    ev.apply_instance_env(); sc = SceneClient(a.host)
    env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=60, realtime=True, img_server_ip=a.host, head_rgbd=True,
                   head_rgbd_port=int(os.environ.get("AALTO_RGBD_PORT", "55560")))
    ov = {"perception.classify_shape": True, "perception.triangle_pose": True, "grasp.yaw_align": True, "grasp.yaw_phi_deg": 105}
    pol = ClassicalPolicy(config=load_config(None, ov, None), safety=env.filter.cfg, time_limit=60, log=None)
    res = []
    try:
        for psi_deg, y_pel, arm in CASES:
            lay = [{"color": "red" if arm == "left" else "blue", "shape": "triangle", "x": 0.45 + y_pel, "y": 0.16, "yaw": float(np.radians(psi_deg))}]
            env_mod.sample_layout = lambda rng, _l=lay, **kw: [dict(o) for o in _l]
            ev.home_arms(env, sc); obs = env.reset(); time.sleep(1.5); obs = env._obs()
            r = {"psi": psi_deg, "y": y_pel, "arm": arm}
            q = next(o for o in sc.state()["objects"] if o["active"])["quat_xyzw"]
            t = float(np.degrees(np.arctan2(2 * (q[3] * q[2] + q[0] * q[1]), 1 - 2 * (q[1] ** 2 + q[2] ** 2))))
            r.update(mesh_yaw=t, exp_mesh_yaw=w(-90 - psi_deg), apex_heading=w(t + 90), exp_apex=w(-psi_deg))
            r["c1"] = abs(w(r["mesh_yaw"] - r["exp_mesh_yaw"])) <= 3 and abs(w(r["apex_heading"] - r["exp_apex"])) <= 3
            print(f"psi={psi_deg}: sim mesh yaw t={t:.1f} expected {r['exp_mesh_yaw']:.1f} -> apex heading {r['apex_heading']:.1f} expected {r['exp_apex']:.1f}  c1={r['c1']}")
            dets = pol.perception.detect(obs["head_rgb"], obs["head_depth"])
            if not dets or getattr(dets[0], "apex_yaw", None) is None:
                r.update(c2=False, c3=False, note=f"no detection / no apex_yaw ({len(dets)} dets, shape={getattr(dets[0], 'shape', None) if dets else None})")
                print("   ", r["note"]); res.append(r); continue
            d = dets[0]; ay = float(np.degrees(d.apex_yaw))
            r.update(apex_yaw=ay, conf=float(d.apex_conf), err_mod120=m120(ay + psi_deg), shape=d.shape,
                     minarearect_yaw=None if d.yaw is None else float(np.degrees(d.yaw)))
            r["c2"] = abs(r["err_mod120"]) <= 10 and r["conf"] >= 0.5
            print(f"   perception apex_yaw={ay:.1f} conf={r['conf']:.2f}; diff to -psi mod 120 = {r['err_mod120']:.1f}; shape={d.shape}; c2={r['c2']}")
            pol.planner.mapping = {"red": "left", "blue": "right", "yellow": "center"}
            p = pol.planner.plan(d, env.filter.arm_cmd.copy(), env.filter.hand_cmd.copy())
            hints = [round(float(np.degrees(h)), 1) for h in pol.planner.yaw_hint_angles(d.apex_yaw, arm)]
            exp = expected_heading(-psi_deg, arm)
            mt = re.match(r"yaw-aligned \((-?\d+) deg, err (\d+)", str(p.reason))
            X, E = (float(mt.group(1)), float(mt.group(2))) if mt else (None, None)
            r.update(plan_ok=bool(p.ok), plan_side=p.side, plan_reason=p.reason, hints=hints, expected_headings=exp, X=X, E=E)
            r["c3"] = bool(mt and E <= 12 and any(abs(w(X - e)) <= 3 for e in exp))
            print(f"   plan: {p.side} {p.reason}; hint angles {hints}; expected headings {exp}; c3={r['c3']}")
            res.append(r)
    finally:
        env.close()
    ok = all(r.get(k) for r in res for k in ("c1", "c2", "c3")) and len(res) == len(CASES)
    print("VERDICT:", "PASS" if ok else "FAIL", {f"psi{r['psi']}{r['arm'][0]}": [r.get(k) for k in ("c1", "c2", "c3")] for r in res})
    from pathlib import Path
    Path(a.out).parent.mkdir(parents=True, exist_ok=True); json.dump({"verdict": ok, "cases": res}, open(a.out, "w"), indent=1, default=str)


if __name__ == "__main__":
    main()
