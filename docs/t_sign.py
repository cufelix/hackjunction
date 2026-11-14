# t_sign.py  (python -m aaltonomous.sim.t_sign), env isaac, sim running (Aalto-Sort)
# Single-prism sign test + shape diagnostics (needs the "never classify from 2d blob; shape diag" patch).
# Prints ALL detections per case, picks the det matching the layout object by colour + nearest top xy (pelvis),
# and saves the top-face xy of every failing case to runs/t_sign_top_psi{psi}_{arm}.npy.
import os
import time

import numpy as np

from aaltonomous.sim.scene_client import SceneClient
from aaltonomous.sim import evaluate as ev
from aaltonomous.interface import env as env_mod
from aaltonomous.interface.env import AaltoEnv
from aaltonomous.baselines.classical.config import load_config
from aaltonomous.baselines.classical.policy import ClassicalPolicy

ev.apply_instance_env(); sc = SceneClient("127.0.0.1")
env = AaltoEnv(backend="sim", task="sort_fixed", seed=0, time_limit=60, realtime=True, img_server_ip="127.0.0.1", head_rgbd=True)
ov = {"perception.classify_shape": True, "perception.triangle_pose": True, "perception.shape_diag": True,
      "grasp.yaw_align": True, "grasp.yaw_phi_deg": 105}
pol = ClassicalPolicy(config=load_config(None, ov, None), safety=env.filter.cfg, time_limit=60, log=print)
os.makedirs("runs", exist_ok=True)
w = lambda d: (d + 180) % 360 - 180


def deg(a):
    return None if a is None else round(float(np.degrees(a)), 1)


def diag_str(g):
    if g is None:
        return "None"
    return {k: (round(v, 4) if isinstance(v, float) else v) for k, v in g.items() if k != "top_xy"}


for psi_deg, y_pel, arm in [(0, 0.10, "left"), (-90, 0.10, "left"), (-20, 0.10, "left"), (0, -0.10, "right")]:
    color = "red" if arm == "left" else "blue"
    lay = [{"color": color, "shape": "triangle", "x": 0.45 + y_pel, "y": 0.16, "yaw": np.radians(psi_deg)}]
    env_mod.sample_layout = lambda rng, _l=lay, **kw: [dict(o) for o in _l]
    ev.home_arms(env, sc); obs = env.reset(); time.sleep(1.5); obs = env._obs()
    q = next(o for o in sc.state()["objects"] if o["active"])["quat_xyzw"]
    t = np.degrees(np.arctan2(2*(q[3]*q[2]+q[0]*q[1]), 1-2*(q[1]**2+q[2]**2)))
    print(f"psi={psi_deg} {arm}: sim mesh yaw t={t:.1f} expected {w(-90-psi_deg):.1f} -> apex heading {w(t+90):.1f} expected {w(-psi_deg):.1f}")   # step 1

    # layout (mat frame) -> pelvis: pelvis x = mat_y + 0.14, pelvis y = mat_x - 0.45 (scene.mat_to_pelvis, origin (0.14, -0.45))
    want = pol.scene.mat_to_pelvis([lay[0]["x"], lay[0]["y"], 0.0])[:2]
    print(f"   layout mat=({lay[0]['x']:.3f},{lay[0]['y']:.3f}) -> pelvis xy=({want[0]:+.3f},{want[1]:+.3f})")
    dets, rej = pol.perception.detect(obs["head_rgb"], obs["head_depth"], debug=True)
    for i, d in enumerate(dets):
        print(f"   det[{i}] {d.color} shape={d.shape} src={d.method} top={np.round(d.top, 3).tolist()} "
              f"apex_yaw={deg(d.apex_yaw)} conf={d.apex_conf:.2f} diag={diag_str(d.shape_diag)}")
    for d, why in rej:
        if d is not None:
            print(f"   rej {d.color} src={d.method} top={np.round(d.top, 3).tolist()} -> {why} diag={diag_str(d.shape_diag)}")
    cand = [d for d in dets if d.color == color]
    if not cand:
        print(f"   FAIL: no {color} detection"); continue
    d = min(cand, key=lambda d: float(np.hypot(*(d.top[:2] - want))))
    print(f"   picked {d.color} at {np.hypot(*(d.top[:2] - want)) * 100:.1f} cm from the layout position")

    if d.shape != "triangle" or d.apex_yaw is None:
        top = (d.shape_diag or {}).get("top_xy")
        path = f"runs/t_sign_top_psi{psi_deg}_{arm}.npy"
        np.save(path, np.asarray(top if top is not None else np.zeros((0, 2)), float))
        print(f"   FAIL: shape={d.shape} apex_yaw={deg(d.apex_yaw)} path={(d.shape_diag or {}).get('path')} "
              f"-> top-face xy ({0 if top is None else len(top)} pts) saved to {path}")
        continue
    print(f"   perception apex_yaw={np.degrees(d.apex_yaw):.1f} conf={d.apex_conf:.2f}; diff to -psi mod 120 = {((np.degrees(d.apex_yaw) + psi_deg + 60) % 120) - 60:.1f}; _yaw (minAreaRect, mod 90) = {deg(d.yaw)}")  # step 2
    pol.planner.mapping = {"red": "left", "blue": "right", "yellow": "center"}
    p = pol.planner.plan(d, env.filter.arm_cmd.copy(), env.filter.hand_cmd.copy())
    print("   plan:", p.side, p.reason)                                                                              # step 3: "yaw-aligned (<deg> deg, err <e>)"
    exp = pol.planner.yaw_hint_angles(d.apex_yaw, arm); print("   hint angles (deg):", [round(np.degrees(a), 1) for a in exp])
