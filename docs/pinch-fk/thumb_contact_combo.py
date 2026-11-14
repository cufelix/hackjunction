"""Combined fix search: thumb0/1/2 + point_in_palm at the index/thumb midpoint. Run like thumb_contact.py."""
import itertools, sys
import numpy as np
from aaltonomous.baselines.classical.config import load_config, hand_postures
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene
cfg, spec = load_config(), JointSpec(); kin = ArmKinematics(cfg, spec); post = hand_postures(cfg, spec)
sc = Scene.load(); H = sc.object_height(); S = float(sc.cfg["objects"]["size"]); CLR = cfg["grasp"]["fingertip_clearance"]
L = slice(0, 7); lo7, hi7 = spec.hand_lower[:7], spec.hand_upper[:7]
def pts(h):
    kin._fk(np.zeros(14), np.r_[h, np.zeros(7)]); M = kin.data.oMf[kin.palm["left"]]
    org = np.array([kin.data.oMf[f].translation for f in kin.fingers["left"]])
    return (np.vstack([org, kin._tips("left")]) - M.translation) @ M.rotation
pre = pts(post["preshape"][L])
def box(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw); x, y = c*x+s*y, -s*x+c*y; q = np.abs([x, y]) - S/2
    return np.linalg.norm(np.maximum(q, 0)) + min(max(q[0], q[1]), 0)
def ext(d2, w):
    dh = abs(w) - H/2
    return max(d2, dh) if d2 <= 0 and dh <= 0 else np.hypot(max(d2, 0), max(dh, 0))
def tri(yaw):
    r = 2*S/3; return np.array([[r*np.cos(yaw+k*2*np.pi/3), r*np.sin(yaw+k*2*np.pi/3)] for k in range(3)])
def poly(x, y, V):
    p = np.array([x, y]); d, ins = np.inf, True
    for i in range(3):
        a, b = V[i], V[(i+1) % 3]; e = b-a; t = np.clip((p-a)@e/(e@e), 0, 1); d = min(d, np.linalg.norm(p-a-t*e))
        if e[0]*(p-a)[1]-e[1]*(p-a)[0] < 0: ins = False
    return -d if ins else d
SH = {"cube0": lambda x, y: box(x, y, 0), "cube45": lambda x, y: box(x, y, np.pi/4), "cyl": lambda x, y: np.hypot(x, y)-S/2}
for yd in (0, 30, 60, 90): SH[f"tri{yd}"] = (lambda V: lambda x, y: poly(x, y, V))(tri(np.radians(yd)))
# palm frame at the planner pose (tilt 0): u = palm x, v = palm z (index side), w = palm y (up)
def sd(tips, p):
    low = (pre[:, 1] - p[1]).min(); zg = max(H/2, CLR - low)
    c = p - np.array([0, zg - H/2, 0]); rel = (tips - c)[:, [0, 2, 1]]
    return zg, {k: np.array([ext(f(r[0], r[1]), r[2]) for r in rel])*1000 for k, f in SH.items()}
cl = post["closed"][L]; out = []
for t0, t1, t2 in itertools.product(np.arange(-1.0, 1.01, 0.1), np.arange(-0.7, 1.04, 0.05), np.arange(0.02, 1.73, 0.05)):
    h = cl.copy(); h[:3] = t0, t1, t2
    tp = pts(h)[3:]; th, ix = tp[2], tp[0]
    if np.linalg.norm(th - ix) > 0.06: continue
    for py in (-0.084, -0.0887):
        for fz in (0.0, 0.1, -0.1):  # centre on the midpoint, nudged along v
            p = 0.5*(th+ix); p[1] = py; p[2] += fz*abs(ix[2]-th[2])
            zg, d = sd(tp, p)
            if zg > H - 0.008: continue
            if all(-8 <= d[k][i] <= -3 for k in ("cube0", "cyl") for i in (0, 2)) and all(d[k][1] > 0 for k in SH):
                _, dp = sd(pre[3:], p)
                if all((dp[k] > 0).all() for k in SH):
                    cost = sum(abs(d[k][i]+5) for k in ("cube0", "cyl", "cube45") for i in (0, 2)) + 2*np.abs(h[:3]-cl[:3]).sum()
                    out.append((cost, h.copy(), p, zg, d, np.linalg.norm(th-ix)))
out.sort(key=lambda t: t[0]); print("feasible", len(out))
for c, h, p, zg, d, dist in out[:4]:
    print(f"thumb {np.round(h[:3],2)} p {np.round(p*1000,2)} zg {zg*1000:.1f} thb-idx {dist*1000:.1f} | " +
          " ".join(f"{k}:{d[k][0]:+.1f}/{d[k][1]:+.1f}/{d[k][2]:+.1f}" for k in SH))
if out:
    _, h, p, zg, _, _ = out[0]
    print("sweep cube0/cyl/cube45 idx,thb first touch:")
    fr = np.round(np.arange(0, 1.0001, 0.05), 2)
    rows = [sd(pts((1-f)*post["preshape"][L]+f*h)[3:], p)[1] for f in fr]
    for k in ("cube0", "cube45", "cyl", "tri0", "tri90"):
        print(" ", k, [(n, next((float(f) for f, r in zip(fr, rows) if r[k][i] <= 0), None)) for i, n in enumerate(("idx", "mid", "thb"))])
    low = min(((pts((1-f)*post["preshape"][L]+f*h) - p)[:, 1]).min() + zg for f in fr)
    print(f"  lowest finger point above counter while closing {low*1000:.1f} mm; preshape tips 2D footprint thumb:",
          {k: round(SH[k](*((pre[5]-p)[[0, 2]]))*1000, 1) for k in SH})
