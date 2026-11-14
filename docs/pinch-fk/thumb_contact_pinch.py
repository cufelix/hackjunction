"""pinch_v1: closed_tight with |thumb-index| = 41 / 39 mm, point_in_palm = jaw midpoint, checks. Run like thumb_contact.py."""
import itertools
import numpy as np
import types
C = types.SimpleNamespace()
_src = open(__file__.replace("thumb_contact_pinch", "thumb_contact_combo")).read().split("cl = post[")[0]
_g = {}; exec(_src, _g); C.__dict__.update(_g)  # helpers only, no search
from aaltonomous.baselines.classical.config import home_arm
kin, post, L, cl = C.kin, C.post, C.L, C.post["closed"][C.L]
lo7, hi7 = C.lo7, C.hi7
print("\n=== pinch_v1 search; limits (left, rad):", {C.spec.hand_names[i]: (round(lo7[i], 3), round(hi7[i], 3)) for i in range(7)})
def search(target, joints):
    grids = {0: np.arange(-0.6, 0.61, 0.05), 1: np.arange(-0.70, 1.04, 0.025), 2: np.arange(0.0, 1.745, 0.025),
             5: np.arange(-1.5, -0.79, 0.05), 6: np.arange(-1.2, -0.39, 0.05)}
    out = []
    for vals in itertools.product(*[grids[j] for j in joints]):
        h = cl.copy(); h[list(joints)] = vals
        if np.any(h < lo7 + 0.02) or np.any(h > hi7 - 0.02): continue
        tp = C.pts(h)[3:]; d = np.linalg.norm(tp[2] - tp[0])
        if abs(d - target) > 0.001: continue
        mid = 0.5 * (tp[2] + tp[0])
        zg, s = C.sd(tp, mid)
        if not all(s[k][1] > 0 for k in C.SH): continue
        cost = abs(mid[2] - 0.015) * 1000 + abs(s["cube0"][0] - s["cube0"][2]) + 3 * np.abs(h - cl).sum()
        out.append((cost, h, mid, zg, s, d))
    out.sort(key=lambda t: t[0]); return out
res = {}
for target in (0.041, 0.039):
    for joints in ((1, 2), (0, 1, 2), (1, 2, 5, 6)):
        r = search(target, joints)
        print(f"target {target*1000:.0f} mm, joints {[C.spec.hand_names[j].replace('left_hand_','') for j in joints]}: {len(r)} solutions")
        for cost, h, mid, zg, s, d in r[:2]:
            print(f"   hand {np.round(h, 3).tolist()} dist {d*1000:.1f} mid {np.round(mid*1000, 2)} | " +
                  " ".join(f"{k}:{s[k][0]:+.1f}/{s[k][1]:+.1f}/{s[k][2]:+.1f}" for k in C.SH))
        if r and target not in res: res[target] = r[0]
home = home_arm(C.cfg, C.spec)
for target, (cost, h, mid, zg, s, d) in res.items():
    p = np.round(mid, 4)
    print(f"\n### {target*1000:.0f} mm  closed_tight {np.round(h, 3).tolist()}  point_in_palm L {p.tolist()} R {(p*[1,-1,1]).tolist()}")
    zg, s = C.sd(C.pts(h)[3:], p)
    print("  pen (idx/mid/thb):", " ".join(f"{k}:{s[k][0]:+.1f}/{s[k][1]:+.1f}/{s[k][2]:+.1f}" for k in C.SH))
    fr = np.round(np.arange(0, 1.0001, 0.05), 2)
    rows = [C.sd(C.pts((1-f)*post["preshape"][L]+f*h)[3:], p)[1] for f in fr]
    print("  first touch idx/thb:", {k: tuple(next((float(f) for f, r in zip(fr, rows) if r[k][i] <= 0), None) for i in (0, 2)) for k in C.SH})
    low = min(((C.pts((1-f)*post["preshape"][L]+f*h) - p)[:, 1]).min() + zg for f in fr)
    print(f"  lowest finger point while closing {low*1000:.1f} mm above counter")
    kin.offset = {"left": p, "right": p * np.array([1, -1, 1.0])}
    for side, y in (("left", 0.10), ("right", -0.10)):
        kin.hint = np.array([1.0, 0, 0]); z = C.sc.counter_z + 0.5 * C.H
        for _ in range(4):
            q, ok, err, tilt = kin.solve(side, np.r_[0.30, y, z], home, top_down=True, extra_seeds=[home[kin.sl(side)]])
            pp = kin.hand_points(side, q, post["preshape"])[2:]; c = (pp[:, 2] - C.sc.counter_z).min()
            if c >= C.CLR - 1e-3: break
            z += C.CLR - c
        gp, _ = kin.grasp_pose(side, q); tips = kin.hand_points(side, q, post["preshape"])[5:]
        print(f"  IK {side} (0.30,{y:+.2f}): ok {ok} err {err*1000:.3f} mm tilt {np.degrees(tilt):.2f} deg grasp z {(z-C.sc.counter_z)*1000:.1f} preshape clr {c*1000:.1f} mm")
        for n, t in zip(("idx", "mid", "thb"), tips):
            r = t - gp; fp = min(C.SH[k](r[0], r[1]) for k in C.SH)
            print(f"     preshape {n} rel centre (pelvis x,y) [{r[0]*1000:+.1f},{r[1]*1000:+.1f}] z {(t[2]-C.sc.counter_z)*1000:.1f}  min footprint clearance {fp*1000:+.1f} mm")
