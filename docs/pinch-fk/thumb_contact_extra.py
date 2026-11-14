"""Addendum to thumb_contact.py: midpoint candidate, descent footprint check, IK both arms.
Run: PYTHONPATH=repo_env venv_env/bin/python thumb_contact_extra.py"""
import sys
import numpy as np

sys.argv = sys.argv[:1]
import thumb_contact as T  # noqa: E402  (re-runs the main analysis; keep one process at a time)

L = T.L
MID = np.array([0.08285, -0.0887, 0.01425])
M = np.diag([1.0, -1.0, 1.0])


def ik_both(p):
    out = []
    for side, y in (("left", 0.10), ("right", -0.10)):
        T.kin.offset = {"left": p, "right": p * M.diagonal()}
        h = np.array([1.0, 0, 0]); T.kin.hint = h
        zg = T.zc + 0.5 * T.H
        for _ in range(4):
            q, ok, err, tilt = T.kin.solve(side, np.r_[0.30, y, zg], T.home, top_down=True,
                                           extra_seeds=[T.home[T.ArmKinematics.sl(side)]])
            pts = T.kin.hand_points(side, q, T.post["preshape"])[2:]
            c = (pts[:, 2] - T.zc).min()
            if c >= T.CLR - 1e-3:
                break
            zg += T.CLR - c
        gp, _ = T.kin.grasp_pose(side, q)
        tips = T.kin.hand_points(side, q, T.post["preshape"])[5:] - np.r_[gp[:2], 0]
        out.append((side, ok, err * 1000, np.degrees(tilt), (zg - T.zc) * 1000, c * 1000, tips))
    T.kin.offset = {"left": T.P0, "right": T.P0 * M.diagonal()}
    return out


for label, p in (("current", T.P0), ("midpoint", MID)):
    print(f"\n==== {label} point_in_palm {np.round(p*1000, 2)} mm")
    for side, ok, err, tilt, zgmm, c, tips in ik_both(p):
        print(f"  IK {side:5s} (0.30,{'+' if side=='left' else '-'}0.10): ok {ok} err {err:.2f} mm tilt {tilt:.1f} deg"
              f"  grasp z {zgmm:.1f} mm  preshape clr {c:.1f} mm")
        if side == "left":
            print("  PRESHAPE tips rel. object centre, pelvis-aligned (x fwd, y left, z above counter) mm:")
            for n, t in zip(T.NAMES, tips):
                xy = t[:2]
                fp = {k: f(xy[0], xy[1]) * 1000 for k, f in T.SHAPES.items()}  # pelvis x ~ finger dir u
                print(f"    {n:6s} [{t[0]*1000:6.1f} {t[1]*1000:6.1f} {(t[2]-T.zc)*1000:6.1f}]  footprint sd2D " +
                      " ".join(f"{k}:{v:+.1f}" for k, v in fp.items()))
    zg = T.raise_needed(p, T.up, T.pre_pts)
    for name, h in (("closed", T.post["closed"][L]),):
        rel, sd = T.sdists(T.hand_pts(h)[3:], p, T.up, T.B, zg)
        print(f"  {name} signed dist mm idx/mid/thb: " + "  ".join(f"{k}:{sd[k][0]:+.1f}/{sd[k][1]:+.1f}/{sd[k][2]:+.1f}" for k in T.SHAPES))
        print(f"  {name} tips rel centre (u,v,w) mm: " + " ".join(str(np.round(r*1000, 1)) for r in rel))
    T.sweep(p, T.post["closed"][L], zg, f"CLOSED @ {label}")

# closed_tight for the midpoint: thumb + index flexion
print("\n==== closed_tight search at midpoint (thumb0-2, index0-1)")
import itertools
base = T.post["closed"][L].copy(); lo7, hi7 = T.spec.hand_lower[:7], T.spec.hand_upper[:7]
zg = T.raise_needed(MID, T.up, T.pre_pts)
best = []
for t1, t2, i0, i1 in itertools.product(np.arange(-0.3, 0.71, 0.1), np.arange(0.2, 1.51, 0.1),
                                        np.arange(-1.2, -0.79, 0.1), np.arange(-1.0, -0.39, 0.1)):
    h = base.copy(); h[1], h[2], h[5], h[6] = t1, t2, i0, i1
    if np.any(h < lo7 + 0.02) or np.any(h > hi7 - 0.02):
        continue
    sd = T.sdists(T.hand_pts(h)[3:], MID, T.up, T.B, zg)[1]
    if all(-8 <= sd[k][i] <= -3 for k in ("cube0", "cyl") for i in (0, 2)) and all(sd[k][1] > 0 for k in T.SHAPES):
        best.append((sum(abs(sd[k][i] + 5) for k in ("cube0", "cyl") for i in (0, 2)) + 0.3 * np.abs(h - base).sum(), h, sd))
best.sort(key=lambda t: t[0])
print(f"  feasible {len(best)}")
for _, h, sd in best[:3]:
    print(f"  hand {np.round(h, 2)}  " + "  ".join(f"{k}:{sd[k][0]:+.1f}/{sd[k][1]:+.1f}/{sd[k][2]:+.1f}" for k in T.SHAPES))
if best:
    T.sweep(MID, best[0][1], zg, "CLOSED_TIGHT @ midpoint")

# relaxed 3a: cube0 idx & thb in [-8,-3], middle outside; best achievable cylinder penetration
print("\n==== 3a relaxed: cube0 ok, maximise min(idx,thb) cylinder penetration")
res = []
for x in np.arange(0.075, 0.092, 0.0005):
    for z in np.arange(0.0, 0.03, 0.0005):
        p = np.array([x, -0.084, z]); r = T.score(p)
        if r is None:
            continue
        zg_, sd, sdp = r
        if all(-8 <= sd["cube0"][i] <= -3 for i in (0, 2)) and all(sd[k][1] > 0 for k in T.SHAPES) and all((sdp[k] > 0).all() for k in T.SHAPES):
            res.append((max(sd["cyl"][0], sd["cyl"][2]), p, sd))
res.sort(key=lambda t: t[0])
for c, p, sd in res[:2]:
    print(f"  p {np.round(p*1000,1)} cyl worst {c:+.1f}  " + "  ".join(f"{k}:{sd[k][0]:+.1f}/{sd[k][1]:+.1f}/{sd[k][2]:+.1f}" for k in T.SHAPES))
