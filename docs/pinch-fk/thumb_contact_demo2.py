"""pinch_demo: IK, preshape clearance, descent footprint, right-hand mirror of thumb0."""
import numpy as np
from aaltonomous.baselines.classical.config import mirror_left_to_right, home_arm
_src = open(__file__.replace("thumb_contact_demo2", "thumb_contact_combo")).read().split("cl = post[")[0]
_g = {}; exec(_src, _g); kin, SH, C_cfg, spec, sc, H, CLR = _g["kin"], _g["SH"], _g["cfg"], _g["spec"], _g["sc"], _g["H"], _g["CLR"]
D_CL = np.array([0.00, 0.92, 0.25, -0.50, -1.41, -0.49, -1.21]); D_PRE = np.array([-0.67, 0.92, 0.15, -0.28, -0.02, -0.19, -0.05])
P = np.array([0.1094, -0.0694, 0.0142]); M = np.array([1, -1, 1.0])
for nm, l in (("closed", D_CL), ("preshape", D_PRE)):
    tl = kin.tips_in_palm("left", np.r_[l, np.zeros(7)])
    r1 = mirror_left_to_right(l); r2 = r1.copy(); r2[0] = l[0]
    e1 = np.abs(kin.tips_in_palm("right", np.r_[np.zeros(7), r1]) - tl * M).max()
    e2 = np.abs(kin.tips_in_palm("right", np.r_[np.zeros(7), r2]) - tl * M).max()
    print(f"{nm}: right via config mirror {np.round(r1,2).tolist()} tip err {e1*1000:.1f} mm; thumb0 sign kept err {e2*1000:.2f} mm")
home = home_arm(C_cfg, spec)
kin.offset = {"left": P, "right": P * M}
for side, y in (("left", 0.10), ("right", -0.10)):
    hpre = _g["post"]["preshape"]  # team decision: OUR preshape
    kin.hint = np.array([1.0, 0, 0]); z = sc.counter_z + 0.5 * H
    for _ in range(4):
        q, ok, err, tilt = kin.solve(side, np.r_[0.30, y, z], home, top_down=True, extra_seeds=[home[kin.sl(side)]])
        pp = kin.hand_points(side, q, hpre)[2:]; c = (pp[:, 2] - sc.counter_z).min()
        if c >= CLR - 1e-3: break
        z += CLR - c
    gp, _ = kin.grasp_pose(side, q); tips = kin.hand_points(side, q, hpre)[5:]
    print(f"IK {side}: ok {ok} err {err*1000:.3f} mm tilt {np.degrees(tilt):.2f} grasp z {(z-sc.counter_z)*1000:.1f} (mid {H/2*1000:.1f}) preshape clr {c*1000:.1f}")
    for n, t in zip(("idx", "mid", "thb"), tips):
        r = t - gp; fp = {k: round(f(r[0], r[1]) * 1000, 1) for k, f in SH.items()}
        print(f"   preshape {n} rel centre [{r[0]*1000:+.1f},{r[1]*1000:+.1f}] z {(t[2]-sc.counter_z)*1000:.1f} footprint sd {fp}")
