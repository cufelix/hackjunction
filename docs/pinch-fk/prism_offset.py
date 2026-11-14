"""Prism fix without yaw alignment: keep the hand heading, SHIFT the object/grasp point along the clamp axis.
Run from outside the repo: PYTHONPATH=repo_env venv_env/bin/python prism_offset.py

Conventions as prism_yaw_sweep.py: pelvis frame, real top-down LEFT IK pose, h = finger heading (palm x),
rel = apex heading - h (CCW from above), 0..120.
u = clamp axis = horizontal unit vector index tip -> thumb tip (closed posture).
d = shift of the OBJECT along u relative to the grasp point (+ toward thumb) == grasp point moved by -d*u.
Assumes the hand pose (orientation) is unchanged by a few-mm grasp-point shift (pure translation)."""
import numpy as np

from aaltonomous.baselines.classical.config import load_config, hand_postures, home_arm
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

cfg, spec = load_config(), JointSpec()
kin, post, home = ArmKinematics(cfg, spec), hand_postures(cfg, spec), home_arm(cfg, spec)
sc = Scene.load()
zc = sc.counter_z
CLR = float(cfg["grasp"]["fingertip_clearance"])
ALT = H = 0.045
R_IN, R_CIRC = ALT / 3, 2 * ALT / 3
PRE_L = post["preshape"][:7]
FR = np.linspace(0, 1, 201)
FINGER_J = {0: (5, 6), 1: (3, 4), 2: (0, 1, 2)}
POSTURES = {
    "pinch_v1": (np.array([0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]), np.array([0.0862, -0.0864, 0.0143])),
    "pinch_demo": (np.array([0.00, 0.92, 0.25, -0.50, -1.41, -0.49, -1.21]), np.array([0.1094, -0.0694, 0.0142])),
}
RELS = np.arange(0, 120, 2)  # 0..118 (120 == 0)
DS = np.arange(-15, 15.001, 0.25) / 1000
print(f"inradius {R_IN*1000:.1f} mm, circumradius {R_CIRC*1000:.1f} mm")


def tri2(x, y, A):
    V = np.array([[R_CIRC * np.cos(A + k * 2 * np.pi / 3), R_CIRC * np.sin(A + k * 2 * np.pi / 3)] for k in range(3)])
    p = np.array([x, y]); d, ins = np.inf, True
    for i in range(3):
        a, b = V[i], V[(i + 1) % 3]; e = b - a; t = np.clip((p - a) @ e / (e @ e), 0, 1)
        d = min(d, np.linalg.norm(p - a - t * e))
        if e[0] * (p - a)[1] - e[1] * (p - a)[0] < 0:
            ins = False
    return -d if ins else d


def sdist(pt, c, A):
    d2 = tri2(pt[0] - c[0], pt[1] - c[1], A); dh = abs(pt[2] - c[2]) - H / 2
    return max(d2, dh) if d2 <= 0 and dh <= 0 else np.hypot(max(d2, 0), max(dh, 0))


def pose(p):
    kin.offset = {"left": p, "right": p * np.array([1, -1, 1.0])}
    kin.hint = np.array([1.0, 0, 0])
    z = zc + 0.5 * H
    for _ in range(4):
        q, ok, err, tilt = kin.solve("left", np.r_[0.30, 0.10, z], home, top_down=True, extra_seeds=[home[kin.sl("left")]])
        c = (kin.hand_points("left", q, post["preshape"])[2:, 2] - zc).min()
        if c >= CLR - 1e-3:
            break
        z += CLR - c
    gp, _ = kin.grasp_pose("left", q)
    R = kin.data.oMf[kin.palm["left"]].rotation.copy()
    return q, gp, R, ok


def tip_traj(q, cl7):
    out = []
    for f in FR:
        h14 = np.zeros(14); h14[:7] = PRE_L + f * (cl7 - PRE_L)
        out.append(kin.hand_points("left", q, h14)[5:])
    return np.array(out)


def full_eval(traj, centre, A, cl7):
    D = np.array([[sdist(t, centre, A) for t in row] for row in traj]) * 1000
    leads = {}
    for i, js in FINGER_J.items():
        hit = np.flatnonzero(D[:, i] <= 0)
        if len(hit):
            f = FR[hit[0]]
            for j in js:
                leads[j] = abs((1 - f) * (cl7[j] - PRE_L[j]))
    return D[-1], (max(leads.values()) if leads else 0.0), D[:, 1].min()


def extent(A, u_ang):
    """extent [min,max] (mm) of the triangle along the axis at angle u_ang, relative to the centroid."""
    pr = [R_CIRC * np.cos(A + k * 2 * np.pi / 3 - u_ang) for k in range(3)]
    return min(pr) * 1000, max(pr) * 1000


def interp_cyc(tab_x, tab_y, x, period=120):
    x = np.mod(x, period)
    xs = np.r_[tab_x, tab_x[0] + period]; ys = np.r_[tab_y, tab_y[0]]
    return np.interp(x, xs, ys)


summary = {}
for name, (cl7, p) in POSTURES.items():
    q, gp, R, ok = pose(p)
    h = np.arctan2(R[1, 0], R[0, 0])
    centre0 = np.r_[gp[:2], zc + H / 2]
    traj = tip_traj(q, cl7)
    tips = traj[-1]  # closed tips idx/mid/thb
    v = tips[2, :2] - tips[0, :2]; u = v / np.linalg.norm(v); u3 = np.r_[u, 0]
    u_ang = np.arctan2(u[1], u[0])
    jaw = np.linalg.norm(v) * 1000
    print(f"\n==== {name} LEFT: IK ok {ok}, heading {np.degrees(h):.1f} deg, clamp axis (idx->thb) at {np.degrees(u_ang - h):+.1f} deg"
          f" from heading, horiz tip gap {jaw:.1f} mm, grasp point to tips along u: idx {(tips[0,:2]-gp[:2])@u*1000:+.1f}"
          f" thb {(tips[2,:2]-gp[:2])@u*1000:+.1f} mm")

    # final signed distances over (rel, d) grid  -> F[rel, d, tip] (mm)
    F = np.zeros((len(RELS), len(DS), 3))
    for i, rd in enumerate(RELS):
        A = h + np.radians(rd)
        for k, d in enumerate(DS):
            c = centre0 + d * u3
            F[i, k] = [sdist(t, c, A) * 1000 for t in tips]
    pen = np.minimum(-F[:, :, 0], -F[:, :, 2])  # min penetration (mm)

    # 1+2: per-yaw table
    mids, dstar, rows = [], [], []
    print("  rel  ext_min ext_max width  midoff |  d*    idx    thb   minpen |  -mid  d0: idx  thb")
    for i, rd in enumerate(RELS):
        lo, hi = extent(h + np.radians(rd), u_ang)
        mo = (lo + hi) / 2
        best = pen[i].max(); ks = np.flatnonzero(pen[i] >= best - 1e-6); k = ks[len(ks) // 2]
        k0 = np.argmin(np.abs(DS))
        mids.append(mo); dstar.append(DS[k] * 1000)
        print(f"  {rd:3d}  {lo:+6.1f}  {hi:+6.1f}  {hi-lo:5.1f}  {mo:+6.2f} | {DS[k]*1000:+5.2f} {F[i,k,0]:+6.1f} {F[i,k,2]:+6.1f}  {best:+5.1f}"
              f" | {-mo:+6.2f}  {F[i,k0,0]:+6.1f} {F[i,k0,2]:+6.1f}")
    mids, dstar = np.array(mids), np.array(dstar)
    # fit d* = b + a*mid
    Afit = np.c_[mids, np.ones_like(mids)]; (a, b), *_ = np.linalg.lstsq(Afit, dstar, rcond=None)
    resid = dstar - (a * mids + b)
    print(f"  midoff range {mids.min():+.2f}..{mids.max():+.2f} mm, width {min(extent(h+np.radians(r),u_ang)[1]-extent(h+np.radians(r),u_ang)[0] for r in RELS):.1f}.."
          f"{max(extent(h+np.radians(r),u_ang)[1]-extent(h+np.radians(r),u_ang)[0] for r in RELS):.1f} mm;"
          f" d* range {dstar.min():+.2f}..{dstar.max():+.2f} mm; fit d* = {a:+.3f}*midoff {b:+.2f} mm (rms resid {np.sqrt((resid**2).mean()):.2f},"
          f" max {np.abs(resid).max():.2f}); corr(d*, -midoff) {np.corrcoef(dstar, -mids)[0,1]:+.3f}")
    print(f"  max achievable min-penetration over yaws: {pen.max(1).min():+.1f}..{pen.max(1).max():+.1f} mm;"
          f" yaws with best >= 1 mm: {(pen.max(1) >= 1).mean()*100:.0f}%")

    # 3: constant offset
    cov1 = (pen >= 1).mean(0); cov0 = (pen >= 0).mean(0)
    kbest = max(range(len(DS)), key=lambda k: (cov1[k], cov0[k], pen[:, k].min()))
    # report the centre of the plateau of best (cov1,cov0)
    plate = [k for k in range(len(DS)) if cov1[k] == cov1[kbest] and cov0[k] == cov0[kbest]]
    kbest = plate[len(plate) // 2]
    dc = DS[kbest]
    print(f"  CONSTANT d = {dc*1000:+.2f} mm (plateau {DS[plate[0]]*1000:+.2f}..{DS[plate[-1]]*1000:+.2f}): both>=1mm {cov1[kbest]*100:.0f}%,"
          f" both>=0 {cov0[kbest]*100:.0f}% (d=0: {cov1[np.argmin(np.abs(DS))]*100:.0f}% / {cov0[np.argmin(np.abs(DS))]*100:.0f}%)")
    good = [i for i in range(len(RELS)) if pen[i, kbest] >= 1]
    worst_mid, worst_lead = np.inf, 0.0
    for i in good:
        fin, lmax, mmin = full_eval(traj, centre0 + dc * u3, h + np.radians(RELS[i]), cl7)
        worst_mid, worst_lead = min(worst_mid, mmin), max(worst_lead, lmax)
    if good:
        print(f"    covered rels {[int(RELS[i]) for i in good]}; middle min over close {worst_mid:+.1f} mm, max lead {worst_lead:.3f} rad")
    # coverage curve
    print("    cov(d): " + " ".join(f"{d*1000:+.0f}:{cov1[k]*100:.0f}" for k, d in enumerate(DS) if abs(d * 1000 % 2) < 1e-6))

    # 4: yaw-dependent offset, with perception error e (perceived rel = true + e)
    tab_x = np.arange(0, 120, 10)
    tab_d = interp_cyc(RELS, dstar, tab_x)
    tab_d = np.array([dstar[list(RELS).index(x)] for x in tab_x])

    def coverage(dfun, e):
        p1 = p0 = 0; worst_mid, worst_lead = np.inf, 0.0
        for i, rd in enumerate(RELS):
            d = dfun(rd + e) / 1000
            A = h + np.radians(rd)
            pn = min(-sdist(tips[0], centre0 + d * u3, A), -sdist(tips[2], centre0 + d * u3, A)) * 1000
            p1 += pn >= 1; p0 += pn >= 0
        return p1 / len(RELS) * 100, p0 / len(RELS) * 100

    mid_of = lambda r: np.mean(extent(h + np.radians(r), u_ang))
    funcs = {
        "exact d*(rel) 2deg": lambda r: interp_cyc(RELS, dstar, r),
        "LUT 10deg (lin interp)": lambda r: interp_cyc(tab_x, tab_d, r),
        f"fit {a:+.2f}*midoff{b:+.2f}": lambda r: a * mid_of(r) + b,
        f"-midoff + {b:+.2f}": lambda r: -mid_of(r) + b,
        "constant": lambda r: dc * 1000,
    }
    print("  YAW-DEPENDENT coverage both>=1mm / both>=0 (%) vs perception yaw error e (deg):")
    print("    " + " " * 28 + "  ".join(f"e={e:+3d}" for e in (-10, -5, 0, 5, 10)))
    for nm, fn in funcs.items():
        cs = [coverage(fn, e) for e in (-10, -5, 0, 5, 10)]
        print(f"    {nm:28s}" + "  ".join(f"{c1:3.0f}/{c0:3.0f}" for c1, c0 in cs))
    # full check of exact d* at e=0
    wm, wl = np.inf, 0.0
    for i, rd in enumerate(RELS):
        if pen[i].max() >= 1:
            fin, lmax, mmin = full_eval(traj, centre0 + dstar[i] / 1000 * u3, h + np.radians(rd), cl7)
            wm, wl = min(wm, mmin), max(wl, lmax)
    print(f"    exact d*: middle min {wm:+.1f} mm, max lead {wl:.3f} rad over covered yaws")
    print("  LUT d(rel) mm, rel = (apex_yaw - finger heading) mod 120: " + ", ".join(f"{x}:{y:+.1f}" for x, y in zip(tab_x, tab_d)))
    # why slope != -1: tips are not on the centroid line. chord of the triangle along the tips' own line
    nrm = np.array([-u[1], u[0]])
    w = [((t[:2] - gp[:2]) @ nrm) * 1000 for t in tips]
    print(f"  tips perpendicular to u (mm, from grasp pt): idx {w[0]:+.1f} mid {w[1]:+.1f} thb {w[2]:+.1f};"
          f" tip z above counter idx {(tips[0,2]-zc)*1000:.1f} thb {(tips[2,2]-zc)*1000:.1f} (prism top {H*1000:.0f})")
    wl = (w[0] + w[2]) / 2 / 1000
    cm = []
    for rd in RELS:
        s = np.linspace(-0.04, 0.04, 1601)
        ins = [s_ for s_ in s if tri2(*(s_ * u + wl * nrm), h + np.radians(rd)) <= 0]
        cm.append((min(ins) + max(ins)) / 2 * 1000 if ins else np.nan)
    cm = np.array(cm)
    print(f"  chord-midpoint on the tips' line: corr(d*, -chordmid) {np.corrcoef(dstar, -cm)[0,1]:+.3f},"
          f" max |d* + chordmid| {np.nanmax(np.abs(dstar + cm)):.2f} mm")
    # 2D shift (along u AND perpendicular) for the best-case posture check
    best2 = []
    for i, rd in enumerate(RELS):
        A = h + np.radians(rd); bb = -np.inf
        for dn in np.arange(-15, 15.01, 1) / 1000:
            for d in np.arange(-15, 15.01, 1) / 1000:
                c = centre0 + d * u3 + dn * np.r_[nrm, 0]
                bb = max(bb, min(-sdist(tips[0], c, A), -sdist(tips[2], c, A)) * 1000)
        best2.append(bb)
    best2 = np.array(best2)
    print(f"  2D shift (1 mm grid, +-15): best min-pen {best2.min():+.1f}..{best2.max():+.1f} mm, yaws >= 1 mm {(best2>=1).mean()*100:.0f}%")
    summary[name] = (dc, cov1[kbest], cov0[kbest], a, b)
print("\nSUMMARY", {k: tuple(round(float(x), 3) for x in v) for k, v in summary.items()})
