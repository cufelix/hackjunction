"""Prism apex-yaw sweep for two pinch postures: where do thumb AND index both penetrate (two-sided pinch)?
Run from outside the repo: PYTHONPATH=repo_env venv_env/bin/python prism_yaw_sweep.py

Everything is evaluated in the PELVIS frame at a real IK pose (top-down, planner clearance raise), so the yaw
convention is the planner's directly:
  h   = finger heading = atan2 of palm x in pelvis xy (as planner._finger_err)
  A   = prism apex heading in pelvis xy (CCW from above, x fwd, y left)
  rel = A - h  (apex relative to the finger direction, CCW from above)   -> swept 0..120 deg
  phi = h - A = -rel (mod 120)  -> grasp.yaw_phi_deg for the LEFT arm (heading = apex + phi)
Also printed: the earlier scripts' palm-frame yaw (combo `sd`: angle from palm x towards palm +z) for comparison.
Assumptions: point fingertips at kinematics.TIP, rigid contact (no stop at touch), linear joint interpolation from OUR
preshape to the closed posture, object fixed: centroid xy = grasp point, standing on the counter (centre z = zc + H/2),
no yaw_tri_shift."""
import numpy as np

from aaltonomous.baselines.classical.config import load_config, hand_postures, home_arm, mirror_left_to_right
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

cfg, spec = load_config(), JointSpec()
kin, post, home = ArmKinematics(cfg, spec), hand_postures(cfg, spec), home_arm(cfg, spec)
sc = Scene.load()
zc, H, S = sc.counter_z, sc.object_height(), float(sc.cfg["objects"]["size"])
CLR = float(cfg["grasp"]["fingertip_clearance"])
ALT, HGT = 0.045, 0.045
print(f"scene size {S*1000:.1f} height {H*1000:.1f} (task: altitude {ALT*1000:.0f}, side {ALT*2/np.sqrt(3)*1000:.1f}, height {HGT*1000:.0f})")
H = HGT
PRE_L = post["preshape"][:7]
PRE_R = post["preshape"][7:]
FR = np.linspace(0, 1, 201)
FINGER_J = {0: (5, 6), 1: (3, 4), 2: (0, 1, 2)}  # tip row (index, middle, thumb) -> LEFT joint indices
M = np.array([1, -1, 1.0])
POSTURES = {
    "pinch_v1": (np.array([0.0, 0.0, 1.2, -1.0, -0.6, -1.0, -0.6]), np.array([0.0862, -0.0864, 0.0143])),
    "pinch_demo": (np.array([0.00, 0.92, 0.25, -0.50, -1.41, -0.49, -1.21]), np.array([0.1094, -0.0694, 0.0142])),
}
wrap = lambda d: (d + 180) % 360 - 180


def tri2(x, y, A):
    """signed 2D distance to the equilateral triangle (altitude ALT, centroid 0, one vertex at heading A)."""
    r = 2 * ALT / 3
    V = np.array([[r * np.cos(A + k * 2 * np.pi / 3), r * np.sin(A + k * 2 * np.pi / 3)] for k in range(3)])
    p = np.array([x, y]); d, ins = np.inf, True
    for i in range(3):
        a, b = V[i], V[(i + 1) % 3]; e = b - a; t = np.clip((p - a) @ e / (e @ e), 0, 1)
        d = min(d, np.linalg.norm(p - a - t * e))
        if e[0] * (p - a)[1] - e[1] * (p - a)[0] < 0:
            ins = False
    return -d if ins else d


def sdist(pt, c, A):
    """signed distance (m) of a pelvis point to the extruded prism centred at c (centre at mid-height)."""
    d2 = tri2(pt[0] - c[0], pt[1] - c[1], A); dh = abs(pt[2] - c[2]) - H / 2
    return max(d2, dh) if d2 <= 0 and dh <= 0 else np.hypot(max(d2, 0), max(dh, 0))


def pose(side, p, pre14):
    kin.offset = {"left": p, "right": p * M}
    kin.hint = np.array([1.0, 0, 0])
    y = 0.10 if side == "left" else -0.10
    z = zc + 0.5 * H
    for _ in range(4):
        q, ok, err, tilt = kin.solve(side, np.r_[0.30, y, z], home, top_down=True, extra_seeds=[home[kin.sl(side)]])
        c = (kin.hand_points(side, q, pre14)[2:, 2] - zc).min()
        if c >= CLR - 1e-3:
            break
        z += CLR - c
    gp, _ = kin.grasp_pose(side, q)
    R = kin.data.oMf[kin.palm[side]].rotation.copy()
    return q, gp, R, ok, np.degrees(tilt), c


def tip_traj(side, q, pre7, cl7):
    """(len(FR), 3 tips, 3) pelvis-frame tips along the linear close."""
    out = []
    for f in FR:
        h14 = np.zeros(14); h7 = pre7 + f * (cl7 - pre7)
        if side == "left":
            h14[:7] = h7
        else:
            h14[7:] = h7
        out.append(kin.hand_points(side, q, h14)[5:])
    return np.array(out)


def evaluate(traj, centre, A, pre7, cl7):
    """final signed distances (mm) idx/mid/thb, max lead (rad), min middle sd over the close (mm)."""
    D = np.array([[sdist(t, centre, A) for t in row] for row in traj]) * 1000  # (F, 3)
    leads = {}
    for i, js in FINGER_J.items():
        hit = np.flatnonzero(D[:, i] <= 0)
        if len(hit):
            f = FR[hit[0]]
            for j in js:
                leads[j] = abs((1 - f) * (cl7[j] - pre7[j]))
    return D[-1], (max(leads.values()) if leads else 0.0), D[:, 1].min(), leads


def best_mirror(cl7):
    """right-hand posture that mirrors the left tips: config mirror, or config mirror with thumb0 sign kept."""
    tl = kin.tips_in_palm("left", np.r_[cl7, np.zeros(7)])
    r1 = mirror_left_to_right(cl7); r2 = r1.copy(); r2[0] = cl7[0]
    errs = [np.abs(kin.tips_in_palm("right", np.r_[np.zeros(7), r]) - tl * M).max() for r in (r1, r2)]
    k = int(np.argmin(errs))
    return (r1, r2)[k], errs[k], ("config mirror", "thumb0 sign kept")[k]


summary = {}
for name, (cl7, p) in POSTURES.items():
    q, gp, R, ok, tilt, c = pose("left", p, post["preshape"])
    h = np.arctan2(R[1, 0], R[0, 0])
    centre = np.r_[gp[:2], zc + H / 2]
    pz = R[:, 2]; zh = wrap(np.degrees(np.arctan2(pz[1], pz[0]) - h))
    print(f"\n==== {name} LEFT: IK ok {ok} tilt {tilt:.2f} deg, grasp z {(gp[2]-zc)*1000:.1f} mm, preshape clr {c*1000:.1f} mm,"
          f" heading {np.degrees(h):.1f} deg; palm up axis z-comp of R: x {R[2,0]:+.2f} y {R[2,1]:+.2f} z {R[2,2]:+.2f};"
          f" palm +z at {zh:+.0f} deg from heading -> combo yaw = {'+' if zh > 0 else '-'}rel")
    traj = tip_traj("left", q, PRE_L, cl7)
    rows = []
    print("  rel(deg) combo  idx_mm  mid_mm  thb_mm  maxlead  mid_min  both>=1mm")
    for rd in range(0, 121, 2):
        A = h + np.radians(rd)
        fin, lmax, mmin, _ = evaluate(traj, centre, A, PRE_L, cl7)
        both = fin[0] <= -1 and fin[2] <= -1
        rows.append((rd, fin, lmax, mmin, both))
        cy = int(round(np.sign(zh) * rd)) % 120
        print(f"  {rd:4d}    {cy:4d}  {fin[0]:+6.1f}  {fin[1]:+6.1f}  {fin[2]:+6.1f}  {lmax:6.3f}  {mmin:+6.1f}  {'YES' if both else ''}")
    # windows (cyclic over 120)
    flags = [r[4] for r in rows[:-1]]  # 0..118, 120 == 0
    n = len(flags); wins = []
    if all(flags):
        wins = [(0, n - 1)]
    elif any(flags):
        start = next(i for i in range(n) if flags[i] and not flags[i - 1])
        i = start
        for k in range(n):
            j = (start + k) % n
            if flags[j] and not flags[j - 1]:
                s0 = j
            if flags[j] and not flags[(j + 1) % n]:
                wins.append((s0, j))
    res = []
    for s0, e0 in wins:
        idx = [(s0 + k) % n for k in range(((e0 - s0) % n) + 1)]
        lo, hi = rows[idx[0]][0], rows[idx[-1]][0]
        cdeg = (lo + ((hi - lo) % 120) / 2) % 120
        A = h + np.radians(cdeg)
        fin, lmax, mmin, leads = evaluate(traj, centre, A, PRE_L, cl7)
        wl = max(rows[i][2] for i in idx)
        phi = (-cdeg) % 120
        res.append((lo, hi, cdeg, fin, lmax, wl, mmin, phi, leads))
        jl = " ".join(f"{spec.hand_names[j].replace('left_hand_', '')}:{v:.3f}" for j, v in sorted(leads.items()))
        print(f"  WINDOW rel {lo}..{hi} deg (width {((hi-lo)%120)} deg) centre {cdeg:.0f}: idx/mid/thb {fin[0]:+.1f}/{fin[1]:+.1f}/{fin[2]:+.1f} mm,"
              f" lead at centre {lmax:.3f} [{jl}], max lead in window {wl:.3f}, middle min {mmin:+.1f}"
              f" -> phi = heading - apex = {phi:.0f} deg (left heading = apex + {phi:.0f})")
    if not res:
        b = max(rows, key=lambda r: min(-r[1][0], -r[1][2]))
        print(f"  NO WINDOW. best rel {b[0]}: idx {b[1][0]:+.1f} thb {b[1][2]:+.1f} mm")
    summary[name] = res

    # right-arm cross-check at each window centre: phi -> -phi, apex = heading_R + phi
    for lo, hi, cdeg, fin, lmax, wl, mmin, phi, _ in res:
        r7, merr, how = best_mirror(cl7)
        qR, gpR, RR, okR, tiltR, cR = pose("right", p, post["preshape"])
        hR = np.arctan2(RR[1, 0], RR[0, 0])
        AR = hR - np.radians(-phi)  # heading = apex + s*phi, s = -1 -> apex = heading + phi
        trajR = tip_traj("right", qR, PRE_R, r7)
        # FINGER_J holds LEFT indices; right DDS order is thumb0-2, index0-1, middle0-1
        cl_l, pre_l = cl7, PRE_L
        D = np.array([[sdist(t, np.r_[gpR[:2], zc + H / 2], AR) for t in row] for row in trajR]) * 1000
        print(f"  RIGHT check (posture via {how}, tip mirror err {merr*1000:.2f} mm; preshape mirror via config):"
              f" IK ok {okR} tilt {tiltR:.2f}, heading {np.degrees(hR):.1f}, apex {np.degrees(AR):.1f} (= heading + {phi:.0f}):"
              f" idx/mid/thb {D[-1,0]:+.1f}/{D[-1,1]:+.1f}/{D[-1,2]:+.1f} mm vs left {fin[0]:+.1f}/{fin[1]:+.1f}/{fin[2]:+.1f};"
              f" first touch idx/thb frac {next((FR[i] for i in range(len(FR)) if D[i,0] <= 0), None)}/"
              f"{next((FR[i] for i in range(len(FR)) if D[i,2] <= 0), None)}")

print("\n==== SUMMARY (left arm; right uses -phi)")
for name, res in summary.items():
    if not res:
        print(f"  {name}: no two-sided window -> no yaw alignment makes the prism two-sided")
    for lo, hi, cdeg, fin, lmax, wl, mmin, phi, _ in res:
        print(f"  {name}: rel {lo}..{hi} centre {cdeg:.0f}, idx {fin[0]:+.1f} thb {fin[2]:+.1f} mm, lead centre {lmax:.3f}"
              f" window max {wl:.3f} rad -> grasp.yaw_phi_deg = {phi:.0f} (equiv. {phi-120:.0f})")
