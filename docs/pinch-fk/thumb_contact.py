"""Does the Dex3 thumb ever touch the object in the classical grasp? FK + signed distances, LEFT hand.
Run from outside the repo: PYTHONPATH=repo_env venv_env/bin/python thumb_contact.py
Assumptions: fingertip = point at kinematics.TIP in the distal link (rigid point contact, no pad geometry),
object centre = grasp point in xy, at mid-height on the counter (z from the planner's raise rule),
prism centred on its centroid. Object frame: u = finger direction (horizontal), v = sideways (index side +), w = up."""
import itertools
import numpy as np

from aaltonomous.baselines.classical.config import load_config, hand_postures, home_arm
from aaltonomous.baselines.classical.kinematics import ArmKinematics
from aaltonomous.interface.safety import JointSpec
from aaltonomous.scene.scene import Scene

cfg, spec = load_config(), JointSpec()
kin, post, home = ArmKinematics(cfg, spec), hand_postures(cfg, spec), home_arm(cfg, spec)
sc = Scene.load()
zc, H = sc.counter_z, sc.object_height()
S = float(sc.cfg["objects"]["size"])
g = cfg["grasp"]
CLR = float(g["fingertip_clearance"])
P0 = np.asarray(kin.offset["left"], float)
NAMES = ("index", "middle", "thumb")
L = slice(0, 7)
print(f"object size {S*1000:.0f} mm, height {H*1000:.0f} mm, counter_z {zc:.4f}, point_in_palm {P0*1000}")
lo_h = np.array(kin.model.lowerPositionLimit[kin.hand_iq]); hi_h = np.array(kin.model.upperPositionLimit[kin.hand_iq])
for i in range(7):
    print(f"  {spec.hand_names[i]:26s} URDF [{lo_h[i]:+.3f}, {hi_h[i]:+.3f}]  spec [{spec.hand_lower[i]:+.3f}, {spec.hand_upper[i]:+.3f}]")


def hand_pts(h7):
    """palm-frame (m): rows 0-2 distal link origins, 3-5 tips (index, middle, thumb)."""
    kin._fk(np.zeros(14), np.r_[h7, np.zeros(7)])
    M = kin.data.oMf[kin.palm["left"]]
    org = np.array([kin.data.oMf[f].translation for f in kin.fingers["left"]])
    return (np.vstack([org, kin._tips("left")]) - M.translation) @ M.rotation


# ---------------------------------------------------------------- typical pose: palm orientation + planner raise
def pose_frame(xy, p):
    kin.offset["left"] = np.asarray(p, float)
    kin.hint = np.array([1.0, 0, 0])
    z = zc + g["height_frac"] * H
    for _ in range(4):
        q, ok, err, tilt = kin.solve("left", np.r_[xy, z], home, top_down=True, extra_seeds=[home[L]])
        pts = kin.hand_points("left", q, post["preshape"])[2:]
        c = (pts[:, 2] - zc).min()
        if c >= CLR - 1e-3:
            break
        z += CLR - c
    kin._fk(q)
    R = kin.data.oMf[kin.palm["left"]].rotation.copy()
    kin.offset["left"] = P0
    return z - zc, R, ok, np.degrees(tilt), c


def frame_from_R(R):
    up = R.T @ np.array([0, 0, 1.0])            # world up in palm coordinates
    u = R[:, 0].copy(); u[2] = 0; u /= np.linalg.norm(u)  # finger direction, horizontal (world)
    w = np.array([0, 0, 1.0]); v = np.cross(w, u)
    B = np.vstack([u, v, w]) @ R                 # palm -> object-frame rotation
    if (B @ np.array([0, 0, 1.0]))[1] < 0:        # make +v the index side (palm +z)
        B[1] *= -1
    return up, B


def raise_needed(p, up, pre_pts):
    """planner rule in palm frame: grasp height above counter = max(H/2, CLR - lowest preshape point rel. grasp pt)."""
    low = ((pre_pts - p) @ up).min()
    return max(g["height_frac"] * H, CLR - low)


# ---------------------------------------------------------------- signed distance functions (object frame, m)
def sd_extrude(d2, w):
    dh = abs(w) - H / 2
    if d2 <= 0 and dh <= 0:
        return max(d2, dh)
    return np.hypot(max(d2, 0), max(dh, 0))


def sd_box2(x, y, yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    x, y = c * x + s * y, -s * x + c * y
    q = np.abs([x, y]) - S / 2
    return np.linalg.norm(np.maximum(q, 0)) + min(max(q[0], q[1]), 0)


def sd_poly2(x, y, V):
    p = np.array([x, y]); n = len(V); dmin, inside = np.inf, True
    for i in range(n):
        a, b = V[i], V[(i + 1) % n]
        e = b - a; t = np.clip((p - a) @ e / (e @ e), 0, 1)
        dmin = min(dmin, np.linalg.norm(p - a - t * e))
        if e[0] * (p - a)[1] - e[1] * (p - a)[0] < 0:  # CCW polygon: right of an edge = outside
            inside = False
    return -dmin if inside else dmin


def tri(yaw):
    r = 2 * S / 3  # centroid -> apex for altitude S
    return np.array([[r * np.cos(yaw + k * 2 * np.pi / 3), r * np.sin(yaw + k * 2 * np.pi / 3)] for k in range(3)])


SHAPES = {"cube0": lambda x, y: sd_box2(x, y, 0.0), "cube45": lambda x, y: sd_box2(x, y, np.pi / 4),
          "cyl": lambda x, y: np.hypot(x, y) - S / 2}
for yd in (0, 30, 60, 90):
    SHAPES[f"tri{yd}"] = (lambda V: (lambda x, y: sd_poly2(x, y, V)))(tri(np.radians(yd)))


def sdists(tips_palm, p, up, B, zg):
    """signed distance (mm) of each tip to each shape. object centre (palm) = p - (zg - H/2) * up."""
    c = p - (zg - H / 2) * up
    rel = (tips_palm - c) @ B.T
    return rel, {k: np.array([sd_extrude(f(r[0], r[1]), r[2]) for r in rel]) * 1000 for k, f in SHAPES.items()}


# ---------------------------------------------------------------- run
pre_pts, cls_pts = hand_pts(post["preshape"][L]), hand_pts(post["closed"][L])
print("\n== typical poses (left arm, fingers forward): grasp-point height above counter after the clearance raise")
frames = []
for xy in ((0.30, 0.10), (0.25, 0.15), (0.35, 0.05)):
    zg, R, ok, tilt, c = pose_frame(np.array(xy), P0)
    frames.append((zg, R))
    print(f"  xy {xy}: ok {ok} tilt {tilt:.1f} deg  grasp z {zg*1000:.1f} mm (mid-height {H/2*1000:.1f})  preshape min clr {c*1000:.1f}")
zg, R = frames[0]
up, B = frame_from_R(R)
print(f"  up in palm {np.round(up, 3)}; palm-frame raise estimate {raise_needed(P0, up, pre_pts)*1000:.1f} mm")

tc = cls_pts[3:]
print(f"\n(1) CLOSED: thumb-index tip distance {np.linalg.norm(tc[2]-tc[0])*1000:.1f} mm, index-middle {np.linalg.norm(tc[1]-tc[0])*1000:.1f}")
rel, sd = sdists(tc, P0, up, B, zg)
for i, n in enumerate(NAMES):
    print(f"  {n:6s} tip rel. object centre (u fwd, v idx-side, w up) mm {np.round(rel[i]*1000, 1)}")
print("  signed distance mm (neg = penetration):  " + "  ".join(f"{k:>7s}" for k in SHAPES))
for i, n in enumerate(NAMES):
    print(f"  {n:6s}" + " " * 34 + "  ".join(f"{sd[k][i]:7.1f}" for k in SHAPES))


def sweep(p, hcl, zg_, label):
    print(f"\n(2) sweep PRESHAPE->{label} (signed distance mm per tip; first-touch fraction)")
    fr = np.round(np.arange(0, 1.0001, 0.05), 2)
    rows = []
    for f in fr:
        h = (1 - f) * post["preshape"][L] + f * hcl
        rows.append(sdists(hand_pts(h)[3:], p, up, B, zg_)[1])
    for k in SHAPES:
        first = []
        for i, n in enumerate(NAMES):
            d = np.array([r[k][i] for r in rows])
            hit = np.flatnonzero(d <= 0)
            first.append(f"{n} {fr[hit[0]]:.2f}" if len(hit) else f"{n} never(min {d.min():+.1f})")
        print(f"  {k:6s} first touch: " + ", ".join(first))
    print("  cube0 per fraction (idx/mid/thb):", " | ".join(f"{f:.2f}:{r['cube0'][0]:+.0f}/{r['cube0'][1]:+.0f}/{r['cube0'][2]:+.0f}"
                                                         for f, r in zip(fr, rows) if round(f * 20) % 2 == 0))
    # lowest tip height above counter during the close (planner allows 3 mm while closing)
    lows = [((hand_pts((1 - f) * post["preshape"][L] + f * hcl) - p) @ up).min() + zg_ for f in fr]
    print(f"  lowest finger point above counter while closing: {min(lows)*1000:.1f} mm")


sweep(P0, post["closed"][L], zg, "CLOSED")


# ---------------------------------------------------------------- (3a) new point_in_palm
def score(p, hcl=post["closed"][L]):
    zg_ = raise_needed(p, up, pre_pts)
    if zg_ > H - 0.008:
        return None
    tcl = hand_pts(hcl)[3:]
    sd = sdists(tcl, p, up, B, zg_)[1]
    sdp = sdists(pre_pts[3:], p, up, B, zg_)[1]
    return zg_, sd, sdp


print("\n(3a) grid over point_in_palm x, z (y fixed / varied): need idx & thb in [-8,-3] for cube0, cube45, cyl;"
      " middle tip outside (>0) all shapes; preshape tips outside all shapes")
best = []
for x, y, z in itertools.product(np.arange(0.070, 0.100, 0.001), (-0.084, -0.080, -0.088), np.arange(0.0, 0.030, 0.001)):
    p = np.array([x, y, z]); r = score(p)
    if r is None:
        continue
    zg_, sd, sdp = r
    ok = all(-8 <= sd[k][0] <= -3 and -8 <= sd[k][2] <= -3 for k in ("cube0", "cyl")) and \
        all(sd[k][1] > 0 for k in SHAPES) and all((sdp[k] > 0).all() for k in SHAPES)
    if ok:
        bal = sum(abs(sd[k][0] - sd[k][2]) for k in ("cube0", "cyl")) + sum(abs(sd[k][i] + 5) for k in ("cube0", "cyl") for i in (0, 2))
        best.append((bal, p, zg_, sd))
best.sort(key=lambda t: t[0])
print(f"  feasible candidates: {len(best)}")
for bal, p, zg_, sd in best[:3]:
    print(f"  p {np.round(p*1000, 1)} mm  grasp z {zg_*1000:.1f}  " +
          "  ".join(f"{k}:{sd[k][0]:+.1f}/{sd[k][1]:+.1f}/{sd[k][2]:+.1f}" for k in SHAPES))
if best:
    PA = best[0][1]
    zgA, RA, okA, tiltA, cA = pose_frame(np.array((0.30, 0.10)), PA)
    print(f"  IK check p={np.round(PA*1000,1)}: ok {okA} tilt {tiltA:.1f} grasp z {zgA*1000:.1f} preshape clr {cA*1000:.1f} mm")
    sweep(PA, post["closed"][L], raise_needed(PA, up, pre_pts), "CLOSED with 3a point")

# ---------------------------------------------------------------- (3b) closed_tight: more thumb flexion
print("\n(3b) closed_tight grid over thumb_0/1/2 (fingers as closed), current point_in_palm")
base = post["closed"][L].copy()
lo7, hi7 = spec.hand_lower[:7], spec.hand_upper[:7]
cands = []
for t0, t1, t2 in itertools.product(np.arange(-0.4, 0.41, 0.1), np.arange(lo7[1], hi7[1] + 1e-9, 0.05),
                                    np.arange(lo7[2], hi7[2] + 1e-9, 0.05)):
    h = base.copy(); h[0], h[1], h[2] = t0, t1, t2
    if np.any(h < lo7 + 0.02) or np.any(h > hi7 - 0.02):
        continue
    tcl = hand_pts(h)[3:]
    sd = sdists(tcl, P0, up, B, zg)[1]
    if all(-8 <= sd[k][2] <= -3 for k in ("cube0", "cyl")):
        cands.append((abs(sd["cube0"][2] + 5) + abs(sd["cyl"][2] + 5) + 0.5 * np.abs(h[:3] - base[:3]).sum(), h, sd))
cands.sort(key=lambda t: t[0])
print(f"  thumb limits spec: t0 [{lo7[0]:+.2f},{hi7[0]:+.2f}] t1 [{lo7[1]:+.2f},{hi7[1]:+.2f}] t2 [{lo7[2]:+.2f},{hi7[2]:+.2f}];"
      f" feasible: {len(cands)}")
for _, h, sd in cands[:3]:
    print(f"  thumb {np.round(h[:3], 2)}  " + "  ".join(f"{k}:{sd[k][0]:+.1f}/{sd[k][1]:+.1f}/{sd[k][2]:+.1f}" for k in SHAPES))
if cands:
    sweep(P0, cands[0][1], zg, "CLOSED_TIGHT")
