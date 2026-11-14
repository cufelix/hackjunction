"""Centroid bias of the perceived TOP FACE, using the real classical pipeline copied verbatim
from scripts/measure_shape_feature.py (constants, surface(), and pipeline() internals unchanged).
The only change: top_points() returns the metric top-face points instead of the area ratio."""
import itertools, numpy as np, cv2

K = np.array([[604.827, 0, 325.042], [0, 604.298, 250.927], [0, 0, 1.0]])
T_po = np.array([[0.0, -0.738455, 0.674302, 0.053660],
                 [-1.0, 0.0, 0.0, 0.017530],
                 [0.0, -0.674302, -0.738455, 0.473870],
                 [0.0, 0.0, 0.0, 1.0]])
T_op = np.linalg.inv(T_po)
SIZE = HGT = 0.045
Z0 = 0.01
ERODE, TOP_BAND, TOP_PCT, MIN_DEPTH_PX = 2, 0.006, 92, 25
DN_MM, DN_REL, HOLE = 1.0, 0.002, 0.01
W, H = 640, 480

def surface(shape, yaw, n=140):
    if shape == "cube":
        t = np.linspace(-.5, .5, n)
        gx, gy = np.meshgrid(t, t); top = np.c_[gx.ravel(), gy.ravel()] * SIZE
        e = np.linspace(-.5, .5, n) * SIZE
        ring = np.vstack([np.c_[e, np.full(n, -SIZE/2)], np.c_[e, np.full(n, SIZE/2)],
                          np.c_[np.full(n, -SIZE/2), e], np.c_[np.full(n, SIZE/2), e]])
    elif shape == "cylinder":
        a = np.linspace(0, 2*np.pi, 4*n, endpoint=False)
        rr = np.sqrt(np.linspace(0, 1, n)) * (SIZE/2)
        top = np.vstack([np.c_[np.cos(a[::4])*r, np.sin(a[::4])*r] for r in rr])
        ring = np.c_[np.cos(a), np.sin(a)] * (SIZE/2)
    else:
        side = 2*SIZE/np.sqrt(3); R = side/np.sqrt(3)
        V = np.c_[np.cos(np.radians([90, 210, 330])), np.sin(np.radians([90, 210, 330]))] * R
        u, v = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n))
        u, v = u.ravel(), v.ravel(); keep = u + v <= 1
        u, v = u[keep], v[keep]
        top = V[0] + u[:, None]*(V[1]-V[0]) + v[:, None]*(V[2]-V[0])
        e = np.linspace(0, 1, n)
        ring = np.vstack([V[i] + e[:, None]*(V[(i+1) % 3]-V[i]) for i in range(3)])
    c, s = np.cos(yaw), np.sin(yaw)
    Rz = np.array([[c, s], [-s, c]])
    top, ring = top @ Rz, ring @ Rz
    zs = np.linspace(0, HGT, n)
    sides = np.vstack([np.c_[ring, np.full(len(ring), z)] for z in zs[::4]])
    return np.vstack([np.c_[top, np.full(len(top), HGT)], sides])

def top_points(shape, px, py, yaw, rng):
    """pipeline() internals, but returns the metric top-face points (pelvis frame) or None."""
    P = surface(shape, yaw) + np.array([px, py, Z0])
    Pc = (T_op[:3, :3] @ P.T).T + T_op[:3, 3]
    uv = (K @ Pc.T).T
    u = np.round(uv[:, 0]/uv[:, 2]).astype(int); v = np.round(uv[:, 1]/uv[:, 2]).astype(int)
    z = Pc[:, 2]
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], z[ok]
    zbuf = np.full((H, W), np.inf)
    np.minimum.at(zbuf, (v, u), z)
    mask = np.isfinite(zbuf).astype(np.uint8)
    k = 2*ERODE + 1
    core = cv2.erode(mask, np.ones((k, k), np.uint8))
    if core.sum() < 0.25*mask.sum():
        core = mask
    vv, uu = np.nonzero(core)
    d = zbuf[vv, uu] * 1000.0
    d = d + rng.normal(0, 1, d.shape) * (DN_MM + DN_REL*d)
    d[rng.random(d.shape) < HOLE] = 0.0
    good = d > 0
    if good.sum() < MIN_DEPTH_PX:
        return None
    uu, vv, d = uu[good], vv[good], d[good]/1000.0
    ray = np.c_[(uu - K[0, 2])/K[0, 0], (vv - K[1, 2])/K[1, 1], np.ones(len(uu))] * d[:, None]
    Pp = (T_po[:3, :3] @ ray.T).T + T_po[:3, 3]
    z_top = np.percentile(Pp[:, 2], TOP_PCT)
    top = Pp[Pp[:, 2] >= z_top - TOP_BAND]
    if len(top) < 5:
        return None
    return top

# ---- convention check: where does the apex (vertex starting at +y) actually point? ----
def rot(yaw):
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, s], [-s, c]])

R_TRI = 2*SIZE/3
for yd in (0, 30, 90):
    yaw = np.radians(yd)
    S = surface("triangle", yaw)[:, :2]
    d_plus = np.min(np.linalg.norm(S - R_TRI*np.array([np.cos(np.radians(90+yd)), np.sin(np.radians(90+yd))]), axis=1))
    d_minus = np.min(np.linalg.norm(S - R_TRI*np.array([np.cos(np.radians(90-yd)), np.sin(np.radians(90-yd))]), axis=1))
    print(f"convention yaw={yd:3d}: surface has vertex at angle 90+yaw within {d_plus*1000:.4f} mm, "
          f"at 90-yaw within {d_minus*1000:.4f} mm")

# ---- measurement ----
XS = np.linspace(0.27, 0.33, 3); YS = np.linspace(-0.15, 0.15, 3)
POS = list(itertools.product(XS, YS))
YAWS_DEG = np.arange(0, 121, 10)
DRAWS = 20

def run(shape):
    rng = np.random.default_rng(7)
    rows, drops, dead = [], 0, 0
    for px, py in POS:
        for yd in YAWS_DEG:
            yaw = np.radians(yd)
            bs = []
            for _ in range(DRAWS):
                top = top_points(shape, px, py, yaw, rng)
                if top is None:
                    drops += 1
                    continue
                bs.append(top[:, :2].mean(axis=0) - np.array([px, py]))
            if not bs:
                dead += 1
                continue
            b = np.mean(bs, axis=0) * 1000.0           # mm, averaged over noise draws
            rows.append((px*1000, py*1000, yd, b[0], b[1]))
    return np.array(rows), drops, dead

def summarise(name, rows, drops, dead):
    b = rows[:, 3:5]; mag = np.linalg.norm(b, axis=1)
    print(f"{name}: samples kept {len(rows)}, dead (pos,yaw) {dead}, "
          f"dropped draws {drops}/{len(rows)*DRAWS + drops}")
    return mag

tri, tdrop, tdead = run("triangle")
cube, cdrop, cdead = run("cube")
cyl, ydrop, ydead = run("cylinder")

# triangle decomposition: apex direction after yaw, verified above = [-sin yaw, cos yaw]
yaw_t = np.radians(tri[:, 2])
apex = np.stack([-np.sin(yaw_t), np.cos(yaw_t)], axis=1)
perp = np.stack([-apex[:, 1], apex[:, 0]], axis=1)
b_t = tri[:, 3:5]
along = np.sum(b_t*apex, axis=1)
across = np.sum(b_t*perp, axis=1)
mag_t = np.linalg.norm(b_t, axis=1)
i = np.argmax(mag_t)

mag_c = np.linalg.norm(cube[:, 3:5], axis=1)
mag_y = np.linalg.norm(cyl[:, 3:5], axis=1)

print()
print(f"CONVENTION: apex direction after yaw = angle 90+yaw deg (unit = [-sin yaw, cos yaw])")
print(f"TRIANGLE bias (mm): |bias| mean {mag_t.mean():.3f} median {np.median(mag_t):.3f} max {mag_t.max():.3f}")
print(f"  along apex: mean {along.mean():.3f} (sign: + = toward apex), min {along.min():.3f}, max {along.max():.3f}")
print(f"  perpendicular: mean {across.mean():.3f}, max |N| {np.abs(across).max():.3f}")
print(f"  worst single sample: |bias| {mag_t[i]:.3f} mm at x={tri[i,0]:.1f} y={tri[i,1]:.1f} "
      f"yaw={tri[i,2]:.0f}deg, along-apex {along[i]:.3f}")
print(f"CUBE bias (mm): |bias| mean {mag_c.mean():.3f} max {mag_c.max():.3f}")
print(f"CYLINDER bias (mm): |bias| mean {mag_y.mean():.3f} max {mag_y.max():.3f}")
ctrl = (mag_c.mean() + mag_y.mean())/2
print(f"TRIANGLE-SPECIFIC BIAS: {mag_t.mean() - ctrl:+.3f} mm (triangle mean minus mean of cube and cylinder means)")
print(f"  most negative along-apex (toward base face): {along.min():.3f} mm; "
      f"cube max {mag_c.max():.3f}, cylinder max {mag_y.max():.3f}")
print(f"DROPS: triangle {tdrop}+{tdead} dead, cube {cdrop}+{cdead} dead, cylinder {ydrop}+{ydead} dead")

# ---- direction of bias in the pelvis xy plane ----
print()
print("DIRECTION (pelvis xy, mm; -x = toward robot / camera side)")
for name, rows in (("triangle", tri), ("cube", cube), ("cylinder", cyl)):
    b = rows[:, 3:5]
    mean_b = b.mean(axis=0)
    ang = np.degrees(np.arctan2(mean_b[1], mean_b[0]))
    unit = mean_b / np.linalg.norm(mean_b)
    print(f"  {name}: mean bias vector ({mean_b[0]:+.3f}, {mean_b[1]:+.3f}) mm, "
          f"unit ({unit[0]:+.3f}, {unit[1]:+.3f}), angle {ang:+.1f} deg, "
          f"mean -x component {-mean_b[0]:+.3f} mm, "
          f"samples with bias_x<0 (camera-ward) {np.mean(b[:,0]<0)*100:.0f}%")
