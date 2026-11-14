"""Per-sample estimator bias (real perception pipeline, copied from centroid_bias.py).
usage: regress_xy.py <variant> <shape>   -> regr/<variant>_<shape>.npy with rows
[px, py, yaw_deg, dx_unweighted, dy_unweighted, dx_depth2_weighted, dy_depth2_weighted] (mm, positions in m).
Estimators: unweighted mean of top-band xy; depth^2-weighted mean (perception.py _from_depth: w=(d/1000)^2)."""
import sys, itertools, numpy as np, cv2

VARIANTS = {
    "base":     dict(ERODE=2, TOP_BAND=0.006, DN_MM=1.0, DN_REL=0.002, HOLE=0.01),
    "noerode":  dict(ERODE=0, TOP_BAND=0.006, DN_MM=1.0, DN_REL=0.002, HOLE=0.01),
    "wideband": dict(ERODE=2, TOP_BAND=0.02,  DN_MM=1.0, DN_REL=0.002, HOLE=0.01),
    "nonoise":  dict(ERODE=2, TOP_BAND=0.006, DN_MM=0.0, DN_REL=0.0,   HOLE=0.0),
}
VAR, SHAPE = sys.argv[1], sys.argv[2]
cfg = VARIANTS[VAR]
ERODE, TOP_BAND, DN_MM, DN_REL, HOLE = cfg["ERODE"], cfg["TOP_BAND"], cfg["DN_MM"], cfg["DN_REL"], cfg["HOLE"]
TOP_PCT, MIN_DEPTH_PX = 92, 25

K = np.array([[604.827, 0, 325.042], [0, 604.298, 250.927], [0, 0, 1.0]])
T_po = np.array([[0.0, -0.738455, 0.674302, 0.053660],
                 [-1.0, 0.0, 0.0, 0.017530],
                 [0.0, -0.674302, -0.738455, 0.473870],
                 [0.0, 0.0, 0.0, 1.0]])
T_op = np.linalg.inv(T_po)
SIZE = HGT = 0.045
Z0 = 0.01
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
    """Same pipeline internals as measure_shape_feature.pipeline; returns (top xyz pelvis, depth_m of top pts) or None."""
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
    core = cv2.erode(mask, np.ones((k, k), np.uint8)) if ERODE > 0 else mask
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
    sel = Pp[:, 2] >= z_top - TOP_BAND
    if sel.sum() < 5:
        return None
    return Pp[sel], d[sel]

XS = np.linspace(0.26, 0.34, 3); YS = np.linspace(-0.22, 0.22, 11)
YAWS_DEG = np.arange(0, 121, 10)
DRAWS = 20
rng = np.random.default_rng(7)
rows, drops = [], 0
for px, py in itertools.product(XS, YS):
    for yd in YAWS_DEG:
        yaw = np.radians(yd)
        eu, ew = [], []
        for _ in range(DRAWS):
            r = top_points(SHAPE, px, py, yaw, rng)
            if r is None:
                drops += 1
                continue
            top, dm = r
            xy = top[:, :2]
            w = dm**2                                  # perception.py: w = (d/1000)**2
            eu.append(xy.mean(0) - [px, py])
            ew.append((xy*w[:, None]).sum(0)/w.sum() - [px, py])
        if not eu:
            continue
        mu, mw = np.mean(eu, axis=0)*1000, np.mean(ew, axis=0)*1000
        rows.append([px, py, yd, mu[0], mu[1], mw[0], mw[1]])
np.save(f"/tmp/claude-1000/-home-felix-projects-hackjunction/811dcd44-f22a-4c64-9008-914b85c21b0a/scratchpad/regr/{VAR}_{SHAPE}.npy", np.array(rows))
print(f"{VAR} {SHAPE}: rows {len(rows)} draws dropped {drops}")
