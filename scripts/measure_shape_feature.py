"""Where must the shape feature be computed? Runs the REAL classical pipeline on synthetic
RGB-D of the three solids: rasterise -> z-buffer -> train.yaml depth noise -> deproject ->
erode -> top_percentile/top_band -> extent ratio (area / minAreaRect area).

Every constant is read from the starter repo:
  calibration.yaml head intrinsics + T_pelvis_optical, scene.yaml objects.size/counter/layout,
  classical/config.yaml erode_px 2, top_band 0.006, top_percentile 92, min_valid_depth_px 25,
  sim/dr/train.yaml depth_noise_mm 1.0, depth_noise_rel 0.002, depth_hole_frac 0.01
"""
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
    """Dense point sample of the solid's full surface (top ring + sides + top face interior)."""
    if shape == "cube":
        t = np.linspace(-.5, .5, n)
        gx, gy = np.meshgrid(t, t); top = np.c_[gx.ravel(), gy.ravel()] * SIZE
        e = np.linspace(-.5, .5, n) * SIZE
        ring = np.vstack([np.c_[e, np.full(n, -SIZE/2)], np.c_[e, np.full(n, SIZE/2)],
                          np.c_[np.full(n, -SIZE/2), e], np.c_[np.full(n, SIZE/2), e]])
    elif shape == "cylinder":
        a = np.linspace(0, 2*np.pi, 4*n, endpoint=False)
        rr = np.sqrt(np.linspace(0, 1, n)) * (SIZE/2)
        top = np.vstack([np.c_[np.cos(a[:n])*r, np.sin(a[:n])*r] for r in rr])
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

def pipeline(shape, px, py, yaw, rng, where):
    P = surface(shape, yaw) + np.array([px, py, Z0])
    Pc = (T_op[:3, :3] @ P.T).T + T_op[:3, 3]
    uv = (K @ Pc.T).T
    u = np.round(uv[:, 0]/uv[:, 2]).astype(int); v = np.round(uv[:, 1]/uv[:, 2]).astype(int)
    z = Pc[:, 2]
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], z[ok]
    zbuf = np.full((H, W), np.inf)
    np.minimum.at(zbuf, (v, u), z)                      # z-buffer: nearest surface wins
    mask = np.isfinite(zbuf).astype(np.uint8)
    if where == "blob":                                  # Oliver's classify_blob input
        return ratio_of(np.argwhere(mask > 0)[:, ::-1].astype(np.float32)), int(mask.sum())
    k = 2*ERODE + 1
    core = cv2.erode(mask, np.ones((k, k), np.uint8))    # perception.py erode_px
    if core.sum() < 0.25*mask.sum():
        core = mask
    vv, uu = np.nonzero(core)
    d = zbuf[vv, uu] * 1000.0                            # mm, as the obs carries it
    d = d + rng.normal(0, 1, d.shape) * (DN_MM + DN_REL*d)
    d[rng.random(d.shape) < HOLE] = 0.0                  # train.yaml depth_hole_frac
    good = d > 0
    if good.sum() < MIN_DEPTH_PX:
        return np.nan, int(good.sum())
    uu, vv, d = uu[good], vv[good], d[good]/1000.0
    ray = np.c_[(uu - K[0, 2])/K[0, 0], (vv - K[1, 2])/K[1, 1], np.ones(len(uu))] * d[:, None]
    Pp = (T_po[:3, :3] @ ray.T).T + T_po[:3, 3]          # back to the pelvis frame: METRIC
    z_top = np.percentile(Pp[:, 2], TOP_PCT)
    top = Pp[Pp[:, 2] >= z_top - TOP_BAND]
    if len(top) < 5:
        return np.nan, len(top)
    return ratio_of(top[:, :2].astype(np.float32)), len(top)

def ratio_of(pts):
    hull = cv2.convexHull(pts)
    a = cv2.contourArea(hull)
    (_, _), (w, h), _ = cv2.minAreaRect(hull)
    return a / max(w*h, 1e-9)

XS, YS = np.linspace(0.26, 0.35, 3), np.linspace(-0.17, 0.17, 5)
YAWS = np.radians(np.linspace(-45, 45, 7))
CUBE_MIN, CYL_MIN = 0.88, 0.66
def label(r): return "cube" if r >= CUBE_MIN else "cylinder" if r >= CYL_MIN else "triangle"

for where, name in (("blob", "A) 2D PIXEL BLOB  (what classify_blob is given today)"),
                    ("metric", "B) METRIC TOP-FACE POINTS  (top[:, :2], same data _yaw gets)")):
    print(f"\n{name}")
    print(f"  {'shape':10} {'ratio range':>22} {'median':>8} {'px':>6}   accuracy @ 0.88/0.66")
    for shape in ("cube", "cylinder", "triangle"):
        rng = np.random.default_rng(7)
        out = [pipeline(shape, x, y, yw, rng, where) for x, y, yw in itertools.product(XS, YS, YAWS)]
        r = np.array([o[0] for o in out]); n = np.array([o[1] for o in out])
        r = r[np.isfinite(r)]
        got = [label(v) for v in r]
        acc = sum(g == shape for g in got)/len(got)
        wrong = {}
        for g in got:
            if g != shape: wrong[g] = wrong.get(g, 0) + 1
        extra = "  -> " + ", ".join(f"{k} {100*v/len(got):.0f}%" for k, v in sorted(wrong.items())) if wrong else ""
        print(f"  {shape:10} {r.min():.3f} - {r.max():.3f}  {np.median(r):8.3f} {int(np.median(n)):6d}   {acc*100:5.1f}%{extra}")
