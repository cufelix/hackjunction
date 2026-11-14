"""Timi's open defect: a blob cut by the bin-hull mask is not at the image edge, so a border
guard on image bounds misses it. Does the cut actually mislabel, and does a hull-AREA floor
catch it before it does?

Reuses the measured pipeline from scripts/measure_shape_feature.py unchanged; the only new
thing is a straight-line cut applied to the METRIC top-face points, which is what the mask
cut does to them.
"""
import itertools, sys
import numpy as np, cv2

sys.path.insert(0, "/home/felix/projects/hackjunction/scripts")
import importlib.util
spec = importlib.util.spec_from_file_location(
    "msf", "/home/felix/projects/hackjunction/scripts/measure_shape_feature.py")
# the module prints its table on import, so capture and drop that
import io, contextlib
msf = importlib.util.module_from_spec(spec)
with contextlib.redirect_stdout(io.StringIO()):
    spec.loader.exec_module(msf)

IDEAL_MM2 = {"cube": 45.0**2, "cylinder": np.pi*22.5**2,
             "triangle": np.sqrt(3)/4 * (2*45/np.sqrt(3))**2}


def top_points(shape, px, py, yaw, rng):
    """The metric top-face points, before ratio_of -- a copy of pipeline()'s metric branch."""
    P = msf.surface(shape, yaw) + np.array([px, py, msf.Z0])
    Pc = (msf.T_op[:3, :3] @ P.T).T + msf.T_op[:3, 3]
    uv = (msf.K @ Pc.T).T
    u = np.round(uv[:, 0]/uv[:, 2]).astype(int); v = np.round(uv[:, 1]/uv[:, 2]).astype(int)
    z = Pc[:, 2]
    ok = (u >= 0) & (u < msf.W) & (v >= 0) & (v < msf.H)
    u, v, z = u[ok], v[ok], z[ok]
    zbuf = np.full((msf.H, msf.W), np.inf)
    np.minimum.at(zbuf, (v, u), z)
    mask = np.isfinite(zbuf).astype(np.uint8)
    k = 2*msf.ERODE + 1
    core = cv2.erode(mask, np.ones((k, k), np.uint8))
    if core.sum() < 0.25*mask.sum():
        core = mask
    vv, uu = np.nonzero(core)
    d = zbuf[vv, uu] * 1000.0
    d = d + rng.normal(0, 1, d.shape) * (msf.DN_MM + msf.DN_REL*d)
    d[rng.random(d.shape) < msf.HOLE] = 0.0
    good = d > 0
    if good.sum() < msf.MIN_DEPTH_PX:
        return None
    uu, vv, d = uu[good], vv[good], d[good]/1000.0
    ray = np.c_[(uu - msf.K[0, 2])/msf.K[0, 0], (vv - msf.K[1, 2])/msf.K[1, 1],
                np.ones(len(uu))] * d[:, None]
    Pp = (msf.T_po[:3, :3] @ ray.T).T + msf.T_po[:3, 3]
    z_top = np.percentile(Pp[:, 2], msf.TOP_PCT)
    top = Pp[Pp[:, 2] >= z_top - msf.TOP_BAND]
    return top[:, :2].astype(np.float32) if len(top) >= 5 else None


def cut(pts, frac, ang):
    """Drop the `frac` of points furthest along direction `ang`: one straight mask edge."""
    if frac <= 0:
        return pts
    c = pts.mean(0)
    t = (pts - c) @ np.array([np.cos(ang), np.sin(ang)], dtype=np.float32)
    keep = t <= np.quantile(t, 1.0 - frac)
    return pts[keep]


def feat(pts):
    hull = cv2.convexHull(pts)
    a = float(cv2.contourArea(hull))
    (_, _), (w, h), _ = cv2.minAreaRect(hull)
    return a/max(w*h, 1e-9), a*1e6          # ratio, area in mm^2


def label(r):
    return "cube" if r >= 0.90 else "cylinder" if r >= 0.70 else "triangle"


XS, YS = np.linspace(0.26, 0.35, 3), np.linspace(-0.17, 0.17, 3)
YAWS = np.radians(np.linspace(-45, 45, 5))
ANGS = np.radians(np.linspace(0, 330, 12))
FRACS = (0.0, 0.10, 0.20, 0.30, 0.40, 0.50)

print("Hull area of the UNCUT top face, mm^2 (ideal in brackets)")
base = {}
for shape in ("cube", "cylinder", "triangle"):
    rng = np.random.default_rng(7)
    A = []
    for x, y, yw in itertools.product(XS, YS, YAWS):
        p = top_points(shape, x, y, yw, rng)
        if p is not None:
            A.append(feat(p)[1])
    A = np.array(A); base[shape] = A
    print(f"  {shape:9} min {A.min():7.0f}  median {np.median(A):7.0f}  max {A.max():7.0f}"
          f"   [{IDEAL_MM2[shape]:.0f}]")

FLOOR = None
print("\nCut fraction -> label accuracy, and how many wrong labels an area floor would catch")
print(f"  {'shape':9} {'cut':>5} {'acc':>6} {'wrong':>6} {'wrong & area<floor':>20} {'median area':>12}")
floor = base["triangle"].min() * 0.90        # 10 % under the thinnest honest triangle
for shape in ("cube", "cylinder", "triangle"):
    for fr in FRACS:
        rng = np.random.default_rng(11)
        got, areas, caught = [], [], 0
        for x, y, yw in itertools.product(XS, YS, YAWS):
            p = top_points(shape, x, y, yw, rng)
            if p is None:
                continue
            for ang in ANGS:
                q = cut(p, fr, ang)
                if len(q) < 30:
                    continue
                r, a = feat(q)
                lb = label(r); got.append(lb); areas.append(a)
                if lb != shape and a < floor:
                    caught += 1
        n = len(got); wrong = sum(g != shape for g in got)
        print(f"  {shape:9} {fr*100:4.0f}% {100*(n-wrong)/n:5.1f}% {wrong:6d} "
              f"{caught:20d} {np.median(areas):12.0f}")
print(f"\narea floor used = 0.90 * thinnest honest triangle = {floor:.0f} mm^2")
