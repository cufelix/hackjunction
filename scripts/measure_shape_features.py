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

# ---- everything above is verbatim from measure_shape_feature.py; below is new (copy in scratchpad) ----
def ratio_of(pts):
    hull = cv2.convexHull(pts); a = cv2.contourArea(hull)
    (_, _), (w, h), _ = cv2.minAreaRect(hull)
    return a / max(w*h, 1e-9)

XS, YS = np.linspace(0.26, 0.35, 3), np.linspace(-0.17, 0.17, 5)
YAWS = np.radians(np.linspace(-45, 45, 7))
MIN_PTS = 30
SHAPES = ("cube", "cylinder", "triangle")

def top_points(shape, px, py, yaw, rng, band):
    global TOP_BAND
    TOP_BAND = band
    P = surface(shape, yaw) + np.array([px, py, Z0])
    Pc = (T_op[:3, :3] @ P.T).T + T_op[:3, 3]
    uv = (K @ Pc.T).T
    u = np.round(uv[:, 0]/uv[:, 2]).astype(int); v = np.round(uv[:, 1]/uv[:, 2]).astype(int)
    z = Pc[:, 2]; ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    u, v, z = u[ok], v[ok], z[ok]
    zbuf = np.full((H, W), np.inf); np.minimum.at(zbuf, (v, u), z)
    mask = np.isfinite(zbuf).astype(np.uint8)
    k = 2*ERODE + 1
    core = cv2.erode(mask, np.ones((k, k), np.uint8))
    if core.sum() < 0.25*mask.sum(): core = mask
    vv, uu = np.nonzero(core)
    d = zbuf[vv, uu]*1000.0
    d = d + rng.normal(0, 1, d.shape)*(DN_MM + DN_REL*d)
    d[rng.random(d.shape) < HOLE] = 0.0
    good = d > 0
    if good.sum() < MIN_DEPTH_PX: return None
    uu, vv, d = uu[good], vv[good], d[good]/1000.0
    ray = np.c_[(uu-K[0, 2])/K[0, 0], (vv-K[1, 2])/K[1, 1], np.ones(len(uu))]*d[:, None]
    Pp = (T_po[:3, :3] @ ray.T).T + T_po[:3, 3]
    z_top = np.percentile(Pp[:, 2], TOP_PCT)
    top = Pp[Pp[:, 2] >= z_top - band]
    return top[:, :2].astype(np.float32)

def harmonics(pts, how="max", nb=72):
    q = pts - pts.mean(0)
    th = np.arctan2(q[:, 1], q[:, 0]); r = np.hypot(q[:, 0], q[:, 1])
    b = ((th + np.pi)/(2*np.pi)*nb).astype(int) % nb
    rad = np.full(nb, np.nan)
    for i in range(nb):
        s = r[b == i]
        if len(s): rad[i] = s.max() if how == "max" else s.mean()
    ok = np.isfinite(rad)
    if ok.sum() < 8: return np.nan, np.nan
    x = np.arange(nb)
    rad = np.interp(x, x[ok], rad[ok], period=nb)
    c = np.abs(np.fft.rfft(rad))
    return c[3]/c[0], c[4]/c[0]

def feats(pts):
    hull = cv2.convexHull(pts); per = cv2.arcLength(hull, True); a = cv2.contourArea(hull)
    f = {"ratio": ratio_of(pts), "iso": 4*np.pi*a/per**2}
    for e in (0.02, 0.04, 0.06):
        f[f"nv{e}"] = len(cv2.approxPolyDP(hull, e*per, True))
    f["h3max"], f["h4max"] = harmonics(pts, "max")
    f["h3mean"], f["h4mean"] = harmonics(pts, "mean")
    return f

def collect(band):
    out = {}
    for shape in SHAPES:
        rng = np.random.default_rng(7)
        rows, ns, ab = [], [], 0
        for x, y, yw in itertools.product(XS, YS, YAWS):
            p = top_points(shape, x, y, yw, rng, band)
            if p is None or len(p) < MIN_PTS: ab += 1; ns.append(0 if p is None else len(p)); continue
            ns.append(len(p)); rows.append(feats(p))
        out[shape] = ({k: np.array([r[k] for r in rows]) for k in rows[0]} if rows else {}, np.array(ns), ab)
    return out

def label(r): return "cube" if r >= 0.88 else "cylinder" if r >= 0.66 else "triangle"
def mmm(a): return f"{a.min():.3f}/{np.median(a):.3f}/{a.max():.3f}"

def gap(hi, lo):   # hi class feature array should sit above lo class array
    return hi.min() - lo.max()

if __name__ == "__main__":
    print("SYNTHETIC data. grid = 3x5 positions x 7 yaws = 105 per shape, rng seed 7 per shape.")
    print("\nMEASUREMENT 1: ratio min/median/max | acc@.88/.66 | median n_top | abstain(<30)")
    res = {}
    for band in (0.002, 0.003, 0.004, 0.006, 0.008, 0.010):
        res[band] = collect(band)
        for s in SHAPES:
            d, n, ab = res[band][s]; r = d["ratio"]
            acc = np.mean([label(v) == s for v in r])
            print(f"  band {band*1000:4.0f}mm {s:9} {mmm(r)}  acc {acc*100:5.1f}%  n_top {int(np.median(n)):4d}  abstain {ab}")
        c, y, t = (res[band][s][0]["ratio"] for s in SHAPES)
        print(f"     gaps: cube-cyl {c.min()-y.max():+.3f}  cyl-tri {y.min()-t.max():+.3f}")

    print("\nMEASUREMENT 2 @ band 6mm")
    R = res[0.006]; F = {s: R[s][0] for s in SHAPES}
    def block(name, key, order, note=""):
        # order: classes from high feature to low
        print(f"\n[{name}] {note}")
        for s in SHAPES: print(f"  {s:9} min/med/max {mmm(F[s][key])}")
        g1 = gap(F[order[0]][key], F[order[1]][key]); g2 = gap(F[order[1]][key], F[order[2]][key])
        t1 = (F[order[0]][key].min()+F[order[1]][key].max())/2; t2 = (F[order[1]][key].min()+F[order[2]][key].max())/2
        m = min(g1, g2)
        print(f"  order {'>'.join(order)}; thresholds {t1:.3f} / {t2:.3f}; gaps {g1:+.3f} / {g2:+.3f}; margin {m:+.3f} = {100*m/min(abs(t1),abs(t2)) if False else 100*m/ (t1 if g1<g2 else t2):+.1f}% of its threshold")
        return m, 100*m/(t1 if g1<g2 else t2)
    ranks = {}
    ranks["ratio"] = block("ratio", "ratio", ("cube", "cylinder", "triangle"))
    ranks["iso"] = block("isoperimetric", "iso", ("cylinder", "cube", "triangle"))
    for e in (0.02, 0.04, 0.06):
        k = f"nv{e}"
        print(f"\n[approxPolyDP eps={e}*perim] vertex-count histogram")
        for s in SHAPES:
            v, c = np.unique(F[s][k], return_counts=True); print(f"  {s:9} {dict(zip(v.tolist(), c.tolist()))}")
        exp = {"triangle": lambda v: v == 3, "cube": lambda v: v == 4, "cylinder": lambda v: v >= 5}
        acc = {s: np.mean(exp[s](F[s][k])) for s in SHAPES}
        print("  correct rate: " + ", ".join(f"{s} {a*100:.0f}%" for s, a in acc.items()))
        g1 = F["cylinder"][k].min() - F["cube"][k].max(); g2 = F["cube"][k].min() - F["triangle"][k].max()
        m = min(g1, g2); print(f"  thresholds 4.5 / 3.5; gaps(cyl-cube) {g1:+d} (cube-tri) {g2:+d}; margin {m:+d} vertices = {100*m/3.5:+.0f}%")
        ranks[k] = (float(m), 100*m/3.5)
    for how in ("max", "mean"):
        k3, k4 = f"h3{how}", f"h4{how}"
        print(f"\n[harmonics, radial {how}-per-bin, 72 bins]")
        for s in SHAPES: print(f"  {s:9} |c3|/|c0| {mmm(F[s][k3])}   |c4|/|c0| {mmm(F[s][k4])}")
        gT = F["triangle"][k3].min() - max(F["cube"][k3].max(), F["cylinder"][k3].max())
        tT = (F["triangle"][k3].min() + max(F["cube"][k3].max(), F["cylinder"][k3].max()))/2
        gC = F["cube"][k4].min() - F["cylinder"][k4].max()
        tC = (F["cube"][k4].min() + F["cylinder"][k4].max())/2
        # decision rule: h3>=tT -> triangle; else h4>=tC -> cube; else cylinder
        ok = {}
        for s in SHAPES:
            lab = np.where(F[s][k3] >= tT, "triangle", np.where(F[s][k4] >= tC, "cube", "cylinder"))
            ok[s] = np.mean(lab == s)
        m = min(gT, gC); pct = 100*m/(tT if gT < gC else tC)
        print(f"  tri-vs-rest on c3: thr {tT:.3f} gap {gT:+.3f};  cube-vs-cyl on c4: thr {tC:.3f} gap {gC:+.3f}")
        print(f"  margin {m:+.3f} = {pct:+.1f}% of its threshold; accuracy " + ", ".join(f"{s} {v*100:.0f}%" for s, v in ok.items()))
        ranks[f"harm-{how}"] = (m, pct)
    print("\nRANKING by margin % of threshold (best first):")
    for k, (m, p) in sorted(ranks.items(), key=lambda kv: -kv[1][1]): print(f"  {k:12} margin {m:+.3f}  ({p:+.1f}%)")
