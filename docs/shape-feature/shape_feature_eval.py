#!/usr/bin/env python3
"""Shape-feature eval on a real dataset. numpy + cv2 only.  Usage:
    python3 -I shape_feature_eval.py shape_features_dataset.npz
    python3 -I shape_feature_eval.py --selftest
Feature = convex-hull area / minAreaRect area. Phase 0 schema, 1 bands+accuracy, 2 border truncation, 3 summary.
Constants from scripts/measure_shape_feature.py: top_percentile 92, top_band 0.006 m, min_valid_depth_px 25.
(erode_px 2 acts on a pixel mask before deprojection; it cannot be replayed on a bare cloud, so it is skipped.)"""
import io, sys, contextlib
import numpy as np
import cv2

NAN = float("nan")
SHAPES = ("cube", "cylinder", "triangle")
TOP_PCT, TOP_BAND, MIN_PTS = 92, 0.006, 25
SYN = {"blob": {"cube": (0.806, 0.982), "cylinder": (0.823, 0.853), "triangle": (0.624, 0.730)},
       "met": {"cube": (0.942, 0.987), "cylinder": (0.783, 0.828), "triangle": (0.542, 0.608)}}
SRC = {"blob": "2D blob", "met": "metric top-face"}
PAIRS = {"deployed .66/.88": (0.66, 0.88), "midpoint .70/.90": (0.70, 0.90)}  # (cyl_min, cube_min)
LOW = 0.66

# ---------- feature ----------
def ratio_of(pts):
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    p = p[np.isfinite(p).all(1)]
    if len(p) < 3:
        return NAN
    hull = cv2.convexHull(p)
    (_, _), (w, h), _ = cv2.minAreaRect(hull)
    return float(cv2.contourArea(hull) / (w * h)) if w * h > 1e-12 else NAN

def max_angle(pts):
    """Largest vertex angle (deg) of the min-area ENCLOSING TRIANGLE of the hull.

    Second-stage discriminator for the ratio < 0.66 branch only, where hull/minAreaRect is ~0.5 for
    EVERY triangle and so cannot tell an equilateral prism face from a corner-cut cube. Geometry:
    an equilateral face is 60 deg; any corner-cut of a rectangle leaves a right triangle, >= 90 deg.
    Synthetic (ideal polygons + 0.4 mm rms xy): prism 61.8, prism with a corner clipped 66.4,
    cube cut on the diagonal 90.4, cube cut at 70% of the diagonal 90.6. Meaningless when the region
    is not triangular, hence the ratio gate.
    """
    p = np.asarray(pts, np.float32).reshape(-1, 2)
    p = p[np.isfinite(p).all(1)]
    if len(p) < 3:
        return NAN
    try:
        _, tri = cv2.minEnclosingTriangle(cv2.convexHull(p))
        if tri is None:
            return NAN
        T = np.asarray(tri, np.float64).reshape(3, 2)
    except cv2.error:
        return NAN
    out = []
    for i in range(3):
        v1, v2 = T[(i + 1) % 3] - T[i], T[(i + 2) % 3] - T[i]
        n = np.linalg.norm(v1) * np.linalg.norm(v2)
        if n < 1e-12:
            return NAN
        out.append(np.degrees(np.arccos(np.clip(np.dot(v1, v2) / n, -1.0, 1.0))))
    return float(max(out))

def touches_border(st, hw, b=1):  # same convention as policy/shape_from_top.py
    x, y, w, h = (int(st[k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
    return x < b or y < b or x + w > hw[1] - b or y + h > hw[0] - b

def blob_feature(m):
    """largest connected component of mask -> (ratio, touches_border) or None"""
    m = np.asarray(m)
    if m.ndim == 3 and m.shape[-1] in (1, 3):
        m = m[..., 0]
    if m.ndim != 2:
        return None
    n, lab, st, _ = cv2.connectedComponentsWithStats((m > 0).astype(np.uint8), connectivity=8)
    if n < 2:
        return None
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    ys, xs = np.nonzero(lab == j)
    return ratio_of(np.c_[xs, ys]), touches_border(st[j], m.shape)

def top_xy(c, pre):
    c = np.asarray(c, float)
    if c.ndim != 2 or c.shape[1] < 2:
        return None
    c = c[np.isfinite(c).all(1)]
    c = c[np.abs(c).sum(1) > 0]
    if len(c) < 5:
        return None
    if np.median(np.abs(c)) > 5:  # looks like mm
        c = c / 1000.0
    if not pre:
        if c.shape[1] < 3 or len(c) < MIN_PTS:
            return None
        c = c[c[:, 2] >= np.percentile(c[:, 2], TOP_PCT) - TOP_BAND]  # assumes z up (pelvis frame)
        if len(c) < 5:
            return None
    return c[:, :2]

# ---------- phase 0 ----------
def load(path):
    try:
        z = np.load(path, allow_pickle=True)
    except Exception as e:
        print(f"cannot load {path!r}: {e!r}")
        return None
    D = {}
    if isinstance(z, np.ndarray):
        D["<array>"] = z
    else:
        for k in z.files:
            try:
                D[k] = z[k]
            except Exception as e:
                print(f"  key {k!r}: UNREADABLE {e!r}")
    for k, v in list(D.items()):  # unwrap a pickled dict stored as 0-d object
        if v.dtype == object and v.ndim == 0 and isinstance(v.item(), dict):
            del D[k]
            for kk, vv in v.item().items():
                try:
                    D[f"{k}.{kk}"] = np.asarray(vv)
                except Exception:
                    o = np.empty(len(vv), object)
                    for i, x in enumerate(vv):
                        o[i] = x
                    D[f"{k}.{kk}"] = o
    return D

def phase0(D):
    print("== PHASE 0: schema ==")
    for k, v in D.items():
        s = f"  {k!r}: shape={v.shape} dtype={v.dtype}"
        try:
            if v.dtype == object and v.size:
                e = v.flat[0]
                s += f" first={type(e).__name__}" + (f" shape={e.shape} dtype={e.dtype}" if hasattr(e, "shape") else f" repr={repr(e)[:70]}")
            elif v.size <= 8:
                s += f" value={v.tolist()}"
            else:
                s += f" sample={v.reshape(-1)[:5].tolist()}"
                if v.dtype.kind in "fiu":
                    f = v[np.isfinite(v)] if v.dtype.kind == "f" else v
                    s += f" min/max={f.min():.4g}/{f.max():.4g}" if f.size else ""
            if v.ndim == 1 and v.size and v.dtype.kind in "USiubO" and not hasattr(v[0], "shape"):
                u, c = np.unique(v, return_counts=True)
                if len(u) <= 8:
                    s += f" counts={dict(zip(u.tolist(), c.tolist()))}"
        except Exception as e:
            s += f" (inspect error {e!r})"
        print(s)
    if not D:
        print("  (empty)")

# ---------- detection ----------
def first_elem(v):
    return v.flat[0] if v.dtype == object and v.size else None

def is_labels(v):
    if v.ndim != 1 or v.size < 3 or v.dtype.kind in "bf":
        return False
    return len(np.unique(v)) <= 8

def is_mask(v):
    e = first_elem(v)
    if e is not None:
        e = np.asarray(e)
        return e.ndim == 2 and min(e.shape) >= 8 and len(np.unique(e)) <= 3
    return v.dtype != object and v.ndim == 3 and min(v.shape[1:]) >= 8 and len(np.unique(v[0])) <= 3

def is_cloud(v):
    e = first_elem(v)
    if e is not None:
        e = np.asarray(e)
        return e.dtype.kind == "f" and e.ndim == 2 and 2 <= e.shape[1] <= 6 and e.shape[0] >= 5
    return v.dtype.kind == "f" and v.ndim == 3 and 2 <= v.shape[2] <= 6 and v.shape[1] >= 5

def is_flag(v):
    return v.ndim == 1 and v.dtype.kind in "biu" and v.size >= 3

NAMES = {"lab": ["label", "labels", "y", "shape_label", "shape", "shapes", "class", "classes", "cls", "target", "gt"],
         "mask": ["mask", "masks", "blob", "blobs", "seg", "silhouette"],
         "top": ["top_points", "top_face", "top_xy", "top_pts", "topface", "top"],
         "cloud": ["cloud", "clouds", "points", "pts", "pc", "pcd", "xyz", "point_cloud", "pointclouds"],
         "touch": ["touches_border", "touch_border", "border", "clipped", "on_border", "truncated", "touch"]}
PRED = {"lab": is_labels, "mask": is_mask, "top": is_cloud, "cloud": is_cloud, "touch": is_flag}

def pick(D, role, used):
    low = {k.lower(): k for k in D}
    cands = [(low[n], "name") for n in NAMES[role] if n in low]
    cands += [(lk, "name~") for n in NAMES[role] if len(n) >= 4 for lk in low.values() if n in lk.lower() and lk.lower() != n]
    if role not in ("touch", "top"):  # top face only by name; any other cloud is derived
        cands += [(k, "heuristic") for k in D]
    for k, how in cands:
        if k in used:
            continue
        try:
            if PRED[role](D[k]):
                used.add(k)
                return k, how
        except Exception:
            pass
    return None, None

def lab_names(D, v):
    if v.dtype.kind in "USO":
        out = []
        for x in v:
            x = x.decode() if isinstance(x, bytes) else str(x)
            xl = x.lower()
            out.append(next((s for s in SHAPES if xl.startswith(s[:3])), x))
        return out, "strings"
    for k, nm in D.items():  # a names array like ['cube','cylinder','triangle']
        if nm.ndim == 1 and nm.dtype.kind in "USO" and 2 <= nm.size <= 6 and k.lower() in ("classes", "class_names", "label_names", "names", "shape_names"):
            names = [str(x).lower() for x in nm]
            return [next((s for s in SHAPES if names[int(i)].startswith(s[:3])), names[int(i)]) if 0 <= int(i) < len(names) else "?" for i in v], f"ints via {k!r}"
    u = sorted(set(int(i) for i in v))
    return [SHAPES[int(i)] if 0 <= int(i) < 3 else "?" for i in v], f"ASSUMED int 0/1/2 = cube/cylinder/triangle (ints seen: {u})"

# ---------- stats ----------
def classify(r, pair):
    cy, cu = pair
    return "cube" if r >= cu else "cylinder" if r >= cy else "triangle"

def vals(recs, s, src, touch=None):
    return np.array([r[src] for r in recs if r["lab"] == s and np.isfinite(r[src]) and (touch is None or r["touch"] is touch)])

def acc(v, s, pair):
    g = [classify(x, pair) for x in v]
    wr = {}
    for x in g:
        if x != s:
            wr[x] = wr.get(x, 0) + 1
    return sum(x == s for x in g) / max(len(g), 1), wr

def fmt_wr(wr):
    return "->" + ",".join(f"{k} {n}" for k, n in sorted(wr.items())) if wr else "->none"

def table(recs, src):
    print(f"  [{SRC[src]}]  (synthetic band alongside)")
    for s in SHAPES:
        v = vals(recs, s, src)
        if not len(v):
            print(f"   {s:9} no samples")
            continue
        lo, hi = SYN[src][s]
        ov = v.min() <= hi and v.max() >= lo
        inb = float(np.mean((v >= lo) & (v <= hi)))
        out = f"   {s:9} n={len(v):3} {v.min():.3f}-{v.max():.3f} med {np.median(v):.3f} | synth {lo:.3f}-{hi:.3f} overlap={'yes' if ov else 'NO'} in-band {inb:.0%}"
        print(out)
        for nm, p in PAIRS.items():
            a, wr = acc(v, s, p)
            print(f"        {nm}: {a:6.1%} {fmt_wr(wr)}")

# ---------- analysis ----------
def analyze(D):
    phase0(D)
    print("\n== PHASE 1: feature comparison ==")
    used, K = set(), {}
    for role in ("lab", "touch", "mask", "top", "cloud"):
        try:
            K[role] = pick(D, role, used)
        except Exception as e:
            K[role] = (None, None)
            print(f"  detect {role} failed: {e!r}")
    for role, (k, how) in K.items():
        print(f"  {role:6} -> {k!r} ({how})" if k else f"  {role:6} -> not found")
    kl = K["lab"][0]
    srcs = {r: D[K[r][0]] for r in ("mask", "top", "cloud") if K[r][0]}
    if not kl or not any(r in srcs for r in ("mask", "top", "cloud")):
        print(f"  Phase 1 needs per-sample labels AND (masks or clouds). Found labels={kl!r}, masks={K['mask'][0]!r}, "
              f"top={K['top'][0]!r}, clouds={K['cloud'][0]!r}. Other (N,3)-like arrays are not usable per-sample.")
        print("  Phases 1-2 skipped: schema above is all this run can offer.")
        return phase3(None, {}, None)
    L, how = lab_names(D, D[kl])
    print(f"  labels: {how}; counts { {s: L.count(s) for s in sorted(set(L))} }")
    N = len(L)
    recs = [dict(lab=L[i], blob=NAN, met=NAN, ang=NAN, touch=None) for i in range(N)]
    if K["touch"][0] and len(D[K["touch"][0]]) == N:
        for i, t in enumerate(D[K["touch"][0]]):
            recs[i]["touch"] = bool(t)
    errs = 0
    for role, arr in srcs.items():
        if len(arr) != N:
            print(f"  skip {role}: length {len(arr)} != N labels {N}")
            continue
        for i in range(N):
            try:
                if role == "mask":
                    f = blob_feature(arr[i])
                    if f:
                        recs[i]["blob"], recs[i]["touch"] = f
                else:
                    xy = top_xy(arr[i], role == "top")
                    if xy is not None:
                        recs[i]["met"] = ratio_of(xy)
                        recs[i]["ang"] = max_angle(xy)
            except Exception:
                errs += 1
    if errs:
        print(f"  {errs} per-sample errors skipped")
    have = [s for s in ("blob", "met") if any(np.isfinite(r[s]) for r in recs)]
    if "met" in have and K["top"][0] is None:
        print(f"  note: top face DERIVED from cloud (pct {TOP_PCT}, band {TOP_BAND} m, >= {MIN_PTS} pts; assumes pelvis frame, z up, metres or mm; no erosion)")
    if not have:
        print("  Phase 1: arrays found but no sample produced a ratio.")
    for s in have:
        table(recs, s)
    head = phase2(recs, have)
    return phase3(recs, have, head)

def phase2(recs, have):
    print("\n== PHASE 2: border truncation ==")
    known = [r for r in recs if r["touch"] is not None]
    if not known:
        print("  Phase 2 needs per-sample masks (for connectedComponentsWithStats) or a boolean border/clipped array; found neither.")
        return None
    if all(r["touch"] for r in known):
        print("  WARNING: 100% touch the border -- masks may be cropped to the bbox, so border test is meaningless")
    head = {}
    for s in SHAPES:
        t = [r["touch"] for r in known if r["lab"] == s]
        print(f"  {s:9} border-touch {sum(t)}/{len(t)}" + (f" = {sum(t)/len(t):.0%}" if t else ""))
    for src in have:
        print(f"  [{SRC[src]}] ratio split by touch  (deployed .66/.88 accuracy)")
        for s in SHAPES:
            for tv, nm in ((True, "touched"), (False, "clear")):
                v = vals(recs, s, src, tv)
                if len(v):
                    a, wr = acc(v, s, PAIRS["deployed .66/.88"])
                    print(f"   {s:9} {nm:8} n={len(v):3} {v.min():.3f}-{v.max():.3f} med {np.median(v):.3f} acc {a:.0%} {fmt_wr(wr)}")
        cu = [r for r in known if r["lab"] == "cube" and np.isfinite(r[src])]
        low = [r for r in cu if r[src] < LOW]
        tl = sum(r["touch"] for r in low)
        nt, nu = sum(r["touch"] for r in cu), sum(not r["touch"] for r in cu)
        head[src] = dict(n=len(cu), low=len(low), tl=tl, rt=sum(r["touch"] for r in cu if r[src] < LOW) / max(nt, 1), ru=(len(low) - tl) / max(nu, 1))
        print(f"  HEADLINE [{SRC[src]}]: cubes < {LOW}: {len(low)}/{len(cu)}; touched border: {tl}/{len(low)}")
    return head

def verdict(h):
    if not h:
        return "insufficient data to tell (no border info or no ratios)"
    if h["n"] == 0:
        return "insufficient data to tell (no labelled cubes with a ratio)"
    if h["low"] < 3:
        return f"insufficient data to tell ({h['low']} cube(s) < {LOW}; cube->triangle not reproduced here)"
    f = h["tl"] / h["low"]
    if f >= 0.7:
        return f"data supports CLIPPING ({f:.0%} of low cubes touched the border)"
    if f <= 0.3:
        return f"data supports NOT clipping ({f:.0%} of low cubes touched the border)"
    return f"MIXED: {f:.0%} of low cubes touched the border"

def phase3(recs, have, head):
    L = ["", "== PHASE 3: SUMMARY (paste-ready) =="]
    if recs is None or not have:
        L.append("no ratios computed: dataset lacks per-sample labels and masks/clouds (see Phase 0 schema)")
    else:
        L.append("shape      n | " + " | ".join(f"{SRC[s]} min-max med acc(.66/.88, .70/.90)" for s in have))
        for sh in SHAPES:
            cells = []
            for s in have:
                v = vals(recs, sh, s)
                if len(v):
                    a1, a2 = (acc(v, sh, p)[0] for p in PAIRS.values())
                    cells.append(f"{v.min():.3f}-{v.max():.3f} {np.median(v):.3f} {a1:.0%}, {a2:.0%}")
                else:
                    cells.append("n/a")
            L.append(f"{sh:9} {sum(r['lab'] == sh for r in recs):3} | " + " | ".join(cells))
        L.append("synthetic: blob cube .806-.982 cyl .823-.853 tri .624-.730; metric cube .942-.987 cyl .783-.828 tri .542-.608")
        if head:
            for s, h in head.items():
                L.append(f"cubes < {LOW} [{SRC[s]}]: {h['low']}/{h['n']}; touched border {h['tl']}/{h['low']} (low-rate touched {h['rt']:.0%}, clear {h['ru']:.0%})")
        else:
            L.append(f"cubes < {LOW}: not computable (no border info)")
        ang_tri = np.array([r["ang"] for r in recs if r["lab"] == "triangle" and np.isfinite(r["ang"])])
        ang_low = np.array([r["ang"] for r in recs
                            if r["lab"] == "cube" and np.isfinite(r["ang"]) and r["met"] < LOW])
        if len(ang_tri) and len(ang_low):
            sep = "SEPARATES" if np.median(ang_low) - np.median(ang_tri) > 15 else "DOES NOT separate"
            L.append(f"max-angle (ratio<{LOW} branch): true prisms med {np.median(ang_tri):.1f} deg "
                     f"(n={len(ang_tri)}) vs cubes<{LOW} med {np.median(ang_low):.1f} deg "
                     f"(n={len(ang_low)}) -> {sep}; synthetic said 61.8 vs 90.4")
        elif len(ang_tri):
            L.append(f"max-angle: true prisms med {np.median(ang_tri):.1f} deg (n={len(ang_tri)}); "
                     f"no cube fell below {LOW}, so the discriminator is untested here")
        else:
            L.append("max-angle: not computable (needs metric top-face points)")
        L.append("VERDICT: " + verdict(head.get("blob") or head.get("met") if head else None))
    print("\n".join(L[:16]))

# ---------- selftest ----------
def selftest():
    g = np.linspace(-1, 1, 41)
    sq = np.array([(x, y) for x in g for y in g])
    t = np.linspace(0, 2 * np.pi, 720, endpoint=False)
    ci = np.c_[np.cos(t), np.sin(t)]
    V = np.array([[0, 1], [-np.sqrt(3) / 2, -.5], [np.sqrt(3) / 2, -.5]])
    tri = np.vstack([V[i] + np.linspace(0, 1, 100)[:, None] * (V[(i + 1) % 3] - V[i]) for i in range(3)])
    rs, rc, rt = ratio_of(sq), ratio_of(ci), ratio_of(tri)
    print(f"square {rs:.3f}  circle {rc:.3f}  triangle {rt:.3f}")
    assert abs(rs - 1) < 1e-3 and abs(rc - np.pi / 4) < 5e-3 and abs(rt - .5) < 1e-3
    def diamond(cx):
        m = np.zeros((120, 120), np.uint8)
        cv2.fillPoly(m, [np.array([[cx, 32], [cx + 28, 60], [cx, 88], [cx - 28, 60]], np.int32)], 1)
        return m
    full, clip = blob_feature(diamond(60)), blob_feature(diamond(112))
    print(f"diamond full {full[0]:.3f} touch={full[1]} | clipped {clip[0]:.3f} touch={clip[1]}")
    assert not full[1] and clip[1] and clip[0] < full[0] - 0.03
    rng = np.random.default_rng(0)
    cloud = np.vstack([np.c_[rng.uniform(-.02, .02, (2000, 2)), np.full(2000, .045)],
                       np.c_[rng.uniform(-.02, .02, (1000, 2)), rng.uniform(0, .04, 1000)]])
    rd = ratio_of(top_xy(cloud, False))
    print(f"derived top face of a box cloud {rd:.3f}")
    assert rd > .95
    D = {"labels": np.array(["cube", "cylinder", "triangle"] * 2), "masks": np.stack([diamond(60), diamond(112)] * 3)}
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        analyze(D)
        analyze({"junk": np.arange(5), "obj": np.empty(2, object)})
    assert "SUMMARY" in buf.getvalue()
    print("analyze() smoke test on tiny + junk schemas: no crash")
    print("selftest ok")

if __name__ == "__main__":
    a = sys.argv[1:]
    if "--selftest" in a:
        selftest()
    elif not a:
        print("usage: python3 -I shape_feature_eval.py <shape_features_dataset.npz>\n       python3 -I shape_feature_eval.py --selftest")
        sys.exit(2)
    else:
        D = load(a[0])
        if D is None:
            sys.exit(1)
        try:
            analyze(D)
        except Exception as e:  # never die without the schema having printed
            print(f"\nanalysis aborted: {e!r}")
