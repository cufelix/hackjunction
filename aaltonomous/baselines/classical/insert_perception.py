"""Lid + hole pose perception for insert_shapes (rung 3). numpy only, no flag of its own (the caller gates it).

Method: model-based fit of the known lid (outline + three holes) to the head depth image.
  1. z_top: the plate top is the largest flat horizontal patch near the prior lid position (z histogram mode).
  2. Every depth pixel is intersected with the plane z = z_top; there the observed depth is classified PLATE (matches
     the plane), LOW (deeper: hole interior / bin / counter) or ignored (nearer: hands, objects; invalid; ambiguous).
     This is exact for an oblique camera (a hole shows the bin floor displaced in xy, but its depth class is right).
  3. A soft raster of the lid template (plate = 1, holes / outside = 0) is looked up at the plane point in lid coords;
     the pose (dx, dy, yaw) maximises sum(sign * (2 f - 1)) by coarse grid + pattern search (sub-mm).
  4. Confidence: per-axis curvature of the score (an axis the visible edges do not pin down, e.g. a lid cut by the
     image border or covered by a hand, gets low confidence) and the agreement fraction. Low -> None (caller falls
     back to the nominal scene.yaml layout, hole_targets()).
The lid geometry is a LidModel (default: Scene.lid_holes(), identical to assets/lid_shapes.stl); pass model= for
another lid (e.g. the printed 200 mm one).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import insert_shapes as ins

HOLE_OF = {"cube": "square", "cylinder": "circle", "triangle": "triangle"}


@dataclass
class HoleEstimate:
    shape: str
    xy: np.ndarray
    yaw: float
    conf: float


@dataclass
class LidEstimate:
    z_top: float
    centre_xy: np.ndarray
    yaw: float
    holes: dict = field(default_factory=dict)
    conf: float = 0.0
    sigma: tuple = (0.0, 0.0, 0.0)   # (x m, y m, yaw rad) rough 1-sigma from the score curvature
    info: dict = field(default_factory=dict)


@dataclass
class LidModel:
    """Lid in PELVIS axes around the lid centre: half = half side of the (square) plate, holes {shape: (centre (2,), polygon (N,2))}
    with polygon vertices relative to the lid centre; ref_yaw {shape: yaw of the hole's reference direction}."""
    half: float
    holes: dict
    ref_yaw: dict

    @classmethod
    def from_scene(cls, scene, half=0.105, offsets_mat=None):
        centre_mat = np.asarray(scene.cfg["bins"]["centers"][scene.lid_bin], float)
        c0 = np.asarray(scene.mat_to_pelvis(np.r_[centre_mat, 0.0])[:2], float)
        t = ins.hole_targets(scene, offsets_mat=offsets_mat)
        holes = {}
        for h in scene.lid_holes():
            s = h["shape"]
            c_mat = h["center"] if not (offsets_mat and s in offsets_mat) else centre_mat + np.asarray(offsets_mat[s], float)
            poly_mat = np.asarray(h["poly"], float) - np.asarray(h["center"], float) + c_mat
            poly = np.asarray(scene.mat_to_pelvis(poly_mat)[:, :2], float) - c0
            holes[s] = (np.asarray(t[s]["xy"], float) - c0, poly)
        return cls(half, holes, {s: t[s]["yaw"] for s in t})

    def raster(self, res=0.0005, pad=0.03):
        n = int(np.ceil((self.half + pad) / res)) * 2 + 1
        ax = (np.arange(n) - n // 2) * res
        X, Y = np.meshgrid(ax, ax, indexing="ij")
        f = (np.abs(X) <= self.half) & (np.abs(Y) <= self.half)
        for _, poly in self.holes.values():
            f &= ~_in_convex(X, Y, poly)
        f = f.astype(np.float32)
        k = 2   # ~1 mm box blur = anti-aliasing so the objective is smooth at sub-mm
        cs = np.pad(f, k, mode="edge")
        out = np.zeros_like(f)
        for i in range(-k, k + 1):
            for j in range(-k, k + 1):
                out += cs[k + i:k + i + n, k + j:k + j + n]
        return out / (2 * k + 1) ** 2, res, n // 2


def _in_convex(X, Y, poly):
    poly = np.asarray(poly, float)
    if np.cross(poly[1] - poly[0], poly[2] - poly[1]) < 0:
        poly = poly[::-1]
    ok = np.ones(X.shape, bool)
    for a, b in zip(poly, np.roll(poly, -1, axis=0)):
        ok &= (b[0] - a[0]) * (Y - a[1]) - (b[1] - a[1]) * (X - a[0]) >= 0
    return ok


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _camera(perception):
    cam = perception.cal.cams[perception.cam]
    return np.asarray(cam.K, float), perception.current_T(), float(getattr(cam, "depth_scale", 0.001))


def _prior(scene, prior, model):
    centre_mat = np.asarray(scene.cfg["bins"]["centers"][scene.lid_bin], float)
    c0 = np.asarray(scene.mat_to_pelvis(np.r_[centre_mat, 0.0])[:2], float)
    if prior is not None:
        return np.asarray(prior.centre_xy, float), float(prior.yaw), float(prior.z_top)
    z = float(scene.mat_to_pelvis([0.0, 0.0, scene.lid_z[1]])[2])
    return c0, 0.0, z


def nominal_lid(scene, model=None, conf=0.5):
    """The scene.yaml lid as a LidEstimate (the fallback when perception is not confident)."""
    model = model or LidModel.from_scene(scene)
    c0, yaw, z = _prior(scene, None, model)
    return _make(model, c0, yaw, z, conf, (0, 0, 0), {"nominal": True})


def _make(model, c, yaw, z, conf, sigma, info):
    R = _rot(yaw)
    holes = {}
    for s, (hc, _) in model.holes.items():
        holes[s] = HoleEstimate(s, np.asarray(c, float) + R @ hc, float(model.ref_yaw[s] + yaw), conf)
    return LidEstimate(float(z), np.asarray(c, float), float(yaw), holes, float(conf), sigma, info)


def _score_fn(px, py, sign, tpl):
    F, res, o = tpl
    n = F.shape[0]

    def score(dx, dy, yaw, cx, cy):
        R = _rot(-yaw)
        qx = R[0, 0] * (px - cx - dx) + R[0, 1] * (py - cy - dy)
        qy = R[1, 0] * (px - cx - dx) + R[1, 1] * (py - cy - dy)
        ix = np.rint(qx / res).astype(np.int64) + o
        iy = np.rint(qy / res).astype(np.int64) + o
        ok = (ix >= 0) & (ix < n) & (iy >= 0) & (iy < n)
        f = np.zeros(px.shape, np.float32)   # outside the raster: not plate
        f[ok] = F[ix[ok], iy[ok]]
        return float(np.sum(sign * (2.0 * f - 1.0)))

    return score


def estimate_lid(rgb, depth, perception, scene, prior=None, model=None, min_conf=0.6, search_m=0.035, search_deg=12.0,
                 level_to_counter=True):
    """Fit the lid to head depth (uint16 mm, 0 = invalid). Returns LidEstimate (pelvis frame) or None if not confident."""
    if depth is None:
        return None
    depth = np.asarray(depth)
    model = model or LidModel.from_scene(scene)
    c0, yaw0, z0 = _prior(scene, prior, model)
    K, T, dscale = _camera(perception)
    H, W = depth.shape[:2]
    u, v = np.meshgrid(np.arange(W), np.arange(H))
    rays = np.stack([(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones_like(u, float)], -1) @ T[:3, :3].T
    o = T[:3, 3]
    zobs = depth.astype(float) * dscale
    valid = depth > 0
    cam_dz = 0.0
    if level_to_counter:
        # the counter plane in the same frame is the height reference: a camera height / pitch error shifts the whole
        # scene (measured +7 mm in sim with the nominal calibration); remove it so the lid is relative to the counter.
        cam_dz = _counter_offset(o, rays, zobs, valid, scene)
        o = o + np.array([0.0, 0.0, -cam_dz])
    # window: pixels whose observed 3D point is within reach of the lid (+ search margin)
    P = o + rays * zobs[..., None]
    reach = model.half + search_m + 0.03
    win = valid & (np.abs(P[..., 0] - c0[0]) < reach + 0.05) & (np.abs(P[..., 1] - c0[1]) < reach + 0.05) \
        & (np.abs(P[..., 2] - z0) < 0.04)
    if win.sum() < 400:
        return None
    # 1. plate top height: mode of z over the central patch (1 mm bins)
    zz = P[..., 2][win & (np.abs(P[..., 0] - c0[0]) < model.half) & (np.abs(P[..., 1] - c0[1]) < model.half)]
    if zz.size < 300:
        return None
    hist, edges = np.histogram(zz, bins=np.arange(z0 - 0.04, z0 + 0.0405, 0.001))
    hs = np.convolve(hist, np.ones(3), mode="same")
    kmax = int(np.argmax(hs))
    sel = zz[(zz > edges[kmax] - 0.0015) & (zz < edges[kmax] + 0.0025)]
    z_top = float(np.median(sel))
    # 2. classify every pixel against the plane z = z_top
    with np.errstate(divide="ignore", invalid="ignore"):
        t = (z_top - o[2]) / rays[..., 2]
    ok = valid & np.isfinite(t) & (t > 0)
    pxy = o[:2] + t[..., None] * rays[..., :2]
    diff = zobs - t
    inwin = ok & (np.abs(pxy[..., 0] - c0[0]) < reach) & (np.abs(pxy[..., 1] - c0[1]) < reach)
    plate = inwin & (np.abs(diff) < 0.005)
    low = inwin & (diff > 0.012)
    use = plate | low
    sign = np.where(plate, 1.0, -1.0)[use].astype(np.float32)
    px, py = pxy[..., 0][use], pxy[..., 1][use]
    if sign.size < 400:
        return None
    tpl = model.raster()
    # coarse grid on a pixel subsample, then pattern search on all pixels
    sub = np.arange(sign.size) % 4 == 0
    sc_sub = _score_fn(px[sub], py[sub], sign[sub], tpl)
    sc = _score_fn(px, py, sign, tpl)
    best = (-1e18, 0, 0, 0)
    ds = np.arange(-search_m, search_m + 1e-9, 0.005)
    for ya in np.radians(np.arange(-search_deg, search_deg + 1e-9, 3.0)):
        for dx in ds:
            for dy in ds:
                s = sc_sub(dx, dy, yaw0 + ya, c0[0], c0[1])
                if s > best[0]:
                    best = (s, dx, dy, ya)
    _, dx, dy, ya = best
    ya += yaw0
    step = np.array([0.004, 0.004, np.radians(2.0)])
    cur = np.array([dx, dy, ya])
    s_cur = sc(*cur, c0[0], c0[1])
    while step[0] > 0.0002:
        improved = False
        for i in range(3):
            for sg in (1, -1):
                cand = cur.copy()
                cand[i] += sg * step[i]
                s = sc(*cand, c0[0], c0[1])
                if s > s_cur:
                    cur, s_cur, improved = cand, s, True
        if not improved:
            step *= 0.5
    dx, dy, yaw = cur
    # 3. confidence: curvature at +-2 mm / +-1 deg + agreement
    n = float(sign.size)
    agree = (s_cur / n + 1.0) / 2.0   # fraction of agreeing pixel votes (soft)
    h = np.array([0.002, 0.002, np.radians(1.0)])
    curv = []
    for i in range(3):
        a, b = cur.copy(), cur.copy()
        a[i] += h[i]
        b[i] -= h[i]
        curv.append(max(s_cur - 0.5 * (sc(*a, c0[0], c0[1]) + sc(*b, c0[0], c0[1])), 1e-6))
    curv = np.array(curv)
    # votes lost per axis for a shift of h; a pose axis is pinned when >= ~25 votes are lost (visible edge length)
    sigma = tuple(float(h[i] * np.sqrt(25.0 / curv[i]) / 2.0) for i in range(3))
    pin = float(np.clip(np.min(curv) / 40.0, 0.0, 1.0))
    conf = float(np.clip((agree - 0.80) / 0.15, 0.0, 1.0) * pin)
    info = {"cam_dz": float(cam_dz), "agree": float(agree), "curv": curv.tolist(), "n": int(n), "dxy": (float(dx), float(dy))}
    c = c0 + np.array([dx, dy])
    est = _make(model, c, yaw, z_top, conf, sigma, info)
    return est if conf >= min_conf else None


def _counter_offset(o, rays, zobs, valid, scene):
    """Measured counter z minus scene.counter_z (m), from the mode of the low points in the object band; 0 if unsure."""
    P = o + rays * zobs[..., None]
    m = valid & (P[..., 0] > 0.22) & (P[..., 0] < 0.42) & (np.abs(P[..., 1]) < 0.25) & (np.abs(P[..., 2] - scene.counter_z) < 0.02)
    z = P[..., 2][m]
    if z.size < 2000:
        return 0.0
    hist, edges = np.histogram(z, bins=np.arange(scene.counter_z - 0.02, scene.counter_z + 0.0201, 0.001))
    k = int(np.argmax(hist))
    sel = z[(z > edges[k] - 0.001) & (z < edges[k] + 0.002)]
    return float(np.median(sel) - scene.counter_z) if sel.size > 1000 else 0.0


def to_targets(lid, scene, min_conf=0.7):
    """HoleTarget dicts (as insert_shapes.hole_targets) from a LidEstimate; holes below min_conf omitted."""
    base = ins.hole_targets(scene, z_lid_top=lid.z_top)
    out = {}
    for s, h in lid.holes.items():
        if h.conf < min_conf or s not in base:
            continue
        d = dict(base[s])
        d["xy"] = np.asarray(h.xy, float)
        d["yaw"] = float(h.yaw)
        out[s] = d
    return out
