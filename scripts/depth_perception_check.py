#!/usr/bin/env python3
"""Offline replica of the starter repo's classical head perception (classical/perception.py).

    python3 depth_perception_check.py <dir with head_seed<N>[_depth].png> [--repo PATCHWORK] | --demo

Per object: colour, px, n_top, METRIC extent ratio (hull area / minAreaRect area of top-band xy), path, pelvis xyz, and
the bias column (mean of top-band points minus minAreaRect centre). Constants come from the repo YAML; stdlib+numpy+cv2.
"""
from __future__ import annotations

import math
import os
import sys
import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np

REPO = Path(os.environ.get("PATCHWORK", "/tmp/claude-1000/-home-felix-projects-hackjunction/"
                           "811dcd44-f22a-4c64-9008-914b85c21b0a/scratchpad/patchwork"))
SEEDS = range(101, 113)
# literals hardcoded in the real code (not in config.yaml)
CORE_MIN_FRAC = 0.25      # perception.py:80  eroded core must keep >= 25 % of the blob else use the blob
KEEP_H_MIN = 0.25         # perception.py:114 keep-mask lower height bound, in object sizes
MIN_TOP_RAW = 5           # perception.py:142 _from_depth gives up (-> plane) below 5 top points
PLANE_AIM = 0.25          # perception.py:161 plane fallback aims 0.25 heights below the top
MIN_TOP_POINTS = 30       # policy/shape_from_top.py:62 ratio needs >= 30 top points, else ABSTAIN
PLANE_BINS = 5            # --plane: radial-distance bins for the residual table
PLANE_DILATE_PX = 9       # --plane: colour blobs are grown by this many px so blob edges never count as counter
PLANE_FLAT_MM = 5.0       # --plane self-test: "flat" = rms below this
PLANE_BOWL_MM = 10.0      # --plane self-test: wrong convention must exceed this rms (measured ~17-19; flat ~0.3)
HIST_EDGES = np.arange(0.50, 1.0001, 0.05)
EXPECT = dict(fx=604.827, fy=604.298, cx=325.042, cy=250.927, ty=0.017530)   # parse-failure tripwire

# ------------------------------------------------------------------ tiny YAML subset reader
def _strip(line: str) -> str:
    q = None
    for i, ch in enumerate(line):
        if q:
            q = None if ch == q else q
        elif ch in "\"'":
            q = ch
        elif ch == "#" and (i == 0 or line[i - 1].isspace()):
            return line[:i].rstrip()
    return line.rstrip()

def _scalar(s: str):
    s = s.strip()
    if s in ("null", "~", ""):
        return None
    if s in ("true", "false"):
        return s == "true"
    try:
        return float(s) if any(ch in s for ch in ".eE") else int(s)
    except ValueError:
        return s.strip("\"'")

def _split(s: str) -> list[str]:
    parts, depth, cur = [], 0, ""
    for ch in s:
        depth += (ch in "[{") - (ch in "]}")
        parts, cur = (parts + [cur], "") if ch == "," and depth == 0 else (parts, cur + ch)
    return [p for p in parts + [cur] if p.strip()]

def _value(s: str):
    s = s.strip()
    if s.startswith("["):
        return [_value(p) for p in _split(s[1:-1])]
    if s.startswith("{"):
        kv = (p.partition(":") for p in _split(s[1:-1]))
        return {k.strip().strip("\"'"): _value(v) for k, _, v in kv}
    return _scalar(s)

def _block(lines: list[tuple[int, str]], i: int, ind: int):
    if lines[i][1].startswith("- "):
        out = []
        while i < len(lines) and lines[i][0] == ind and lines[i][1].startswith("- "):
            out.append(_value(lines[i][1][2:]))
            i += 1
        return out, i
    out = {}
    while i < len(lines) and lines[i][0] == ind:
        k, _, rest = lines[i][1].partition(":")
        i += 1
        if rest.strip():
            out[k.strip()] = _value(rest)
        elif i < len(lines) and lines[i][0] > ind:
            out[k.strip()], i = _block(lines, i, lines[i][0])
        else:
            out[k.strip()] = None
    return out, i

def parse_yaml(text: str) -> dict:
    lines = [(len(ln) - len(ln.lstrip()), _strip(ln).strip()) for ln in text.splitlines() if _strip(ln).strip()]
    return _block(lines, 0, lines[0][0])[0]

# ------------------------------------------------------------------ context (calibration + config + scene)
@dataclass(frozen=True)
class Ctx:
    K: np.ndarray            # 3x3
    T: np.ndarray            # T_pelvis_optical 4x4
    scale: float             # depth_scale (png units -> m)
    cfg: dict                # config.yaml perception block
    scene: dict
    origin: np.ndarray       # mat_origin_in_pelvis
    height: float            # object height = size * height_ratio  (scene.py:105-107)

def load_ctx(repo: Path, cal_path: Path | None = None) -> Ctx:
    root = repo / "aaltonomous"
    cal_path = cal_path or root / "interface/calibration.yaml"
    cam = parse_yaml(cal_path.read_text())["cameras"]["head"]
    i = cam["intrinsics"]
    K = np.array([[i["fx"], 0, i["cx"]], [0, i["fy"], i["cy"]], [0, 0, 1]], float)
    T = np.array(cam["T_pelvis_optical"], float)
    got = dict(fx=K[0, 0], fy=K[1, 1], cx=K[0, 2], cy=K[1, 2], ty=T[1, 3])
    assert T.shape == (4, 4) and all(abs(got[k] - v) < 1e-6 for k, v in EXPECT.items()), \
        f"calibration parse mismatch in {cal_path}: {got}"
    cfg = parse_yaml((root / "baselines/classical/config.yaml").read_text())["perception"]
    scene = parse_yaml((root / "scene/scene.yaml").read_text())
    o = scene["objects"]
    return Ctx(K, T, float(cam.get("depth_scale", 0.001)), cfg, scene,
               np.asarray(scene["counter"]["mat_origin_in_pelvis"], float), float(o["size"]) * float(o.get("height_ratio", 1.0)))

def to_pelvis(c: Ctx, u, v, d_mm) -> np.ndarray:           # calibration.py:62-73
    z = np.asarray(d_mm, float) * c.scale
    p = np.stack([(np.asarray(u, float) - c.K[0, 2]) / c.K[0, 0] * z,
                  (np.asarray(v, float) - c.K[1, 2]) / c.K[1, 1] * z, z], -1)
    return p @ c.T[:3, :3].T + c.T[:3, 3]

def project(c: Ctx, P: np.ndarray) -> np.ndarray:           # calibration.py:75-84
    p = (np.asarray(P, float) - c.T[:3, 3]) @ c.T[:3, :3]
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.stack([c.K[0, 0] * p[..., 0] / p[..., 2] + c.K[0, 2],
                         c.K[1, 1] * p[..., 1] / p[..., 2] + c.K[1, 2], p[..., 2]], -1)

def pelvis_to_mat(c: Ctx, P) -> np.ndarray:                 # scene.py:79-81
    d = np.asarray(P, float) - c.origin
    return np.stack([d[..., 1], d[..., 0], d[..., 2]], -1)

def mat_to_pelvis(c: Ctx, p) -> np.ndarray:                 # scene.py:73-77
    p = np.asarray(p, float)
    return c.origin + np.stack([p[..., 1], p[..., 0], p[..., 2]], -1)

# ------------------------------------------------------------------ masks (perception.py:47-136)
def colour_masks(c: Ctx, rgb: np.ndarray) -> dict[str, np.ndarray]:
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    out, k = {}, int(c.cfg["open_px"])
    for color, ranges in c.cfg["hsv"].items():
        m = np.zeros(hsv.shape[:2], np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        out[color] = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8)) if k > 0 else m
    return out

def bin_half(c: Ctx) -> np.ndarray:
    return np.asarray(c.scene["bins"]["outer_size"], float) / 2 + c.cfg["bin_pixel_margin"]

def bin_hulls_mask(c: Ctx, shape: tuple[int, int]) -> np.ndarray:
    keep, b, half = np.full(shape, 255, np.uint8), c.scene["bins"], bin_half(c)
    for cx, cy in b["centers"].values():
        corners = np.array([[cx + sx * half[0], cy + sy * half[1], z] for sx in (-1, 1) for sy in (-1, 1)
                            for z in (0.0, float(b["wall_height"]))])
        uvz = project(c, mat_to_pelvis(c, corners))
        if np.any(uvz[:, 2] <= 0):
            continue
        cv2.fillConvexPoly(keep, cv2.convexHull(np.round(uvz[:, :2]).astype(np.int32)), 0)
    return keep

def keep_mask(c: Ctx, depth: np.ndarray) -> np.ndarray:
    """255 = pixel may belong to an object. Rejection path 1: 3D bin footprint + bin_pixel_margin (+ height gate)."""
    keep = bin_hulls_mask(c, depth.shape)
    dlo, dhi = c.cfg["depth_range_mm"]
    d = depth.astype(float)
    valid = (d >= dlo) & (d <= dhi)
    if valid.mean() < c.cfg["min_valid_frac"]:
        return keep                       # (almost) no depth: only projected hulls can be removed
    keep[:] = 255
    vv, uu = np.nonzero(valid)
    P = to_pelvis(c, uu, vv, d[vv, uu])
    h = (P[:, 2] - c.origin[2]) / c.height
    pm = pelvis_to_mat(c, P)
    ok = (h >= KEEP_H_MIN) & (h <= c.cfg["height_range"][1])
    for ctr in c.scene["bins"]["centers"].values():
        ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= bin_half(c), axis=1)
    keep[vv[~ok], uu[~ok]] = 0
    return keep

# ------------------------------------------------------------------ per-object estimate
@dataclass(frozen=True)
class Obj:
    color: str
    area: int
    n_depth: int
    n_top: int
    ratio: float
    path: str                 # metric | plane | abstain
    pos: np.ndarray           # pelvis frame, m (what perception.py would return as Detection.top)
    off_mm: float             # |mean(top xy) - minAreaRect centre| in mm
    off_xy: tuple[float, float]
    cos_cam: float            # cosine between that offset and the direction toward the camera (+1 = pulled to camera)
    why: str | None = None    # rejection reason; None = accepted
    seed: str = ""

def extent_ratio(xy: np.ndarray) -> float:                   # policy/shape_from_top.py:classify_top_face
    pts = xy.astype(np.float32)
    if len(pts) < MIN_TOP_POINTS:
        return float("nan")
    hull = cv2.convexHull(pts)
    (_, _), (w, h), _ = cv2.minAreaRect(hull)
    area, rect = float(cv2.contourArea(hull)), float(w) * float(h)
    return area / rect if area > 0 and rect > 0 else float("nan")

def bias(c: Ctx, xy: np.ndarray) -> tuple[float, tuple[float, float], float]:
    (rx, ry), _, _ = cv2.minAreaRect(xy.astype(np.float32))
    off = xy.mean(0) - np.array([rx, ry])
    to_cam = c.T[:2, 3] - xy.mean(0)
    cos = float(off @ to_cam / (np.linalg.norm(off) * np.linalg.norm(to_cam) + 1e-12))
    return float(np.linalg.norm(off) * 1e3), (float(off[0] * 1e3), float(off[1] * 1e3)), cos

def from_depth(c: Ctx, color: str, u, v, d, area: int) -> Obj | None:      # perception.py:138-150
    P = to_pelvis(c, u, v, d)
    z_top = np.percentile(P[:, 2], c.cfg["top_percentile"])
    sel = P[:, 2] >= z_top - c.cfg["top_band"]
    if sel.sum() < MIN_TOP_RAW:
        return None
    top, w = P[sel], (d[sel] / 1000.0) ** 2
    xy = (top[:, :2] * w[:, None]).sum(0) / w.sum() + np.asarray(c.cfg["xy_offset"] or [0, 0], float)
    ratio = extent_ratio(top[:, :2])
    off, off_xy, cos = bias(c, top[:, :2])
    path = "metric" if math.isfinite(ratio) else "abstain"
    return Obj(color, area, len(u), len(top), ratio, path, np.array([xy[0], xy[1], np.median(top[:, 2])]),
               off, off_xy, cos)

def from_plane(c: Ctx, color: str, uv, area: int, n_depth: int) -> Obj:     # perception.py:152-167
    ray = c.T[:3, :3] @ np.array([(uv[0] - c.K[0, 2]) / c.K[0, 0], (uv[1] - c.K[1, 2]) / c.K[1, 1], 1.0])
    z_top = c.origin[2] + c.height
    pos = np.full(3, np.nan)
    if ray[2] < -1e-6:
        p = c.T[:3, 3] + (z_top - PLANE_AIM * c.height - c.T[2, 3]) / ray[2] * ray
        p[:2] += np.asarray(c.cfg["xy_offset"] or [0, 0], float)
        pos = np.array([p[0], p[1], z_top])
    return Obj(color, area, n_depth, 0, float("nan"), "plane", pos, float("nan"), (float("nan"),) * 2, float("nan"))

def reject(c: Ctx, pos: np.ndarray) -> str | None:           # perception.py:169-187 (paths 2 and 3)
    if not np.all(np.isfinite(pos)):
        return "no_3d"
    h = (pos[2] - c.origin[2]) / c.height
    lo, hi = c.cfg["height_range"]
    if not lo <= h <= hi:
        return f"height {h:.2f} sizes"
    pm = pelvis_to_mat(c, pos)
    W, D = c.scene["counter"]["mat_size"]
    m = c.cfg["counter_margin"]
    if not (-m <= pm[0] <= W + m and -m <= pm[1] <= D + m):
        return "off_mat"
    half = np.asarray(c.scene["bins"]["outer_size"], float) / 2 + c.cfg["bin_exclusion_margin"]
    for name, ctr in c.scene["bins"]["centers"].items():
        if np.all(np.abs(pm[:2] - np.asarray(ctr, float)) <= half):
            return f"in_bin:{name}"
    return None

def detect(c: Ctx, rgb: np.ndarray, depth: np.ndarray) -> list[Obj]:      # perception.py:61-95
    cfg, out = c.cfg, []
    dlo, dhi = cfg["depth_range_mm"]
    keep, k, e = keep_mask(c, depth), int(cfg["open_px"]), int(cfg["erode_px"])
    for color, m in colour_masks(c, rgb).items():
        m = m & keep
        if k > 0:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < cfg["min_area_px"]:
                continue
            blob = (lab == i).astype(np.uint8)
            core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
            if core.sum() < CORE_MIN_FRAC * area:
                core = blob
            v, u = np.nonzero(core)
            d = depth[v, u].astype(float)
            ok = (d >= dlo) & (d <= dhi)          # depth 0 (hole) is < dlo, so invalid everywhere
            obj = from_depth(c, color, u[ok], v[ok], d[ok], area) if ok.sum() >= cfg["min_valid_depth_px"] else None
            obj = obj or from_plane(c, color, cent[i], area, int(ok.sum()))
            out.append(replace(obj, why=reject(c, obj.pos)))
    return out

# ------------------------------------------------------------------ IO + report
def load_pair(folder: Path, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rp, dp = folder / f"head_seed{seed}.png", folder / f"head_seed{seed}_depth.png"
    bgr = cv2.imread(str(rp), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError(f"{rp}: cannot read RGB image")
    depth = cv2.imread(str(dp), cv2.IMREAD_UNCHANGED)
    if depth is None or depth.dtype != np.uint16:
        raise ValueError(f"{dp}: depth must be a uint16 PNG (mm), got {None if depth is None else depth.dtype}")
    if depth.shape != bgr.shape[:2]:
        raise ValueError(f"{dp}: depth shape {depth.shape} != RGB shape {bgr.shape[:2]} of {rp}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), depth

HEAD = f"{'seed':>4} {'colour':<6} {'px':>5} {'nd':>5} {'n_top':>5} {'ratio':>6} {'path':<7} " \
       f"{'x':>7} {'y':>7} {'z':>7}  {'off_mm':>6} {'dx':>6} {'dy':>6} {'ang':>5} {'cos_cam':>7}  status"

def fmt(o: Obj) -> str:
    ang = math.degrees(math.atan2(o.off_xy[1], o.off_xy[0])) if o.off_mm == o.off_mm else float("nan")
    return (f"{o.seed:>4} {o.color:<6} {o.area:>5} {o.n_depth:>5} {o.n_top:>5} {o.ratio:>6.3f} {o.path:<7} "
            f"{o.pos[0]:>+7.3f} {o.pos[1]:>+7.3f} {o.pos[2]:>+7.3f}  {o.off_mm:>6.2f} {o.off_xy[0]:>+6.2f} "
            f"{o.off_xy[1]:>+6.2f} {ang:>5.0f} {o.cos_cam:>+7.2f}  {'ok' if o.why is None else 'REJECT ' + o.why}")

def histogram(objs: list[Obj]) -> str:
    rows = []
    for color in dict.fromkeys(o.color for o in objs):
        r = np.array([o.ratio for o in objs if o.color == color and o.why is None and o.path == "metric"])
        rows.append(f"ratio histogram [{color}]  n={len(r)}" + (f"  median={np.median(r):.3f}" if len(r) else ""))
        counts, _ = np.histogram(np.clip(r, HIST_EDGES[0], HIST_EDGES[-1] - 1e-9), HIST_EDGES)
        rows += [f"  {a:.2f}-{b:.2f} | {'#' * n} {n}" for a, b, n in zip(HIST_EDGES, HIST_EDGES[1:], counts)]
    return "\n".join(rows)

def report(objs: list[Obj]) -> None:
    print(HEAD)
    print("\n".join(fmt(o) for o in objs))
    print("\noff_mm/dx/dy/ang = mean(top xy) - minAreaRect centre (pelvis, +x fwd, +y left); cos_cam>0 = toward camera\n")
    print(histogram(objs))

def main(argv: list[str]) -> int:
    if "--demo" in argv:
        return demo() or 0
    repo = Path(argv[argv.index("--repo") + 1]) if "--repo" in argv else REPO
    args = [a for a in argv if not a.startswith("--") and a != str(repo)]
    if not args or not (repo / "aaltonomous").is_dir():
        print(__doc__, f"\nneeds a folder arg and the starter repo at {repo} (--repo)")
        return 2
    folder = Path(args[0])
    cal = folder / "calibration.yaml"
    c = load_ctx(repo, cal if cal.exists() else None)
    print("constants:", {k: v for k, v in c.cfg.items() if k != "hsv"}, "hsv:", c.cfg["hsv"], sep="\n  ")
    objs: list[Obj] = []
    for s in SEEDS:
        if not (folder / f"head_seed{s}.png").exists():
            continue                              # seeds not captured (yet) are skipped silently
        try:
            rgb, depth = load_pair(folder, s)
        except ValueError as err:
            print(f"ERROR {err}")
            return 1
        objs += [replace(o, seed=str(s)) for o in detect(c, rgb, depth)]
        if "--plane" in argv:
            print(f"\n--plane seed {s}")
            plane_check(c, rgb, depth)
    report(objs)
    return 0

# ------------------------------------------------------------------ synthetic self-check
def render(c: Ctx, xy: tuple[float, float], yaw: float, rgb_obj=(200, 30, 30)) -> tuple[np.ndarray, np.ndarray]:
    """Ray-cast a flat counter plus one cube (size/height from scene.yaml) -> uint8 RGB, uint16 mm depth."""
    u, v = np.meshgrid(np.arange(640), np.arange(480))
    dopt = np.stack([(u - c.K[0, 2]) / c.K[0, 0], (v - c.K[1, 2]) / c.K[1, 1], np.ones_like(u, float)], -1)
    d, o = dopt @ c.T[:3, :3].T, c.T[:3, 3]
    z0, s = c.origin[2], float(c.scene["objects"]["size"])
    t_plane = (z0 - o[2]) / d[..., 2]
    ct, st = math.cos(yaw), math.sin(yaw)
    R = np.array([[ct, -st, 0], [st, ct, 0], [0, 0, 1]])                 # box -> world
    lo = (o - np.array([xy[0], xy[1], z0 + c.height / 2])) @ R           # origin in box frame
    dl = d @ R
    half = np.array([s / 2, s / 2, c.height / 2])
    with np.errstate(divide="ignore", invalid="ignore"):
        t1, t2 = (-half - lo) / dl, (half - lo) / dl
    tn, tf = np.minimum(t1, t2).max(-1), np.maximum(t1, t2).min(-1)
    hit = (tn <= tf) & (tn > 0) & (tn < t_plane)
    t = np.where(hit, tn, t_plane)
    rgb = np.where(hit[..., None], np.array(rgb_obj, np.uint8), np.uint8(150))
    return rgb.astype(np.uint8), np.round(t * 1000).astype(np.uint16)    # optical z = t (dopt z = 1)

def demo() -> None:
    c = load_ctx(REPO)
    truth = (0.30, 0.05)
    rgb, depth = render(c, truth, math.radians(20))
    with tempfile.TemporaryDirectory() as tmp:                           # round-trip through real PNG IO
        cv2.imwrite(f"{tmp}/head_seed101.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
        cv2.imwrite(f"{tmp}/head_seed101_depth.png", depth)
        rgb, depth = load_pair(Path(tmp), 101)
        for bad, why in ((depth.astype(np.uint8), "uint16"), (depth[:-1], "shape")):
            cv2.imwrite(f"{tmp}/head_seed102_depth.png", bad)
            cv2.imwrite(f"{tmp}/head_seed102.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
            try:
                load_pair(Path(tmp), 102)
                raise AssertionError(f"bad {why} depth was accepted")
            except ValueError as err:
                assert "head_seed102_depth.png" in str(err)
    objs = detect(c, rgb, depth)
    report([replace(o, seed="demo") for o in objs])
    assert len(objs) == 1 and objs[0].why is None, f"expected exactly one accepted object, got {objs}"
    o = objs[0]
    err_mm = float(np.linalg.norm(o.pos[:2] - np.array(truth)) * 1e3)
    print(f"\nmetric path position error vs synthesised truth: {err_mm:.2f} mm (limit 5)")
    assert o.color == "red" and o.path == "metric", f"path {o.path}"
    assert err_mm < 5.0, f"position off by {err_mm:.2f} mm"
    holes = np.zeros_like(depth)                                         # no depth stream at all -> plane fallback
    p = detect(c, rgb, holes)
    assert len(p) == 1 and p[0].path == "plane" and p[0].why is None, f"plane fallback: {p}"
    print(f"plane fallback (all-zero depth): path={p[0].path} pos={np.round(p[0].pos, 4)} (aims below top by design)")
    print("DEMO OK: all asserts passed")
    demo_plane()

# ------------------------------------------------------------------ --plane: depth convention test (z-along-axis vs radial)
def counter_mask(c: Ctx, rgb: np.ndarray, depth: np.ndarray) -> np.ndarray:
    """Bare counter: depth in depth_range_mm, outside the projected bin hulls (keep_mask itself would also drop the
    counter, its height gate starts at 0.25 sizes), outside every HSV colour blob grown by PLANE_DILATE_PX."""
    dlo, dhi = c.cfg["depth_range_mm"]
    ok = (depth >= dlo) & (depth <= dhi) & (bin_hulls_mask(c, depth.shape) > 0)
    for m in colour_masks(c, rgb).values():
        ok &= cv2.dilate(m, np.ones((PLANE_DILATE_PX, PLANE_DILATE_PX), np.uint8)) == 0
    return ok

def plane_fit(c: Ctx, u, v, d, label: str) -> tuple[float, float]:
    """Deproject (to_pelvis, unchanged) -> SVD plane -> rms / max |resid| mm, plus residual by radial pixel distance."""
    P = to_pelvis(c, u, v, d)
    ctr = P.mean(0)
    n = np.linalg.svd(P - ctr, full_matrices=False)[2][2]
    res = (P - ctr) @ n * 1e3
    r = np.hypot(u - c.K[0, 2], v - c.K[1, 2])
    edges = np.linspace(0, r.max() + 1e-9, PLANE_BINS + 1)
    rms, mx = float(np.sqrt((res ** 2).mean())), float(np.abs(res).max())
    print(f"  [{label}] normal={np.round(n * np.sign(n[2]), 4)} rms={rms:.2f} mm max|res|={mx:.2f} mm")
    for a, b in zip(edges, edges[1:]):
        m = (r >= a) & (r < b)
        if m.any():
            print(f"    r {a:5.0f}-{b:5.0f} px  n={m.sum():>6}  mean_res={res[m].mean():>+8.2f} mm  rms={np.sqrt((res[m] ** 2).mean()):>7.2f} mm")
    return rms, mx

def plane_check(c: Ctx, rgb: np.ndarray, depth: np.ndarray) -> tuple[tuple[float, float], tuple[float, float]]:
    v, u = np.nonzero(counter_mask(c, rgb, depth))
    d = depth[v, u].astype(float)
    k = np.sqrt(1 + ((u - c.K[0, 2]) / c.K[0, 0]) ** 2 + ((v - c.K[1, 2]) / c.K[1, 1]) ** 2)
    print(f"  counter pixels: {len(u)} (valid depth, outside bin hulls, outside colour blobs)")
    a = plane_fit(c, u, v, d, "depth = z along axis (current code)")
    b = plane_fit(c, u, v, d / k, "depth = radial, d/sqrt(1+a^2+b^2)")
    print(f"  VERDICT: {'z-along-axis (current)' if a[0] <= b[0] else 'RADIAL (current code is wrong)'} is flatter "
          f"({min(a[0], b[0]):.2f} vs {max(a[0], b[0]):.2f} mm rms)")
    return a, b

def demo_plane() -> None:
    c = load_ctx(REPO)
    rgb, z_img = render(c, (9.0, 9.0), 0.0)                       # cube far out of view: flat counter only
    u, v = np.meshgrid(np.arange(640), np.arange(480))
    k = np.sqrt(1 + ((u - c.K[0, 2]) / c.K[0, 0]) ** 2 + ((v - c.K[1, 2]) / c.K[1, 1]) ** 2)
    rad_img = np.round(z_img * k).astype(np.uint16)
    print("\n--plane self-test, depth stored as z along axis:")
    (z_ok, _), (z_bad, _) = plane_check(c, rgb, z_img)
    print("--plane self-test, depth stored as radial distance:")
    (r_bad, _), (r_ok, _) = plane_check(c, rgb, rad_img)
    assert z_ok < PLANE_FLAT_MM and r_ok < PLANE_FLAT_MM, f"correct convention not flat: {z_ok:.2f} {r_ok:.2f}"
    assert z_bad > PLANE_BOWL_MM and r_bad > PLANE_BOWL_MM, f"wrong convention not a bowl: {z_bad:.2f} {r_bad:.2f}"
    print(f"PLANE OK: flat {z_ok:.2f}/{r_ok:.2f} mm vs bowl {z_bad:.2f}/{r_bad:.2f} mm rms")

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
