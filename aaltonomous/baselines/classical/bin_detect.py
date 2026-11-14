"""Per-episode bin detection (perception.detect_bins, default OFF): where are the three orange bins really?

    res = detect_bin_row(P, orange, scene, cfg)    # P (N,3) pelvis points of valid depth pixels, orange (N,) bool
    res.layout  -> {name: {"xy": (2,), "yaw": rad, "source": "perceived" | "nominal"}}

Walls are vertical, so every wall / rim point projects in xy onto the wall centre line of its box: a bin is the
boundary of a known square. Steps:
  1. counter plane (fit_counter_plane, shared with Perception.calibrate_bins) -> height of every point;
     candidates = orange-ish points between min_height_m and rim + rim_slack_m, near the nominal row.
  2. coarse: the whole row as one rigid body (dx, dy, yaw). Per yaw step the candidates are rotated into the
     nominal frame, rasterized (occupancy, coarse_cell_m) and matched against the 3-square outline (matchTemplate).
  3. fine: each bin alone (cx, cy, yaw), Gauss-Newton on point-to-nearest-edge distance with a shrinking inlier
     window (objects inside / next to a bin are outliers, not wall).
  4. gate against nominal: every bin within max_offset_m and max_yaw_deg, enough inliers on both wall directions,
     neighbour spacing within max_spacing_err_m, per-bin yaws within max_yaw_spread_deg. Anything fails -> the
     whole layout stays nominal (all-or-nothing) and `reason` says why.
Pelvis frame throughout; yaw = rotation of a bin about z relative to the nominal (axis-aligned) layout.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

DEFAULTS = {
    "pixel_stride": 3,
    "max_plane_points": 6000,     # counter-plane fit subsample
    "max_points": 2500,           # wall-point subsample (runtime)
    "min_height_m": 0.015,        # above the fitted counter plane: drops counter + bin floors
    "rim_slack_m": 0.03,          # up to wall_height + this (real wall height still unmeasured)
    "use_colour": True,
    # orange basin, OpenCV HSV: sim bins H ~12 (DR 15-24), real cardboard H 2-7 -> H 0-26 / 165-180
    "orange_hsv": [[[0, 60, 40], [26, 255, 255]], [[165, 60, 40], [180, 255, 255]]],
    "search_m": 0.15,             # coarse translation window (also the gate)
    "max_offset_m": 0.15,
    "max_yaw_deg": 15.0,
    "coarse_cell_m": 0.01,
    "coarse_yaw_step_deg": 2.5,
    "inlier_m": [0.025, 0.015, 0.008],   # shrinking Gauss-Newton inlier windows (point -> wall line)
    "iters_per_window": 3,
    "min_points_per_bin": 120,    # inliers at the last window
    "min_edge_points": 20,        # inliers on each wall direction (x-normal and y-normal walls)
    "max_spacing_err_m": 0.03,    # |neighbour centre distance - nominal|
    "max_yaw_spread_deg": 6.0,    # per-bin yaw vs row yaw
}


def params(cfg_perception):
    p = dict(DEFAULTS)
    p.update(cfg_perception.get("bin_detect") or {})
    return p


@dataclass
class BinDetection:
    success: bool
    valid_frame: bool
    reason: str
    layout: dict
    row_yaw: float = 0.0
    errors_m: dict = field(default_factory=dict)      # perceived - nominal centre distance per bin
    n_points: int = 0
    elapsed_ms: float = 0.0

    def summary(self):
        if not self.success:
            return f"nominal ({self.reason}), {self.n_points} wall pts, {self.elapsed_ms:.1f} ms"
        parts = [f"{n}=({v['xy'][0]:+.3f},{v['xy'][1]:+.3f}) {np.degrees(v['yaw']):+.1f}deg"
                 for n, v in self.layout.items()]
        return (f"perceived row yaw {np.degrees(self.row_yaw):+.1f}deg " + " ".join(parts) +
                f" | {self.reason}, {self.n_points} wall pts, {self.elapsed_ms:.1f} ms")


def nominal_layout(scene):
    b = scene.cfg["bins"]
    return {n: {"xy": scene.mat_to_pelvis(np.r_[np.asarray(c, float), 0.0])[:2], "yaw": 0.0, "source": "nominal"}
            for n, c in b["centers"].items()}


def half_pelvis(scene):
    """Half outer size of a bin in the pelvis frame (mat x = pelvis y)."""
    return np.asarray(scene.cfg["bins"]["outer_size"], float)[::-1] / 2


def in_footprints(layout, half, xy, margin):
    """{name: (N,) bool} xy (pelvis) inside each (rotated) bin footprint + margin."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    out = {}
    for name, v in layout.items():
        c, s = np.cos(v["yaw"]), np.sin(v["yaw"])
        d = xy - v["xy"]
        q = np.c_[c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]]
        out[name] = np.all(np.abs(q) <= half + margin, axis=1)
    return out


def footprint_corners(v, half):
    """(4, 2) pelvis xy corners of a bin footprint (centre xy, yaw, half size)."""
    c, s = np.cos(v["yaw"]), np.sin(v["yaw"])
    R = np.array([[c, -s], [s, c]])
    loc = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], float) * half
    return v["xy"] + loc @ R.T


def fit_counter_plane(P, scene, window, tol, min_points):
    """z = ax + by + c fitted to the counter points (trimmed). -> (coef, None) or (None, reason)."""
    pm = scene.pelvis_to_mat(P)
    mat = np.asarray(scene.cfg["counter"]["mat_size"], float)
    counter = ((pm[:, 0] >= -0.03) & (pm[:, 0] <= mat[0] + 0.03) &
               (pm[:, 1] >= -0.03) & (pm[:, 1] <= mat[1] + 0.03) &
               (np.abs(P[:, 2] - scene.counter_z) < window))
    plane_points = P[counter]
    if len(plane_points) < min_points:
        return None, "counter plane not visible"
    A = np.c_[plane_points[:, :2], np.ones(len(plane_points))]
    keep = np.ones(len(A), bool)
    coef = np.array([0.0, 0.0, scene.counter_z])
    for _ in range(3):
        if keep.sum() < min_points:
            break
        coef = np.linalg.lstsq(A[keep], plane_points[keep, 2], rcond=None)[0]
        residual = plane_points[:, 2] - A @ coef
        keep = np.abs(residual - np.median(residual)) <= tol
    if keep.sum() < min_points:
        return None, "counter plane fit rejected"
    return coef, None


def orange_mask(rgb, ranges):
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    m = np.zeros(hsv.shape[:2], np.uint8)
    for lo, hi in ranges:
        m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    return m > 0


def _rot(a):
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, -s], [s, c]])


def _coarse_row(Q, centres, half_wall, p):
    """Rigid row pose (yaw, shift) maximizing the outline overlap. -> (yaw, (2,) shift in the nominal frame)."""
    import cv2
    cell, S = float(p["coarse_cell_m"]), float(p["search_m"])
    row_c = centres.mean(0)
    lo = centres.min(0) - half_wall - 2 * cell
    hi = centres.max(0) + half_wall + 2 * cell
    tshape = np.ceil((hi - lo) / cell).astype(int) + 1          # (nx, ny)
    tmpl = np.zeros((tshape[1], tshape[0]), np.uint8)            # rows = y, cols = x
    for c in centres:
        a = np.round((c - half_wall - lo) / cell).astype(int)
        b = np.round((c + half_wall - lo) / cell).astype(int)
        cv2.rectangle(tmpl, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), 1, 1)
    tmpl = cv2.dilate(tmpl, np.ones((3, 3), np.uint8)).astype(np.float32)
    n = int(round(S / cell))
    ilo = lo - n * cell
    ishape = tshape + 2 * n
    best = (-1.0, 0.0, np.zeros(2))
    max_yaw = np.radians(float(p["max_yaw_deg"]))
    step = np.radians(float(p["coarse_yaw_step_deg"]))
    for yaw in np.arange(-max_yaw, max_yaw + 0.5 * step, step):
        q = (Q - row_c) @ _rot(-yaw).T + row_c                  # rotate the data into the nominal frame
        ij = np.round((q - ilo) / cell).astype(int)
        ok = (ij[:, 0] >= 0) & (ij[:, 0] < ishape[0]) & (ij[:, 1] >= 0) & (ij[:, 1] < ishape[1])
        img = np.zeros((ishape[1], ishape[0]), np.float32)
        img[ij[ok, 1], ij[ok, 0]] = 1.0                         # occupancy: near walls do not outweigh far ones
        r = cv2.matchTemplate(img, tmpl, cv2.TM_CCORR)
        _, mx, _, loc = cv2.minMaxLoc(r)
        if mx > best[0]:
            best = (mx, yaw, (np.array(loc, float) - n) * cell)
    return best[1], best[2]


def _refine_bin(Q, c0, yaw0, hw, p):
    """Gauss-Newton fit of one square outline (half wall-centre size hw) -> (centre, yaw, inliers, edge counts)."""
    c, yaw = np.asarray(c0, float).copy(), float(yaw0)
    gate = hw + 0.03
    Q = Q[np.all(np.abs(Q - c) <= np.max(hw) + 0.06, axis=1)]   # this bin's neighbourhood only (runtime)
    for w in p["inlier_m"]:
        for _ in range(int(p["iters_per_window"])):
            R = _rot(yaw)
            q = (Q - c) @ R                                      # local coords
            near = np.all(np.abs(q) <= gate, axis=1)
            qn = q[near]
            if len(qn) < 10:
                return c, yaw, 0, (0, 0)
            ax = (np.abs(qn[:, 1]) / hw[1] > np.abs(qn[:, 0]) / hw[0]).astype(int)   # 0: x-normal wall, 1: y-normal
            sgn = np.sign(qn[np.arange(len(qn)), ax])
            sgn[sgn == 0] = 1
            nloc = np.zeros((len(qn), 2))
            nloc[np.arange(len(qn)), ax] = sgn
            r = np.sum(nloc * qn, axis=1) - hw[ax]
            inl = np.abs(r) <= w
            if inl.sum() < 10:
                return c, yaw, 0, (0, 0)
            nw = nloc @ R.T                                       # world normals
            tw = nw @ _rot(np.pi / 2).T
            d = Q[near] - c
            J = np.c_[-nw, np.sum(tw * d, axis=1)]
            Ji, ri = J[inl], r[inl]
            delta = np.linalg.lstsq(Ji, -ri, rcond=None)[0]
            c, yaw = c + delta[:2], yaw + float(delta[2])
            if np.abs(delta).max() < 1e-5:
                break
    q = (Q - c) @ _rot(yaw)
    near = np.all(np.abs(q) <= gate, axis=1)
    qn = q[near]
    ax = (np.abs(qn[:, 1]) / hw[1] > np.abs(qn[:, 0]) / hw[0]).astype(int)
    r = np.abs(np.abs(qn[np.arange(len(qn)), ax]) - hw[ax])
    inl = r <= p["inlier_m"][-1]
    return c, yaw, int(inl.sum()), (int((inl & (ax == 0)).sum()), int((inl & (ax == 1)).sum()))


def detect_bin_row(P, orange, scene, cfg_perception, started=None):
    """P (N,3) pelvis points of valid depth pixels, orange (N,) bool (or None = no colour gate) -> BinDetection."""
    started = time.perf_counter() if started is None else started
    p = params(cfg_perception)
    nominal = nominal_layout(scene)
    names = list(nominal)
    centres = np.array([nominal[n]["xy"] for n in names])
    b = scene.cfg["bins"]
    half = half_pelvis(scene)
    hw = half - float(b["wall_thickness"]) / 2                    # wall centre line

    def out(success, reason, layout=None, row_yaw=0.0, n=0, errors=None):
        return BinDetection(success, True, reason, layout or nominal_layout(scene), float(row_yaw), errors or {},
                            int(n), (time.perf_counter() - started) * 1000.0)

    sc_cal = cfg_perception.get("self_calibration") or {}
    Pp = P[::max(1, len(P) // int(p["max_plane_points"]))]
    coef, why = fit_counter_plane(Pp, scene, float(sc_cal.get("plane_z_window", 0.06)),
                                  float(sc_cal.get("plane_inlier_m", 0.006)), int(sc_cal.get("min_plane_points", 500)))
    if coef is None:
        return out(False, why)
    height = P[:, 2] - (np.c_[P[:, :2], np.ones(len(P))] @ coef)
    S = float(p["search_m"])
    lo = centres.min(0) - half - S - 0.02
    hi = centres.max(0) + half + S + 0.02
    sel = ((height >= float(p["min_height_m"])) & (height <= float(b["wall_height"]) + float(p["rim_slack_m"])) &
           np.all((P[:, :2] >= lo) & (P[:, :2] <= hi), axis=1))
    if orange is not None and p["use_colour"]:
        sel &= orange
    Q = P[sel, :2]
    n_all = len(Q)
    if len(Q) > int(p["max_points"]):
        Q = Q[np.linspace(0, len(Q) - 1, int(p["max_points"])).astype(int)]
    if len(Q) < 3 * int(p["min_points_per_bin"]):
        return out(False, f"too few orange wall points ({len(Q)})", n=n_all)

    yaw, shift = _coarse_row(Q, centres, hw, p)
    row_c = centres.mean(0)
    R = _rot(yaw)
    layout, errors, yaws, stats = {}, {}, [], []
    for name, n_c in zip(names, centres):
        c0 = row_c + (n_c - row_c + shift) @ R.T
        c, y, n_in, (ex, ey) = _refine_bin(Q, c0, yaw, hw, p)
        layout[name] = {"xy": c, "yaw": float(y), "source": "perceived"}
        errors[name] = float(np.linalg.norm(c - n_c))
        yaws.append(y)
        stats.append(f"{name}:{n_in}({ex}/{ey})")
        if n_in < int(p["min_points_per_bin"]) or min(ex, ey) < int(p["min_edge_points"]):
            return out(False, f"{name}: weak wall fit {n_in} inliers ({ex}/{ey} per wall direction)", n=n_all,
                       errors=errors)
    left_right = layout[names[-1]]["xy"] - layout[names[0]]["xy"]
    nom_lr = centres[-1] - centres[0]
    row_yaw = float(np.arctan2(left_right[1], left_right[0]) - np.arctan2(nom_lr[1], nom_lr[0]))
    row_yaw = (row_yaw + np.pi) % (2 * np.pi) - np.pi
    for name in names:
        if errors[name] > float(p["max_offset_m"]):
            return out(False, f"{name} {errors[name] * 100:.1f} cm from nominal", n=n_all, errors=errors)
        if abs(layout[name]["yaw"]) > np.radians(float(p["max_yaw_deg"])):
            return out(False, f"{name} yaw {np.degrees(layout[name]['yaw']):+.1f} deg", n=n_all, errors=errors)
        if abs(layout[name]["yaw"] - row_yaw) > np.radians(float(p["max_yaw_spread_deg"])):
            return out(False, f"{name} yaw {np.degrees(layout[name]['yaw']):+.1f} deg vs row "
                              f"{np.degrees(row_yaw):+.1f} deg", n=n_all, errors=errors)
    for a, bb, na, nb in zip(names[:-1], names[1:], centres[:-1], centres[1:]):
        d = np.linalg.norm(layout[a]["xy"] - layout[bb]["xy"]) - np.linalg.norm(na - nb)
        if abs(d) > float(p["max_spacing_err_m"]):
            return out(False, f"spacing {a}-{bb} off by {d * 100:+.1f} cm", n=n_all, errors=errors)
    return out(True, "accepted " + " ".join(stats), layout, row_yaw, n_all, errors)
