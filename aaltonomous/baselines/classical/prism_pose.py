"""Robust top-face pose helpers for the classical perception (pure numpy/cv2, not wired into the planner).

    robust_center(xy)            trimmed centre of a point cloud (metres)
    triangle_pose(xy)            (centre_xy, yaw, confidence) of an equilateral-triangle top face, or None
    classify_blob(mask)          ("cube"|"cylinder"|"triangle"|None, area ratio) from the 2D colour blob

Equilateral triangle: the top face has a 3-fold symmetry, so the geometry only fixes the yaw modulo 120 deg. The
returned yaw is the vertex direction (centroid -> a vertex) that lies closest to `prefer_yaw` (default +y = the
scene's rot=0 apex), i.e. in [prefer-60, prefer+60) deg. Which of the three vertices is "the" apex is a convention,
not a measurement; a grasp that is symmetric under 120 deg does not care.

Fit: for a triangle with edge normals n_k = th + k*120deg, every point satisfies p.n_k <= c.n_k + r. The robust
support values h_k(th) (99.5th percentile of p.n_k) give r = mean(h) and c = 2/3 sum h_k n_k exactly (sum n_k = 0),
and sum h_k(th) = 3 r(th) is the size of the smallest enclosing equilateral triangle of orientation th, which is
minimal at the true orientation. Hole-tolerant (only needs points near each edge), no reliance on the mean.
"""
from __future__ import annotations

import numpy as np

# ---- config constants ------------------------------------------------------------------------------------------
TRIANGLE_MIN_POINTS = 40          # below this triangle_pose returns None (see test table: p90 yaw < 10 deg from ~this)
ROBUST_MIN_POINTS = 5
TRIANGLE_ALTITUDE = 0.045         # m, flat-to-apex (scene objects.size); inradius = altitude / 3
SUPPORT_PERCENTILE = 99.5         # robust "max" of the projection onto each edge normal
SIZE_TOLERANCE = 0.5              # |r_fit / r_expected - 1| at which the size term of the confidence reaches 0
CONF_FULL_POINTS = 200            # point count at which the count term of the confidence saturates
# blob classifier
BLOB_MIN_AREA_PX = 60
BLOB_BORDER_PX = 1                # blob pixels within this many px of the image edge -> abstain (clipped blob)
RATIO_CUBE_MIN = 0.88             # area / minAreaRect-area: cube ~1.0, cylinder ~0.785, triangle ~0.5
RATIO_CYL_MIN = 0.66


def robust_center(xy, trim: float = 0.2):
    """Centre of a 2D cloud: coordinate-wise median, then mean of the points within the (1-trim) quantile of the
    distance to it, repeated once. Returns (2,) or None for fewer than ROBUST_MIN_POINTS points."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    if len(xy) < ROBUST_MIN_POINTS:
        return None
    c = np.median(xy, axis=0)
    for _ in range(2):
        d = np.linalg.norm(xy - c, axis=1)
        keep = d <= np.quantile(d, 1.0 - trim)
        c = xy[keep].mean(0)
    return c


def _supports(P, th, q):
    """P (N,2) centred points, th (A,) normal-set angles -> h (A,3) support values."""
    ang = th[:, None] + np.arange(3) * (2 * np.pi / 3)               # (A,3)
    N = np.stack([np.cos(ang), np.sin(ang)], -1).reshape(-1, 2)      # (3A,2)
    return np.percentile(P @ N.T, q, axis=0).reshape(-1, 3)


def triangle_fit(xy, q: float = SUPPORT_PERCENTILE):
    """(centre (2,), normal angle th in [0,120deg), inradius r) of the best-fit equilateral triangle."""
    xy = np.asarray(xy, float).reshape(-1, 2)
    c0 = np.median(xy, axis=0)
    P = xy - c0
    coarse = np.radians(np.arange(0.0, 120.0, 1.0))
    th = coarse[np.argmin(_supports(P, coarse, q).sum(1))]
    fine = th + np.radians(np.arange(-1.0, 1.0001, 0.1))
    h = _supports(P, fine, q)
    i = int(np.argmin(h.sum(1)))
    th, h = fine[i], h[i]
    n = np.stack([np.cos(th + np.arange(3) * 2 * np.pi / 3), np.sin(th + np.arange(3) * 2 * np.pi / 3)], -1)
    c = c0 + (2.0 / 3.0) * (h[:, None] * n).sum(0)
    return c, th % (2 * np.pi / 3), float(h.mean())


def triangle_pose(xy_metres, prefer_yaw: float = np.pi / 2, min_points: int = TRIANGLE_MIN_POINTS):
    """Top-face point cloud (N,2) in metres -> (centre_xy (2,), yaw in [0,2pi), confidence in [0,1]) or None.

    yaw points from the centroid to a vertex (the "apex"), the one closest to prefer_yaw (see module docstring)."""
    xy = np.asarray(xy_metres, float).reshape(-1, 2)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if len(xy) < max(min_points, ROBUST_MIN_POINTS):
        return None
    c, th, r = triangle_fit(xy)
    # vertex directions are opposite to the edge normals
    base = th + np.pi
    k = np.round((prefer_yaw - base) / (2 * np.pi / 3))
    yaw = float((base + k * 2 * np.pi / 3) % (2 * np.pi))
    # confidence: enough points, fitted size matches the known object, minAreaRect offset agrees (h/6 toward vertex)
    r_exp = TRIANGLE_ALTITUDE / 3.0
    f_n = min(1.0, len(xy) / CONF_FULL_POINTS)
    f_s = float(np.clip(1.0 - abs(r / r_exp - 1.0) / SIZE_TOLERANCE, 0.0, 1.0))
    import cv2
    (rx, ry), _, _ = cv2.minAreaRect(xy.astype(np.float32))
    off = np.array([rx, ry]) - robust_center(xy)
    # any of the 3 vertices is valid, so compare against the nearest one
    d = np.arctan2(off[1], off[0]) - yaw
    d = (d + np.pi / 3) % (2 * np.pi / 3) - np.pi / 3
    agree = abs(np.linalg.norm(off) - TRIANGLE_ALTITUDE / 6) < 0.006 and abs(d) < np.radians(30)
    conf = float(f_n * f_s * (0.75 + 0.25 * bool(agree)))
    return c, yaw, conf


def classify_blob(mask, min_area_px: int = BLOB_MIN_AREA_PX, border_px: int = BLOB_BORDER_PX):
    """2D colour blob (uint8/bool mask, single object) -> (label, ratio). label is "cube" | "cylinder" | "triangle",
    or None to abstain: blob touches the image edge (clipped), or is smaller than min_area_px, or is empty.
    ratio = pixel area / minAreaRect area (cube ~1.0, cylinder ~0.785, triangle ~0.5)."""
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    H, W = m.shape
    b = max(int(border_px), 1)
    if m[:b].any() or m[-b:].any() or m[:, :b].any() or m[:, -b:].any():
        return None, float("nan")
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None, float("nan")
    cnt = max(cnts, key=cv2.contourArea)
    area = float(m.sum()) if len(cnts) == 1 else float(cv2.contourArea(cnt))
    if area < min_area_px:
        return None, float("nan")
    _, (w, h), _ = cv2.minAreaRect(cnt)
    ratio = area / max((w + 1.0) * (h + 1.0), 1e-9)       # +1 px: contour runs through pixel centres
    label = "cube" if ratio >= RATIO_CUBE_MIN else "cylinder" if ratio >= RATIO_CYL_MIN else "triangle"
    return label, float(ratio)
