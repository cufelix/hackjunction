"""Robust top-face pose helpers for the classical perception (pure numpy/cv2). Shape decisions live ONLY in shape_id.classify_top_face.

    robust_center(xy)            trimmed centre of a point cloud (metres)
    triangle_pose(xy)            (centre_xy, yaw, confidence) of an equilateral-triangle top face, or None

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
FILL_MIN, FILL_FULL = 0.70, 0.85   # fill term of the confidence: 0 at/below FILL_MIN, 1 at/above FILL_FULL
CONF_FULL_POINTS = 200            # point count at which the count term of the confidence saturates


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


def triangle_pose(xy_metres, prefer_yaw: float = np.pi / 2, min_points: int = TRIANGLE_MIN_POINTS,
                  altitude: float = TRIANGLE_ALTITUDE):
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
    r_exp = altitude / 3.0
    f_n = min(1.0, len(xy) / CONF_FULL_POINTS)
    f_s = float(np.clip(1.0 - abs(r / r_exp - 1.0) / SIZE_TOLERANCE, 0.0, 1.0))
    import cv2
    # fill: hull area / area of the fitted triangle. ~0.9+ for a real triangle, ~0.6 for a cube or cylinder that was
    # mistaken for one (their smallest enclosing triangle is mostly empty). Independent of the object size, so a wrong
    # shape label still ends in low confidence (-> planner falls back to the stock finger hints).
    fill = float(cv2.contourArea(cv2.convexHull(xy.astype(np.float32)))) / (3.0 * np.sqrt(3.0) * r * r)
    f_f = float(np.clip((fill - FILL_MIN) / (FILL_FULL - FILL_MIN), 0.0, 1.0))
    (rx, ry), _, _ = cv2.minAreaRect(xy.astype(np.float32))
    off = np.array([rx, ry]) - robust_center(xy)
    # any of the 3 vertices is valid, so compare against the nearest one
    d = np.arctan2(off[1], off[0]) - yaw
    d = (d + np.pi / 3) % (2 * np.pi / 3) - np.pi / 3
    agree = abs(np.linalg.norm(off) - altitude / 6) < 0.006 and abs(d) < np.radians(30)
    conf = float(f_n * f_s * f_f * (0.75 + 0.25 * bool(agree)))
    return c, yaw, conf
