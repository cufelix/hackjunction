"""perception.prism_orientation: is a triangular prism standing on a triangular face or lying on a rectangular side?

Pure function on the pelvis-frame points of one blob (no state, a few hundred points, well under 1 ms).

Geometry (prism: equilateral triangle, altitude `size`, extruded by `length`):
  standing: footprint = the triangle (convex-hull area / min-area-rect area = 1/2); the top-band points are the whole
            triangle face, so the top face is as wide as the footprint.
  lying:    footprint = a rectangle (fill ~ 1); the top-band points are a thin strip along the ridge, whose width is
            2*(h - z_band_low)/tan(60 deg) for the visible sloped faces.
Decision cues (all derived from the geometry, see PRISM.md):
  fill      = hull area / min-area-rect area of the footprint; the midpoint of 1/2 and 1 is `fill_min` = 0.75.
  strip     = (width of the top band across its own long axis) / (width of the footprint across that axis); the
              midpoint of the expected lying ratio and 1 (standing) is the threshold.
  height    = only when the standing and lying top heights differ by more than 2 top bands (else the cue is silent).
Lying needs fill AND strip (and height, when available) to agree; standing needs the opposite for the two geometry cues;
anything else is unsure (None) and the caller keeps the current behaviour. The footprint aspect ratio is reported but
not used: the lying footprint is `side` x `length` (52 x 45 mm for our objects), nearly square.
"""
from __future__ import annotations

import numpy as np

TAN60 = float(np.tan(np.pi / 3))   # slope of the sloped faces of an equilateral prism (base angle 60 deg)


def _spread(x, lo=1.0, hi=99.0):
    return float(np.percentile(x, hi) - np.percentile(x, lo))


def expected_strip_ratio(altitude, side, top_percentile, band):
    """Top-band width / footprint width of a lying prism. The band starts at the `top_percentile` height of the (uniform
    in z) sloped surface minus `band`; the strip is two slopes of that depth below the ridge, 60 deg from horizontal."""
    depth = (1.0 - top_percentile / 100.0) * altitude + band
    return min(1.0, 2.0 * depth / TAN60 / side)


def prism_orientation(P, top, counter_z, altitude, side, length, band, top_percentile, min_points=25,
                      fill_min=0.75):
    """P (N,3) all blob points, top (M,3) the top band, pelvis frame (m). Returns None when unsure / too few points,
    else {orientation: "standing"|"lying", ridge_yaw (rad, [-pi/2, pi/2), lying only), xy (ridge-strip centre, lying only),
    conf (0..1, smallest normalised cue margin), fill, strip_ratio, aspect, dz}."""
    import cv2
    P, top = np.asarray(P, float), np.asarray(top, float)
    if len(P) < min_points or len(top) < 5 or not np.all(np.isfinite(P)):
        return None
    pts = P[:, :2].astype(np.float32)
    hull = cv2.convexHull(pts)
    (_, _), (rw, rh), _ = cv2.minAreaRect(pts)
    if rw <= 1e-6 or rh <= 1e-6:
        return None
    fill = float(cv2.contourArea(hull) / (rw * rh))
    aspect = float(max(rw, rh) / min(rw, rh))
    t = top[:, :2] - top[:, :2].mean(0)
    w, V = np.linalg.eigh(t.T @ t / len(t))
    u, n = V[:, 1], V[:, 0]                       # major / minor axis of the top band
    strip_w = _spread(t @ n)
    foot_w = _spread((P[:, :2] - top[:, :2].mean(0)) @ n)
    if foot_w <= 1e-6:
        return None
    ratio = strip_w / foot_w
    thr = 0.5 * (expected_strip_ratio(altitude, side, top_percentile, band) + 1.0)
    dz = float(np.median(top[:, 2]) - counter_z)
    # cue margins, positive = lying, negative = standing, each normalised to [-1, 1]
    m_fill = (fill - fill_min) / (1.0 - fill_min) if fill >= fill_min else (fill - fill_min) / (fill_min - 0.5)
    m_strip = (thr - ratio) / thr if ratio <= thr else (thr - ratio) / (1.0 - thr)
    margins = [m_fill, m_strip]
    if abs(length - altitude) >= 2.0 * band:       # standing top = length, lying top = altitude: only then a cue
        margins.append((abs(dz - length) - abs(dz - altitude)) / abs(length - altitude))
    margins = np.clip(margins, -1.0, 1.0)
    lying = bool(np.all(margins > 0))
    standing = bool(margins[0] < 0 and margins[1] < 0 and (len(margins) < 3 or margins[2] <= 0))
    out = {"fill": fill, "strip_ratio": float(ratio), "aspect": aspect, "dz": dz, "margins": [float(m) for m in margins]}
    if lying:
        a = float(np.arctan2(u[1], u[0]))
        out["ridge_yaw"] = float((a + np.pi / 2) % np.pi - np.pi / 2)
        a_ = t @ u
        b_ = t @ n
        c = top[:, :2].mean(0) + u * 0.5 * (np.percentile(a_, 2) + np.percentile(a_, 98)) \
            + n * 0.5 * (np.percentile(b_, 2) + np.percentile(b_, 98))
        out.update(orientation="lying", xy=c, conf=float(margins.min()))
    elif standing:
        out.update(orientation="standing", conf=float(-margins[:2].max()))
    else:
        return None
    return out
