"""Lying triangular prism (rectangular face down, ridge up) from the object's depth points. Pure numpy.

A lying prism has no flat top face: two slanted planes meet at a ridge, so the height profile across it is a tent and
is constant along it. `lying_prism(P, counter_z, size)` -> (centre_xy, ridge_z, ridge_yaw) or None. ridge_yaw in [0, pi).
Used only by perception when `perception.prism_side` is on; shape_id is untouched (an upright prism, cube or cylinder
has a wide top face and never passes the strip test).

Tests, in order: (1) the highest points (ridge_z - STRIP_BAND) form a thin straight strip: width <= STRIP_MAX_W * size,
length in LEN_RANGE * size, length/width >= MIN_ASPECT; (2) max height ~ the triangle altitude (= size) +- HEIGHT_TOL;
(3) away from the strip the surface falls off like a 60 deg roof: median drop / |across| in SLOPE_RANGE (tan 60 = 1.73),
from at least MIN_SLOPE_PTS points. Vertical cube walls (slope >> 3) and flat tops (slope ~ 0) fail (3).
"""
from __future__ import annotations

import numpy as np

STRIP_BAND = 0.007          # m below the robust ridge height that still counts as ridge
RIDGE_PCT = 97.0            # robust "highest point" percentile
STRIP_MAX_W = 0.50          # strip width (m) <= this * size; an upright / cube / cylinder top is >= 0.87 * size
LEN_RANGE = (0.55, 1.50)    # strip length / size (the ridge is one object edge long = size)
MIN_ASPECT = 2.2
HEIGHT_TOL = 0.30           # ridge height within (1 +- tol) * size above the counter
SLOPE_RANGE = (1.0, 2.6)    # tan(60 deg) = 1.73 nominal
MIN_SLOPE_PTS = 30
MIN_STRIP_PTS = 12
SLOPE_ACROSS = (0.006, 0.035)   # |across-ridge distance| window (m) the slope is measured in


def lying_prism(P, counter_z, size):
    P = np.asarray(P, float).reshape(-1, 3)
    P = P[np.all(np.isfinite(P), axis=1)]
    if len(P) < 3 * MIN_STRIP_PTS:
        return None
    z = P[:, 2]
    zr = float(np.percentile(z, RIDGE_PCT))
    if not (1 - HEIGHT_TOL) * size <= zr - counter_z <= (1 + HEIGHT_TOL) * size:
        return None
    S = P[z >= zr - STRIP_BAND, :2]
    if len(S) < MIN_STRIP_PTS:
        return None
    c = np.median(S, axis=0)
    w, V = np.linalg.eigh(np.cov((S - c).T))
    ax = V[:, 1]                                   # principal (ridge) direction
    u, t = (S - c) @ ax, (S - c) @ np.array([-ax[1], ax[0]])
    length = np.percentile(u, 98) - np.percentile(u, 2)
    width = np.percentile(t, 95) - np.percentile(t, 5)
    if not (LEN_RANGE[0] * size <= length <= LEN_RANGE[1] * size and width <= STRIP_MAX_W * size
            and length >= MIN_ASPECT * max(width, 1e-6)):
        return None
    c = c + ax * 0.5 * (np.percentile(u, 98) + np.percentile(u, 2)) + \
        np.array([-ax[1], ax[0]]) * np.median(t)
    ua, ta = (P[:, :2] - c) @ ax, (P[:, :2] - c) @ np.array([-ax[1], ax[0]])
    m = (np.abs(ta) >= SLOPE_ACROSS[0]) & (np.abs(ta) <= SLOPE_ACROSS[1]) & (np.abs(ua) <= 0.5 * length) \
        & (z - counter_z >= 0.006)
    if m.sum() < MIN_SLOPE_PTS:
        return None
    slope = float(np.median((zr - z[m]) / np.abs(ta[m])))
    if not SLOPE_RANGE[0] <= slope <= SLOPE_RANGE[1]:
        return None
    return c, zr, float(np.arctan2(ax[1], ax[0]) % np.pi)
