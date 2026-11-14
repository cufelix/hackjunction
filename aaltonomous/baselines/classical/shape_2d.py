"""perception.shape_2d_v2: 2D cube/cylinder call for real head-camera frames (yellow only; see Perception._apply_shape2d).

`shape2d_features(mask)` is the ONE place the feature definition lives (swap it to change the measurement):
  quad_rms   RMS residual (px) of a quadratic y=f(x) fitted to the top edge (topmost mask pixel of every column)
  n_vert     vertices of cv2.approxPolyDP(largest external contour, eps_frac * perimeter, closed)
  axis_ratio minor/major of cv2.fitEllipse on the upper hull (convex-hull points at or above the centroid row); log only
"""
from __future__ import annotations

import numpy as np

MIN_AREA_PX = 60      # same floor as shape_id.BLOB_MIN_AREA_PX
MIN_COLS = 8          # columns needed for a stable quadratic fit


def shape2d_features(mask, approx_eps_frac=0.02):
    """mask (bool/uint8, any size) -> {"quad_rms", "n_vert", "axis_ratio"}; values None when not measurable."""
    import cv2
    out = {"quad_rms": None, "n_vert": None, "axis_ratio": None}
    m = (np.asarray(mask) > 0).astype(np.uint8)
    if m.ndim != 2 or int(m.sum()) < MIN_AREA_PX:
        return out
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return out
    cnt = max(cnts, key=cv2.contourArea)
    out["n_vert"] = int(len(cv2.approxPolyDP(cnt, approx_eps_frac * cv2.arcLength(cnt, True), True)))
    cols = np.flatnonzero(m.any(axis=0))
    if len(cols) >= MIN_COLS:
        top_y = m[:, cols].argmax(axis=0).astype(float)      # first (topmost) set pixel per column
        x = cols.astype(float)
        out["quad_rms"] = float(np.sqrt(np.mean((np.polyval(np.polyfit(x - x.mean(), top_y, 2), x - x.mean()) - top_y) ** 2)))
    hull = cv2.convexHull(cnt).reshape(-1, 2)
    up = hull[hull[:, 1] <= hull[:, 1].mean()] if len(hull) else hull   # ponytail: hull-vertex mean row, not centroid
    if len(up) >= 5:
        try:
            ax = sorted(cv2.fitEllipse(up.astype(np.float32).reshape(-1, 1, 2))[1])
            out["axis_ratio"] = float(ax[0] / ax[1]) if ax[1] > 0 else None
        except cv2.error:
            pass
    return out


def decide_shape2d(f, quad_rms_max=1.77, n_vert_min=8, cube_n_vert_max=6):
    """features -> "cylinder" | "cube" | None (abstain)."""
    q, n = f.get("quad_rms"), f.get("n_vert")
    if q is None or n is None:
        return None
    if q < quad_rms_max and n >= n_vert_min:
        return "cylinder"
    if q >= quad_rms_max and n <= cube_n_vert_max:
        return "cube"
    return None
