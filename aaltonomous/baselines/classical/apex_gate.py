"""grasp.posture_apex_gate: is a top face that shape_id calls "triangle" really a prism top?

Computation taken from Oliver's offline evaluator apex_conf_eval.py (origin/feat/grasp-yaw e0f1ca2, incl. the mm
fix): apex_conf = prism_pose.triangle_pose(points in metres)[2]; max vertex angle of cv2.minEnclosingTriangle over
the same points in MILLIMETRES (minEnclosingTriangle on metre-scale float points is numerically unreliable).
A prism top is equilateral (all angles 60 deg). A cube top that an occluder (the hand at home) cut down to a
triangle-ish patch keeps one 90 deg corner, so its enclosing triangle has an angle >= 90 deg.
"""
from __future__ import annotations

import numpy as np

from .prism_pose import TRIANGLE_ALTITUDE, triangle_pose


def apex_gate_features(xy_metres, altitude: float = TRIANGLE_ALTITUDE):
    """(apex_conf 0..1, max vertex angle in deg or None) of a top-face point set (N,2) in metres."""
    import cv2
    xy = np.asarray(xy_metres, float).reshape(-1, 2)
    xy = xy[np.all(np.isfinite(xy), axis=1)]
    if not _usable_cloud(xy):
        return 0.0, None
    try:
        return _features(xy, altitude)
    except (cv2.error, ValueError, FloatingPointError, np.linalg.LinAlgError):
        return 0.0, None


def _usable_cloud(xy, eps: float = 1e-9) -> bool:
    """>= 3 distinct, non-collinear points with a hull of non-zero area (metres)."""
    import cv2
    if len(xy) < 3 or len(np.unique(np.round(xy, 9), axis=0)) < 3:
        return False
    c = xy - xy.mean(0)
    if np.linalg.svd(c, compute_uv=False)[-1] < 1e-6:          # < 1 micron spread across the minor axis: collinear
        return False
    return float(cv2.contourArea(cv2.convexHull(xy.astype(np.float32)))) > eps


def _features(xy, altitude):
    import cv2
    tp = triangle_pose(xy, altitude=altitude)
    conf = 0.0 if tp is None else float(tp[2])
    ang = None
    if len(xy) >= 5:
        ok, tri = cv2.minEnclosingTriangle((xy * 1000.0).astype(np.float32).reshape(-1, 1, 2))
        t = np.asarray(tri, float).reshape(-1, 2)
        if t.shape != (3, 2) or not np.all(np.isfinite(t)):
            return 0.0, None
        angs = []
        for k in range(3):
            u, w = t[(k + 1) % 3] - t[k], t[(k + 2) % 3] - t[k]
            c = u @ w / (np.linalg.norm(u) * np.linalg.norm(w) + 1e-12)
            angs.append(float(np.degrees(np.arccos(np.clip(c, -1.0, 1.0)))))
        ang = max(angs)
    return conf, ang


def apex_gate_pass(conf, ang, min_conf: float, max_angle_deg: float) -> bool:
    return ang is not None and conf >= min_conf and ang <= max_angle_deg
