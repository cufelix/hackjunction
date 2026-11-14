"""Geometry-only top-down grasp proposal for an unknown small object.

``depth`` is aligned optical-Z depth in metres. ``K`` is the 3x3 pinhole
matrix and ``T_cam_to_robot`` maps optical-frame points into the robot frame
(+Z up).  The implementation is NumPy-only and bounded by the image size.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

_MIN_VALID_POINTS = 20
_MIN_TOP_POINTS = 3
_TOP_BAND_M = 0.008
_TOP_PERCENTILE = 92.0
_FULL_SUPPORT_POINTS = 40
_MAX_GRASP_WIDTH_M = 0.060
_FLAT_MAD_M = 0.004

__all__ = ["GraspProposal", "propose_grasp"]


@dataclass(frozen=True)
class GraspProposal:
    """A top-down grasp in the robot frame."""

    centre: np.ndarray
    yaw: float
    width: float
    confidence: float
    n_points: int


def propose_grasp(mask, depth, K, T_cam_to_robot) -> GraspProposal | None:
    """Return a robust top-surface grasp, or ``None`` for insufficient data.

    ``yaw`` is the undirected major-axis heading in ``[-pi/2, pi/2)`` and
    ``width`` is the robust extent across its perpendicular (closing) axis.
    Invalid/zero depths are ignored. Inputs are never mutated.
    """
    mask = np.asarray(mask)
    depth = np.asarray(depth, dtype=np.float64)
    K = np.asarray(K, dtype=np.float64)
    T = np.asarray(T_cam_to_robot, dtype=np.float64)
    if mask.ndim != 2 or depth.ndim != 2:
        raise ValueError("mask and depth must be HxW arrays")
    if mask.shape != depth.shape:
        raise ValueError("mask and depth must have the same shape")
    if K.shape != (3, 3):
        raise ValueError("K must be a 3x3 pinhole matrix")
    if T.shape != (4, 4):
        raise ValueError("T_cam_to_robot must be a 4x4 transform")
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])
    if not np.all(np.isfinite(K)) or fx <= 0.0 or fy <= 0.0:
        raise ValueError("K must contain finite positive focal lengths")
    if not np.all(np.isfinite(T)):
        raise ValueError("T_cam_to_robot must be finite")

    valid = (mask > 0) & np.isfinite(depth) & (depth > 0.0)
    if int(np.count_nonzero(valid)) < _MIN_VALID_POINTS:
        return None
    points = _backproject(valid, depth, fx, fy, cx, cy, T)
    points = points[np.all(np.isfinite(points), axis=1)]
    if len(points) < _MIN_VALID_POINTS:
        return None
    top = _top_surface(points)
    if len(top) < _MIN_TOP_POINTS:
        return None

    # A component-wise median can fall in empty space for a concave/ring-like
    # top.  Snap it to the nearest measured top point so the approach target is
    # always supported by the observed object surface.
    median_xy = np.median(top[:, :2], axis=0)
    centre = top[np.argmin(np.sum((top[:, :2] - median_xy) ** 2, axis=1))].copy()
    yaw, minor = _yaw_and_minor(top[:, :2])
    width = _minor_width(top[:, :2], minor)
    confidence = _confidence(len(top), width, _touches_border(mask > 0))
    if not (np.all(np.isfinite(centre)) and np.isfinite(yaw) and np.isfinite(width)):
        return None
    return GraspProposal(np.asarray(centre, dtype=np.float64), float(yaw), float(width),
                         confidence, int(len(top)))


def _backproject(valid, depth, fx, fy, cx, cy, T):
    v, u = np.nonzero(valid)
    z = depth[v, u]
    points = np.column_stack(((u - cx) * z / fx, (v - cy) * z / fy, z))
    return points @ T[:3, :3].T + T[:3, 3]


def _top_surface(points):
    """Select the highest coherent band without following sparse spikes."""
    z = points[:, 2]
    median = float(np.median(z))
    mad = float(np.median(np.abs(z - median)))
    z_ref = median if mad <= _FLAT_MAD_M else float(np.percentile(z, _TOP_PERCENTILE))
    return points[np.abs(z - z_ref) <= _TOP_BAND_M]


def _yaw_and_minor(xy):
    centred = xy - np.mean(xy, axis=0)
    cov = centred.T @ centred / max(len(centred) - 1, 1)
    _values, vectors = np.linalg.eigh(cov)
    major = np.asarray(vectors[:, -1], dtype=np.float64)
    norm = float(np.linalg.norm(major))
    major = major / norm if np.isfinite(norm) and norm > 1e-12 else np.array([1.0, 0.0])
    minor = np.array([-major[1], major[0]])
    return _fold_yaw(float(np.arctan2(major[1], major[0]))), minor


def _fold_yaw(yaw):
    return float((yaw + np.pi / 2.0) % np.pi - np.pi / 2.0)


def _minor_width(xy, minor):
    """Robust physical extent: trim isolated outliers, retain real edges."""
    projection = xy @ minor
    p5, p95 = np.percentile(projection, [5.0, 95.0])
    span = float(p95 - p5)
    if not np.isfinite(span) or span <= 1e-12:
        return 0.0
    margin = 0.20 * span
    inliers = projection[(projection >= p5 - margin) & (projection <= p95 + margin)]
    return span if len(inliers) < 2 else float(np.ptp(inliers))


def _touches_border(mask):
    return bool(mask.size and (mask[0].any() or mask[-1].any() or mask[:, 0].any() or mask[:, -1].any()))


def _confidence(n_points, width, touches_border):
    support = min(1.0, n_points / float(_FULL_SUPPORT_POINTS))
    width_score = (1.0 if width <= _MAX_GRASP_WIDTH_M else
                   max(0.0, 1.0 - (width - _MAX_GRASP_WIDTH_M) / _MAX_GRASP_WIDTH_M))
    return float(np.clip(support * width_score * (0.25 if touches_border else 1.0), 0.0, 1.0))
