"""The ONE place a shape is decided: `classify_top_face`.

    label, conf = classify_top_face(top_xy, blob_mask=None, image_shape=None)

    top_xy       (N,2) top-face points in the pelvis frame (metres), or None / too few points
    blob_mask    full-frame 2D colour blob of the object (uint8/bool), or None
    image_shape  (H, W) of the frame the blob was cut from

    -> (label, conf): label is "cube" | "cylinder" | "triangle" or None = abstain (never guess); conf in [0, 1].

Perception._shape_of is the only caller; nothing else (planner, prism_pose) may decide a shape. To swap in another
classifier (e.g. Felix's), keep this signature and change the single import line in perception.py.

Default implementation: ratio = area / minAreaRect area (cube ~1.0, cylinder ~0.785, triangle ~0.5), measured on the
metric top-face cloud (convex hull area) when it has >= SHAPE_MIN_POINTS points, else on the 2D blob (pixels).
Border guard: a blob touching the image edge is clipped, so its top face is partial -> abstain.
"""
from __future__ import annotations

import numpy as np

SHAPE_MIN_POINTS = 40       # top-face points needed for the metric decision (= prism_pose.TRIANGLE_MIN_POINTS)
CONF_FULL_POINTS = 200      # point count at which the count term of the confidence saturates
BLOB_MIN_AREA_PX = 60
BLOB_BORDER_PX = 1          # blob pixels within this many px of the image edge -> abstain
RATIO_CUBE_MIN = 0.88       # cube ~1.0, cylinder ~0.785, triangle ~0.5
RATIO_CYL_MIN = 0.66
CONF_MARGIN = 0.06          # distance of the ratio to the nearest decision boundary at which conf reaches 1


def _label(ratio):
    """(label, confidence from the margin to the nearest threshold)."""
    label = "cube" if ratio >= RATIO_CUBE_MIN else "cylinder" if ratio >= RATIO_CYL_MIN else "triangle"
    margin = min(abs(ratio - RATIO_CUBE_MIN), abs(ratio - RATIO_CYL_MIN))
    return label, float(min(1.0, margin / CONF_MARGIN))


def _touches_border(mask, image_shape, b=BLOB_BORDER_PX):
    m = np.asarray(mask) > 0
    if m.ndim != 2 or (image_shape is not None and tuple(m.shape) != tuple(image_shape[:2])):
        return True                        # not a full-frame mask: the border cannot be checked -> abstain
    b = max(int(b), 1)
    return bool(m[:b].any() or m[-b:].any() or m[:, :b].any() or m[:, -b:].any())


def _ratio_cloud(xy):
    import cv2
    hull = cv2.convexHull(xy.astype(np.float32))
    (_, _), (w, h), _ = cv2.minAreaRect(hull)
    if w * h < 1e-12:
        return None
    return float(cv2.contourArea(hull)) / float(w * h)


def _ratio_blob(mask):
    import cv2
    m = (np.asarray(mask) > 0).astype(np.uint8)
    cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return None
    cnt = max(cnts, key=cv2.contourArea)
    area = float(m.sum()) if len(cnts) == 1 else float(cv2.contourArea(cnt))
    if area < BLOB_MIN_AREA_PX:
        return None
    _, (w, h), _ = cv2.minAreaRect(cnt)
    return area / max((w + 1.0) * (h + 1.0), 1e-9)      # +1 px: contour runs through pixel centres


def classify_top_face(top_xy, blob_mask=None, image_shape=None):
    """See module docstring. Returns (label | None, confidence)."""
    if blob_mask is not None and _touches_border(blob_mask, image_shape):
        return None, 0.0
    if top_xy is not None:
        xy = np.asarray(top_xy, float).reshape(-1, 2)
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        if len(xy) >= SHAPE_MIN_POINTS:
            r = _ratio_cloud(xy)
            if r is not None:
                label, conf = _label(r)
                return label, conf * min(1.0, len(xy) / CONF_FULL_POINTS)
    if blob_mask is not None:
        r = _ratio_blob(blob_mask)
        if r is not None:
            return _label(r)
    return None, 0.0
