"""Shape classifier for build item #7 -- the DELTA to Oliver's `prism_pose.classify_blob`, not a
second implementation. His discriminator is right; the data it is fed is wrong.

    label, ratio = classify_top_face(top_xy)     # top_xy = metric top-face points, pelvis frame

The feature is the same one he chose: silhouette area / minAreaRect area, which for a convex
region is 1.000 for a square, pi/4 = 0.785 for a circle and 0.500 for a triangle -- scale-free and
rotation-free. The problem is the input. `classify_blob(blob)` is given the 2D colour blob, and a
colour blob is the whole SOLID seen from 42.4 deg off vertical (calibration.yaml head
T_pelvis_optical), so the visible side faces add area the feature was never meant to see.

Measured on the real pipeline (rasterise -> z-buffer -> train.yaml depth noise and holes ->
erode_px 2 -> top_percentile 92 / top_band 0.006 -> deproject), over the real sampling region
(scene.yaml layout: pelvis x 0.26-0.35, y +-0.17, yaw +-45 deg), 105 poses per shape --
reproduce with scripts/measure_shape_feature.py:

                       2D PIXEL BLOB                 METRIC TOP-FACE POINTS
    cube        0.806-0.982  ->  62.9%            0.942-0.987  ->  100.0%
    cylinder    0.822-0.859  -> 100.0%            0.792-0.858  ->  100.0%
    triangle    0.624-0.730  ->  21.0%            0.542-0.608  ->  100.0%
                             (a triangle reads as a cylinder 79% of the time)

So the fix is where the feature is computed, not how. `_from_depth` already has exactly the right
array: `top`, the metric top-face points it hands to `_yaw` (perception.py:149). Deprojected to the
pelvis frame they are the TRUE top face in metres -- no perspective, no shear -- so all three
shapes land on their ideal ratios and the bands separate with room to spare.

Two further properties worth having on the record:

  * **Immune to the dominant error term.** The per-episode camera pose bias (+-10 mm, +-1 deg,
    drawn once per episode in dr.py:102-105) is a rigid transform of the point set, and an area
    RATIO is invariant under rigid transforms. This is the one quantity in the perception stack
    that the camera-pose problem cannot touch.
  * **Blobs should not merge.** scene.yaml layout.min_spacing is 0.08 m against 0.045 m objects,
    so ~35 mm of gap, ~43 px at this range. Touching objects would break any shape feature; the
    scene generator does not produce them.

WIRING (against the starter + Oliver's feat/grasp-yaw branch), 3 lines in perception.py:

    # _from_depth already computes `top`; pass its xy straight in
    def _from_depth(self, color, u, v, d, area, shape=None):
        ...
        if self.classify_shape and shape is None:
            shape = classify_top_face(top[:, :2])[0]        # <-- metric, not the blob
        return Detection(color, ..., shape=shape, ...)

Keep `classify_blob` for the `_from_plane` fallback, where there are no 3D points at all: 61 % is
worse than 100 % and better than nothing. Keep his border abstain exactly as written -- and see
the note on `truncated` at the bottom of this file, because a clipped blob biases the POSITION
too, and that costs -1 rather than 0.
"""
from __future__ import annotations

import numpy as np

# Gap midpoints of the measured bands above, so each boundary carries the same margin:
#   triangle ceiling 0.608 | 0.700 | 0.792 cylinder floor      -> +-0.092
#   cylinder ceiling 0.858 | 0.900 | 0.942 cube floor          -> +-0.042
# Oliver's 0.66 / 0.88 also separate the measured bands, but asymmetrically: 0.88 leaves the
# cylinder only 0.022 of headroom, and the held-out DR is harsher than train.yaml by an unknown
# amount (see BUILD-HANDBOOK A.2 correction 11). Centring the thresholds costs nothing.
RATIO_CUBE_MIN = 0.90
RATIO_CYL_MIN = 0.70
MIN_TOP_POINTS = 30          # measured medians are 599-2768; this only catches a collapsed top face

IDEAL = {"cube": 1.0, "cylinder": float(np.pi / 4), "triangle": 0.5}


def classify_top_face(top_xy, min_points: int = MIN_TOP_POINTS):
    """Metric top-face points (N,2) in the pelvis frame -> (label, ratio).

    label is "cube" | "cylinder" | "triangle", or None to abstain when there are too few points
    or they are degenerate. ratio is area / minAreaRect area; see the table above for its
    measured range per shape.
    """
    import cv2

    pts = np.asarray(top_xy, dtype=np.float32)
    if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < min_points:
        return None, float("nan")
    if not np.all(np.isfinite(pts)):
        pts = pts[np.all(np.isfinite(pts), axis=1)]
        if len(pts) < min_points:
            return None, float("nan")
    hull = cv2.convexHull(pts)
    area = float(cv2.contourArea(hull))
    (_, _), (w, h), _ = cv2.minAreaRect(hull)
    rect = float(w) * float(h)
    if rect <= 0.0 or area <= 0.0:
        return None, float("nan")
    ratio = area / rect
    label = "cube" if ratio >= RATIO_CUBE_MIN else "cylinder" if ratio >= RATIO_CYL_MIN else "triangle"
    return label, float(ratio)


def touches_border(stats_row, shape_hw, border_px: int = 1):
    """True if a connected component touches the image edge, from the stats row
    `connectedComponentsWithStats` already returns -- so this costs nothing.

    The detect loop at perception.py:74 reads only CC_STAT_AREA and throws the rest away. A
    clipped blob is not merely unclassifiable (Oliver's classify_blob already abstains on it):
    its centroid and its top-face median are pulled toward the image interior, so `_from_depth`
    and `_from_plane` both return a BIASED grasp target. A biased target is a miss or a wrong
    bin, which is -1; refusing the object is 0. That makes this worth carrying on the Detection
    rather than consuming inside the classifier.

    ponytail: flag only, no reject. How often an object actually clips the frame is unmeasured,
    and rejecting one costs a guaranteed 0. Wire the flag, measure the rate at step 0, then
    decide whether to reject.
    """
    import cv2

    x, y = int(stats_row[cv2.CC_STAT_LEFT]), int(stats_row[cv2.CC_STAT_TOP])
    w, h = int(stats_row[cv2.CC_STAT_WIDTH]), int(stats_row[cv2.CC_STAT_HEIGHT])
    H, W = int(shape_hw[0]), int(shape_hw[1])
    b = max(int(border_px), 1)
    return x < b or y < b or (x + w) > (W - b) or (y + h) > (H - b)


# Measured bands from scripts/measure_shape_feature.py (105 poses per shape, real pipeline).
# The self-check asserts against these, not against the ideals: the ideals are geometry, these
# are what the pipeline actually produces.
MEASURED = {"cube": (0.942, 0.987), "cylinder": (0.792, 0.858), "triangle": (0.542, 0.608)}

# xy noise for the self-check. NOT the depth sigma: a deprojected point's xy error is dominated
# by pixel quantisation, 0.045 m / ~54 px = 0.83 mm per pixel, so ~0.4 mm rms. Depth error moves
# a point mostly along its ray, which at 42 deg off vertical lands mainly in z, and z is already
# consumed by the top_band selection.
DEMO_XY_NOISE_M = 0.0005


def _demo():
    """Self-check: ideal top faces must land inside their measured bands and get their own label."""
    rng = np.random.default_rng(0)
    s = 0.045

    def square(n):
        return rng.uniform(-s / 2, s / 2, (n, 2))

    def circle(n):
        a = rng.uniform(0, 2 * np.pi, n)
        r = (s / 2) * np.sqrt(rng.random(n))
        return np.c_[np.cos(a) * r, np.sin(a) * r]

    def triangle(n):
        side = 2 * s / np.sqrt(3)
        R = side / np.sqrt(3)
        V = np.c_[np.cos(np.radians([90, 210, 330])), np.sin(np.radians([90, 210, 330]))] * R
        u, v = rng.random(n), rng.random(n)
        flip = u + v > 1
        u[flip], v[flip] = 1 - u[flip], 1 - v[flip]
        return V[0] + u[:, None] * (V[1] - V[0]) + v[:, None] * (V[2] - V[0])

    for name, gen in (("cube", square), ("cylinder", circle), ("triangle", triangle)):
        for yaw in np.radians(np.linspace(-45, 45, 7)):
            c, si = np.cos(yaw), np.sin(yaw)
            pts = gen(1500) @ np.array([[c, si], [-si, c]])
            pts = pts + rng.normal(0, DEMO_XY_NOISE_M, pts.shape)
            label, ratio = classify_top_face(pts)
            assert label == name, f"{name} at yaw {np.degrees(yaw):.0f} -> {label} (ratio {ratio:.3f})"
            lo, hi = MEASURED[name]
            assert lo - 0.05 <= ratio <= hi + 0.05, f"{name} ratio {ratio:.3f} outside measured {lo}-{hi}"
        print(f"  {name:10} ok, ratio {ratio:.3f} (ideal {IDEAL[name]:.3f}, measured {MEASURED[name][0]}-{MEASURED[name][1]})")

    # too few points must abstain rather than guess
    assert classify_top_face(square(5))[0] is None
    assert classify_top_face(np.zeros((0, 2)))[0] is None
    assert classify_top_face(np.full((100, 2), np.nan))[0] is None

    import cv2
    row = np.zeros(5, int)
    row[cv2.CC_STAT_LEFT], row[cv2.CC_STAT_TOP] = 0, 100
    row[cv2.CC_STAT_WIDTH], row[cv2.CC_STAT_HEIGHT] = 50, 50
    assert touches_border(row, (480, 640)) is True
    row[cv2.CC_STAT_LEFT] = 300
    assert touches_border(row, (480, 640)) is False
    print("  border guard ok")
    print("all ok")


if __name__ == "__main__":
    _demo()
