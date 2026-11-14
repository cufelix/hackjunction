"""Head RGB-D -> coloured objects in the pelvis frame. Shape-agnostic: an object is a coloured blob, its grasp
target is the centre of its top face.

    det = Perception(cfg, scene, calibration).detect(obs["head_rgb"], obs["head_depth"])
    # -> [Detection(color, top (3,) pelvis, n_px, n_depth, method, yaw), ...]

Per colour: HSV threshold (config.yaml perception.hsv) -> opening -> connected components. Per blob: erode, take
the pixels with valid depth, deproject them (calibration.yaml intrinsics + T_pelvis_optical), keep the top face
(points within `top_band` of a robust max height) and use the median of those as the top centre. With too few valid
depth pixels (holes, or no depth stream) the blob centroid ray is intersected with the plane at the known object
height instead (method "plane"). Blobs over the bins, off the mat, or at a wrong height are rejected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import time

import numpy as np

from .shape_id import classify_top_face   # the ONE shape decision; swap this import to change the classifier
from .shape_id import classify_top_face_diag   # diagnostics only (perception.shape_diag): never decides anything


@dataclass
class Detection:
    color: str
    top: np.ndarray            # top-face centre, pelvis frame (m)
    n_px: int
    n_depth: int
    method: str                # "depth" | "plane"
    yaw: float | None = None   # in-plane orientation of the top face (rad), info only
    uv: tuple = field(default=(0.0, 0.0))
    shape: str | None = None   # "cube" | "cylinder" | "triangle" | None (unknown)
    apex_yaw: float | None = None  # triangle only: pelvis-frame heading (rad, x fwd, y left) centre -> any one vertex (mod 120 deg)
    apex_conf: float = 0.0     # confidence of apex_yaw, 0..1 (from perception.triangle_pose)
    shape_diag: dict | None = None  # perception.shape_diag only: source, n_top, ratio_cloud, ratio_blob, path, label,
                                    # conf, near_bin, dz_top (top z - counter z, m), top_xy (list of [x, y], pelvis)

    def __repr__(self):
        x, y, z = self.top
        return (f"{self.color:<6} top=({x:+.3f},{y:+.3f},{z:+.3f}) px={self.n_px} depth_px={self.n_depth} "
                f"[{self.method}]")


@dataclass
class BinCalibration:
    """Result of one episode's depth-only alignment to the fixed bin geometry."""
    success: bool
    valid_frame: bool
    confidence: float
    xy_offset: np.ndarray
    n_points: int
    residual_m: float
    elapsed_ms: float
    reason: str


class Perception:
    def __init__(self, cfg, scene, calibration, log=None):
        self.cfg = cfg["perception"]
        self.scene = scene
        self.cal = calibration
        self.cam = self.cfg["camera"]
        self.size = float(self.cfg.get("object_size") or scene.size)
        self.height = float(self.cfg.get("object_height") or scene.object_height())
        # Pelvis-frame xy correction added to every detection. The configured value is the fail-safe fallback.
        self.base_xy_offset = np.asarray(self.cfg.get("xy_offset") or [0.0, 0.0], float)
        self.xy_offset = self.base_xy_offset.copy()
        self.calibration = None
        # opt-in shape / triangle pose (all default OFF = stock behaviour)
        self.classify_shape = bool(self.cfg.get("classify_shape", False))
        self.triangle_pose = bool(self.cfg.get("triangle_pose", False))
        # #7 near-bin guard: abstain on shape when the top face may be cut by the bin exclusion (needs classify_shape)
        self.near_bin_guard = bool(self.cfg.get("near_bin_guard", False))
        # perception.y_affine [a, b] (23-grasp-failure §11b/§13): perceived - true y = a*y + b (pelvis) -> subtract it.
        # Applied after xy_offset in both the depth and the plane path; null = off. Do not combine with a self-
        # calibration that already absorbs the constant b without re-fitting.
        ya = self.cfg.get("y_affine")
        self.y_affine = None if ya is None else (float(ya[0]), float(ya[1]))
        # passive shape diagnostics: Detection.shape_diag + one log line per detection (default OFF, no action change)
        self.shape_diag = bool(self.cfg.get("shape_diag", False))
        # no-blob shape (Oliver f857054): no metric top face -> abstain instead of the 2D-blob ratio (default OFF)
        self.no_blob_shape = bool(self.cfg.get("no_blob_shape", False))
        self.log = log if log is not None else print

    def reset_calibration(self):
        """Restore the configured fallback at the start of every episode."""
        self.xy_offset = self.base_xy_offset.copy()
        self.calibration = None

    def calibration_failure(self, reason, elapsed_ms=0.0):
        """Latch a safe failed result after an unexpected calibration error."""
        self.xy_offset = self.base_xy_offset.copy()
        self.calibration = BinCalibration(False, True, 0.0, self.xy_offset.copy(), 0, np.inf,
                                          float(elapsed_ms), str(reason))
        return self.calibration

    def calibrate_bins(self, depth):
        """Estimate a single XY correction by aligning elevated depth points to the fixed bin walls.

        A plane is fitted to the counter first, so camera z translation and small roll/pitch errors do not turn the
        counter into false wall points. Only a bounded translation is optimized: pose translation and rotation are
        intentionally not separated because they are nearly degenerate at this working distance.
        """
        started = time.perf_counter()
        c = self.cfg.get("self_calibration") or {}

        def result(success, valid, confidence, offset, n_points, residual, reason):
            out = BinCalibration(success, valid, float(confidence), np.asarray(offset, float), int(n_points),
                                 float(residual), (time.perf_counter() - started) * 1000.0, reason)
            self.calibration = out
            return out

        if not c.get("enabled", False):
            return result(False, True, 0.0, self.xy_offset.copy(), 0, np.inf, "disabled")
        if depth is None or np.ndim(depth) != 2:
            return result(False, False, 0.0, self.xy_offset.copy(), 0, np.inf, "no depth frame")
        dlo, dhi = self.cfg["depth_range_mm"]
        valid = (depth >= dlo) & (depth <= dhi)
        if valid.mean() < float(c.get("min_valid_frac", self.cfg["min_valid_frac"])):
            return result(False, False, 0.0, self.xy_offset.copy(), 0, np.inf, "insufficient valid depth")

        stride = max(1, int(c.get("pixel_stride", 3)))
        vv, uu = np.nonzero(valid[::stride, ::stride])
        vv, uu = vv * stride, uu * stride
        P = self.cal.pixel_to_pelvis(self.cam, uu, vv, depth[vv, uu].astype(float))
        sc = self.scene

        # Fit z = ax + by + c to counter points. Bins and objects are positive outliers and are trimmed iteratively.
        pm = sc.pelvis_to_mat(P)
        mat = np.asarray(sc.cfg["counter"]["mat_size"], float)
        counter = ((pm[:, 0] >= -0.03) & (pm[:, 0] <= mat[0] + 0.03) &
                   (pm[:, 1] >= -0.03) & (pm[:, 1] <= mat[1] + 0.03) &
                   (np.abs(P[:, 2] - sc.counter_z) < float(c.get("plane_z_window", 0.06))))
        plane_points = P[counter]
        if len(plane_points) < int(c.get("min_plane_points", 500)):
            return result(False, True, 0.0, self.xy_offset.copy(), 0, np.inf, "counter plane not visible")
        A = np.c_[plane_points[:, :2], np.ones(len(plane_points))]
        keep = np.ones(len(A), bool)
        coef = np.array([0.0, 0.0, sc.counter_z])
        plane_tol = float(c.get("plane_inlier_m", 0.006))
        for _ in range(3):
            if keep.sum() < int(c.get("min_plane_points", 500)):
                break
            coef = np.linalg.lstsq(A[keep], plane_points[keep, 2], rcond=None)[0]
            residual = plane_points[:, 2] - A @ coef
            center = np.median(residual)
            keep = np.abs(residual - center) <= plane_tol
        if keep.sum() < int(c.get("min_plane_points", 500)):
            return result(False, True, 0.0, self.xy_offset.copy(), 0, np.inf, "counter plane fit rejected")

        height = P[:, 2] - (np.c_[P[:, :2], np.ones(len(P))] @ coef)
        b = sc.cfg["bins"]
        gate = float(c.get("max_offset_m", 0.03))
        half_mat = np.asarray(b["outer_size"], float) / 2
        around = np.zeros(len(P), bool)
        centers = []
        for ctr_mat in b["centers"].values():
            ctr = sc.mat_to_pelvis([ctr_mat[0], ctr_mat[1], 0.0])[:2]
            centers.append(ctr)
            around |= np.all(np.abs(P[:, :2] - ctr) <= half_mat[::-1] + gate + 0.01, axis=1)
        wall = (around & (height >= float(c.get("min_wall_height_m", 0.008))) &
                (height <= float(c.get("max_wall_height_m", float(b["wall_height"]) + 0.015))))
        Q = P[wall, :2]
        max_points = int(c.get("max_points", 6000))
        if len(Q) > max_points:
            Q = Q[np.linspace(0, len(Q) - 1, max_points).astype(int)]
        min_points = int(c.get("min_wall_points", 180))
        if len(Q) < min_points:
            return result(False, True, 0.0, self.xy_offset.copy(), len(Q), np.inf, "too few bin wall points")

        # Rasterized distance-to-edge field makes the bounded search deterministic and fast on CPU.
        resolution = float(c.get("grid_resolution_m", 0.001))
        margin = gate + 0.02
        outer_half = half_mat[::-1]
        inner_half = outer_half - float(b["wall_thickness"])
        lo = np.min(np.asarray(centers) - outer_half - margin, axis=0)
        hi = np.max(np.asarray(centers) + outer_half + margin, axis=0)
        shape = np.ceil((hi - lo) / resolution).astype(int) + 1
        import cv2
        edge_image = np.full((shape[1], shape[0]), 255, np.uint8)

        def pixel(xy):
            ij = np.round((np.asarray(xy) - lo) / resolution).astype(int)
            return int(ij[0]), int(ij[1])

        for ctr in centers:
            for half in (outer_half, inner_half):
                x0, y0 = pixel(ctr - half)
                x1, y1 = pixel(ctr + half)
                cv2.rectangle(edge_image, (x0, y0), (x1, y1), 0, 1)
        distance = cv2.distanceTransform(edge_image, cv2.DIST_L2, 5) * resolution
        truncate = float(c.get("score_truncate_m", 0.012))

        def distances(offset):
            ij = np.round((Q + offset - lo) / resolution).astype(int)
            inside = ((ij[:, 0] >= 0) & (ij[:, 0] < shape[0]) &
                      (ij[:, 1] >= 0) & (ij[:, 1] < shape[1]))
            d = np.full(len(Q), truncate)
            d[inside] = distance[ij[inside, 1], ij[inside, 0]]
            return d

        def search(center, span, step):
            values = np.arange(-span, span + 0.5 * step, step)
            best_score, best = np.inf, None
            for dx in values:
                for dy in values:
                    off = center + [dx, dy]
                    if np.linalg.norm(off) > gate:
                        continue
                    d = np.minimum(distances(off), truncate)
                    score = float(np.mean(d * d))
                    if score < best_score:
                        best_score, best = score, off.copy()
            return best

        coarse_step = float(c.get("coarse_step_m", 0.003))
        estimate = search(np.zeros(2), gate, coarse_step)
        if estimate is None:
            return result(False, True, 0.0, self.xy_offset.copy(), len(Q), np.inf, "search outside gate")
        estimate = search(estimate, coarse_step, float(c.get("fine_step_m", 0.0005)))
        if estimate is None or np.linalg.norm(estimate) > gate:
            return result(False, True, 0.0, self.xy_offset.copy(), len(Q), np.inf, "offset outside gate")

        d = distances(estimate)
        inlier_m = float(c.get("inlier_m", 0.006))
        inlier_ratio = float(np.mean(d <= inlier_m))
        residual = float(np.median(d))

        # Require constraints normal to both wall directions; one visible line cannot determine a 2-D shift safely.
        def axis_support(axis):
            normal, tangent = axis, 1 - axis
            supported = np.zeros(len(Q), bool)
            for ctr in centers:
                for half in (outer_half, inner_half):
                    for sign in (-1, 1):
                        edge = ctr[normal] + sign * half[normal]
                        normal_ok = np.abs(Q[:, normal] + estimate[normal] - edge) <= inlier_m
                        tangent_ok = np.abs(Q[:, tangent] + estimate[tangent] - ctr[tangent]) <= half[tangent] + inlier_m
                        supported |= normal_ok & tangent_ok
            return int(supported.sum())

        support_x, support_y = axis_support(0), axis_support(1)
        support_min = int(c.get("min_axis_points", 35))
        point_factor = min(1.0, len(Q) / max(2.0 * min_points, 1.0))
        residual_factor = max(0.0, 1.0 - residual / max(inlier_m, 1e-9))
        axis_factor = min(1.0, min(support_x, support_y) / max(2.0 * support_min, 1.0))
        confidence = float(np.clip(inlier_ratio * point_factor * residual_factor * axis_factor, 0.0, 1.0))
        confidence_min = float(c.get("min_confidence", 0.30))
        if support_x < support_min or support_y < support_min:
            return result(False, True, confidence, self.xy_offset.copy(), len(Q), residual,
                          f"weak axis support x={support_x} y={support_y}; rejected estimate "
                          f"({1000 * estimate[0]:+.1f},{1000 * estimate[1]:+.1f}) mm")
        if inlier_ratio < float(c.get("min_inlier_ratio", 0.35)) or confidence < confidence_min:
            return result(False, True, confidence, self.xy_offset.copy(), len(Q), residual,
                          f"low confidence (inliers={inlier_ratio:.2f}); rejected estimate "
                          f"({1000 * estimate[0]:+.1f},{1000 * estimate[1]:+.1f}) mm")

        self.xy_offset = estimate.copy()
        return result(True, True, confidence, estimate, len(Q), residual,
                      f"accepted; support x={support_x} y={support_y}, inliers={inlier_ratio:.2f}")

    def masks(self, rgb):
        import cv2
        hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
        out = {}
        for color, ranges in self.cfg["hsv"].items():
            m = np.zeros(hsv.shape[:2], np.uint8)
            for lo, hi in ranges:
                m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
            k = int(self.cfg["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            out[color] = m
        return out

    def detect(self, rgb, depth, debug=False, include_bins=False):
        """Detect coloured objects.

        ``include_bins`` keeps pixels over the known bin footprints long enough for
        :meth:`_reject` to label them ``in_bin:<name>``.  They are still returned
        only through the rejected half of the ``debug=True`` result, so existing
        callers cannot accidentally pick an already sorted object.
        """
        import cv2
        c = self.cfg
        dets, rejected = [], []
        dlo, dhi = c["depth_range_mm"]
        keep = self._keep_mask(depth, exclude_bins=not include_bins)
        for color, m in self.masks(rgb).items():
            m = m & keep  # drop bin / counter pixels BEFORE grouping, so an object touching a same-colour bin survives
            k = int(c["open_px"])
            if k > 0:
                m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
            n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
            for i in range(1, n):
                area = int(stats[i, cv2.CC_STAT_AREA])
                if area < c["min_area_px"]:
                    continue
                blob = (lab == i).astype(np.uint8)
                e = int(c["erode_px"])
                core = cv2.erode(blob, np.ones((2 * e + 1, 2 * e + 1), np.uint8)) if e > 0 else blob
                if core.sum() < 0.25 * area:
                    core = blob
                v, u = np.nonzero(core)
                d = depth[v, u].astype(float) if depth is not None else np.zeros(len(u))
                ok = (d >= dlo) & (d <= dhi)
                det = None
                if ok.sum() >= c["min_valid_depth_px"]:
                    det = self._from_depth(color, u[ok], v[ok], d[ok], area, blob)
                if det is None:
                    det = self._from_plane(color, cent[i], area, int(ok.sum()))
                    if det is not None:
                        det.shape, conf, path = self._shape_conf(None, blob)
                        if self.shape_diag:
                            det.shape_diag = self._diag(det, "plane", None, blob, conf, path, None)
                why = self._reject(det, bins_first=include_bins)
                if self.shape_diag and det is not None and det.shape_diag is not None:
                    self._log_diag(det, why)
                if why:
                    rejected.append((det, why))
                else:
                    dets.append(det)
        return (dets, rejected) if debug else dets

    def _keep_mask(self, depth, exclude_bins=True):
        """uint8 mask of pixels that may belong to an object: no valid depth (unknown), or a 3D point at object
        height over the mat and outside every bin footprint."""
        c, sc = self.cfg, self.scene
        shape = depth.shape if depth is not None else (480, 640)
        keep = self._bin_hulls_mask(shape) if exclude_bins else np.full(shape, 255, np.uint8)
        if depth is None:
            return keep
        dlo, dhi = c["depth_range_mm"]
        d = depth.astype(float)
        valid = (d >= dlo) & (d <= dhi)
        if valid.mean() < c["min_valid_frac"]:
            return keep  # (almost) no depth stream: only the projected bin hulls can be removed
        keep[:] = 255    # with depth: holes stay "unknown" (kept); valid pixels are judged in 3D
        vv, uu = np.nonzero(valid)
        P = self.cal.pixel_to_pelvis(self.cam, uu, vv, d[vv, uu])
        h = (P[:, 2] - sc.counter_z) / self.height
        pm = sc.pelvis_to_mat(P)
        # Verification must also see a same-colour object stacked above the
        # 6 cm rim.  _reject() will discard elevated blobs outside bin
        # footprints, while bin blobs remain labelled ``in_bin:*``.
        upper = c["height_range"][1] if exclude_bins else max(c["height_range"][1], 4.0)
        ok = (h >= 0.25) & (h <= upper)
        if exclude_bins:
            half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + c["bin_pixel_margin"]
            for ctr in sc.cfg["bins"]["centers"].values():
                ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1)
        keep[vv[~ok], uu[~ok]] = 0
        return keep

    def _bin_hulls_mask(self, shape):
        """255 everywhere except inside the image hull of each bin box (footprint + margin, counter to rim)."""
        import cv2
        sc = self.scene
        keep = np.full(shape, 255, np.uint8)
        b = sc.cfg["bins"]
        half = np.asarray(b["outer_size"], float) / 2 + self.cfg["bin_pixel_margin"]
        for cx, cy in b["centers"].values():
            corners = np.array([[cx + sx * half[0], cy + sy * half[1], z] for sx in (-1, 1) for sy in (-1, 1)
                                for z in (0.0, float(b["wall_height"]))])
            uvz = self.cal.project(self.cam, sc.mat_to_pelvis(corners))
            if np.any(uvz[:, 2] <= 0):
                continue
            hull = cv2.convexHull(np.round(uvz[:, :2]).astype(np.int32))
            cv2.fillConvexPoly(keep, hull, 0)
        return keep

    def _near_bin(self, top):
        """True if any top-face point (pelvis frame) lies within a bin footprint + bin_pixel_margin + object size.
        Same 3D test (mat frame) as the bin exclusion in the keep mask: there the top face may have been cut by a
        straight line in xy, which turns a cube into a 'cylinder'; such a face must not be classified."""
        sc = self.scene
        pm = sc.pelvis_to_mat(np.asarray(top, float))
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + self.cfg["bin_pixel_margin"] + self.size
        return any(np.any(np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= half, axis=1))
                   for ctr in sc.cfg["bins"]["centers"].values())

    def _y_affine(self, xy):
        """y_corrected = y - (a*y + b): the measured perceived-minus-true error model in the pelvis frame."""
        a, b = self.y_affine
        out = np.array(xy, float)
        out[1] = out[1] - (a * out[1] + b)
        return out

    def _shape_of(self, top_xy, blob, near_bin=False):
        """Shape label, or None (flags off / abstain). Delegates to `classify_top_face` and decides nothing itself;
        `near_bin` (top face possibly cut by the bin exclusion) only forces an abstain."""
        return self._shape_conf(top_xy, blob, near_bin)[0]

    def _shape_conf(self, top_xy, blob, near_bin=False):
        """(label, conf, path): path is "off" / "abstain_near_bin" when decided here, else None (= the seam's)."""
        if not (self.classify_shape or self.triangle_pose):
            return None, 0.0, "off"
        if near_bin:
            return None, 0.0, "abstain_near_bin"
        shape = None if blob is None else blob.shape
        label, conf = classify_top_face(top_xy, blob, shape, use_blob=not self.no_blob_shape)
        return label, float(conf), None

    def _diag(self, det, source, top_xy, blob, conf, path, near_bin):
        """Passive shape diagnostics (perception.shape_diag). Reads only; never changes det's fields."""
        try:
            _, _, info = classify_top_face_diag(top_xy, blob, None if blob is None else blob.shape,
                                                use_blob=not self.no_blob_shape)
        except Exception as e:   # diagnostics must never break detection
            info = {"path": f"diag_error:{e!r}", "n_top": 0, "ratio_cloud": None, "ratio_blob": None}
        if path is None:          # the seam decided: report the default classifier's path
            path = info["path"]
            if source == "plane" and path == "abstain_few_points":
                path = "abstain_plane"
        top_list = None
        if top_xy is not None:
            top_list = np.asarray(top_xy, float).reshape(-1, 2).tolist()
        return {"source": source, "n_top": int(info["n_top"]), "ratio_cloud": info["ratio_cloud"],
                "ratio_blob": info["ratio_blob"], "path": path, "label": det.shape, "conf": float(conf),
                "near_bin": near_bin, "dz_top": float(det.top[2] - self.scene.counter_z), "top_xy": top_list}

    def _log_diag(self, det, why=None):
        g = det.shape_diag

        def f(x, n=3):
            return "None" if x is None else f"{x:.{n}f}"
        self.log(f"[shape_diag] {det.color} src={g['source']} n_top={g['n_top']} r_cloud={f(g['ratio_cloud'])} "
                 f"r_blob={f(g['ratio_blob'])} path={g['path']} label={g['label']} conf={g['conf']:.2f} "
                 f"near_bin={g['near_bin']} dz_top={f(g['dz_top'], 4)}" + (f" rej={why}" if why else ""))

    def _from_depth(self, color, u, v, d, area, blob=None):
        P = self.cal.pixel_to_pelvis(self.cam, u, v, d)
        z_top = np.percentile(P[:, 2], self.cfg["top_percentile"])
        top = P[P[:, 2] >= z_top - self.cfg["top_band"]]
        if len(top) < 5:
            return None
        # pixels sample the top face unevenly (perspective): weight each by its footprint ~ depth^2
        w = (d[P[:, 2] >= z_top - self.cfg["top_band"]] / 1000.0) ** 2
        xy = (top[:, :2] * w[:, None]).sum(0) / w.sum()
        z = float(np.median(top[:, 2]))
        near_bin = self.near_bin_guard and self.classify_shape and self._near_bin(top)   # near-bin guard: abstain on shape only
        shape, conf, path = self._shape_conf(top[:, :2], blob, near_bin)
        apex_yaw, apex_conf = None, 0.0
        if shape == "triangle" and self.triangle_pose:
            from .prism_pose import robust_center, triangle_pose
            tp = triangle_pose(top[:, :2], altitude=self.size)
            if tp is not None:
                apex_yaw, apex_conf = float(tp[1]), float(tp[2])
            rc = robust_center(top[:, :2])
            if rc is not None:
                xy = rc  # unweighted, trimmed: the weighted mean is biased by the triangle's uneven pixel sampling
        xy = xy + self.xy_offset
        if self.y_affine is not None:
            xy = self._y_affine(xy)
        det = Detection(color, np.array([xy[0], xy[1], z]), area, len(u), "depth", _yaw(top[:, :2]),
                        (float(np.median(u)), float(np.median(v))))
        det.shape = shape
        if apex_yaw is not None:
            det.apex_yaw, det.apex_conf = apex_yaw, apex_conf
        if self.shape_diag:
            nb = bool(near_bin) if (self.classify_shape and self.near_bin_guard) else _safe_near_bin(self, top)
            det.shape_diag = self._diag(det, "depth", top[:, :2], blob, conf, path, nb)
        return det

    def _from_plane(self, color, uv, area, n_depth):
        """Centroid ray of the blob intersected with the horizontal plane at the object's top."""
        T = self.cal._T(self.cam)
        K = self.cal.cams[self.cam].K
        ray_opt = np.array([(uv[0] - K[0, 2]) / K[0, 0], (uv[1] - K[1, 2]) / K[1, 1], 1.0])
        ray = T[:3, :3] @ ray_opt
        o = T[:3, 3]
        z_top = self.scene.counter_z + self.height
        # the blob centroid lies between top face and front face: aim a bit below the top
        z_aim = z_top - 0.25 * self.height
        if ray[2] >= -1e-6:
            return Detection(color, np.array([np.nan, np.nan, np.nan]), area, n_depth, "plane")
        s = (z_aim - o[2]) / ray[2]
        p = o + s * ray
        p[:2] += self.xy_offset
        if self.y_affine is not None:
            p[:2] = self._y_affine(p[:2])
        return Detection(color, np.array([p[0], p[1], z_top]), area, n_depth, "plane", None, (float(uv[0]), float(uv[1])))

    def _reject(self, det, bins_first=False):
        """Rejection reason or None. Stock order: height, off_mat, in_bin. `bins_first` (#6 verification look) checks
        the bin footprints before the height so a stacked / rim object is labelled in_bin:<name>, not height."""
        if det is None or not np.all(np.isfinite(det.top)):
            return "no_3d"
        c, sc = self.cfg, self.scene
        h = (det.top[2] - sc.counter_z) / self.height
        lo, hi = c["height_range"]
        if not bins_first and not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        pm = sc.pelvis_to_mat(det.top)
        W, D = sc.cfg["counter"]["mat_size"]
        m = c["counter_margin"]
        if not (-m <= pm[0] <= W + m and -m <= pm[1] <= D + m):
            return "off_mat"
        bm = c["bin_exclusion_margin"]
        half = np.asarray(sc.cfg["bins"]["outer_size"], float) / 2 + bm
        for name, ctr in sc.cfg["bins"]["centers"].items():
            if np.all(np.abs(pm[:2] - np.asarray(ctr, float)) <= half):
                return f"in_bin:{name}"
        if bins_first and not lo <= h <= hi:
            return f"height {h:.2f} sizes"
        return None


def _safe_near_bin(perc, top):
    """Diagnostics only: the near-bin test even when the guard is off (classify_shape false)."""
    try:
        return bool(perc._near_bin(top))
    except Exception:
        return None


def _yaw(xy):
    """Principal direction of the top-face points, folded to [-pi/4, pi/4) (square symmetry). Info only."""
    if len(xy) < 10:
        return None
    import cv2
    (_, _), (_, _), ang = cv2.minAreaRect(xy.astype(np.float32))
    a = np.radians(ang)
    return float((a + np.pi / 4) % (np.pi / 2) - np.pi / 4)
