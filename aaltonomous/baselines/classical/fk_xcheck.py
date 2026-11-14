"""RGB-only head-extrinsic cross-check against FK points on the robot's Dex3 hands."""
from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np


@dataclass(frozen=True)
class FkXCheckResult:
    visible: bool
    pitch_deg: float
    yaw_deg: float
    confidence: float
    n_points: int
    error_px: float
    elapsed_ms: float
    reason: str
    online_pitch_deg: float | None = None
    disagreement_deg: float | None = None


def _axis_rotation(pitch_deg, yaw_deg):
    """Pelvis-frame head correction: yaw about +z, then pitch about +y."""
    p, y = np.radians([pitch_deg, yaw_deg])
    cp, sp, cy, sy = np.cos(p), np.sin(p), np.cos(y), np.sin(y)
    ry = np.array([[cp, 0.0, sp], [0.0, 1.0, 0.0], [-sp, 0.0, cp]])
    rz = np.array([[cy, -sy, 0.0], [sy, cy, 0.0], [0.0, 0.0, 1.0]])
    return rz @ ry


def project_with_correction(points, calibration, cam, pitch_deg=0.0, yaw_deg=0.0):
    """Project pelvis points through a pitch/yaw-corrected copy of the configured extrinsic."""
    points = np.asarray(points, float).reshape(-1, 3)
    c, T = calibration.cams[cam], calibration._T(cam)
    R = _axis_rotation(pitch_deg, yaw_deg) @ T[:3, :3]
    q = (points - T[:3, 3]) @ R
    z = q[:, 2]
    with np.errstate(divide="ignore", invalid="ignore"):
        u = c.K[0, 0] * q[:, 0] / z + c.K[0, 2]
        v = c.K[1, 1] * q[:, 1] / z + c.K[1, 2]
    return np.c_[u, v, z]


class FkHandXCheck:
    """Bounded image-space search. ``estimate`` is safe to run in a worker thread."""

    def __init__(self, cfg, calibration, cam="head"):
        self.cfg = cfg
        self.cal = calibration
        self.cam = cam

    def estimate(self, rgb, points, depth=None, online_pitch_deg=None):
        started = time.perf_counter()

        def result(visible, pitch=0.0, yaw=0.0, confidence=0.0, n=0, error=np.inf, reason=""):
            disagreement = None if online_pitch_deg is None else abs(float(pitch) - float(online_pitch_deg))
            if disagreement is not None and disagreement > 2.0:
                confidence *= max(0.1, 2.0 / disagreement)
                reason += f"; online disagreement {disagreement:.1f} deg"
            return FkXCheckResult(bool(visible), float(pitch), float(yaw), float(np.clip(confidence, 0.0, 1.0)),
                                  int(n), float(error), (time.perf_counter() - started) * 1000.0, reason,
                                  None if online_pitch_deg is None else float(online_pitch_deg), disagreement)

        if rgb is None or np.ndim(rgb) != 3 or np.shape(rgb)[2] != 3:
            return result(False, reason="no RGB frame")
        points = np.asarray(points, float).reshape(-1, 3)
        if not len(points) or not np.isfinite(points).all():
            return result(False, reason="no finite FK points")
        try:
            mask = self._hand_mask(np.asarray(rgb), points, depth)
        except Exception as exc:
            return result(False, reason=f"mask error: {type(exc).__name__}: {exc}")
        if mask is None or not np.any(mask):
            return result(False, reason="no hand blob")

        import cv2
        distance = cv2.distanceTransform((~mask).astype(np.uint8), cv2.DIST_L2, 5)
        coarse = self._search(points, distance,
                              np.arange(-float(self.cfg["pitch_range_deg"]),
                                        float(self.cfg["pitch_range_deg"]) + 0.01, 1.0),
                              np.arange(-float(self.cfg["yaw_range_deg"]),
                                        float(self.cfg["yaw_range_deg"]) + 0.01, 1.0))
        if coarse is None:
            return result(False, reason="FK points outside image")
        _, pitch, yaw, distances, n = coarse
        fine_step = float(self.cfg.get("fine_step_deg", 0.25))
        fine = self._search(points, distance,
                            np.arange(pitch - 1.0, pitch + 1.01, fine_step),
                            np.arange(yaw - 1.0, yaw + 1.01, fine_step),
                            pitch_bounds=(-float(self.cfg["pitch_range_deg"]), float(self.cfg["pitch_range_deg"])),
                            yaw_bounds=(-float(self.cfg["yaw_range_deg"]), float(self.cfg["yaw_range_deg"])))
        if fine is not None:
            _, pitch, yaw, distances, n = fine
        error = float(np.sqrt(np.mean(np.square(distances))))
        match_px = float(self.cfg.get("match_px", 8.0))
        coverage = float(np.mean(distances <= match_px))
        support = min(1.0, n / float(self.cfg.get("good_points", 6)))
        confidence = coverage * support * float(np.exp(-0.5 * (error / match_px) ** 2))
        min_points = int(self.cfg.get("min_points", 3))
        if n < min_points or coverage < float(self.cfg.get("min_coverage", 0.5)):
            return result(False, pitch, yaw, confidence * 0.25, n, error, "insufficient hand support")
        return result(True, pitch, yaw, confidence, n, error, "accepted")

    def _search(self, points, distance, pitches, yaws, pitch_bounds=None, yaw_bounds=None):
        h, w = distance.shape
        truncate = float(self.cfg.get("score_truncate_px", 80.0))
        best = None
        for pitch in pitches:
            if pitch_bounds and not pitch_bounds[0] - 1e-9 <= pitch <= pitch_bounds[1] + 1e-9:
                continue
            for yaw in yaws:
                if yaw_bounds and not yaw_bounds[0] - 1e-9 <= yaw <= yaw_bounds[1] + 1e-9:
                    continue
                uvz = project_with_correction(points, self.cal, self.cam, pitch, yaw)
                ok = ((uvz[:, 2] > 0.05) & (uvz[:, 0] >= 0) & (uvz[:, 0] < w) &
                      (uvz[:, 1] >= 0) & (uvz[:, 1] < h))
                if not np.any(ok):
                    continue
                ij = np.rint(uvz[ok, :2]).astype(int)
                ij[:, 0] = np.clip(ij[:, 0], 0, w - 1)
                ij[:, 1] = np.clip(ij[:, 1], 0, h - 1)
                d = np.minimum(distance[ij[:, 1], ij[:, 0]], truncate)
                score = float(np.mean(d * d))
                # Stable tie break: a smaller correction is preferable when the hand mask gives a flat minimum.
                key = (score, abs(float(pitch)) + abs(float(yaw)), abs(float(yaw)))
                if best is None or key < best[0]:
                    best = (key, float(pitch), float(yaw), d, int(ok.sum()))
        return None if best is None else (best[0][0], *best[1:])

    def _hand_mask(self, rgb, points, depth):
        import cv2
        c = self.cfg
        x = np.asarray(rgb, np.uint8)
        hi, lo = x.max(axis=2).astype(np.int16), x.min(axis=2).astype(np.int16)
        max_v, max_s = int(c.get("max_v", 105)), int(c.get("max_s", 100))
        low_sat = (hi == 0) | ((hi - lo) * 255 <= max_s * np.maximum(hi, 1))
        raw = ((hi <= max_v) & low_sat).astype(np.uint8)
        k = int(c.get("open_px", 3))
        if k > 1:
            raw = cv2.morphologyEx(raw, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
        n, lab, stats, centroids = cv2.connectedComponentsWithStats(raw, connectivity=8)
        nominal = project_with_correction(points, self.cal, self.cam)[:, :2]
        h, w = raw.shape
        visible = ((nominal[:, 0] >= -float(c.get("association_px", 170))) &
                   (nominal[:, 0] < w + float(c.get("association_px", 170))) &
                   (nominal[:, 1] >= -float(c.get("association_px", 170))) &
                   (nominal[:, 1] < h + float(c.get("association_px", 170))))
        nominal = nominal[visible]
        out = np.zeros_like(raw, bool)
        min_area, max_area = int(c.get("min_area_px", 25)), int(c.get("max_area_px", 14000))
        association = float(c.get("association_px", 170))
        for i in range(1, n):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < min_area or area > max_area or not len(nominal):
                continue
            radius = 0.5 * max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
            if np.min(np.linalg.norm(nominal - centroids[i], axis=1)) > association + radius:
                continue
            component = lab == i
            if not self._depth_agrees(component, centroids[i], points, depth):
                continue
            out |= component
        return out

    def _depth_agrees(self, component, centroid, points, depth):
        """Depth is only a rejection cue when enough component pixels contain a valid measurement."""
        if depth is None or np.shape(depth) != component.shape:
            return True
        cam = self.cal.cams[self.cam]
        d = np.asarray(depth)[component].astype(float) * float(cam.depth_scale)
        dlo, dhi = self.cfg.get("depth_range_m", [0.15, 2.0])
        d = d[(d >= dlo) & (d <= dhi)]
        if len(d) < int(self.cfg.get("min_valid_depth_px", 20)):
            return True
        uvz = project_with_correction(points, self.cal, self.cam)
        j = int(np.argmin(np.linalg.norm(uvz[:, :2] - np.asarray(centroid), axis=1)))
        return abs(float(np.median(d)) - float(uvz[j, 2])) <= float(self.cfg.get("depth_tolerance_m", 0.15))
