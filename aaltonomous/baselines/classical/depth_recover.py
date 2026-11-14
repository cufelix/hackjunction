"""Cheap depth-supported recovery for HSV misses.

This module is imported only when ``perception.depth_blob_recover`` is on.  It
uses the existing counter-plane fit and geometry, clusters a stride-sampled
height map, and emits only blobs whose RGB pixels confidently match one of the
active closed-set HSV classes.  Unknown colours are deliberately not returned.
"""
from __future__ import annotations

import time

import cv2
import numpy as np

from .perception import colour_confidence


def _relaxed_ranges(ranges, sat_floor):
    """Copy active HSV ranges while lowering (never raising) their S floor."""
    out = {}
    for name, bands in ranges.items():
        out[name] = []
        for lo, hi in bands:
            low = list(lo)
            low[1] = min(float(low[1]), float(sat_floor))
            out[name].append((low, list(hi)))
    return out


def _known_colour(pixels, ranges, cfg):
    """Return (closed-set label, confidence), or (None, 0) on ambiguity.

    ``pixels`` are the HSV samples at the depth component's cells.  Using the
    sampled support avoids expanding every cell to a stride-by-stride block
    merely to classify colour, and (unlike that expansion) cannot mix counter
    pixels into a thin or diagonal footprint.
    """
    pixels = np.asarray(pixels, dtype=np.uint8).reshape(-1, 3)
    if len(pixels) < int(cfg["min_colour_px"]):
        return None, 0.0
    # Pick the only plausible class with the active relaxed thresholds first.
    # Calling the richer existing confidence function only for that class saves
    # two full blob scans in the <5 ms recovery budget.
    purities = []
    for name, bands in ranges.items():
        inside = np.zeros(len(pixels), bool)
        for lo, hi in bands:
            inside |= np.all((pixels >= np.asarray(lo)) & (pixels <= np.asarray(hi)), axis=1)
        purities.append((float(inside.mean()), name))
    purities.sort(reverse=True)
    best_purity, label = purities[0]
    runner_up = purities[1][0] if len(purities) > 1 else 0.0
    if best_purity - runner_up < float(cfg["min_colour_margin"]):
        return None, 0.0
    # Reuse the normal colour confidence calculation on the compact sample.
    # Its work is then independent of the full 640x480 image/blob size.
    confidence = colour_confidence(pixels[:, None, :], np.ones((len(pixels), 1), bool),
                                   ranges, label, cfg["min_colour_px"])
    if confidence < float(cfg["min_colour_conf"]):
        return None, float(confidence)
    return label, float(confidence)


def _matches_colour_detection(component, centroid, records, stride, cfg):
    """True when IoU or image-centroid distance explains this depth blob."""
    for det, mask in records:
        other = np.asarray(mask)[::stride, ::stride] > 0
        inter = int(np.count_nonzero(component & other))
        if inter:
            union = int(np.count_nonzero(component | other))
            if union and inter / union >= float(cfg["match_iou"]):
                return True
        if np.hypot(float(det.uv[0]) - centroid[0], float(det.uv[1]) - centroid[1]) <= float(cfg["match_centroid_px"]):
            return True
    return False


def _shape_hint(det, component, heights, cfg):
    """Conservative optional footprint/flatness hint for a recovered blob."""
    if not cfg.get("shape_hint", False) or det.shape is not None:
        return
    vv, uu = np.nonzero(component)
    if len(uu) < 8:
        return
    xy = np.c_[uu, vv].astype(float)
    xy -= xy.mean(0)
    cov = np.cov(xy.T)
    values, vectors = np.linalg.eigh(cov)
    if values[0] <= 1e-9:
        return
    aspect = float(np.sqrt(values[1] / values[0]))
    proj = xy @ vectors
    span = np.ptp(proj, axis=0) + 1.0
    fill = float(np.clip(len(xy) / max(float(span[0] * span[1]), 1.0), 0.0, 1.0))
    top = heights >= np.percentile(heights, 80)
    flatness = float(np.std(heights[top])) if np.count_nonzero(top) >= 5 else np.inf
    if flatness > float(cfg["shape_flatness_m"]):
        return
    label, conf = None, 0.0
    if aspect >= float(cfg["shape_prism_aspect"]) and fill < 0.82:
        label, conf = "triangle", 0.55
    # A coarse, perspective-skewed height map cannot safely separate a square
    # from a circle: both are compact and near-isotropic at stride 6.  Abstain
    # instead of attaching a wrong shape to a colour-confident candidate.  A
    # strongly elongated, flat footprint remains a useful optional prism hint.
    if label is not None:
        det.shape, det.shape_conf, det.shape_path = label, conf, "depth_blob_hint"


def recover_depth_blobs(perception, rgb, depth, hsv, colour_records=(), arm_mask=None):
    """Return ``(accepted, rejected, elapsed_ms, stats)`` for one RGB-D frame."""
    started = time.perf_counter()
    accepted, rejected = [], []
    stats = {"components": 0, "matched": 0, "unknown": 0, "accepted": 0, "plane": "known"}
    cfg = perception.depth_recover_cfg
    if depth is None or np.ndim(depth) != 2 or np.asarray(depth).shape != np.asarray(rgb).shape[:2]:
        return accepted, rejected, (time.perf_counter() - started) * 1000.0, stats

    raw = np.asarray(depth)
    if np.issubdtype(raw.dtype, np.floating) and not np.isfinite(raw).all():
        raw = np.nan_to_num(raw, nan=0.0, posinf=0.0, neginf=0.0)
    stride = int(cfg["stride"])
    sampled = raw[::stride, ::stride]
    dlo, dhi = perception.cfg["depth_range_mm"]
    valid = (sampled >= dlo) & (sampled <= dhi)
    vv, uu = np.nonzero(valid)
    if len(uu) < int(cfg["min_plane_points"]):
        return accepted, rejected, (time.perf_counter() - started) * 1000.0, stats

    plane_cfg = {"pixel_stride": stride, "plane_z_window": float(cfg["plane_z_window"]),
                 "plane_inlier_m": float(cfg["plane_inlier_m"]),
                 "min_plane_points": int(cfg["min_plane_points"]), "min_valid_frac": 0.05}
    points, plane, _ = perception._counter_plane(raw, plane_cfg)
    if points is None or len(points) != len(uu):
        points = perception._pixel_to_pelvis(uu * stride, vv * stride, sampled[vv, uu])
        plane = np.array([0.0, 0.0, perception.scene.counter_z], float)
    else:
        stats["plane"] = "fit"

    heights = points[:, 2] - (np.c_[points[:, :2], np.ones(len(points))] @ plane)
    scene = perception.scene
    pm = scene.pelvis_to_mat(points)
    width, depth_m = np.asarray(scene.cfg["counter"]["mat_size"], float)
    margin = float(perception.cfg["counter_margin"])
    foreground = ((heights >= float(cfg["min_height_m"])) & (heights <= float(cfg["max_height_m"])) &
                  (pm[:, 0] >= -margin) & (pm[:, 0] <= width + margin) &
                  (pm[:, 1] >= -margin) & (pm[:, 1] <= depth_m + margin))

    half = np.asarray(scene.cfg["bins"]["outer_size"], float) / 2 + float(perception.cfg["bin_pixel_margin"])
    for centre in perception._bin_centres_mat().values():
        foreground &= ~np.all(np.abs(pm[:, :2] - centre) <= half, axis=1)
    if arm_mask is not None and np.asarray(arm_mask).shape == raw.shape:
        # FK is a projected surface-point mask and may be sparse at full
        # resolution.  Max-pool it into the same cells as the height map so a
        # model point anywhere in a sampled cell excludes that cell.
        arm = np.asarray(arm_mask, bool)
        rows = np.arange(0, arm.shape[0], stride)
        cols = np.arange(0, arm.shape[1], stride)
        arm_cells = np.logical_or.reduceat(np.logical_or.reduceat(arm, rows, axis=0), cols, axis=1)
        foreground &= ~arm_cells[vv, uu]

    height_map = np.zeros(sampled.shape, np.float32)
    height_map[vv, uu] = heights.astype(np.float32)
    binary = np.zeros(sampled.shape, np.uint8)
    binary[vv[foreground], uu[foreground]] = 1
    count, labels, cc_stats, centroids = cv2.connectedComponentsWithStats(binary, connectivity=8)
    relaxed = _relaxed_ranges(perception.hsv, cfg["sat_floor"])

    for idx in range(1, count):
        cells = int(cc_stats[idx, cv2.CC_STAT_AREA])
        if not int(cfg["min_blob_cells"]) <= cells <= int(cfg["max_blob_cells"]):
            continue
        stats["components"] += 1
        component = labels == idx
        centroid = (float(centroids[idx, 0] * stride), float(centroids[idx, 1] * stride))
        if _matches_colour_detection(component, centroid, colour_records, stride, cfg):
            stats["matched"] += 1
            continue

        cv, cu = np.nonzero(component)
        full_v, full_u = cv * stride, cu * stride
        label, confidence = _known_colour(hsv[full_v, full_u], relaxed, cfg)
        if label is None:
            stats["unknown"] += 1
            continue

        # Only accepted components need a full-resolution mask for the legacy
        # Detection/optional-shape metadata.  Unknowns never pay this cost.
        blob = np.repeat(np.repeat(component, stride, axis=0), stride, axis=1)
        blob = blob[:raw.shape[0], :raw.shape[1]].astype(np.uint8)
        full_d = raw[full_v, full_u]
        area = cells * stride * stride
        det = perception._from_depth(label, full_u, full_v, full_d, area, blob, raw)
        if det is None:
            continue
        det.color_conf = confidence
        # Leave the legacy hue/saturation post-gate metadata unset: this path
        # has already passed its own stricter closed-set confidence + margin
        # check with the explicitly relaxed saturation floor. Reapplying the
        # colour-path floors would immediately discard the misses it recovers.
        _shape_hint(det, component, height_map[component], cfg)
        reason = perception._reject(det)
        if reason:
            rejected.append((det, f"depth_recover:{reason}"))
        else:
            accepted.append(det)
            stats["accepted"] += 1

    return accepted, rejected, (time.perf_counter() - started) * 1000.0, stats
