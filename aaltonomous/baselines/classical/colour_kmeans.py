"""Deterministic, bounded per-episode HSV k-means for object colour ranges.

Only hue bounds are changed.  Saturation/value bounds and the robust ranges are
the fail-safe whenever three clean colour clusters are not present.
"""
from __future__ import annotations

import numpy as np


COLOURS = ("red", "yellow", "blue")
MIN_CLUSTER_PX = 120
MIN_PURITY = 0.70
MIN_HUE_GAP = 8.0
MAX_PIXELS = 6000
MAX_ITERS = 12


def _intervals(ranges):
    """Return robust hue intervals in red-unwrapped coordinates."""
    out = {}
    for colour in COLOURS:
        rs = ranges.get(colour)
        if not rs:
            return None
        if len(rs) == 1:
            lo, hi = int(rs[0][0][0]), int(rs[0][1][0])
            out[colour] = (lo, hi, rs[0])
        elif colour == "red" and len(rs) == 2:
            low = min(rs, key=lambda r: int(r[0][0]))
            high = max(rs, key=lambda r: int(r[0][0]))
            out[colour] = (int(high[0][0]) - 180, int(low[1][0]), low)
        else:
            return None
    return out


def _inside(px, ranges, colour):
    inside = np.zeros(len(px), bool)
    for lo, hi in ranges[colour]:
        lo, hi = np.asarray(lo), np.asarray(hi)
        inside |= np.all((px >= lo) & (px <= hi), axis=1)
    return inside


def _emit(lo, hi, base):
    """Emit OpenCV hue ranges while retaining the robust S/V limits."""
    (_, s0, v0), (_, s1, v1) = base
    lo, hi = int(lo), int(hi)
    if lo < 0:
        return [[[0, int(s0), int(v0)], [hi, int(s1), int(v1)]],
                [[180 + lo, int(s0), int(v0)], [180, int(s1), int(v1)]]]
    return [[[lo, int(s0), int(v0)], [hi, int(s1), int(v1)]]]


def fit_hue_ranges(hsv_pixels, robust_ranges):
    """Fit k=3 in HSV and re-centre robust class hue ranges.

    The computation is deterministic and capped at ``MAX_PIXELS`` and
    ``MAX_ITERS``.  Clusters are assigned red/yellow/blue by unwrapped hue
    order.  ``robust_ranges`` itself is returned on any ambiguity.
    """
    px = np.asarray(hsv_pixels, dtype=np.uint8).reshape(-1, 3)
    info = {"status": "fallback", "reason": "", "centres": [], "sizes": [],
            "purities": [], "ranges": {}}
    iv = _intervals(robust_ranges)
    if iv is None:
        info["reason"] = "unsupported robust range layout"
        return robust_ranges, info

    # Initial robust memberships are used only to seed k-means and diagnose
    # ambiguity.  The final class assignment is strictly by cluster hue order.
    membership = np.full(len(px), -1, int)
    for i, colour in enumerate(COLOURS):
        hit = _inside(px, robust_ranges, colour)
        membership[(membership < 0) & hit] = i
    keep = membership >= 0
    px, membership = px[keep], membership[keep]
    counts = np.bincount(membership, minlength=3)
    if len(px) < 3 * MIN_CLUSTER_PX or np.any(counts < MIN_CLUSTER_PX):
        info["reason"] = f"insufficient three-class support {counts.tolist()}"
        return robust_ranges, info

    if len(px) > MAX_PIXELS:
        take = np.linspace(0, len(px) - 1, MAX_PIXELS).astype(int)
        px, membership = px[take], membership[take]

    hue = px[:, 0].astype(float)
    hue[hue >= 150.0] -= 180.0
    # Hue is the discriminating channel; S and V still participate without
    # letting differently lit faces split one hue class into multiple clusters.
    feat = np.c_[hue, px[:, 1].astype(float) / 12.0, px[:, 2].astype(float) / 12.0]
    centres = np.vstack([np.median(feat[membership == i], axis=0) for i in range(3)])
    labels = np.full(len(feat), -1, int)
    for _ in range(MAX_ITERS):
        new_labels = np.argmin(((feat[:, None, :] - centres[None, :, :]) ** 2).sum(axis=2), axis=1)
        if np.array_equal(new_labels, labels):
            labels = new_labels
            break
        labels = new_labels
        if any(not np.any(labels == i) for i in range(3)):
            info["reason"] = "empty cluster"
            return robust_ranges, info
        centres = np.vstack([feat[labels == i].mean(axis=0) for i in range(3)])

    order = np.argsort(centres[:, 0])
    cluster_h = centres[order, 0]
    sizes = np.array([(labels == i).sum() for i in order], int)
    purities = np.array([np.mean(membership[labels == i] == rank)
                         for rank, i in enumerate(order)], float)
    info.update(centres=[float(h) for h in cluster_h], sizes=sizes.tolist(),
                purities=[float(p) for p in purities])
    if np.any(sizes < MIN_CLUSTER_PX):
        info["reason"] = f"small cluster {sizes.tolist()}"
        return robust_ranges, info
    if np.any(purities < MIN_PURITY):
        info["reason"] = f"impure clusters {purities.round(3).tolist()}"
        return robust_ranges, info
    if np.any(np.diff(cluster_h) < MIN_HUE_GAP):
        info["reason"] = f"hue clusters too close {cluster_h.round(2).tolist()}"
        return robust_ranges, info
    for colour, h in zip(COLOURS, cluster_h):
        lo, hi, _ = iv[colour]
        if not lo <= h <= hi:
            info["reason"] = f"{colour} centre {h:.1f} outside robust [{lo},{hi}]"
            return robust_ranges, info

    bounds = {}
    for colour, h in zip(COLOURS, cluster_h):
        lo, hi, _ = iv[colour]
        shift = int(round(h - 0.5 * (lo + hi)))
        bounds[colour] = [lo + shift, hi + shift]
    # Preserve widths unless neighbouring re-centred ranges overlap.  In that
    # case their measured hue midpoint is the safest non-overlapping boundary.
    for i, (left, right) in enumerate(zip(COLOURS, COLOURS[1:])):
        if bounds[left][1] >= bounds[right][0]:
            boundary = int(np.floor(0.5 * (cluster_h[i] + cluster_h[i + 1])))
            bounds[left][1] = min(bounds[left][1], boundary)
            bounds[right][0] = max(bounds[right][0], boundary + 1)

    out = {}
    for colour in COLOURS:
        lo, hi, base = iv[colour]
        new_lo, new_hi = bounds[colour]
        if new_lo > new_hi:
            info["reason"] = f"degenerate {colour} range"
            return robust_ranges, info
        out[colour] = _emit(new_lo, new_hi, base)
        info["ranges"][colour] = [int(new_lo), int(new_hi)]
    info["status"], info["reason"] = "adapted", "three unambiguous HSV clusters"
    return out, info
