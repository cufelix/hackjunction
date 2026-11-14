"""Bare / glossy table helpers (flags perception.glare_reject, perception.reflection_reject, perception.local_bg).

Pure functions on HSV frames and blob masks; perception.py owns the depth test (is a pixel on the counter plane) and
passes it in as a callable, so nothing here touches calibration. Every function fails open (returns "no change")."""
from __future__ import annotations

import cv2
import numpy as np


def glare_kill_mask(hsv, v_min, s_max, max_px, dilate_px, halo_s_max, on_plane, max_comps=40):
    """uint8 mask (1 = remove from the colour masks) or None. Core = V >= v_min and S <= s_max (a specular highlight is
    white, never an object colour). Each SMALL core component (<= max_px) that `on_plane(ys, xs)` accepts is removed
    together with its halo: pixels within dilate_px whose S <= halo_s_max (the blend of highlight and table colour
    reads as low-S yellow). Big white areas and highlights on an object top (not on the plane) are left alone."""
    core = ((hsv[..., 2] >= v_min) & (hsv[..., 1] <= s_max)).astype(np.uint8)
    if not core.any():
        return None
    n, lab, stats, _ = cv2.connectedComponentsWithStats(core, connectivity=8)
    if n <= 1:
        return None
    order = [i for i in np.argsort(-stats[1:, cv2.CC_STAT_AREA]) + 1 if stats[i, cv2.CC_STAT_AREA] <= max_px]
    kill = np.zeros(core.shape, np.uint8)
    any_kill = False
    for i in order[:max_comps]:
        x, y, w, h = (int(stats[i, k]) for k in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        ys, xs = np.nonzero(lab[y:y + h, x:x + w] == i)
        ys, xs = ys + y, xs + x
        if on_plane(ys, xs):
            kill[ys, xs] = 1
            any_kill = True
    if not any_kill:
        return None
    if dilate_px > 0:
        k = 2 * dilate_px + 1
        halo = cv2.dilate(kill, np.ones((k, k), np.uint8)) & (hsv[..., 1] <= halo_s_max).astype(np.uint8)
        kill |= halo
    return kill


def _blob_info(crop, x0, y0, hsv):
    """crop: bool bbox-sized mask whose top-left is (x0, y0) in the frame."""
    ys, xs = np.nonzero(crop)
    ys, xs = ys + y0, xs + x0
    h, w = crop.shape
    return {"x0": x0, "y0": y0, "x1": x0 + w, "y1": y0 + h, "w": w, "h": h, "cx": x0 + 0.5 * w,
            "area": len(xs), "s": float(np.median(hsv[ys, xs, 1])), "ys": ys, "xs": xs, "crop": crop}


def reflection_indices(blobs, hsv, p, height_of):
    """Indices into `blobs` ((crop, x0, y0) per same-colour component) that are mirror images of a stronger blob.
    B is a reflection of A when: B lies below A in the image with a gap <= max_gap_px; centred under it (dx <=
    dx_frac * width); similar width and not taller than 1.5 A; median S of B <= s_ratio * median S of A; the vertically
    flipped crop of A overlaps B with IoU >= min_iou; and depth does not contradict: `height_of(ys, xs)` (object
    heights above the counter, None = no valid depth) is None or < h_max (a reflection sits at / below the plane)."""
    infos = [_blob_info(c, x0, y0, hsv) for c, x0, y0 in blobs]
    out = set()
    wlo, whi = p["width_ratio"]
    for bi, B in enumerate(infos):
        for ai, A in enumerate(infos):
            if ai == bi or A["area"] < B["area"] * 0.5:
                continue
            if B["y0"] < A["y0"] + 0.5 * A["h"] or B["y0"] - A["y1"] > p["max_gap_px"]:
                continue
            if abs(B["cx"] - A["cx"]) > p["dx_frac"] * max(A["w"], B["w"]):
                continue
            if not wlo <= B["w"] / max(A["w"], 1) <= whi or B["h"] > 1.5 * A["h"]:
                continue
            if B["s"] > p["s_ratio"] * A["s"]:
                continue
            ca = cv2.resize(A["crop"][::-1].astype(np.uint8), (B["w"], B["h"]), interpolation=cv2.INTER_NEAREST) > 0
            cb = B["crop"]
            union = int((ca | cb).sum())
            if union == 0 or (ca & cb).sum() / union < p["min_iou"]:
                continue
            h = height_of(B["ys"], B["xs"])
            if h is not None and h >= p["h_max"]:
                continue
            out.add(bi)
            break
    return out


def low_contrast_with_ring(hsv, blob_u8, other_u8, ring_px, dh, ds, min_ring=30):
    """True when the blob's median (H, S) matches the ring just around it (ring minus every colour-mask pixel): the
    'object' has the colour of the surface it lies on. Local statistic: no global background / mat model.
    blob_u8 / other_u8: full-frame 0/1 masks (other = union of all colour masks, blob included)."""
    ys, xs = np.nonzero(blob_u8)
    if not len(ys):
        return False
    pad = ring_px + 1
    y0, y1 = max(0, ys.min() - pad), min(blob_u8.shape[0], ys.max() + pad + 1)
    x0, x1 = max(0, xs.min() - pad), min(blob_u8.shape[1], xs.max() + pad + 1)
    b = blob_u8[y0:y1, x0:x1]
    k = 2 * ring_px + 1
    ring = (cv2.dilate(b, np.ones((k, k), np.uint8)) > 0) & (other_u8[y0:y1, x0:x1] == 0)
    px = hsv[y0:y1, x0:x1][ring]
    if len(px) < min_ring:
        return False
    mh, ms = float(np.median(px[:, 0])), float(np.median(px[:, 1]))
    bh, bs = float(np.median(hsv[ys, xs, 0])), float(np.median(hsv[ys, xs, 1]))
    d = abs(bh - mh)
    d = min(d, 180.0 - d)
    return d <= dh and abs(bs - ms) <= ds
