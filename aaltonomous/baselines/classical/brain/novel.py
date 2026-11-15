"""brain.find_to_detection: a find() point that matches no classical detection becomes a NOVEL Detection built from the
depth blob at that point. Minimal bridge only (no learned segmentation): region grow on depth above the counter plane,
clipped to the VLM box, 3D centroid / footprint / height, colour name from the pixels, grasp from generic_grasp.

Nothing here moves anything or touches a known detection: the result is a Detection with novel=True whose bin is
decided only by the brain's validated pick_place (planner.lookup_detection_bin returns None for it)."""
from __future__ import annotations

import numpy as np

from ..perception import Detection

SEED_RADIUS_PX = 24          # nearest above-plane pixel to the VLM point may be this far away
BOX_PAD_PX = 6               # the VLM box is rough: pad it a little
MAX_EXTENT_M = 0.20          # nothing farther than this (xy) from the seed joins the region
MIN_REGION_PX = 60
MAX_REGION_PX = 40000


def colour_name(rgb_pixels):
    """Coarse colour name from the median HSV of an (N, 3) RGB uint8 pixel array. Pure, no model."""
    import cv2
    px = np.asarray(rgb_pixels, np.uint8).reshape(-1, 1, 3)
    if px.size == 0:
        return "unknown"
    h, s, v = (float(np.median(c)) for c in cv2.cvtColor(px, cv2.COLOR_RGB2HSV).reshape(-1, 3).T)
    if v < 50:
        return "black"
    if s < 40:
        return "white" if v > 190 else "grey"
    deg = h * 2.0
    for hi, name in ((15, "red"), (40, "orange"), (70, "yellow"), (165, "green"), (200, "cyan"), (260, "blue"),
                     (300, "purple"), (345, "pink")):
        if deg < hi:
            return name
    return "red"


def _grow(mask, seed_uv, box):
    """Connected component (8-neighbourhood) of `mask` that holds the pixel nearest to seed_uv, inside the padded box.
    -> (region uint8 mask, seed (u, v)) or (None, None)."""
    import cv2
    h, w = mask.shape
    m = (mask > 0).astype(np.uint8)
    if box is not None and len(box) == 4:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        win = np.zeros_like(m)
        win[max(y1 - BOX_PAD_PX, 0):min(y2 + BOX_PAD_PX, h), max(x1 - BOX_PAD_PX, 0):min(x2 + BOX_PAD_PX, w)] = 1
        m &= win
    vs, us = np.nonzero(m)
    if len(us) == 0:
        return None, None
    d2 = (us - seed_uv[0]) ** 2 + (vs - seed_uv[1]) ** 2
    k = int(np.argmin(d2))
    if d2[k] > SEED_RADIUS_PX ** 2:
        return None, None
    n, lab = cv2.connectedComponents(m, connectivity=8)
    region = (lab == lab[vs[k], us[k]]).astype(np.uint8)
    return region, (int(us[k]), int(vs[k]))


def novel_detection(perception, rgb, depth, point_uv, label="", box=None, cfg=None):
    """-> Detection(novel=True) or None (no blob above the counter plane at the point).

    perception: the policy's Perception (camera K, pose, depth range, counter plane, xy_offset). depth: raw head depth as
    the camera delivers it (mm). point_uv: VLM point in image pixels; box: VLM box [x1, y1, x2, y2] or None.
    cfg: the brain config dict (novel_min_height_m, novel_max_width_m). A detection whose grasp is not graspable (flat,
    too wide, no support) is returned with ungraspable=True so the caller can mark it skip."""
    cfg = cfg or {}
    min_h = float(cfg.get("novel_min_height_m", 0.012))
    max_w = float(cfg.get("novel_max_width_m", 0.060))
    cam = perception.cal.cams[perception.cam]
    raw = np.asarray(depth)
    dlo, dhi = perception.cfg["depth_range_mm"]
    valid = np.isfinite(raw) & (raw >= dlo) & (raw <= dhi)
    vv, uu = np.nonzero(valid)
    if len(uu) < MIN_REGION_PX:
        return None
    P = perception._pixel_to_pelvis(uu, vv, raw[vv, uu].astype(float))
    counter_z = float(perception.scene.counter_z)
    hgt = np.full(raw.shape, -1.0)
    hgt[vv, uu] = P[:, 2] - counter_z
    above = valid & (hgt >= min_h) & (hgt <= 0.30)
    try:                                           # bins / walls are not objects (same mask detect() uses)
        keep = perception._keep_mask(raw, exclude_bins=True)
        above &= keep > 0
    except Exception:
        pass
    region, seed = _grow(above.astype(np.uint8), (float(point_uv[0]), float(point_uv[1])), box)
    if region is None:
        return None
    seed_xyz = np.full(3, np.nan)
    i = np.nonzero((uu == seed[0]) & (vv == seed[1]))[0]
    if len(i):
        seed_xyz = P[i[0]]
    sel = (region[vv, uu] > 0)
    pts = P[sel]
    if np.all(np.isfinite(seed_xyz[:2])):          # never a sprawling region: cap the xy extent around the seed
        near = np.linalg.norm(pts[:, :2] - seed_xyz[:2], axis=1) <= MAX_EXTENT_M
        keepv, keepu = vv[sel][near], uu[sel][near]
        pts = pts[near]
        region = np.zeros_like(region)
        region[keepv, keepu] = 1
    if not MIN_REGION_PX <= len(pts) <= MAX_REGION_PX:
        return None
    top_z = float(np.percentile(pts[:, 2], 95))
    height = top_z - counter_z
    xy_c = pts[:, :2].mean(axis=0)
    cov = np.cov((pts[:, :2] - xy_c).T) if len(pts) > 2 else np.eye(2) * 1e-6
    _, vec = np.linalg.eigh(cov)
    ext = []
    for k in (1, 0):                               # major, minor extent (robust: 5..95 percentile)
        proj = (pts[:, :2] - xy_c) @ vec[:, k]
        ext.append(float(np.percentile(proj, 95) - np.percentile(proj, 5)))
    foot = (round(ext[0], 4), round(ext[1], 4))
    rgb_np = np.asarray(rgb)
    ys, xs = np.nonzero(region)
    top_band = pts[:, 2] >= top_z - 0.015
    sel_v, sel_u = vv[sel], uu[sel]
    if np.all(np.isfinite(seed_xyz[:2])):
        sel_v, sel_u = sel_v[near], sel_u[near]
    colour = colour_name(rgb_np[sel_v[top_band], sel_u[top_band]] if top_band.any() else rgb_np[ys, xs])
    uv = (float(xs.mean()), float(ys.mean()))

    xy_off = np.asarray(perception.xy_offset, float)
    det = Detection(color=colour, top=np.array([xy_c[0] + xy_off[0], xy_c[1] + xy_off[1], top_z]), n_px=int(region.sum()),
                    n_depth=int(len(pts)), method="depth", uv=uv, shape=None, generic_required=True,
                    novel=True, label=str(label)[:80], footprint=foot, height_m=round(float(height), 4))
    det.ungraspable = False
    if height < float(cfg.get("novel_min_grasp_height_m", 0.02)):
        det.ungraspable = True                     # flat: the fingers cannot close on it
    try:
        from ..generic_grasp import propose_grasp
        depth_m = np.where(valid, raw, 0.0).astype(np.float64) * float(cam.depth_scale)
        prop = propose_grasp(region, depth_m, cam.K, perception._T())
    except Exception:
        prop = None
    if prop is None or prop.width <= 0.005 or prop.width > max_w or prop.confidence < 0.3:
        det.ungraspable = True                     # too wide / unsupported: the policy leaves it (0 points, never -1)
        if prop is not None:
            det.generic_confidence = float(prop.confidence)
        return det
    centre = np.asarray(prop.centre, float).copy()
    centre[:2] += perception.xy_offset
    if getattr(perception, "y_affine", None) is not None:
        centre[:2] = perception._y_affine(centre[:2])
    det.top, det.yaw = centre, float(prop.yaw)
    det.generic_width, det.generic_confidence = float(prop.width), float(prop.confidence)
    det.shape_path = "generic"
    return det
