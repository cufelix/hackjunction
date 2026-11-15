"""Depth-first novel-object proposals and deterministic instruction grounding.

This module is deliberately independent of the known HSV detector.  The caller
invokes it only behind ``perception.novel_objects`` and gives it the already
accepted known detections.  A component with known-object dimensions and a
nearby known-colour detection is left byte-for-byte on the old path; an
elongated/large component supersedes a coincident colour blob so a red marker
cannot silently become a red cube.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np


PALETTE = ("black", "white", "grey", "red", "orange", "yellow", "green", "blue", "purple", "brown")
SHAPE_WORDS = {
    "round": ("ball", "sphere", "orb", "round"),
    "elongated": ("marker", "pen", "pencil", "rod", "stick", "wand", "spoon", "spatula", "bar"),
    "with-handle": ("pan", "cup", "mug", "handled", "handle", "skillet", "saucepan", "pot", "jug", "kettle", "ladle"),
    "box-like": ("carton", "packet"),
}
_GENERIC_NOUNS = {"object", "item", "thing", "piece", "one", "ones", "objects", "items", "things", "pieces",
                  "blocks", "block", "shapes", "shape", "stuff", "cube", "cubes", "cylinder", "cylinders",
                  "triangle", "triangles", "prism", "prisms"}
_KNOWN_FAMILY = ("red", "orange", "yellow", "blue", "purple")   # known-task colours the existing detector owns
_ROBOT_COLOURS = ("black", "grey", "white")   # G1 hands are black, forearms white/grey
_KNOWN_SHAPES = ("cube", "cylinder", "triangle")
_BIN = re.compile(r"\b(left|centre|center|middle|mid|central|right)(?:[- ]hand)?(?:\s+(?:bins?|box(?:es)?|crates?|buckets?|baskets?|containers?|trays?|bowls?))?\b", re.I)
_BIN_ALIAS = {"centre": "center", "middle": "center", "mid": "center", "central": "center"}
_WORDS = re.compile(r"[a-z]+", re.I)
# One filler/reserved vocabulary for match_instruction and unknown_words.
_FILLER = frozenset(set(PALETTE) | _GENERIC_NOUNS | {
    "put", "place", "move", "pick", "take", "drop", "sort", "the", "a", "an",
    "in", "into", "to", "on", "bin", "left", "right", "center", "centre", "middle", "mid", "central", "hand",
    "please", "up", "and", "small", "little", "large", "big", "toy", "cube", "cylinder", "prism",
    "can", "could", "would", "you", "kindly", "it", "of", "with", "onto", "inside", "over", "there", "that",
    "this", "now", "then", "go", "should", "frying", "sauce", "metal", "wooden", "plastic", "kitchen", "my", "its",
    "goes", "belongs", "belong", "end", "ends", "must", "need", "needs", "want", "wants", "set", "toss", "throw",
    "deliver", "transfer", "carry", "bring", "all", "every", "each", "any", "them", "those", "these", "from",
    "where", "is", "are", "be", "get", "gets", "let", "has", "have", "which", "what", "there", "their", "s", "i",
    "me", "we", "side", "bucket", "basket", "container", "tray",
    "box", "boxes", "crate", "crates", "buckets", "baskets", "containers", "trays", "bowl", "bowls", "bins"})
_RESERVED = _FILLER | frozenset(w for ws in SHAPE_WORDS.values() for w in ws)


def novel_noun_named(text):
    """True when the instruction names a novel object: a shape-synonym noun (plural allowed) from SHAPE_WORDS only."""
    low = str(text or "").lower().replace("gray", "grey")
    words = set(_WORDS.findall(low))
    return any(w in words or f"{w}s" in words for ws in SHAPE_WORDS.values() for w in ws)


def _unknown(words):
    """Words that are neither reserved nor a plural of a reserved word (e.g. "pans", "mugs")."""
    return sorted(w for w in words if w not in _RESERVED and not (w.endswith("s") and w[:-1] in _RESERVED))


@dataclass(frozen=True)
class NovelMatch:
    """Result of deterministic grounding. ``brain_candidates`` is closed set."""

    matched: object | None
    target_bin: str | None
    brain_candidates: tuple
    unresolved_noun: str | None = None


def _fold_yaw(yaw):
    return float((float(yaw) + np.pi / 2.0) % np.pi - np.pi / 2.0)


def _bottom_only(blob):
    """Touches the bottom image edge but not the top/left/right ones: an object close to the robot, cut by the view."""
    return bool(blob[-1].any()) and not (blob[0].any() or blob[:, 0].any() or blob[:, -1].any())


def _near_frac(P, arm_pts, dist_m, max_pts=400):
    """Fraction of component points (pelvis frame) within dist_m of the posed robot surface."""
    P = np.asarray(P, float)[:, :3]
    if len(P) > max_pts:
        P = P[np.linspace(0, len(P) - 1, max_pts).astype(int)]
    try:
        from scipy.spatial import cKDTree
        d, _ = cKDTree(arm_pts).query(P, k=1, distance_upper_bound=dist_m)
        return float(np.mean(np.isfinite(d)))
    except ImportError:
        hit = np.zeros(len(P), bool)
        for i in range(0, len(arm_pts), 2000):
            A = arm_pts[i:i + 2000]
            hit |= (((P[:, None, :] - A[None, :, :]) ** 2).sum(-1) <= dist_m ** 2).any(1)
        return float(hit.mean())


def _uv_in(blob, uv, stride):
    """True if full-res pixel uv lies on the (strided) component mask."""
    u, v = int(round(float(uv[0]) / stride)), int(round(float(uv[1]) / stride))
    return 0 <= v < blob.shape[0] and 0 <= u < blob.shape[1] and bool(blob[v, u])


def dominant_colour(rgb, mask):
    """Broad, illumination-tolerant colour name from a component's pixels."""
    px = np.asarray(rgb, np.uint8)[np.asarray(mask, bool)]
    if not len(px):
        return "grey"
    # Median suppresses highlights, shadows and the counter fringe.  Convert a
    # 1x1 RGB image so hue semantics exactly match the rest of perception.
    med = np.median(px, axis=0).astype(np.uint8)
    h, s, v = cv2.cvtColor(med.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0].astype(int)
    if v < 48:
        return "black"
    if s < 35:
        return "white" if v >= 205 else "grey"
    if 4 <= h <= 18 and v < 150:
        return "brown"
    if h < 8 or h >= 170:
        return "red"
    if h < 20:
        return "orange"
    if h < 38:
        return "yellow"
    if h < 88:
        return "green"
    if h < 132:
        return "blue"
    if h < 165:
        return "purple"
    return "red"


def _axes_and_footprint(xy):
    xy = np.asarray(xy, np.float32).reshape(-1, 2)
    rect = cv2.minAreaRect(xy)
    footprint = cv2.boxPoints(rect).astype(float)
    centred = xy.astype(float) - np.mean(xy, axis=0)
    cov = centred.T @ centred / max(len(xy) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    major = vecs[:, int(np.argmax(vals))]
    if major[0] < 0:
        major = -major
    minor = np.array([-major[1], major[0]])
    pmaj, pmin = xy @ major, xy @ minor
    dims = np.array([np.percentile(pmaj, 98) - np.percentile(pmaj, 2),
                     np.percentile(pmin, 98) - np.percentile(pmin, 2)], float)
    if dims[1] > dims[0]:
        dims = dims[::-1]
        major, minor = minor, -major
    return footprint, major, minor, dims


def _handle_pixels(blob, max_aspect, extension):
    """Return (mask, image-axis) for a thick body plus one thin protrusion."""
    yy, xx = np.nonzero(blob)
    if len(xx) < 20:
        return None, None
    x0, x1, y0, y1 = xx.min(), xx.max() + 1, yy.min(), yy.max() + 1
    crop = np.ascontiguousarray(blob[y0:y1, x0:x1].astype(np.uint8))
    dist = cv2.distanceTransform(crop, cv2.DIST_L2, 5)
    cy, cx = np.unravel_index(int(np.argmax(dist)), dist.shape)
    radius = float(dist[cy, cx])
    if radius < 2.0:
        return None, None
    pts = np.column_stack(np.nonzero(crop))[:, ::-1].astype(float)  # x,y
    delta = pts - [cx, cy]
    radial = np.linalg.norm(delta, axis=1)
    aspect = max(crop.shape) / max(1.0, min(crop.shape))
    if aspect > float(max_aspect) or radial.max() < float(extension) * radius:
        return None, None
    axis = delta[int(np.argmax(radial))]
    axis /= max(np.linalg.norm(axis), 1e-9)
    perp = np.array([-axis[1], axis[0]])
    along, across = delta @ axis, delta @ perp
    # Perspective turns a circle into an ellipse, but it remains two-sided.
    # A handle is a strongly one-sided extension from its thick body.
    if float(along.max()) < 1.35 * max(float(-along.min()), 1.0):
        return None, None
    # Past the thick body's tangent.  A box corner never reaches this gate;
    # the rectangular pan handle contributes a coherent patch.
    take = (along > 0.90 * radius) & (np.abs(across) < 0.75 * radius)
    if take.sum() < max(12, int(0.02 * len(pts))) or np.percentile(along[take], 90) < 1.30 * radius:
        return None, None
    out = np.zeros_like(blob, bool)
    hp = pts[take].astype(int)
    out[hp[:, 1] + y0, hp[:, 0] + x0] = True
    return out, axis


def _shape(blob, dims, cfg):
    contours, _ = cv2.findContours(blob.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour = max(contours, key=cv2.contourArea)
    area, perimeter = float(cv2.contourArea(contour)), float(cv2.arcLength(contour, True))
    circularity = 0.0 if perimeter <= 0 else 4.0 * np.pi * area / (perimeter * perimeter)
    aspect = float(dims[0] / max(dims[1], 1e-6))
    if aspect >= float(cfg.get("elongated_aspect", 2.2)):
        return "elongated", None, None, circularity
    handle, image_axis = (None, None)
    if aspect >= float(cfg.get("handle_min_aspect", 1.5)):
        handle, image_axis = _handle_pixels(blob, cfg.get("handle_max_aspect", 2.5),
                                            cfg.get("handle_extension", 1.45))
    if handle is not None:
        return "with-handle", handle, image_axis, circularity
    if circularity >= float(cfg.get("round_circularity", 0.70)) and aspect <= float(cfg.get("round_aspect", 1.45)):
        return "round", None, None, circularity
    return "box-like", None, None, circularity


def augment_detections(perception, rgb, depth, known, robot_q=None):
    """Return known detections plus depth components classified as novel.

    Malformed/missing depth returns the original list object contents unchanged.
    All work is bounded by image size and connected-component count.
    """
    if depth is None or np.ndim(depth) != 2 or rgb is None or np.shape(rgb)[:2] != np.shape(depth):
        return list(known)
    cfg = perception.cfg.get("novel") or {}
    raw_full = np.asarray(depth)
    stride = max(1, int(cfg.get("pixel_stride", 2)))
    raw = raw_full[::stride, ::stride]
    rgb_small = np.asarray(rgb)[::stride, ::stride]
    dlo, dhi = perception.cfg["depth_range_mm"]
    valid = np.isfinite(raw) & (raw >= dlo) & (raw <= dhi)
    if valid.mean() < float(perception.cfg.get("min_valid_frac", 0.2)):
        return list(known)

    vv, uu = np.nonzero(valid)
    points = np.full((*raw.shape, 3), np.nan)
    points[vv, uu] = perception._pixel_to_pelvis(uu * stride, vv * stride, raw[vv, uu].astype(float))
    h = points[vv, uu, 2] - perception.scene.counter_z
    hlo, hhi = map(float, cfg.get("height_m", (0.01, 0.25)))
    sel = (h >= hlo) & (h <= hhi)
    vv, uu = vv[sel], uu[sel]
    pm = perception.scene.pelvis_to_mat(points[vv, uu])
    counter = perception.scene.cfg["counter"]
    centre = np.asarray(counter["mat_size"], float) / 2.0
    half = np.asarray(counter["counter_size"], float) / 2.0 - float(perception.cfg.get("counter_margin", 0.02))
    ok = np.all(np.abs(pm[:, :2] - centre) <= half, axis=-1)
    bin_half = np.asarray(perception.scene.cfg["bins"]["outer_size"], float) / 2.0 + float(perception.cfg["bin_pixel_margin"])
    for ctr in perception._bin_centres_mat().values():
        ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= bin_half, axis=-1)
    fp = bool(cfg.get("fp_filter", True))
    if fp:
        # FP filter (b): bin walls/rims project a few mm outside the nominal footprint (depth edges, perspective,
        # runtime shift error), so the novel path excludes a wider band than the known detector's bin_pixel_margin.
        wide = np.asarray(perception.scene.cfg["bins"]["outer_size"], float) / 2.0 + float(cfg.get("bin_margin_m", 0.03))
        for ctr in perception._bin_centres_mat().values():
            ok &= ~np.all(np.abs(pm[:, :2] - np.asarray(ctr, float)) <= wide, axis=-1)
    good = np.zeros(raw.shape, bool)
    good[vv[ok], uu[ok]] = True
    mask = good.astype(np.uint8)
    arm_ring = arm_pts = None
    # max |arm_dq| (rad/s) the policy saw with this frame; unknown counts as moving (no 3-D exemption)
    speed = getattr(perception, "novel_arm_speed", None)
    still = speed is not None and float(speed) <= float(cfg.get("still_arm_rad_s", 0.1))
    if robot_q is not None:
        if fp:
            # FP filter (a): the robot silhouette is ALWAYS filled + dilated here, independent of the colour gate's
            # fk settings (sim_overrides switch those off, which left the arms/hands as novel blobs).
            arm = perception._fk_arm_mask(raw_full.shape, *robot_q, dilate_px=int(cfg.get("fk_dilate_px", 10)),
                                          fill=True)
        else:
            arm = perception._fk_arm_mask(raw_full.shape, *robot_q)
        if arm is not None:
            mask[arm[::stride, ::stride]] = 0
            if fp and float(cfg.get("near_arm_m", 0.04)) > 0:
                try:   # (e) robot surface in the pelvis frame (same FK the mask used), for 3-D proximity
                    arm_pts = np.concatenate([P for P, _ in perception._arm_surface.posed_links(*robot_q)], axis=0)
                except Exception:
                    arm_pts = None
            if fp and int(cfg.get("near_arm_px", 40)) > 0:
                # (e) ring just outside the FK silhouette: where a lagging/approximate FK leaves robot fringes.
                r = max(1, int(cfg.get("near_arm_px", 40)) // stride)
                small = arm[::stride, ::stride].astype(np.uint8)
                arm_ring = (cv2.dilate(small, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1)))
                            > 0) & (small == 0)
        elif fp:
            return list(known)   # FK mask failed = arms would read as objects: no novel proposals this frame
    k = max(0, int(cfg.get("morph_px", 3)))
    if k:
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    n, labels, stats, cents = cv2.connectedComponentsWithStats(mask, connectivity=8)
    scale_area = stride * stride
    min_area = max(8, int(np.ceil(float(cfg.get("min_area_px", 45)) / scale_area)))
    max_area = int(np.ceil(float(cfg.get("max_area_px", 30000)) / scale_area))
    known_lo, known_hi = map(float, cfg.get("known_size_m", (0.035, 0.070)))
    match_px = float(cfg.get("known_match_px", 45.0))
    novels, superseded = [], set()
    from .perception import Detection

    for i in range(1, n):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not min_area <= area <= max_area:
            continue
        blob = labels == i
        use = blob & valid
        if int(use.sum()) < min_area:
            continue
        P = points[use]
        if len(P) < 20 or not np.all(np.isfinite(P)):
            continue
        footprint, major, minor, dims = _axes_and_footprint(P[:, :2])
        cu, cv = float(cents[i, 0] * stride), float(cents[i, 1] * stride)
        nearby = [d for d in known if np.hypot(float(d.uv[0]) - cu, float(d.uv[1]) - cv) <= match_px]
        known_geometry = known_lo <= dims[0] <= known_hi and known_lo <= dims[1] <= known_hi
        colour = dominant_colour(rgb_small, blob)
        if fp:
            border = bool(blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any())
            height98 = float(np.percentile(P[:, 2], 98.0) - perception.scene.counter_z)
            if border and cfg.get("drop_border", True) and not (
                    cfg.get("border_keep_bottom", True) and still and _bottom_only(blob)
                    and arm_pts is not None
                    and _near_frac(P, arm_pts, float(cfg.get("near_arm_m", 0.04))) < float(cfg.get("near_arm_frac", 0.3))):
                continue   # (c) partial view at the image edge: robot body, table edge, objects half out of view
            if height98 < float(cfg.get("min_height_m", 0.015)) or dims[1] < float(cfg.get("min_width_m", 0.015)):
                continue   # (c) flat (shadow, mat print, table edge) or a thin strip (mask fringe, wall edge)
            if arm_ring is not None:
                frac = float(cfg.get("near_arm_frac", 0.3) if colour in _ROBOT_COLOURS
                             else cfg.get("near_arm_frac_coloured", 0.3))
                if float((blob & arm_ring).sum()) >= frac * float(blob.sum()):
                    # (e) blob hugging the FK silhouette in the image: the robot's own hand/forearm, or a fingertip
                    # fringe that picked up the bin/object colour behind it -- unless it is clearly away from the
                    # robot in 3-D (an object on the table right below a hovering hand, e.g. the pan's disc).
                    near_m = float(cfg.get("near_arm_m", 0.04))
                    # the 3-D exemption needs a still arm: while the arm moves the image lags the joints, so the
                    # hand's own pixels sit several cm from the FK surface.
                    if (arm_pts is None or near_m <= 0 or not still
                            or _near_frac(P, arm_pts, near_m) >= frac):
                        continue
            # (d) a blob at a known detection of the SAME known-family colour stays known unless it is clearly longer
            # than any known object (a 10 cm marker or a pan next to a cube). Merges with arms/bin walls, which made
            # a red cube read as 'red with-handle', are removed by (a)/(b) before this point.
            same = [d for d in nearby if d.color in _KNOWN_FAMILY and d.color == colour]
            if colour in _KNOWN_FAMILY:
                inside = [d for d in known if d.color == colour and _uv_in(blob, d.uv, stride)]
                if len(inside) >= 2:
                    continue   # (f) two same-colour known objects touching read as one larger blob
                if (inside or same) and arm_ring is not None and bool((blob & arm_ring).any()):
                    continue   # (f) a known object merged with the hand fringe; the known detector already has it
            elongated = dims[0] / max(dims[1], 1e-6) >= float(cfg.get("elongated_aspect", 2.2))
            if same and not elongated and dims[0] < float(cfg.get("supersede_same_colour_ratio", 1.4)) * known_hi:
                continue   # elongated (marker) keeps the stock supersede_min_ratio rule below
        known_family = colour in _KNOWN_FAMILY
        # Never use the novel path to bypass a rejected/ambiguous known-task
        # colour.  Known-size family-colour geometry belongs exclusively to
        # the existing detector, even when that detector prudently abstains.
        if known_geometry and (nearby or known_family):
            continue
        # Only a duplicate of the same object may supersede a known detection: a novel blob of another colour never
        # supersedes a known-family cube/cylinder/triangle, which stays with the existing detector.
        if any(d.color in _KNOWN_FAMILY and d.shape in _KNOWN_SHAPES and d.color != colour for d in nearby):
            continue
        # only a clearly larger component may supersede a coincident colour detection.
        if nearby and dims[0] <= known_hi * float(cfg.get("supersede_min_ratio", 1.15)):
            continue

        label, handle_mask, image_axis, circularity = _shape(blob, dims, cfg)
        object_centre_xy = np.median(P[:, :2], axis=0)
        centre_xy = object_centre_xy.copy()
        yaw = _fold_yaw(np.arctan2(major[1], major[0]))
        width = float(dims[1])
        grasp_kind = "pinch"
        handle_axis = None
        ungraspable_reason = None
        if label == "with-handle" and handle_mask is not None:
            huse = handle_mask & valid
            HP = points[huse]
            if len(HP) >= 10:
                _, hmajor, hminor, hdims = _axes_and_footprint(HP[:, :2])
                # body centre (handle pixels excluded): the handle axis points outward from it; grasp >= rim + handle_rim_margin_m (4 cm)
                BP = points[blob & valid & ~handle_mask]
                body_c = np.median(BP[:, :2], axis=0) if len(BP) >= 10 else object_centre_xy
                u = hmajor if float((np.median(HP[:, :2], axis=0) - body_c) @ hmajor) >= 0 else -hmajor
                w = np.array([-u[1], u[0]])
                rel = HP[:, :2] - body_c
                DP = BP[:, :2] - body_c
                # disc pixels only: handle pixels the handle mask missed sit in the handle band and would push the rim out to the tip
                DP = DP[np.abs(DP @ w - float(np.median(rel @ w))) > float(hdims[1]) / 2 + 0.01]
                rim = max(float(np.percentile(DP @ u, 100.0)), float(_axes_and_footprint(DP)[3][0]) / 2) if len(DP) >= 10 else float(dims[0]) / 2
                s_min = rim + float(cfg.get("handle_rim_margin_m", 0.04))
                if float((rel @ u).max()) < s_min + 0.01:
                    ungraspable_reason = "handle too short/occluded"
                else:
                    s = max(float(np.median(rel @ u)), s_min)
                    centre_xy = body_c + s * u + float(np.median(rel @ w)) * w
                    width = float(hdims[1])
                handle_axis = hmajor
                yaw = _fold_yaw(np.arctan2(hmajor[1], hmajor[0]))
                grasp_kind = "handle"
        hand_open = getattr(perception, "hand_opening_m", None)
        if grasp_kind == "pinch" and hand_open is not None and width > float(hand_open):
            ungraspable_reason = "wider than the hand, no handle"
        top_z = float(np.percentile(P[:, 2], 95.0))
        height = float(np.percentile(P[:, 2], 98.0) - perception.scene.counter_z)
        support = min(1.0, len(P) / 200.0)
        border = bool(blob[0].any() or blob[-1].any() or blob[:, 0].any() or blob[:, -1].any())
        confidence = float(np.clip(support * (0.35 if border else 1.0), 0.0, 1.0))
        det = Detection(colour, np.r_[centre_xy, top_z], area * scale_area, len(P) * scale_area, "novel_depth",
                        yaw, (cu, cv), shape=label,
                        shape_conf=min(1.0, max(0.5, circularity)))
        det.novel = True
        det.footprint_corners = footprint
        det.footprint = (float(dims[0]), float(dims[1]))
        det.height_m = round(float(height), 4)
        det.label = f"{colour} {label}"
        det.pca_axes = np.vstack([major, minor])
        det.footprint_dims = dims
        det.obstacle_center = object_centre_xy
        det.height = height
        det.handle_axis = handle_axis
        det.grasp_kind = grasp_kind
        det.generic_required = True
        det.generic_width = width
        det.generic_confidence = confidence
        det.graspable = ungraspable_reason is None and confidence >= float(perception.cfg.get(
            "novel_grasp_min_confidence", perception._root_cfg["grasp"].get("generic_min_confidence", 0.5)))
        det.ungraspable = not det.graspable
        det.ungraspable_reason = ungraspable_reason
        det.novel_target_bin = None
        det.novel_matched = False
        novels.append(det)
        superseded.update(id(d) for d in nearby)
    return [d for d in known if id(d) not in superseded] + novels


def _target_bin(low, colour):
    """Bin for one object: the first bin named after that colour in the sentence, else the first bin named at all."""
    bins = [(m.start(), _BIN_ALIAS.get(m.group(1), m.group(1))) for m in _BIN.finditer(low)]
    if not bins:
        return None
    anchor = re.search(rf"\b{re.escape(colour)}\b", low) if colour else None
    if anchor is not None:
        after = [target for pos, target in bins if pos > anchor.start()]
        if after:
            return after[0]
    return bins[0][1]


def match_instruction(text, candidates):
    """Ground a terse instruction onto exactly one visible novel candidate.

    A deterministic match requires a destination bin and at least a colour or
    recognised shape noun.  Ambiguity and unknown nouns never choose an object.
    The latter returns only the visible novel set for an optional async brain.
    """
    low = str(text or "").lower().replace("gray", "grey")
    target = _target_bin(low, None)
    colours = [c for c in PALETTE if re.search(rf"\b{re.escape(c)}\b", low)]
    shapes = [shape for shape, words in SHAPE_WORDS.items() if any(re.search(rf"\b{w}s?\b", low) for w in words)]
    unknown = _unknown(set(_WORDS.findall(low)))
    pool = [d for d in candidates if getattr(d, "novel", False)]
    matches = [d for d in pool if (not colours or d.color in colours) and (not shapes or d.shape in shapes)]
    deterministic = target is not None and (colours or shapes) and not unknown and len(matches) == 1
    if deterministic:
        return NovelMatch(matches[0], _target_bin(low, matches[0].color), (), None)
    unresolved = unknown[0] if unknown else None
    return NovelMatch(None, target, tuple(pool) if unresolved and target is not None else (), unresolved)



def unknown_words(text):
    """Words of the instruction that are neither colours, bins, shape synonyms nor filler: an open-vocabulary instruction."""
    low = str(text or "").lower().replace("gray", "grey")
    return _unknown(set(_WORDS.findall(low)))


def named_noun(text):
    """-> (noun, colour_named): the first recognised shape/category noun of the instruction (e.g. "ball") and whether a
    colour word is also present.  Used by the universal VLM cross-check of a shape-noun-only deterministic match."""
    low = str(text or "").lower().replace("gray", "grey")
    words = _WORDS.findall(low)
    noun = next((w for w in words if any(w in ws for ws in SHAPE_WORDS.values())), None)
    return noun, any(c in words for c in PALETTE)
