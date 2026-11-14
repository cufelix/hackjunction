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
    "box-like": ("box", "carton", "packet", "block"),
}
_GENERIC_NOUNS = {"object", "item", "thing", "piece", "one"}
_KNOWN_FAMILY = ("red", "orange", "yellow", "blue", "purple")   # known-task colours the existing detector owns
_KNOWN_SHAPES = ("cube", "cylinder", "triangle")
_BIN = re.compile(r"\b(left|centre|center|middle|mid|central|right)(?:[- ]hand)?(?:\s+bin)?\b", re.I)
_BIN_ALIAS = {"centre": "center", "middle": "center", "mid": "center", "central": "center"}
_WORDS = re.compile(r"[a-z]+", re.I)
# One filler/reserved vocabulary for match_instruction and unknown_words.
_FILLER = frozenset(set(PALETTE) | _GENERIC_NOUNS | {
    "put", "place", "move", "pick", "take", "drop", "sort", "the", "a", "an",
    "in", "into", "to", "on", "bin", "left", "right", "center", "centre", "middle", "mid", "central", "hand",
    "please", "up", "and", "small", "little", "large", "big", "toy", "cube", "cylinder", "prism",
    "can", "could", "would", "you", "kindly", "it", "of", "with", "onto", "inside", "over", "there", "that",
    "this", "now", "then", "go", "should", "frying", "sauce", "metal", "wooden", "plastic", "kitchen", "my", "its"})
_RESERVED = _FILLER | frozenset(w for ws in SHAPE_WORDS.values() for w in ws)


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
    good = np.zeros(raw.shape, bool)
    good[vv[ok], uu[ok]] = True
    mask = good.astype(np.uint8)
    if robot_q is not None:
        arm = perception._fk_arm_mask(raw_full.shape, *robot_q)
        if arm is not None:
            mask[arm[::stride, ::stride]] = 0
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
        if label == "with-handle" and handle_mask is not None:
            huse = handle_mask & valid
            HP = points[huse]
            if len(HP) >= 10:
                _, hmajor, hminor, hdims = _axes_and_footprint(HP[:, :2])
                centre_xy, width = np.median(HP[:, :2], axis=0), float(hdims[1])
                handle_axis = hmajor
                yaw = _fold_yaw(np.arctan2(hmajor[1], hmajor[0]))
                grasp_kind = "handle"
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
        det.graspable = confidence >= float(perception.cfg.get("novel_grasp_min_confidence",
                                                               perception._root_cfg["grasp"].get("generic_min_confidence", 0.5)))
        det.ungraspable = not det.graspable
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
