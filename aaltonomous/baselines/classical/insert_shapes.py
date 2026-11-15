"""policy.insert_shapes (rung 3): geometry + decisions for "put each object through the hole of the same shape".

Pure helpers (numpy only, no URDF): shape -> hole, hole pose in the pelvis frame from the scene (scene.yaml is the
single source: Scene.lid_holes()), held-object yaw alignment, per-object abstain decision. The motion itself is
Planner.plan_insert (it reuses the stock grasp / carry / lower / release machinery, left arm only).

Frames: a hole polygon lives in mat coordinates; pelvis = (0.14 + y_mat, -0.45 + x_mat), i.e. a mat angle t becomes the
pelvis angle pi/2 - t. Hole yaw symmetry: square 90 deg, equilateral triangle 120 deg, circle none.

Why most objects abstain (measured, 2026-11-15, planner IK, left arm, top-down, release height): the grasp point stops
at pelvis x ~ 0.485 (y 0.26) whatever the height, joint margin or tilt budget; the holes are at x = 0.522 (triangle),
0.574 (circle, square). A held object cannot be centred over any hole, so every plan fails its IK and the object is
left on the counter (0 points). The code is complete for the day the reach exists (other arm geometry); until then it
costs nothing: an unreachable hole is an abstention, never a release over the lid (-1).
"""
from __future__ import annotations

import numpy as np

SHAPE_HOLE = {"cube": "square", "cylinder": "circle", "triangle": "triangle"}   # object shape -> hole name
SYMMETRY_DEG = {"cube": 90.0, "cylinder": None, "triangle": 120.0}              # yaw symmetry of object AND hole


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def fold(angle, sym_deg):
    """Angle folded to [-sym/2, sym/2); None symmetry (circle) -> 0.0 (any yaw is fine)."""
    if sym_deg is None:
        return 0.0
    s = np.radians(sym_deg)
    return float((angle + s / 2) % s - s / 2)


def mat_to_pelvis_angle(t):
    return float(np.pi / 2 - t)


def hole_targets(scene, z_lid_top=None, offsets_mat=None, margin=None):
    """{shape: {hole, xy (pelvis), yaw (pelvis rad of the hole's reference direction), sym_deg, z_top, margin}}.

    yaw: cube = 0 (sides along the pelvis axes), triangle = pi (apex towards the robot, -x), circle = 0 (irrelevant).
    z_top: lid plate top in the pelvis frame (nominal counter unless z_lid_top is given).
    margin: per-side clearance between object and hole (hole_clearance / 2, or `margin`).
    offsets_mat {shape: [dx, dy]}: hole centre as an offset (mat axes, m) from the lid-bin centre instead of the
    scene.yaml layout, for a physically different lid (e.g. the printed 200 mm lid). None = scene.yaml."""
    z_top = float(scene.mat_to_pelvis([0.0, 0.0, scene.lid_z[1]])[2]) if z_lid_top is None else float(z_lid_top)
    margin = float(scene.cfg["lid"]["hole_clearance"]) / 2.0 if margin is None else float(margin)
    centre = np.asarray(scene.cfg["bins"]["centers"][scene.lid_bin], float)
    out = {}
    for h in scene.lid_holes():
        shape = h["shape"]
        c_mat = h["center"]
        if offsets_mat and shape in offsets_mat:
            c_mat = centre + np.asarray(offsets_mat[shape], float)
        xy = np.asarray(scene.mat_to_pelvis(np.r_[c_mat, 0.0])[:2], float)
        ref_mat = {"cube": np.pi / 2, "cylinder": 0.0, "triangle": -np.pi / 2}[shape]   # cube: a side along +y_mat; triangle: apex towards -y_mat
        out[shape] = {"hole": SHAPE_HOLE[shape], "xy": xy, "yaw": mat_to_pelvis_angle(ref_mat),
                      "sym_deg": SYMMETRY_DEG[shape], "z_top": z_top, "margin": margin}
    return out


def decide(det, icfg):
    """(ok, reason) for one detection: abstain unless the shape is a confident, metric top-face label.

    icfg = cfg["insert"]: min_shape_conf, require_metric. A wrong label puts the object over the wrong hole (-1/0)."""
    shape = getattr(det, "shape", None)
    if shape not in SHAPE_HOLE:
        return False, f"no confident shape (label {shape!r})"
    conf = float(getattr(det, "shape_conf", 0.0) or 0.0)
    if conf < float(icfg.get("min_shape_conf", 0.7)):
        return False, f"shape {shape} confidence {conf:.2f} < {float(icfg.get('min_shape_conf', 0.7)):.2f}"
    if icfg.get("require_metric", True) and getattr(det, "shape_path", None) != "metric":
        return False, f"shape {shape} path {getattr(det, 'shape_path', None)!r} is not metric"
    if getattr(det, "prism_orientation", None) == "lying":
        return False, "lying prism: no upright carry"
    top = np.asarray(getattr(det, "top", [np.nan] * 3), float)
    if top.shape != (3,) or not np.all(np.isfinite(top)):
        return False, "non-finite object position"
    return True, ""


def object_yaw(det, shape):
    """In-plane yaw (pelvis rad) of the object's reference direction, or None.
    cube: top-face rectangle yaw (mod 90); triangle: apex yaw (mod 120, needs perception.triangle_pose); cylinder: 0."""
    if shape == "cylinder":
        return 0.0
    if shape == "cube":
        y = getattr(det, "yaw", None)
    else:
        y = getattr(det, "apex_yaw", None)
        if y is not None and float(getattr(det, "apex_conf", 0.0)) < 0.5:
            y = None
    if y is None or not np.isfinite(y):
        return None
    return float(y)


def grasp_headings(shape, obj_yaw, side, natural_deg=25.0, max_dev_deg=75.0, phi_deg=105.0, max_n=2):
    """Candidate palm-x headings (pelvis rad) for the pick, best (closest to the arm's natural direction) first.

    cube: fingers close across a pair of faces, heading = yaw + k*90. triangle: apex + s*phi + k*120 (the calibrated
    yaw_align geometry, s = +1 left arm). cylinder: [None] = stock hints (any heading holds a round object)."""
    if shape == "cylinder":
        return [None]
    s = 1.0 if side == "left" else -1.0
    ref = -s * np.radians(natural_deg)
    if shape == "cube":
        base, step = obj_yaw, np.pi / 2
    else:
        base, step = obj_yaw + s * np.radians(phi_deg), 2 * np.pi / 3
    out = []
    for k in range(int(round(2 * np.pi / step))):
        a = _wrap(base + k * step)
        d = abs(_wrap(a - ref))
        if d <= np.radians(max_dev_deg):
            out.append((d, a))
    return [a for _, a in sorted(out)][:max_n]


def drop_heading(shape, obj_yaw, hole_yaw, a_grasp, side, natural_deg=25.0, max_dev_deg=75.0):
    """Palm-x heading at the release that turns the held object onto the hole yaw, or None when it cannot be reached.

    The object turns rigidly with the hand: final object yaw = obj_yaw + (a_drop - a_grasp). Needs
    fold(final - hole_yaw) = 0 modulo the symmetry, so a_drop = a_grasp + (hole_yaw - obj_yaw) + k * symmetry; the k
    nearest the arm's natural direction (within max_dev_deg) wins. Circle: keep the grasp heading (None)."""
    sym = SYMMETRY_DEG[shape]
    if sym is None or a_grasp is None:
        return None
    s = 1.0 if side == "left" else -1.0
    ref = -s * np.radians(natural_deg)
    step = np.radians(sym)
    base = a_grasp + (hole_yaw - obj_yaw)
    cands = [_wrap(base + k * step) for k in range(int(round(360.0 / sym)))]
    cands = [(abs(_wrap(a - ref)), a) for a in cands]
    cands = [c for c in cands if c[0] <= np.radians(max_dev_deg)]
    return min(cands)[1] if cands else None


def drop_candidates(shape, obj_yaw, hole_yaw, a_grasp):
    """EVERY palm-x release heading (pelvis rad, wrapped) that turns the held object onto the hole yaw: a_grasp + (hole_yaw -
    obj_yaw) + k * symmetry. The reach-gated planner keeps the ones the arm really attains at the hole."""
    sym = SYMMETRY_DEG[shape]
    if sym is None or a_grasp is None:
        return []
    step = np.radians(sym)
    base = a_grasp + (hole_yaw - obj_yaw)
    return [_wrap(base + k * step) for k in range(int(round(360.0 / sym)))]


def yaw_error_deg(shape, obj_yaw, a_grasp, a_drop, hole_yaw):
    """Residual yaw error (deg, folded by the symmetry) of the held object at the release; 0 for a circle."""
    sym = SYMMETRY_DEG[shape]
    if sym is None:
        return 0.0
    final = obj_yaw + (a_drop - a_grasp)
    return float(abs(np.degrees(fold(final - hole_yaw, sym))))


def release_gap_ok(held_bottom_z, z_top, gap_min=0.002, gap_max=0.015):
    """The object bottom must clear the lid plate (no scraping) but not fall far (it tips over the edge)."""
    g = held_bottom_z - z_top
    return bool(gap_min <= g <= gap_max)
