"""Tests for the shape seam shape_id.classify_top_face (the only place a shape is decided). Run:
    python -m pytest aaltonomous/baselines/classical/test_shape_id.py -q"""
import cv2
import numpy as np

from .shape_id import SHAPE_MIN_POINTS, classify_top_face

KINDS = ("cube", "cylinder", "triangle")


def _blob(kind, size=40, shift=(0, 0), canvas=(120, 160), rot=0.3):
    m = np.zeros(canvas, np.uint8)
    c = np.array([canvas[1] / 2 + shift[0], canvas[0] / 2 + shift[1]])
    if kind == "cube":
        pts = size / 2 * np.array([[1, 1], [-1, 1], [-1, -1], [1, -1]])
    elif kind == "triangle":
        R = size * 2 / 3
        pts = R * np.array([[np.cos(a), np.sin(a)] for a in 2 * np.pi * np.arange(3) / 3 + 1.2])
    if kind == "cylinder":
        cv2.circle(m, tuple(int(v) for v in c), size // 2, 255, -1)
    else:
        Rm = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
        cv2.fillPoly(m, [np.round(pts @ Rm.T + c).astype(np.int32)], 255)
    return m


def _cloud(kind, side=0.045, n=300, seed=0, rot=0.4, c=(0.3, 0.1)):
    """Uniform top-face points (m): cube side = cylinder diameter = triangle altitude = side."""
    rng = np.random.default_rng(seed)
    if kind == "cylinder":
        a, r = rng.uniform(0, 2 * np.pi, n), side / 2 * np.sqrt(rng.uniform(0, 1, n))
        return np.stack([r * np.cos(a), r * np.sin(a)], -1) + c
    if kind == "cube":
        return _rotate(rng.uniform(-side / 2, side / 2, (n, 2)), rot) + c
    V = side * 2 / 3 * np.array([[np.cos(a), np.sin(a)] for a in rot + 2 * np.pi * np.arange(3) / 3])
    return rng.dirichlet(np.ones(3), n) @ V + c


def _rotate(p, rot):
    R = np.array([[np.cos(rot), -np.sin(rot)], [np.sin(rot), np.cos(rot)]])
    return p @ R.T


def test_cloud_path_all_shapes():
    for kind in KINDS:
        for rot in (0.0, 0.3, 0.785, 1.9):
            lab, conf = classify_top_face(_cloud(kind, rot=rot))
            assert lab == kind and conf > 0.5, (kind, rot, lab, conf)


def test_cloud_path_wins_over_wrong_blob():
    """With enough points the metric cloud decides, not the blob."""
    lab, _ = classify_top_face(_cloud("triangle"), _blob("cube", 40), (120, 160))
    assert lab == "triangle"


def test_blob_path_all_shapes():
    for kind in KINDS:
        for size in (30, 40, 60):
            for rot in (0.0, 0.3, 0.785):
                lab, conf = classify_top_face(None, _blob(kind, size, rot=rot), (120, 160))
                assert lab == kind and conf > 0.3, (kind, size, rot, lab, conf)


def test_few_points_fall_back_to_blob_or_abstain():
    few = _cloud("triangle", n=SHAPE_MIN_POINTS - 1)
    assert classify_top_face(few, _blob("cube", 40), (120, 160))[0] == "cube"      # blob decides
    assert classify_top_face(few) == (None, 0.0)                                   # nothing to decide with
    assert classify_top_face(None) == (None, 0.0)
    assert classify_top_face(np.full((100, 2), np.nan)) == (None, 0.0)


def test_border_guard_abstains_even_with_a_good_cloud():
    for shift in ((70, 0), (-65, 0), (0, 50), (0, -50)):
        m = _blob("cube", 40, shift=shift)
        assert classify_top_face(None, m, (120, 160)) == (None, 0.0), shift
        assert classify_top_face(_cloud("cube"), m, (120, 160)) == (None, 0.0), shift   # top face is clipped too


def test_blob_edge_cases():
    assert classify_top_face(None, _blob("cube", 6), (120, 160))[0] is None            # too small
    assert classify_top_face(None, np.zeros((50, 50), np.uint8), (50, 50))[0] is None  # empty
    assert classify_top_face(None, _blob("cube", 40), (480, 640))[0] is None           # not a full-frame mask


def test_confidence_low_near_a_threshold():
    # a square with one corner clipped sits near the cube/cylinder boundary: it may be labelled but not confidently
    p = _cloud("cube", n=400)
    q = p[(p[:, 0] + p[:, 1]) < (p[:, 0] + p[:, 1]).max() - 0.012]
    lab, conf = classify_top_face(q)
    assert conf < classify_top_face(p)[1]
