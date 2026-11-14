"""perception.shape_2d_v2 (FLAG, default off): yellow 2D cube/cylinder from the blob top edge. Run:
    python -m pytest aaltonomous/baselines/classical/test_shape_2d_v2.py -q"""
import numpy as np
import cv2

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import Renderer
from .perception import Detection, Perception
from .shape_2d import decide_shape2d, shape2d_features
from .test_classical import scene_objects

SC, CAL = Scene.load(), Calibration.load()
H, W = 240, 320


def _cyl():
    m = np.zeros((H, W), np.uint8)
    cv2.ellipse(m, (160, 80), (40, 30), 0, 0, 360, 255, -1)      # top ellipse
    cv2.rectangle(m, (120, 80), (200, 95), 255, -1)              # body
    return m


def _cube():
    """Rotated square top face: the top edge is a corner, not a smooth arc."""
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, cv2.boxPoints(((160, 100), (70, 70), 45)).astype(np.int32), 255)
    return m


def _para():
    m = np.zeros((H, W), np.uint8)
    cv2.fillConvexPoly(m, cv2.boxPoints(((160, 100), (80, 60), 20)).astype(np.int32), 255)
    return m


def _axis_aligned_rect():
    m = np.zeros((H, W), np.uint8)
    cv2.rectangle(m, (120, 60), (200, 120), 255, -1)
    return m


def _label(m):
    return decide_shape2d(shape2d_features(m))


def test_cylinder():
    assert _label(_cyl()) == "cylinder"


def test_cube_and_parallelogram():
    assert _label(_cube()) == "cube"
    assert _label(_para()) == "cube"
    assert _label(_axis_aligned_rect()) is None     # flat top edge: quad_rms ~ 0 -> neither band (spec rule)


def test_tiny_and_empty_abstain():
    m = np.zeros((H, W), np.uint8)
    m[50:54, 50:54] = 255
    assert _label(m) is None and _label(np.zeros((H, W), np.uint8)) is None


def test_ambiguous_abstains():
    f = {"quad_rms": 3.0, "n_vert": 7}      # between the bands
    assert decide_shape2d(f) is None
    assert decide_shape2d({"quad_rms": 1.0, "n_vert": 7}) is None
    assert decide_shape2d({"quad_rms": None, "n_vert": 9}) is None
    rng = np.random.default_rng(0)
    noise = (rng.random((H, W)) > 0.5).astype(np.uint8) * 255
    assert _label(noise) is None


def test_axis_ratio_logged_value():
    f = shape2d_features(_cyl())
    assert f["axis_ratio"] is not None and 0 < f["axis_ratio"] <= 1


def _perc(flag, lines=None):
    cfg = load_config(overlay="none", overrides={"perception.classify_shape": True, "perception.shape_2d_v2": flag})
    return Perception(cfg, SC, CAL, log=(lines.append if lines is not None else None) or (lambda *_: None))


def _det(color):
    return Detection(color, np.zeros(3), 500, 300, "plane", None, (0.0, 0.0))


def test_blob_path_replaced_for_yellow_only():
    for color, expect in (("yellow", "cube"), ("orange", "cube"), ("red", "cube")):
        p, lines = _perc(True), []
        p.log = lines.append
        d = _det(color)
        d.shape = "cylinder"            # what a clipped blob ratio would say
        p._last_path = "blob"
        if color == "yellow":
            p._apply_shape2d(d, _cube())
            assert d.shape == "cube" and any(l.startswith("[shape2d] color=yellow") for l in lines)
        else:
            p._log_orange_candidate(_cube()) if color == "orange" else None
            assert d.shape == "cylinder"


def test_metric_shape_kept_disagreement_logged():
    p, lines = _perc(True), []
    p.log = lines.append
    d = _det("yellow")
    d.shape = "cylinder"
    p._last_path = "metric"
    p._apply_shape2d(d, _cube())
    assert d.shape == "cylinder" and "DISAGREE" in lines[0]


def _run(flag):
    rng = np.random.default_rng(3)
    objs = scene_objects(rng)
    rgb, depth = Renderer(SC, CAL).render(objs, rng, depth_noise_mm=1.0)
    lines = []
    p = Perception(load_config(overlay="none", overrides={"perception.classify_shape": True,
                                                          "perception.shape_2d_v2": flag}), SC, CAL, log=lines.append)
    return [(d.color, d.shape, tuple(np.round(d.top, 6))) for d in p.detect(rgb, depth)], lines


def test_flag_off_identical_and_flag_on_runs():
    off, loff = _run(False)
    on, lon = _run(True)
    assert not any("[shape2d]" in l for l in loff)
    assert [(c, t) for c, _, t in on] == [(c, t) for c, _, t in off]          # positions never change
    assert all(s_on == s_off for (c, s_on, _), (_, s_off, _) in zip(on, off) if c != "yellow")
    assert all("[shape2d]" in l for l in lon if "shape2d" in l)
    if any(c == "yellow" for c, _, _ in on):
        assert any("color=yellow" in l for l in lon)
