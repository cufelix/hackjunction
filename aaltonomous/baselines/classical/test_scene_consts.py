"""Scene-size dependent constants follow scene/calibration (not hard-coded 0.045). Run:
    python -m pytest aaltonomous/baselines/classical/test_scene_consts.py -q"""
import copy
from types import SimpleNamespace

import numpy as np

from aaltonomous.scene.scene import Scene

from .config import load_config
from .perception import Perception
from .prism_pose import triangle_pose

SC, CFG = Scene.load(), load_config()
FX = 600.0


def _cal(depth_scale=0.001):
    cam = SimpleNamespace(K=np.array([[FX, 0, 320.0], [0, FX, 240.0], [0, 0, 1.0]]), depth_scale=depth_scale)
    return SimpleNamespace(cams={"head": cam})


def _perc(size=None, depth_scale=0.001):
    cfg = copy.deepcopy(CFG)
    cfg["perception"]["object_size"] = size
    return Perception(cfg, SC, _cal(depth_scale))


def _area(p, z_mm=500.0, ds=None):
    blob = np.ones((10, 10), np.uint8)
    depth = np.full((10, 10), z_mm)
    return p.expected_area_px(blob, (5, 5), depth)


def test_split_area_follows_scene_size():
    coef = CFG["perception"]["split"]["area_coef"]
    assert CFG["perception"]["split"]["object_size"] is None
    assert np.isclose(_area(_perc()), coef * (FX * SC.size / 0.5) ** 2)
    a45, a50 = _area(_perc(0.045)), _area(_perc(0.05))
    assert np.isclose(a50 / a45, (0.05 / 0.045) ** 2)
    assert np.isclose(a50, coef * (FX * 0.05 / 0.5) ** 2)


def test_depth_scale_used():
    # raw depth 500 units with scale 0.002 -> 1.0 m, area / 4 vs 0.5 m
    assert np.isclose(_area(_perc(0.045, 0.002)) * 4, _area(_perc(0.045, 0.001)))


def test_triangle_altitude_scales():
    rng = np.random.default_rng(0)
    pts = {}
    for alt in (0.045, 0.05):
        a = rng.uniform(size=(400, 3))
        a /= a.sum(1, keepdims=True)
        r = alt * 2 / 3
        V = r * np.array([[np.cos(t), np.sin(t)] for t in (np.pi / 2, np.pi / 2 + 2.094395, np.pi / 2 + 4.18879)])
        pts[alt] = a @ V
    conf = {a: triangle_pose(pts[a], altitude=a)[2] for a in pts}
    assert conf[0.05] > 0.5 and conf[0.045] > 0.5
    # a 0.05 triangle judged with the old fixed 0.045 altitude loses size-term confidence
    assert triangle_pose(pts[0.05], altitude=0.05)[2] > triangle_pose(pts[0.05], altitude=0.045)[2]
