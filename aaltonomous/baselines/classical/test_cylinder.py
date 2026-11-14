"""Cylinders in the mock world (yellow, cube-sized) and the grasp.cylinder_no_yaw_vary flag. Run:
    python -m unittest aaltonomous.baselines.classical.test_cylinder -v
The mock world already renders shape "cylinder" (mock_world.Renderer._cylinder); the layouts here are forced with
sample_layout(shapes=("cylinder",)), recoloured yellow as in the real data.
"""
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from aaltonomous.interface import env as envmod
from aaltonomous.interface.tasks import sample_layout as _sample_layout
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import make_mock_env
from .policy import ClassicalPolicy


def _cylinders_only(rng, **kw):
    kw = {k: v for k, v in kw.items() if k not in ("n_min", "n_max", "shapes")}
    lay = _sample_layout(rng, n_min=3, n_max=3, shapes=("cylinder",), **kw)
    for c in lay:
        c["color"] = "yellow"
    return lay


def _run_episode(overlay, seed, layout=_cylinders_only, limit=20000):
    with mock.patch.object(envmod, "sample_layout", layout):
        env = make_mock_env(seed=seed)
        env.time_limit = 120.0          # as run.py
        o = env.reset()
    pol = ClassicalPolicy(config=load_config(overlay=overlay), log=None, safety=env.filter.cfg, time_limit=120.0)
    pol.reset({"instruction": o["instruction"], "time_limit": 120.0})
    for _ in range(limit):
        o, info = env.step(pol.act(o))
        if info["episode_over"]:
            break
    return env.backend.outcomes(pol.mapping)


class TestCylinderMock(unittest.TestCase):
    def test_mock_cylinder_is_detected_as_cylinder(self):
        L = Scene.load().cfg["layout"]
        centre = (float(np.mean(L["region_x"])), float(np.mean(L["region_y"])))

        def one(rng, **kw):
            return [{"color": "yellow", "shape": "cylinder", "x": round(centre[0], 3), "y": round(centre[1], 3),
                     "yaw": 0.0}]
        with mock.patch.object(envmod, "sample_layout", one):
            env = make_mock_env(seed=0)
            o = env.reset()
        pol = ClassicalPolicy(config=load_config(overlay="cand_r17"), log=None, safety=env.filter.cfg)
        out = env.backend.images()
        dets = pol.perception.detect(out["head_rgb"], out["head_depth"])
        self.assertEqual([(d.color, d.shape) for d in dets], [("yellow", "cylinder")])

    def test_three_cylinders_pick_and_place_on_three_seeds(self):
        for seed in (0, 1, 2):
            outc = _run_episode("cand_r17", seed)
            placed = sum(x in ("correct", "wrong") for x in outc)
            self.assertGreaterEqual(placed, 2, f"seed {seed}: {outc}")


class TestCylinderNoYawVary(unittest.TestCase):
    HIST_KEYS = ("color", "xy", "yaw", "posture", "side", "held")

    def _hist(self, posture, yaw=0.4, side="right"):
        return [{"color": "yellow", "xy": np.array([0.30, 0.0]), "yaw": yaw, "posture": posture, "side": side,
                 "held": False}]

    def _variants(self, overlay_cfg, label, posture, yaw=0.4):
        pol = ClassicalPolicy(config=overlay_cfg, log=None)
        d = SimpleNamespace(color="yellow", shape=label, top=np.array([0.30, 0.0, 0.0]))
        return pol._variants(d, self._hist(posture, yaw))

    def test_flag_on_gives_no_yaw_variants_for_a_cylinder(self):
        cfg = load_config(overlay="ab_cylinder")
        self.assertTrue(cfg["grasp"]["cylinder_no_yaw_vary"])
        for posture in ("closed_tight", "closed_demo"):
            vs = self._variants(cfg, "cylinder", posture)
            self.assertTrue(all(v["yaw"] is None for v in vs), (posture, vs))

    def test_flag_off_keeps_the_yaw_variants(self):
        cfg = load_config(overlay="cand_r17")
        vs = self._variants(cfg, "cylinder", "closed_tight")
        self.assertTrue(any(v["yaw"] is not None for v in vs), vs)

    def test_flag_off_is_identical_to_no_key_and_flag_on_only_drops_cylinder_yaw(self):
        off = load_config(overlay="cand_r17")
        absent = load_config(overlay="cand_r17")
        absent["grasp"].pop("cylinder_no_yaw_vary", None)
        on = load_config(overlay="cand_r17")
        on["grasp"]["cylinder_no_yaw_vary"] = True
        for label in ("cylinder", "cube", "triangle", None):
            for posture in ("closed_tight", "closed_demo"):
                base = self._variants(off, label, posture)
                self.assertEqual(self._variants(absent, label, posture), base, (label, posture))
                if label == "cylinder":
                    self.assertEqual(self._variants(on, label, posture),
                                     [v for v in base if v["yaw"] is None])
                else:
                    self.assertEqual(self._variants(on, label, posture), base, (label, posture))


if __name__ == "__main__":
    unittest.main()
