"""Offline (CPU, no Isaac) tests of the venue twin DR variant `venue_20261115` and that the default stays untouched."""
import json
import os
import unittest
from unittest import mock

import numpy as np

from aaltonomous.scene.scene import Scene
from aaltonomous.sim.aalto_task import dr
from aaltonomous.sim.aalto_task import layout as L

OBJS = L.object_names(Scene.load())


class VenueTwin(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sc = Scene.load()
        cls.cfg = dr.load("venue_20261115")

    def test_parses(self):
        v = self.cfg["venue"]
        self.assertEqual(self.cfg["name"], "venue_20261115")
        for k in ("table_rgb", "bin_outer_rgb", "bin_inner_rgb"):
            self.assertEqual(len(v[k]), 3)
            self.assertTrue(all(0 <= c <= 255 for c in v[k]))
        self.assertEqual(set(v["object_rgb"]), set(self.sc.cfg["objects"]["colors"]))
        # the point of the twin: outer paper is red-orange (R >> G > B), inner is a low-saturation brown
        r, g, b = v["bin_outer_rgb"]
        self.assertTrue(r > 2 * g > 2 * b * 0.9 and r > 150)
        ri, gi, bi = v["bin_inner_rgb"]
        self.assertLess(max(v["bin_inner_rgb"]) - min(v["bin_inner_rgb"]), 60)

    def test_sample_deterministic_and_complete(self):
        a = dr.sample(self.cfg, 7, OBJS, self.sc.bin_names)
        b = dr.sample(self.cfg, 7, OBJS, self.sc.bin_names)
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))
        V = a["venue"]
        self.assertEqual(set(V["bins"]), set(self.sc.bin_names))
        self.assertEqual(set(V["objects"]), set(OBJS))
        self.assertNotEqual(a["venue"], dr.sample(self.cfg, 8, OBJS, self.sc.bin_names)["venue"])
        for m in [V["table"]] + [x for bb in V["bins"].values() for x in bb.values()] + list(V["objects"].values()):
            self.assertTrue(all(0.5 < c < 1.3 for c in m))

    def test_venue_stream_does_not_shift_other_sections(self):
        a = dr.sample(self.cfg, 3, OBJS, self.sc.bin_names)
        nov = {k: v for k, v in self.cfg.items() if k != "venue"}
        b = dr.sample(nov, 3, OBJS, self.sc.bin_names)
        self.assertNotIn("venue", b)
        a.pop("venue")
        self.assertEqual(json.dumps(a, sort_keys=True), json.dumps(b, sort_keys=True))

    def test_default_unchanged(self):
        with mock.patch.dict(os.environ, {}, clear=False):
            os.environ.pop("AALTO_DR", None)
            self.assertIsNone(dr.load())                       # DR (and the venue twin) off unless asked for
        tr = dr.load("train")
        self.assertNotIn("venue", tr)
        self.assertNotIn("venue", dr.sample(tr, 1, OBJS, self.sc.bin_names))
        self.assertEqual(list(self.sc.cfg["bins"]["rgb"]), [230, 110, 30])   # nominal bin colour not edited
        self.assertEqual(self.sc.cfg["objects"]["colors"]["red"], [200, 30, 30])

    def test_venue_color_linear(self):
        self.assertEqual(dr.venue_color([255, 255, 255]), [1.0, 1.0, 1.0])
        self.assertAlmostEqual(dr.venue_color([128, 128, 128])[0], 0.2158, places=3)
        self.assertAlmostEqual(dr.venue_color([128, 0, 0], srgb=False)[0], 128 / 255)
        self.assertEqual(dr.venue_color([250, 0, 0], [2, 1, 1], srgb=False)[0], 1.0)

    def test_liners_sit_on_inner_faces(self):
        bc = self.sc.cfg["bins"]
        ox, oy = bc["outer_size"]
        t, h = bc["wall_thickness"], bc["wall_height"]
        specs = dr.bin_liner_specs(bc)
        cx, cy = bc["centers"]["center"]
        centre = np.asarray(self.sc.mat_to_pelvis([cx, cy, 0.0]))
        for part, mat_off in (("wall_xlo", (-ox / 2 + t / 2, 0)), ("wall_xhi", (ox / 2 - t / 2, 0)),
                              ("wall_ylo", (0, -oy / 2 + t / 2)), ("wall_yhi", (0, oy / 2 - t / 2))):
            wall = np.asarray(self.sc.mat_to_pelvis([cx + mat_off[0], cy + mat_off[1], 0.0]))
            liner = wall + np.asarray(specs[part]["center"])
            # liner centre is closer to the bin centre than the wall centre, by half wall + half liner
            self.assertAlmostEqual(np.linalg.norm(wall - centre) - np.linalg.norm(liner - centre),
                                   t / 2 + dr.LINER_EPS / 2, places=9)
            self.assertEqual(specs[part]["size"][2], h)
            self.assertEqual(min(specs[part]["size"][:2]), dr.LINER_EPS)


if __name__ == "__main__":
    unittest.main()
