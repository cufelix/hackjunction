"""Bare / glossy table flags: perception.glare_reject, perception.reflection_reject, perception.local_bg (all default OFF).
Synthetic frames: the mock renderer's mat-with-grid frame vs a bare glossy table (uniform brown-grey, specular
highlights, mirror images of the objects below them). Run:
    python -m unittest aaltonomous.baselines.classical.test_bare_table -v
"""
import time
import unittest

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import FLAGS, load_config
from .mock_world import Renderer
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
H_TOP = SC.object_height()
COLOURS = ("red", "yellow", "blue")
XY = {"red": (0.25, 0.20), "yellow": (0.45, 0.24), "blue": (0.65, 0.20)}   # mat xy, in front of the bins
BARE = ("perception.glare_reject.enabled", "perception.reflection_reject.enabled", "perception.local_bg.enabled")


def _top(xy, h):
    return SC.mat_to_pelvis([float(xy[0]), float(xy[1]), h])


def _objects(colours=COLOURS):
    return [dict(color=c, shape="cube", yaw=0.0, size=SC.size, height=H_TOP, pos=_top(XY[c], H_TOP / 2)) for c in colours]


def _perception(**flags):
    ov = {"perception.hsv_set": "robust_v2"}
    ov.update(flags)
    return Perception(load_config(overlay="none", overrides=ov), SC, CAL, log=lambda *_: None)


def _all_on():
    return _perception(**{k: True for k in BARE})


def _is_table(rgb0):
    """Pixels of the empty render that are counter (the grey mat); bins / floor excluded."""
    r, g, b = (rgb0[..., i].astype(int) for i in range(3))
    return (np.abs(r - g) < 6) & (np.abs(g - b) < 8) & (r >= 170)


def mat_frame(colours=COLOURS):
    return Renderer(SC, CAL).render(_objects(colours))


def bare_frame(colours=COLOURS, highlights=(), reflections=False, seed=0):
    """Glossy bare table: brown-grey gradient + noise instead of the grey mat; `highlights` = (u, v, r) white specular
    spots with a yellowish halo (depth holes: a glossy surface returns no depth); `reflections`: mirror image of every
    object below it, half-blended with the table, depth hole."""
    rng = np.random.default_rng(seed)
    rend = Renderer(SC, CAL)
    rgb0, d0 = rend.render([])
    rgb, depth = rend.render(_objects(colours))
    tab = _is_table(rgb0)
    base = np.zeros(rgb.shape, float)
    grad = np.linspace(0.92, 1.08, rgb.shape[0])[:, None]
    base[...] = np.array([128.0, 112.0, 98.0]) * grad[..., None]
    base += rng.normal(0, 2.0, rgb.shape)
    rgb = rgb.copy()
    objmask = np.any(rgb != rgb0, axis=2)
    rgb[tab & ~objmask] = np.clip(base, 0, 255).astype(np.uint8)[tab & ~objmask]
    depth = depth.copy()
    yy, xx = np.mgrid[0:rgb.shape[0], 0:rgb.shape[1]]
    for u, v, r in highlights:
        dist = np.hypot(xx - u, yy - v)
        halo, core = (dist <= r + 6) & tab & ~objmask, (dist <= r) & tab & ~objmask
        rgb[halo] = (255, 246, 200)       # H ~ 25, S ~ 50, V 255: reads as pale yellow
        rgb[core] = (255, 255, 252)       # V 255, S ~ 3
        depth[halo] = 0
    if reflections:
        n, lab, st, _ = cv2.connectedComponentsWithStats(objmask.astype(np.uint8))
        for i in range(1, n):
            x, y, w, h = (int(st[i, k]) for k in range(4))
            crop = rgb[y:y + h, x:x + w][::-1]
            m = (lab[y:y + h, x:x + w] == i)[::-1]
            y2 = y + h + 6
            if y2 + h > rgb.shape[0]:
                continue
            dst = rgb[y2:y2 + h, x:x + w]
            dst[m] = (0.5 * dst[m] + 0.5 * crop[m]).astype(np.uint8)
            depth[y2:y2 + h, x:x + w][m] = 0
    return rgb, depth


def _colours(dets):
    return sorted(d.color for d in dets)


class TestRegistered(unittest.TestCase):
    def test_flags_registered_and_off_by_default(self):
        cfg = load_config(overlay="none")
        for k in BARE:
            self.assertIn(k, FLAGS)
            self.assertFalse(cfg["perception"][k.split(".")[1]]["enabled"])

    def test_overlay_turns_all_three_on_and_nothing_else_vs_r21(self):
        a, b = load_config(overlay="finals_r21"), load_config(overlay="finals_r22_bare")
        for k in BARE:
            self.assertTrue(b["perception"][k.split(".")[1]]["enabled"])
            a["perception"][k.split(".")[1]]["enabled"] = True
        a["overlay"] = b["overlay"]
        self.assertEqual(a, b)


class TestDetectionOnBothSurfaces(unittest.TestCase):
    def test_all_three_colours_on_mat_and_bare_flags_off_and_on(self):
        for frame in (mat_frame, bare_frame):
            rgb, depth = frame()
            for p in (_perception(), _all_on()):
                self.assertEqual(_colours(p.detect(rgb, depth)), sorted(COLOURS), (frame.__name__, p.glare_reject))

    def test_all_three_colours_without_depth(self):
        for frame in (mat_frame, bare_frame):
            rgb, _ = frame()
            self.assertEqual(_colours(_all_on().detect(rgb, None)), sorted(COLOURS), frame.__name__)

    def test_each_single_colour_on_bare_table(self):
        for c in COLOURS:
            rgb, depth = bare_frame((c,))
            self.assertEqual(_colours(_all_on().detect(rgb, depth)), [c])


class TestGlare(unittest.TestCase):
    HL = ((250, 350, 9), (450, 430, 7))

    def test_glare_reads_as_yellow_with_the_flag_off(self):
        rgb, depth = bare_frame(("red", "blue"), highlights=self.HL)
        self.assertIn("yellow", _colours(_perception().detect(rgb, depth)))   # the failure this flag exists for

    def test_glare_not_detected_with_the_flag_on(self):
        rgb, depth = bare_frame(("red", "blue"), highlights=self.HL)
        p = _perception(**{"perception.glare_reject.enabled": True})
        self.assertEqual(_colours(p.detect(rgb, depth)), ["blue", "red"])
        self.assertGreater(p.bare_stats["glare_px"], 0)

    def test_glare_not_detected_without_depth(self):
        rgb, _ = bare_frame(("red", "blue"), highlights=self.HL)
        p = _perception(**{"perception.glare_reject.enabled": True})
        self.assertEqual(_colours(p.detect(rgb, None)), ["blue", "red"])

    def test_highlight_on_an_object_top_is_not_removed(self):
        """A highlight at object height (depth valid) is not 'on the counter plane': the yellow object survives."""
        rgb, depth = bare_frame(("yellow",))
        p = _perception(**{"perception.glare_reject.enabled": True})
        ys, xs = np.nonzero(np.any(rgb != bare_frame(())[0], axis=2))
        u, v = int(xs.mean()), int(ys.mean())
        rgb[v - 3:v + 4, u - 3:u + 4] = (255, 255, 250)
        self.assertEqual(_colours(p.detect(rgb, depth)), ["yellow"])


class TestReflection(unittest.TestCase):
    def test_mirror_blobs_are_extra_objects_with_the_flag_off(self):
        rgb, depth = bare_frame(reflections=True)
        self.assertGreater(len(_perception().detect(rgb, depth)), 3)

    def test_mirror_blobs_rejected_with_the_flag_on(self):
        rgb, depth = bare_frame(reflections=True)
        p = _perception(**{"perception.reflection_reject.enabled": True})
        self.assertEqual(_colours(p.detect(rgb, depth)), sorted(COLOURS))
        self.assertGreaterEqual(p.bare_stats["reflection"], 2)

    def test_mirror_blobs_rejected_without_depth(self):
        rgb, _ = bare_frame(reflections=True)
        p = _perception(**{"perception.reflection_reject.enabled": True})
        self.assertEqual(_colours(p.detect(rgb, None)), sorted(COLOURS))

    def test_two_real_same_colour_objects_with_depth_both_kept(self):
        rend = Renderer(SC, CAL)
        objs = [dict(color="red", shape="cube", yaw=0.0, size=SC.size, height=H_TOP, pos=_top(xy, H_TOP / 2))
                for xy in ((0.45, 0.20), (0.45, 0.10))]   # one behind the other
        rgb, depth = rend.render(objs)
        p = _perception(**{"perception.reflection_reject.enabled": True})
        self.assertEqual(_colours(p.detect(rgb, depth)), ["red", "red"])


class TestLocalBg(unittest.TestCase):
    def test_table_coloured_patch_rejected_object_kept(self):
        rgb, depth = bare_frame(("blue",))
        # an orange-ish patch of table colour in a depth hole: H/S equal to its surroundings
        rgb[150:200, 100:160] = (150, 108, 70)
        rgb[100:260, 60:200] = np.where(np.all(rgb[100:260, 60:200] == (150, 108, 70), axis=2)[..., None],
                                        rgb[100:260, 60:200], (150, 108, 70))
        depth[100:260, 60:200] = 0
        on = _perception(**{"perception.local_bg.enabled": True})
        self.assertEqual(_colours(on.detect(rgb, depth)), ["blue"])


class TestFlagsOffIdentical(unittest.TestCase):
    def test_flags_off_equals_stock_config_without_the_blocks(self):
        cfg = load_config(overlay="none", overrides={"perception.hsv_set": "robust_v2"})
        for k in BARE:
            cfg["perception"].pop(k.split(".")[1])
        stock = Perception(cfg, SC, CAL, log=lambda *_: None)
        off = _perception()
        for frame in (mat_frame, lambda: bare_frame(highlights=((250, 350, 9),), reflections=True)):
            rgb, depth = frame()
            a, ra = off.detect(rgb, depth, debug=True)
            b, rb = stock.detect(rgb, depth, debug=True)
            self.assertEqual([repr(d) for d in a], [repr(d) for d in b])
            self.assertEqual([(repr(d), w) for d, w in ra], [(repr(d), w) for d, w in rb])

    def test_flags_on_does_not_change_a_clean_mat_frame(self):
        rgb, depth = mat_frame()
        a = _perception().detect(rgb, depth)
        b = _all_on().detect(rgb, depth)
        self.assertEqual([repr(d) for d in a], [repr(d) for d in b])

    def test_garbage_input_never_raises(self):
        p = _all_on()
        rgb = np.random.default_rng(0).integers(0, 256, (480, 640, 3), dtype=np.uint8)
        p.detect(rgb, np.zeros((480, 640), np.uint16))
        p.detect(rgb, None)


class TestTiming(unittest.TestCase):
    def test_added_cost_per_frame_under_3ms(self):
        rgb, depth = bare_frame(highlights=((250, 350, 9), (450, 430, 7)), reflections=True)
        off, on = _perception(), _all_on()

        def best(p, n=15):
            ts = []
            for _ in range(n):
                t = time.perf_counter()
                p.detect(rgb, depth)
                ts.append(time.perf_counter() - t)
            return min(ts)
        best(off, 3), best(on, 3)
        added = (best(on) - best(off)) * 1000
        self.assertLess(added, 3.0, f"added {added:.2f} ms/frame")


if __name__ == "__main__":
    unittest.main()
