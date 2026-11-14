"""Real-camera colour flags (hf_perception 2026-11-14): perception.hsv_set robust + perception.split_blobs. Run:
    python -m unittest aaltonomous.baselines.classical.test_real_colour -v

HSV values below are MEASURED medians/percentiles (OpenCV H 0-180), not guesses:
  real HF head cam (zupermiez/g1_pick_cube, 60 frames, 336 hand-labelled objects): orange pieces H 13 (9-15),
  yellow H 40 (22-50, top faces overexposed), violet H 111 (108-123); bins H 6 (2-7); counter S ~27
  lab 2026-11-11 (config_lab.yaml): orange H 11-14 S 205-250, yellow H 25 S 185, purple H 127 S 97
  sim DR head frames (seeds 101-112): red H 0-7 / 170-179, yellow H 23-31, blue H 100-116 (S >= 73)
"""
import copy
import unittest

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import load_config
from .mock_world import Renderer
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
STOCK = load_config(overlay="none")
ROBUST = load_config(overlay="none", overrides={"perception.hsv_set": "robust"})
SPLIT = load_config(overlay="none", overrides={"perception.split_blobs": True})
BOTH = load_config(overlay="none", overrides={"perception.hsv_set": "robust", "perception.split_blobs": True})

# (name, (H, S, V), expected class with the robust set)
SAMPLES = [
    ("real orange piece", (13, 160, 250), "red"), ("real orange, shadow side", (9, 140, 150), "red"),
    ("lab orange", (12, 230, 200), "red"),
    ("real yellow top (overexposed)", (40, 85, 250), "yellow"), ("real yellow side", (24, 160, 200), "yellow"),
    ("real yellow top, pale", (47, 52, 250), "yellow"), ("lab yellow", (25, 185, 200), "yellow"),
    ("real violet", (111, 150, 240), "blue"), ("real violet, dark side", (120, 130, 110), "blue"),
    ("lab purple", (127, 97, 150), "blue"),
    ("sim red", (3, 134, 207), "red"), ("sim red wrap", (178, 134, 207), "red"),
    ("sim yellow", (28, 171, 210), "yellow"), ("sim yellow top", (25, 88, 170), "yellow"),
    ("sim blue", (107, 133, 193), "blue"), ("sim blue edge", (100, 80, 150), "blue"),
    ("real counter", (10, 27, 140), None), ("real counter, bright", (98, 27, 200), None),
    ("black hand", (105, 40, 40), None), ("white/grey", (0, 5, 200), None),
]


def _patch(hsv):
    img = np.zeros((40, 40, 3), np.uint8)
    img[:] = hsv
    return cv2.cvtColor(img, cv2.COLOR_HSV2RGB)


def _classes(perc, rgb):
    return sorted(c for c, m in perc.masks(rgb).items() if (m[10:30, 10:30] > 0).mean() > 0.5)


class TestRobustHSV(unittest.TestCase):
    def test_flags_off_uses_the_stock_ranges(self):
        p = Perception(STOCK, SC, CAL)
        self.assertIs(p.hsv, STOCK["perception"]["hsv"])
        self.assertFalse(p.split_blobs)
        rgb = np.random.default_rng(0).integers(0, 256, (120, 160, 3), dtype=np.uint8)
        a, b = p.masks(rgb), Perception(load_config(), SC, CAL).masks(rgb)
        for k in a:
            self.assertTrue(np.array_equal(a[k], b[k]))

    def test_measured_colours_land_in_the_right_class(self):
        p = Perception(ROBUST, SC, CAL)
        for name, hsv, want in SAMPLES:
            # the HSV->RGB->HSV round trip moves H/S by ~1: the expected class must survive it
            self.assertEqual(_classes(p, _patch(hsv)), [want] if want else [], name)

    def test_stock_misses_the_real_pieces(self):
        """Documents the bug the flag fixes: stock red stops at H 8, stock yellow at H 35."""
        p = Perception(STOCK, SC, CAL)
        self.assertEqual(_classes(p, _patch((13, 160, 250))), [])        # real orange piece -> nothing
        self.assertEqual(_classes(p, _patch((40, 85, 250))), [])         # real yellow top -> nothing

    def test_robust_classes_are_disjoint(self):
        p = Perception(ROBUST, SC, CAL)
        hsv = np.stack(np.meshgrid(np.arange(180), np.arange(256), [255]), -1).reshape(-1, 1, 3).astype(np.uint8)
        hsv = np.concatenate([hsv, hsv.copy()])
        hsv[len(hsv) // 2:, 0, 2] = 120
        rgb = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        p.cfg = dict(p.cfg, open_px=0)
        m = p.masks(rgb)
        self.assertEqual(int(sum((v > 0).astype(int) for v in m.values()).max()), 1)

    def test_robust_bins_are_cut_in_3d_for_sim_and_real_bin_colours(self):
        """Every bin colour overlaps some class of the robust set; the depth bin-footprint cut must remove it."""
        for name, rgb_bin in (("sim nominal", [230, 110, 30]), ("real HF H~6", [235, 75, 35]),
                              ("sim DR H~20", [235, 170, 60])):
            sc = copy.deepcopy(SC)
            sc.cfg["bins"]["rgb"] = rgb_bin
            rgb, depth = Renderer(sc, CAL).render([])
            for cfg in (ROBUST, BOTH):
                self.assertEqual(Perception(cfg, sc, CAL).detect(rgb, depth), [], name)
                self.assertEqual(Perception(cfg, sc, CAL).detect(rgb, None), [], name + " (no depth: hulls)")

    def test_robust_finds_sim_objects_like_stock(self):
        rng = np.random.default_rng(3)
        L = SC.cfg["layout"]
        h = SC.object_height()
        for _ in range(4):
            objs, used = [], []
            for col in ("red", "yellow", "blue"):
                while True:
                    x, y = rng.uniform(*L["region_x"]), rng.uniform(*L["region_y"])
                    if all(np.hypot(x - a, y - b) > 0.09 for a, b in used):
                        break
                used.append((x, y))
                objs.append(dict(color=col, shape="cube", yaw=rng.uniform(-0.8, 0.8), size=SC.size, height=h,
                                 pos=SC.mat_to_pelvis([x, y, h / 2])))
            rgb, depth = Renderer(SC, CAL).render(objs, rng, depth_noise_mm=1.0, hole_frac=0.05)
            a = Perception(STOCK, SC, CAL).detect(rgb, depth)
            for cfg in (ROBUST, BOTH):
                b = Perception(cfg, SC, CAL).detect(rgb, depth)
                self.assertEqual(sorted(d.color for d in b), ["blue", "red", "yellow"])
                for d in b:
                    ref = next(q for q in a if q.color == d.color)
                    self.assertLess(np.linalg.norm(d.top - ref.top), 0.002)


class TestSplitBlobs(unittest.TestCase):
    def _two_touching(self, gap=0.0, shapes=("cube", "cylinder"), color="red"):
        h = SC.object_height()
        x0, y0 = 0.40, 0.15                       # inside layout.region (away from the bin hulls)
        objs = []
        for k, shp in enumerate(shapes):
            objs.append(dict(color=color, shape=shp, yaw=0.0, size=SC.size, height=h,
                             pos=SC.mat_to_pelvis([x0 + k * (SC.size + gap), y0, h / 2])))
        return objs

    def test_touching_same_colour_objects_merge_without_and_split_with_the_flag(self):
        for shapes in (("cube", "cylinder"), ("cube", "cube"), ("cylinder", "cylinder")):
            objs = self._two_touching(shapes=shapes)
            rgb, depth = Renderer(SC, CAL).render(objs)
            self.assertEqual(len(Perception(STOCK, SC, CAL).detect(rgb, depth)), 1)  # the bug: one merged blob
            dets = Perception(SPLIT, SC, CAL).detect(rgb, depth)
            self.assertEqual(len(dets), 2, shapes)
            for o in objs:
                top = o["pos"] + [0, 0, o["height"] / 2]
                err = min(np.linalg.norm(d.top[:2] - top[:2]) for d in dets)
                self.assertLess(err, 0.012, f"{shapes} {o['shape']} err {err * 1000:.1f} mm")

    def test_split_without_depth_uses_the_plane_estimate(self):
        objs = self._two_touching(shapes=("cube", "cube"), color="blue")
        rgb, _ = Renderer(SC, CAL).render(objs)
        self.assertEqual(len(Perception(SPLIT, SC, CAL).detect(rgb, None)), 2)

    def test_single_objects_are_never_split(self):
        rng = np.random.default_rng(7)
        h = SC.object_height()
        L = SC.cfg["layout"]
        for _ in range(12):
            x, y = rng.uniform(*L["region_x"]), rng.uniform(*L["region_y"])
            for shp in ("cube", "cylinder", "triangle"):
                ob = dict(color="yellow", shape=shp, yaw=rng.uniform(-0.8, 0.8), size=SC.size, height=h,
                          pos=SC.mat_to_pelvis([x, y, h / 2]))
                rgb, depth = Renderer(SC, CAL).render([ob])
                a = Perception(STOCK, SC, CAL).detect(rgb, depth)
                b = Perception(SPLIT, SC, CAL).detect(rgb, depth)
                self.assertEqual(len(b), len(a), shp)
                for p, q in zip(a, b):
                    self.assertTrue(np.array_equal(p.top, q.top))

    def test_split_blob_geometry(self):
        """Synthetic masks with the plane-free fallback area: two / three touching discs split, a big disc does not."""
        p = Perception(load_config(overlay="none", overrides={"perception.split_blobs": True,
                                                               "perception.split.fallback_area_px": 2000}), SC, None)
        def discs(centres, r=25):
            m = np.zeros((480, 640), np.uint8)
            for c in centres:
                cv2.circle(m, c, r, 1, -1)
            return m
        for centres in ([(200, 200), (248, 200)], [(200, 200), (248, 200), (224, 242)]):
            m = discs(centres)
            parts = p.split_blob(m, int(m.sum()), np.array([0.0, 0.0]), None)
            self.assertEqual(len(parts), len(centres))
            got = sorted(tuple(np.round(c).astype(int)) for _, _, c in parts)
            for (gx, gy), (cx, cy) in zip(got, sorted(centres)):
                self.assertLess(np.hypot(gx - cx, gy - cy), 6)
            self.assertEqual(sum(a for _, a, _ in parts), int(m.sum()))   # every pixel in exactly one part
        big = discs([(300, 300)], r=45)                                   # 6360 px > 1.5 x 2000, one peak
        self.assertEqual(len(p.split_blob(big, int(big.sum()), np.array([300.0, 300.0]), None)), 1)

    def test_split_blob_many_peaks_is_capped(self):
        """A noisy blob with many distance peaks never yields more parts than max_parts / round(area / expected)."""
        p = Perception(load_config(overlay="none", overrides={"perception.split_blobs": True,
                                                               "perception.split.fallback_area_px": 2000}), SC, None)
        m = np.zeros((480, 640), np.uint8)
        for i in range(12):                                   # 12 touching bumps on a bar: area ~ 3.6 x expected
            cv2.circle(m, (100 + 14 * i, 200), 9, 1, -1)
        area = int(m.sum())
        parts = p.split_blob(m, area, np.array([0.0, 0.0]), None)
        self.assertLessEqual(len(parts), min(3, max(1, round(area / 2000))))
        self.assertEqual(sum(a for _, a, _ in parts), area)


if __name__ == "__main__":
    unittest.main()


class TestRejectOrange(unittest.TestCase):
    ON = load_config(overlay="none", overrides={"perception.hsv_set": "robust", "perception.reject_orange": True})

    def test_off_is_unchanged(self):
        p = Perception(ROBUST, SC, CAL)
        self.assertIs(p.hsv, ROBUST["perception"]["hsv_robust"])

    def test_orange_basin_rejected_objects_kept(self):
        p = Perception(self.ON, SC, CAL)
        for hsv in [(12, 200, 220), (10, 120, 250), (16, 150, 200), (20, 60, 240), (21, 200, 200), (24, 70, 240)]:
            self.assertEqual(_classes(p, _patch(hsv)), [], hsv)
        for hsv, cls in [((3, 134, 207), "red"), ((178, 134, 207), "red"), ((28, 171, 210), "yellow"),
                         ((25, 88, 170), "yellow"), ((40, 85, 250), "yellow"), ((107, 133, 193), "blue")]:
            self.assertEqual(_classes(p, _patch(hsv)), [cls], hsv)

    def test_pose_error_phantoms_removed(self):
        T = CAL._T("head").copy()
        T[0, 3] += 0.01                      # 1 cm camera pose error: bin hulls no longer cover the rims
        for rgb_bin in ([230, 110, 30], [235, 170, 60]):
            sc = copy.deepcopy(SC)
            sc.cfg["bins"]["rgb"] = rgb_bin
            rgb, depth = Renderer(sc, CAL, T_pelvis_optical=T).render([])
            self.assertGreater(len(Perception(ROBUST, sc, CAL).detect(rgb, depth)), 0)
            self.assertEqual(Perception(self.ON, sc, CAL).detect(rgb, depth), [])
