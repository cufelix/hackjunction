"""perception.edge_blob (FLAG, default off): colour blobs cut by the image edge. Synthetic mock_world renders only. Run:
    python -m unittest aaltonomous.baselines.classical.test_edge_blob -v
"""
import unittest

import cv2
import numpy as np

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene

from .config import FLAGS, load_config
from .mock_world import Renderer
from .perception import Perception

SC, CAL = Scene.load(), Calibration.load()
RENDER = Renderer(SC, CAL)
RGB0, _ = RENDER.render([])
IMG_H = RGB0.shape[0]
CUT_XY = (0.21, 0.00)     # pelvis xy: the bottom image edge cuts this cube (bbox reaches the last row)
FREE_XY = (0.30, 0.00)    # pelvis xy: fully in frame
lines = []


def _perc(flag=None, **over):
    ov = {} if flag is None else {"perception.edge_blob": flag}
    ov.update({f"perception.{k}": v for k, v in over.items()})
    return Perception(load_config(overlay="none", overrides=ov), SC, CAL, log=lines.append)


def _scene(xy, color="red"):
    ob = dict(color=color, shape="cube", yaw=0.0, size=SC.size, height=SC.object_height(),
              pos=np.array([xy[0], xy[1], SC.counter_z + SC.object_height() / 2]))
    return RENDER.render([ob])


def _blob(rgb):
    """bbox [x, y, w, h, area] and centroid of the object's pixels (what the 2D path would use)."""
    m = np.any(rgb != RGB0, axis=2).astype(np.uint8)
    _, _, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return stats[i], cent[i]


def _cm(a, b):
    return 100.0 * float(np.linalg.norm(np.asarray(a, float)[:2] - np.asarray(b, float)[:2]))


class TestEdgeBlob(unittest.TestCase):
    def setUp(self):
        lines.clear()

    def test_flag_registered_and_off_by_default(self):
        self.assertIn("perception.edge_blob", FLAGS)
        self.assertFalse(load_config(overlay="none")["perception"]["edge_blob"])

    def test_cut_blob_with_depth_uses_top_face_not_biased_centroid(self):
        rgb, depth = _scene(CUT_XY)
        st, cent = _blob(rgb)
        self.assertEqual(st[1] + st[3] - 1, IMG_H - 1)          # bbox touches the bottom row
        dets = _perc(True).detect(rgb, depth)
        self.assertEqual(len(dets), 1)
        det = dets[0]
        self.assertTrue(det.edge_cut)
        self.assertEqual(det.method, "depth")
        fixed_cm = _cm(det.top, (*CUT_XY, SC.counter_z + SC.object_height()))
        centroid_cm = _cm(_perc(True)._from_plane("red", cent, int(st[4]), 0).top, (*CUT_XY, 0.0))
        print(f"\nedge_blob errors: centroid(plane) {centroid_cm:.2f} cm, fixed(depth top face) {fixed_cm:.2f} cm")
        self.assertLess(fixed_cm, centroid_cm)
        self.assertLess(fixed_cm, 2.0)   # measured 1.6 cm; the 1 cm target is NOT met by the top-face path alone
        self.assertIn("[edge_blob] cut=1 depth=1 rejected=0 sides=['bottom']", lines)

    def test_cut_blob_without_depth_is_rejected(self):
        rgb, _ = _scene(CUT_XY)
        dets, rej = _perc(True).detect(rgb, np.zeros(RGB0.shape[:2], np.uint16), debug=True)
        self.assertEqual([d for d in dets if d.color == "red"], [])
        self.assertIn("edge_blob:cut", [why for _, why in rej])
        self.assertIn("[edge_blob] cut=1 depth=0 rejected=1 sides=['bottom']", lines)

    def test_cut_blob_without_depth_flag_off_still_uses_plane(self):
        rgb, _ = _scene(CUT_XY)
        dets = _perc(False).detect(rgb, np.zeros(RGB0.shape[:2], np.uint16))
        red = [d for d in dets if d.color == "red"]
        self.assertEqual([d.method for d in red], ["plane"])
        self.assertFalse(red[0].edge_cut)

    def test_blob_not_touching_the_edge_is_unchanged(self):
        rgb, depth = _scene(FREE_XY)
        st, _ = _blob(rgb)
        self.assertLess(st[1] + st[3] - 1, IMG_H - 1)
        on, off = _perc(True).detect(rgb, depth, debug=True), _perc(False).detect(rgb, depth, debug=True)
        self.assertEqual([repr(d) for d in on[0]], [repr(d) for d in off[0]])
        self.assertEqual([(repr(d), w) for d, w in on[1]], [(repr(d), w) for d, w in off[1]])
        self.assertFalse(any(d.edge_cut for d in on[0]))
        self.assertIn("[edge_blob] cut=0 depth=0 rejected=0 sides=['bottom']", lines)

    def test_sides_top_only_ignores_bottom_cut(self):
        rgb, depth = _scene(CUT_XY)
        dets = _perc(True, edge_blob_sides=["top"]).detect(rgb, depth)
        self.assertFalse(any(d.edge_cut for d in dets))

    def test_flag_off_is_identical_to_no_key(self):
        stock_cfg = load_config(overlay="none")
        stock_cfg["perception"].pop("edge_blob")
        stock = Perception(stock_cfg, SC, CAL, log=lambda *_: None)
        off = _perc(False)
        for xy in (CUT_XY, FREE_XY):
            rgb, depth = _scene(xy)
            a, b = off.detect(rgb, depth, debug=True), stock.detect(rgb, depth, debug=True)
            self.assertEqual([repr(d) for d in a[0]], [repr(d) for d in b[0]])
            self.assertEqual([(repr(d), w) for d, w in a[1]], [(repr(d), w) for d, w in b[1]])
            self.assertFalse(any(d.edge_cut for d in a[0]))


if __name__ == "__main__":
    unittest.main()
