"""Draw what the novel path understands about an unseen object: outline, handle, grasp, closing line, text panel.

    python tools/generality_demo/render.py --synthetic theta [--theta 45] [--out PNG]
    python tools/generality_demo/render.py --npz <dump.npz> --index i [--out PNG]
    python tools/generality_demo/render.py --all-synthetic [--out PNG]   (2x2: theta 0, 45, 90, plain disc)

Runs the same perception novel path as test_handle_grasp.TestPanHandleGrasp.novels(). Only cv2 drawing; no policy changes.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from aaltonomous.baselines.classical import novel_objects as no  # noqa: E402
from aaltonomous.baselines.classical.config import hand_postures, home_arm  # noqa: E402
from aaltonomous.baselines.classical.perception import Perception  # noqa: E402
from aaltonomous.baselines.classical.planner import Planner  # noqa: E402
from aaltonomous.baselines.classical.test_handle_grasp import SC, CAL, pan_frame  # noqa: E402
from aaltonomous.baselines.classical.test_novel_objects import novel_config  # noqa: E402

DEMO_DIR = "/tmp/claude-1000/-home-felix-projects-hackjunction/811dcd44-f22a-4c64-9008-914b85c21b0a/scratchpad/demo"
HANDLE_BGR = (255, 0, 255)   # magenta: handle pixels (contrast on the green disc)
GRASP_BGR = (0, 0, 255)      # red: grasp point and finger closing line
OUTLINE_BGR = (255, 255, 255)


def _setup():
    cfg = novel_config()
    cfg["perception"]["novel"]["fp_filter"] = False            # as in TestPanHandleGrasp.setUpClass
    cfg["perception"]["novel"]["max_area_px"] = 60000
    planner = Planner(cfg, SC, hand_postures(cfg), home_arm(cfg))
    return cfg, planner


def detect_novel(cfg, planner, rgb, depth):
    """Run the novel path; return (perception, novel dets, [(blob, handle_mask)] in the same order as the dets)."""
    records = []
    orig = no._shape

    def spy(blob, dims, c):
        out = orig(blob, dims, c)
        records.append((blob.copy(), out[1]))
        return out

    perception = Perception(cfg, SC, CAL, log=lambda *_: None)
    perception.hand_opening_m = planner.generic_hand_opening
    no._shape = spy
    try:
        dets = perception.detect(rgb, depth)
    finally:
        no._shape = orig
    novel = [d for d in dets if getattr(d, "novel", False)]
    if len(records) != len(novel):          # order contract broken: draw no handle rather than the wrong one
        records = [(None, None)] * len(novel)
    return perception, novel, records


def _full(mask, shape):
    return cv2.resize(mask.astype(np.uint8), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)


def draw(bgr, perception, det, blob, handle):
    h, w = bgr.shape[:2]
    if blob is not None:
        m = _full(blob, bgr.shape)
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(bgr, cnts, -1, OUTLINE_BGR, 2)
    if handle is not None:
        hm = _full(handle, bgr.shape)
        bgr[hm] = (0.4 * bgr[hm] + 0.6 * np.array(HANDLE_BGR)).astype(np.uint8)
    if det.grasp_kind == "handle":
        yaw = float(det.yaw)
        n = np.array([-np.sin(yaw), np.cos(yaw), 0.0])        # closing direction: across the handle (perp to det.yaw)
        half = 0.5 * float(det.generic_width)
        a = perception._project(det.top + half * n)
        b = perception._project(det.top - half * n)
        pa, pb = (int(a[0]), int(a[1])), (int(b[0]), int(b[1]))
        cv2.arrowedLine(bgr, pa, pb, GRASP_BGR, 2, tipLength=0.15)
        cv2.arrowedLine(bgr, pb, pa, GRASP_BGR, 2, tipLength=0.15)
    g = perception._project(det.top)
    cv2.circle(bgr, (int(g[0]), int(g[1])), 7, GRASP_BGR, -1)
    cv2.circle(bgr, (int(g[0]), int(g[1])), 9, OUTLINE_BGR, 1)
    return bgr


def panel(bgr, det, caption):
    if det is None:
        lines = [caption, "no novel object"]
    else:
        width_label = "handle width" if det.grasp_kind == "handle" else "object width"
        status = "graspable" if det.graspable else (f"ungraspable: {det.ungraspable_reason}" if det.ungraspable_reason
                  else f"ungraspable: low confidence {det.generic_confidence:.2f} (object touches image border)")
        lines = [caption, "unseen object", f"shape: {det.shape}", f"grasp_kind: {det.grasp_kind}",
                 f"{width_label}: {1000 * float(det.generic_width):.0f} mm", status]
    band = np.zeros((18 * len(lines) + 16, bgr.shape[1], 3), np.uint8)
    for i, t in enumerate(lines):
        cv2.putText(band, t, (10, 26 + 18 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255) if i else (120, 220, 255), 1, cv2.LINE_AA)
    return np.vstack([bgr, band])


def render_frame(rgb, depth, cfg, planner, caption):
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR).copy()
    perception, novel, records = detect_novel(cfg, planner, rgb, depth)
    det = max(novel, key=lambda d: d.n_px) if novel else None
    blob, handle = (None, None)
    if det is not None:
        blob, handle = records[novel.index(det)]
        bgr = draw(bgr, perception, det, blob, handle)
    return panel(bgr, det, caption)


def synthetic(theta):
    rgb, depth, *_ = pan_frame(theta)
    return rgb, depth


def main(argv=None):
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--synthetic", nargs="?", const="none", help="theta in degrees, or 'none' for a plain disc")
    src.add_argument("--npz")
    src.add_argument("--all-synthetic", action="store_true")
    ap.add_argument("--index", type=int, default=0)
    ap.add_argument("--theta", type=float, default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--out-dir", default=DEMO_DIR)
    a = ap.parse_args(argv)
    cfg, planner = _setup()
    out_dir = Path(a.out_dir)
    if a.all_synthetic:
        cells = []
        for theta, name in ((0, "theta 0"), (45, "theta 45"), (90, "theta 90"), (None, "plain disc")):
            rgb, depth = synthetic(theta)
            cells.append(render_frame(rgb, depth, cfg, planner, name))
        top = np.hstack(cells[:2])
        bot = np.hstack(cells[2:])
        h = max(top.shape[0], bot.shape[0])
        pad = lambda x: np.vstack([x, np.zeros((h - x.shape[0], x.shape[1], 3), np.uint8)])  # noqa: E731
        grid = np.hstack([pad(top), np.zeros((h, 6, 3), np.uint8), pad(bot)])
        cv2.imwrite(a.out or str(out_dir / "pan_grid.png"), grid)
        print(a.out or str(out_dir / "pan_grid.png"))
        return
    if a.npz:
        data = np.load(a.npz)
        rgb, depth = data["rgb"][a.index], data["depth"][a.index]
        name = f"{Path(a.npz).stem}[{a.index}]"
    else:
        th = None if a.synthetic in (None, "none") else float(a.synthetic)
        rgb, depth = synthetic(th)
        name = "plain disc" if th is None else f"theta {th:g}"
    img = render_frame(rgb, depth, cfg, planner, name)
    target = a.out or str(out_dir / ("pan_" + name.replace(" ", "_").replace("[", "_").replace("]", "") + ".png"))
    cv2.imwrite(target, img)
    print(target)


if __name__ == "__main__":
    main()
