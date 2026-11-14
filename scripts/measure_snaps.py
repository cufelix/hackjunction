#!/usr/bin/env python3
"""Measure the 2D colour-blob solidity ratio on real head-camera PNGs.

ratio = blob pixel area / minAreaRect(convexHull(blob)) area. Ideal: square ~1.0, disc ~0.785, triangle ~0.5.
Predicted 2D-blob bands from synthetic data: cube 0.806-0.982, cylinder 0.822-0.859, triangle 0.624-0.730.
Labels: cube >= hi, cylinder >= lo, else triangle (reported at 0.88/0.66 and at 0.90/0.70).
Pipeline mirrors perception.py: RGB->HSV inRange (union of ranges) -> open -> 8-conn components -> min_area_px.
The ORANGE bins are not in the hsv block, so they are not segmented. Blobs touching the border are skipped.

LIMIT: these PNGs have no depth, so this CANNOT verify the metric top-face claim (cube 0.942-0.987,
cylinder 0.792-0.858, triangle 0.542-0.608); it tests only the 2D-blob half. No per-blob ground truth, so
no shape is guessed: read the ratio distribution against the predicted bands yourself.

Usage: python3 measure_snaps.py [dir]   (default: .)   Needs numpy + cv2 only. Always exits 0.
"""
import glob
import os
import sys

import cv2
import numpy as np

# starter repo aaltonomous/baselines/classical/config.yaml perception.hsv lines 19-21, open_px line 23, min_area_px line 24
HSV = {
    "red": [([0, 70, 70], [8, 255, 255]), ([170, 70, 70], [180, 255, 255])],
    "yellow": [([18, 70, 90], [35, 255, 255])],
    "blue": [([100, 70, 50], [130, 255, 255])],
}
OPEN_PX, MIN_AREA = 3, 120  # erode_px=2 (line 22) only feeds the depth core, not the blob area, so unused here


def label(r, hi, lo):
    return "cube" if r >= hi else "cyl" if r >= lo else "tri"


def blobs(img):
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)  # imread is BGR; perception.py gets RGB
    for color, ranges in HSV.items():
        m = np.zeros((h, w), np.uint8)
        for lo, hi in ranges:
            m |= cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((OPEN_PX, OPEN_PX), np.uint8))
        n, lab, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, n):
            x, y, bw, bh, area = (int(v) for v in st[i])
            if area < MIN_AREA:
                continue
            pts = cv2.findNonZero((lab == i).astype(np.uint8))
            (_, _), (rw, rh), _ = cv2.minAreaRect(cv2.convexHull(pts))
            ratio = area / ((rw + 1) * (rh + 1))  # +1: hull of pixel centres is 1 px short per side
            border = x <= 1 or y <= 1 or x + bw >= w - 1 or y + bh >= h - 1
            yield color, area, ratio, (x, y, bw, bh), border


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(d, "*.png")))
    if not files:
        print(f"no PNGs in {d}")
    rows, skipped, bad = [], 0, []
    print("image color px ratio @.88/.66 @.90/.70 border bbox(x,y,w,h)")
    for f in files:
        img = cv2.imread(f)
        if img is None:
            bad.append(os.path.basename(f))
            continue
        try:
            for color, area, r, bb, border in blobs(img):
                name = os.path.basename(f)
                if border:
                    skipped += 1
                else:
                    rows.append((color, r))
                print(f"{name} {color} {area} {r:.3f} {label(r, .88, .66)} {label(r, .90, .70)} "
                      f"{'BORDER' if border else '-'} {bb}")
        except Exception as e:  # keep going on any per-image failure
            bad.append(f"{os.path.basename(f)} ({e})")
    print(f"\n== summary: {len(files)} files, unreadable/failed: {bad or 'none'}; border-skipped blobs: {skipped}")
    if not rows:
        print("no usable blobs")
        return
    rs = np.array([r for _, r in rows])
    print("blobs per colour:", {c: sum(1 for k, _ in rows if k == c) for c in HSV}, "total", len(rs))
    print(f"ratio min/median/max: {rs.min():.3f} / {np.median(rs):.3f} / {rs.max():.3f}")
    print("histogram (0.05 buckets):")
    for lo in np.arange(0.40, 1.0001, 0.05):
        c = int(((rs >= lo) & (rs < lo + 0.05 + (1e-9 if lo > 0.949 else 0))).sum())
        if c:
            print(f"  {lo:.2f}-{lo + .05:.2f} {c:3d} {'#' * c}")
    for hi, lo in ((.88, .66), (.90, .70)):
        L = [label(r, hi, lo) for r in rs]
        print(f"bands @{hi}/{lo}: cube {L.count('cube')}  cyl {L.count('cyl')}  tri {L.count('tri')}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("error:", e)
    sys.exit(0)
