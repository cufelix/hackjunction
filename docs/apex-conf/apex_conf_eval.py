#!/usr/bin/env python3
"""Does triangle_pose's apex_conf separate real triangles from cubes/cylinders? (offline, CPU, no sim)

Usage:  python3 -I apex_conf_eval.py DATASET.npz [--altitude 0.045] [--xy-cols 0,1] [--frame pelvis]
Needs numpy + opencv (cv2). prism_pose.py (same folder) is a verbatim copy of the one in our patch stack
(docs/grasp-yaw-review.patch -> baselines/classical/prism_pose.py), so the numbers are the deployed function.

Dataset layout: auto-detected. Accepted forms (first match wins):
  * an object array of per-sample point clouds under one of: clouds, top_xy, xy, points, top, pts
    (each (N,2) or (N,3) in METRES; for (N,3) the first two columns are used unless --xy-cols says otherwise)
  * one concatenated (M,2|3) array under one of those names plus per-sample lengths ('lengths'|'n'|'counts')
    or start offsets ('offsets'|'starts')
  * per-sample keys like cloud_0, cloud_1 ...
Labels: one of labels, shape, shapes, label, y (strings cube/cylinder/triangle or ints 0/1/2 = cube/cylinder/triangle;
pass --int-labels "cube,cylinder,triangle" to override the int order).
If nothing matches, the script prints every key with its dtype and shape and exits, so the layout can be given.

Output (aggregates only): per shape n, None (too few points) count, apex_conf min/p10/median/p90/max, share >= 0.5;
the separation gap min(triangle) - max(cube, cylinder); confusion at thresholds 0.3/0.5/0.7; and a per-sample CSV
(apex_conf.csv) with index, label, n_points, apex_conf, fitted inradius, fill, so outliers can be inspected.
"""
import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import prism_pose as P  # noqa: E402

NAMES = ("clouds", "top_xy", "xy", "points", "top", "pts")
LABELS = ("labels", "shape", "shapes", "label", "y")


def load(path, xy_cols):
    d = np.load(path, allow_pickle=True)
    keys = list(d.keys())
    clouds = None
    for k in NAMES:
        if k in d:
            a = d[k]
            if a.dtype == object:
                clouds = list(a)
            elif a.ndim == 3:
                clouds = list(a)
            elif a.ndim == 2:
                ln = next((d[x] for x in ("lengths", "n", "counts") if x in d), None)
                st = next((d[x] for x in ("offsets", "starts") if x in d), None)
                if ln is not None:
                    st = np.r_[0, np.cumsum(ln)[:-1]]
                if st is not None:
                    en = list(st[1:]) + [len(a)]
                    clouds = [a[s:e] for s, e in zip(st, en)]
            break
    if clouds is None:
        ck = sorted([k for k in keys if k.startswith("cloud_")], key=lambda s: int(s.split("_")[1]))
        if ck:
            clouds = [d[k] for k in ck]
    lab = next((d[k] for k in LABELS if k in d), None)
    if clouds is None or lab is None:
        print("could not detect layout; keys:")
        for k in keys:
            print(f"  {k}: dtype={d[k].dtype} shape={d[k].shape}")
        sys.exit(2)
    c0, c1 = xy_cols
    clouds = [np.asarray(c, float)[:, [c0, c1]] for c in clouds]
    return clouds, list(lab)


def norm_label(x, order):
    if isinstance(x, (bytes, np.bytes_)):
        x = x.decode()
    if isinstance(x, (int, np.integer)):
        return order[int(x)]
    s = str(x).lower()
    for n in ("cube", "cylinder", "triangle"):
        if n in s or (n == "triangle" and ("tri" in s or "prism" in s)):
            return n
    return s


def dist(v):
    v = sorted(x for x in v if x is not None)
    if not v:
        return "n=0"
    q = lambda p: v[min(len(v) - 1, int(p * (len(v) - 1) + 0.5))]  # noqa: E731
    return f"min {v[0]:.3f} p10 {q(.1):.3f} med {q(.5):.3f} p90 {q(.9):.3f} max {v[-1]:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("npz")
    ap.add_argument("--altitude", type=float, default=P.TRIANGLE_ALTITUDE)
    ap.add_argument("--xy-cols", default="0,1")
    ap.add_argument("--int-labels", default="cube,cylinder,triangle")
    ap.add_argument("--min-points", type=int, default=P.TRIANGLE_MIN_POINTS)
    a = ap.parse_args()
    clouds, labs = load(a.npz, [int(x) for x in a.xy_cols.split(",")])
    order = a.int_labels.split(",")
    import cv2
    rows, by = [], {}
    for i, (xy, l) in enumerate(zip(clouds, labs)):
        l = norm_label(l, order)
        xy = xy[np.all(np.isfinite(xy), axis=1)]
        tp = P.triangle_pose(xy, min_points=a.min_points, altitude=a.altitude)
        conf = None if tp is None else tp[2]
        r = fill = None
        if len(xy) >= 5:
            _, _, r = P.triangle_fit(xy)
            fill = float(cv2.contourArea(cv2.convexHull(xy.astype(np.float32)))) / (3.0 * np.sqrt(3.0) * r * r)
        rows.append((i, l, len(xy), conf, r, fill))
        by.setdefault(l, []).append(conf)
    print(f"samples {len(rows)}; altitude {a.altitude} m; min_points {a.min_points}")
    for l in sorted(by):
        v = by[l]
        ok = [x for x in v if x is not None]
        hi = sum(x >= 0.5 for x in ok)
        print(f"  {l:9s} n={len(v):3d} None={len(v) - len(ok):2d} apex_conf {dist(v)} | >=0.5: {hi}/{len(v)}")
    tri = [x for x in by.get("triangle", []) if x is not None]
    oth = [x for l in ("cube", "cylinder") for x in by.get(l, []) if x is not None]
    if tri and oth:
        print(f"separation gap min(triangle) - max(cube,cyl) = {min(tri) - max(oth):+.3f}  (positive = perfectly separable)")
        for t in (0.3, 0.5, 0.7):
            tp_ = sum(x >= t for x in tri)
            fp = sum(x >= t for x in oth)
            print(f"  threshold {t}: triangles kept {tp_}/{len(by['triangle'])}, cube/cyl passing as triangle {fp}/{len(oth)}")
    out = Path(a.npz).with_name("apex_conf.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["index", "label", "n_points", "apex_conf", "inradius_m", "fill"])
        w.writerows(rows)
    print("per-sample table:", out)


if __name__ == "__main__":
    main()
