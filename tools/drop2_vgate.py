"""drop_2 V-gate + sim-colour targets (CONTRACT v2 lane: OpenCode = drop_2 analysis).
Classes: bin walls (v<0.38H), tape (0.38H<=v<0.45H), orange objects (mat comps, shape-filtered),
purple, yellow. Reports V/S percentiles + mean RGB per class (sim scene.yaml overlay targets).
Run: venv/bin/python drop2_vgate.py
"""
import argparse
import glob
import json
from pathlib import Path

import cv2
import numpy as np

_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default="/var/tmp/hf_drop2")
ROOT = Path(_ap.parse_args().root)
H, W = 480, 640
V_BIN, V_MAT = int(0.38 * H), int(0.45 * H)
rows = []
for p in sorted(glob.glob(str(ROOT / "frames" / "*b.png"))):
    bgr = cv2.imread(p)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    om = cv2.inRange(hsv, (0, 60, 40), (26, 255, 255)) | cv2.inRange(hsv, (165, 60, 40), (180, 255, 255))
    om = cv2.morphologyEx(om, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    ym = cv2.inRange(hsv, (20, 60, 60), (40, 255, 255))
    ym = cv2.morphologyEx(ym, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    pm = cv2.inRange(hsv, (100, 50, 50), (160, 255, 255))
    pm = cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    vv = np.arange(H)[:, None]
    classes = {"bin_walls": (om > 0) & (vv < V_BIN), "tape": (om > 0) & (vv >= V_BIN) & (vv < V_MAT)}
    n, lab, st, cen = cv2.connectedComponentsWithStats(((om > 0) & (vv >= V_MAT)).astype(np.uint8), 8)
    for i in range(1, n):
        x, y, w, h, a = st[i]
        if a >= 1500 and h >= 35 and w <= 160:          # compact object, not wall leak / tape
            classes.setdefault("orange_object", np.zeros((H, W), bool))
            classes["orange_object"] |= lab == i
    for name, m in (("purple", pm), ("yellow", ym)):
        n2, lab2, st2, _ = cv2.connectedComponentsWithStats((m & (vv >= V_BIN)).astype(np.uint8), 8)
        for i in range(1, n2):
            if st2[i, 4] >= 800:
                classes.setdefault(name, np.zeros((H, W), bool))
                classes[name] |= lab2 == i
    row = {"frame": Path(p).stem}
    for cname, sel in classes.items():
        if sel.sum() < 200:
            continue
        v = hsv[..., 2][sel].astype(float)
        s = hsv[..., 1][sel].astype(float)
        h = hsv[..., 0][sel].astype(float)
        mean_rgb = rgb[sel].mean(0).round(0).astype(int).tolist()
        row[cname] = {"n": int(sel.sum()),
                      "V": [round(float(x), 0) for x in np.percentile(v, [5, 50, 95])],
                      "S_p50": round(float(np.median(s)), 0),
                      "H": [round(float(x), 1) for x in np.percentile(h, [5, 50, 95])],
                      "mean_RGB": mean_rgb}
    rows.append(row)

agg = {}
for r in rows:
    for c, d in r.items():
        if c == "frame":
            continue
        a = agg.setdefault(c, {"V5": [], "V50": [], "V95": [], "S": [], "H50": [], "RGB": [], "n_frames": 0})
        a["V5"].append(d["V"][0]); a["V50"].append(d["V"][1]); a["V95"].append(d["V"][2])
        a["S"].append(d["S_p50"]); a["H50"].append(d["H"][1]); a["RGB"].append(d["mean_RGB"])
        a["n_frames"] += 1
summary = {}
for c, a in agg.items():
    summary[c] = {"n_frames": a["n_frames"],
                  "V_p5_of_p5": round(float(np.min(a["V5"])), 0),
                  "V_p50_mean": round(float(np.mean(a["V50"])), 0),
                  "V_p95_of_p95": round(float(np.max(a["V95"])), 0),
                  "S_p50_mean": round(float(np.mean(a["S"])), 0),
                  "H_p50_mean": round(float(np.mean(a["H50"])), 1),
                  "mean_RGB": np.mean(a["RGB"], 0).round(0).astype(int).tolist()}
json.dump({"per_frame": rows, "summary": summary}, open(ROOT / "vgate_results.json", "w"), indent=1)
print("== V-gate + colour targets (48 mid-episode frames) ==")
for c, s in summary.items():
    print(f"{c:14} {json.dumps(s)}")
if "orange_object" in summary and "tape" in summary:
    print("\nV separation: object V_p5(min over frames) =", summary["orange_object"]["V_p5_of_p5"],
          "vs tape V_p95(max) =", summary["tape"]["V_p95_of_p95"])
