"""drop_2 hue analysis v2 — ASSUMPTION-FREE (drop_2 geometry differs from scene.yaml):
- spatial split: bin region v < 0.38H, mat region v > 0.45H (verified visually on e00a/e00b);
- orange object components on the mat (cube + prism both present) vs orange bin walls;
- purple coverage vs configured blue (mask-independent);
- placement outcome by TEMPORAL DIFF (final vs 20% frame): new orange inside a bin = placed object;
  bin identity from per-episode column gaps in the baseline frame (no nominal positions).
Run: venv/bin/python drop2_hue2.py
"""
import argparse
import glob
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # repo root
from aaltonomous.baselines.classical.config import load_config   # noqa: E402

_ap = argparse.ArgumentParser()
_ap.add_argument("--root", default="/var/tmp/hf_drop2")
ROOT = Path(_ap.parse_args().root)
(ROOT / "ov2").mkdir(exist_ok=True)
HSV = load_config(overlay="none")["perception"]["hsv"]
BLUE = HSV["blue"]
manifest = {m["episode"]: m for m in json.load(open(ROOT / "manifest.json"))}
H, W = 480, 640
V_BIN, V_MAT = int(0.38 * H), int(0.45 * H)


def masks(rgb):
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    om = cv2.inRange(hsv, (0, 60, 40), (26, 255, 255)) | cv2.inRange(hsv, (165, 60, 40), (180, 255, 255))
    om = cv2.morphologyEx(om, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    pm = cv2.inRange(hsv, (100, 50, 50), (160, 255, 255))
    pm = cv2.morphologyEx(pm, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    return hsv, om, pm


def comps(m, min_area, min_h=0):
    n, lab, st, cen = cv2.connectedComponentsWithStats(m.astype(np.uint8), 8)
    return [(lab == i, st[i], cen[i]) for i in range(1, n) if st[i, 4] >= min_area and st[i, 3] >= min_h]


def hstats(hsv, sel):
    h = hsv[..., 0][sel].astype(float)
    if len(h) < 30:
        return None
    return {k: round(float(v), 1) for k, v in zip(("p5", "p50", "p95"), np.percentile(h, [5, 50, 95]))} | {"n": len(h)}


def bin_bounds(om_base):
    """u boundaries between bins from gaps (columns without orange) in the baseline bin region."""
    pres = (om_base[:V_BIN] > 0).mean(0) > 0.3
    gaps, start = [], None
    for u in range(W):
        if not pres[u] and start is None:
            start = u
        elif pres[u] and start is not None:
            if u - start >= 8:
                gaps.append((start + u) // 2)
            start = None
    b = [0] + gaps + [W]
    return [(b[i], b[i + 1]) for i in range(len(b) - 1) if b[i + 1] - b[i] > 60]


def bin_of(u, bounds):
    for i, (lo, hi) in enumerate(bounds):
        if lo <= u < hi:
            return ["left", "center", "right"][i] if len(bounds) == 3 else f"bin{i}"
    return None


frames = {}
for p in sorted(glob.glob(str(ROOT / "frames" / "*.png"))):
    name = Path(p).stem
    e, tag = int(name[1:3]), name[3]
    bgr = cv2.imread(p)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    hsv, om, pm = masks(rgb)
    row = {"frame": name, "episode": e, "tag": tag, "task": manifest[e]["task"]}
    row["bin_hue"] = hstats(hsv, (om > 0) & (np.arange(H)[:, None] < V_BIN))
    mat = np.arange(H)[:, None] >= V_MAT
    objs = comps((om > 0) & mat, 1500, min_h=20)
    row["objects"] = [{"hue": hstats(hsv, sel), "area": int(st[4]),
                       "uv": [int(c[0]), int(c[1])]} for sel, st, c in objs]
    pcomps = comps(pm, 800)
    if pcomps:
        sel, st, c = max(pcomps, key=lambda t: t[1][4])
        h, s, v = (hsv[..., i][sel] for i in range(3))
        ok = np.zeros(len(h), bool)
        for lo, hi in BLUE:
            ok |= (h >= lo[0]) & (h <= hi[0]) & (s >= lo[1]) & (v >= lo[2])
        row["purple"] = {"hue": hstats(hsv, sel), "area": int(st[4]), "uv": [int(c[0]), int(c[1])],
                         "in_config_blue": round(float(ok.mean()), 3),
                         "in_bin_region": bool(c[1] < V_BIN)}
    frames[name] = (row, om, pm, bgr)

rows = [f[0] for f in frames.values()]
# ---- placement outcome by temporal diff
outcomes = []
for e, man in manifest.items():
    base, fin = frames.get(f"e{e:02d}a"), frames.get(f"e{e:02d}z")
    if not base or not fin:
        continue
    om_b, om_f, pm_f = base[1], fin[1], fin[2]
    bounds = bin_bounds(om_b)
    want = next(t for t in ("left", "center", "right") if t in man["task"])
    place, how = None, None
    if "purple" in man["task"]:
        pc = comps(pm_f[:V_BIN], 800)
        if pc:
            sel, st, c = max(pc, key=lambda t: t[1][4])
            place, how = bin_of(c[0], bounds), "purple_in_bin"
    else:
        new = (om_f > 0) & ~(om_b > 0)
        new[:V_BIN] &= True
        new[V_BIN:] = False
        new = cv2.morphologyEx(new.astype(np.uint8) * 255, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        nc = comps(new, 800)
        if nc:
            sel, st, c = max(nc, key=lambda t: t[1][4])
            place, how = bin_of(c[0], bounds), "new_orange_in_bin"
    seen_mat = bool(frames[f"e{e:02d}z"][0]["objects"]) or "purple" in frames[f"e{e:02d}z"][0]
    status = ("placed_correct" if place == want else "wrong_bin" if place
              else "not_placed" if seen_mat else "object_not_visible")
    outcomes.append({"episode": e, "task": man["task"], "status": status, "bin": place,
                     "how": how, "n_bins_detected": len(bounds)})
    if e < 6:                                            # a few annotated finals
        out = fin[3].copy()
        cv2.line(out, (0, V_BIN), (W, V_BIN), (255, 255, 0), 1)
        cv2.line(out, (0, V_MAT), (W, V_MAT), (0, 255, 255), 1)
        for lo, hi in bounds:
            cv2.line(out, (lo, 0), (lo, V_BIN), (255, 0, 255), 1)
        cv2.putText(out, f"{status} -> {place} (want {want})", (8, H - 12), 0, 0.55, (255, 255, 255), 1)
        cv2.imwrite(str(ROOT / "ov2" / f"e{e:02d}z.png"), out)

counts = {}
for o in outcomes:
    counts[o["status"]] = counts.get(o["status"], 0) + 1

# ---- aggregate hues by task
agg = {}
for r in rows:
    if r["tag"] == "z":
        continue
    a = agg.setdefault(r["task"], {"obj": [], "bin": [], "gap": [], "purple_cov": [], "n_obj": []})
    if r["bin_hue"]:
        a["bin"].append(r["bin_hue"]["p50"])
    for o in r["objects"]:
        if o["hue"]:
            a["obj"].append(o["hue"]["p50"])
            if r["bin_hue"]:
                a["gap"].append(o["hue"]["p5"] - r["bin_hue"]["p95"])
    a["n_obj"].append(len(r["objects"]))
    if r.get("purple"):
        a["purple_cov"].append(r["purple"]["in_config_blue"])
summary = {}
for t, a in agg.items():
    s = {"n_frames": len(a["n_obj"]), "mean_orange_objects_per_frame": round(float(np.mean(a["n_obj"])), 2)}
    for k, v in (("obj_hue_p50", a["obj"]), ("bin_hue_p50", a["bin"]), ("sep_gap_H", a["gap"])):
        if v:
            s[k] = {"mean": round(float(np.mean(v)), 1), "std": round(float(np.std(v)), 1),
                    "min": round(float(np.min(v)), 1), "max": round(float(np.max(v)), 1)}
    if a["purple_cov"]:
        s["purple_in_config_blue"] = round(float(np.mean(a["purple_cov"])), 3)
    summary[t] = s

json.dump({"per_frame": rows, "by_task": summary, "outcomes": outcomes, "status_counts": counts},
          open(ROOT / "hue_results_v2.json", "w"), indent=1)
print("== by task (v2, assumption-free) ==")
for t, s in summary.items():
    print(f"\n{t}\n  {json.dumps(s)}")
print("\n== placement outcomes (temporal diff, coarse) ==", counts)
print("bins detected per episode:", sorted({o["n_bins_detected"] for o in outcomes}))
print(f"saved {ROOT / 'hue_results_v2.json'}")
