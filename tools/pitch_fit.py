#!/usr/bin/env python3
"""Camera-pitch fit from head frames: is the real-frame bin offset a camera-pose artefact? (Oliver, lane 3.)

Answer on the 15 local HF cam_high frames: YES. Fitted pitch -7.5 +/- 1.8 deg (head camera pitched UP vs the
URDF calibration), rim rms ~20 cm -> 2-3 cm, and it VARIES per episode (e12 ~ -4 deg vs e01 ~ -9 deg) — so it
needs per-episode ONLINE calibration, not a baked constant. See docs/HANDOFF-pitch-fit.md.

Model: bins are at the NOMINAL scene.yaml positions. The head camera may deviate from calibration.yaml by a
pitch dtheta about its own optical x-axis (T' = T @ Rx(dtheta)) and optionally a translation. A far-rim pixel
(u, v) is a ray through the perturbed pose intersected with the rim plane z = counter_z + wall_height; at the
true pose those intersections land on the nominal far-rim lines (x = bin_x + half, y within the bin span).
Fit = robust (trimmed) point-to-rim-line distance. The differential y-shift of the two side bins vs the centre
bin is the pitch signature that a pure row translation cannot produce.

CPU-only (numpy, cv2, scipy). Importable: RimFit / far_rim_bins / rx for reuse in online calibration.

    python3 tools/pitch_fit.py                                  # synthetic validation + /var/tmp/hf_g1/frames
    python3 tools/pitch_fit.py --synth-only                     # validate the fitter on known pitch offsets
    python3 tools/pitch_fit.py --frames DIR --out results.json  # any directory of head RGB pngs
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import cv2
import numpy as np
from scipy.optimize import minimize

from aaltonomous.interface.calibration import Calibration
from aaltonomous.scene.scene import Scene
from aaltonomous.baselines.classical.mock_world import Renderer

SC, CAL = Scene.load(), Calibration.load()
T = CAL._T("head")
K = CAL.cams["head"].K
R, O = T[:3, :3].copy(), T[:3, 3].copy()
HALF = np.asarray(SC.cfg["bins"]["outer_size"], float) / 2
NOM = {n: SC.mat_to_pelvis([c[0], c[1], 0.0])[:2] for n, c in SC.cfg["bins"]["centers"].items()}
NAMES = ["left", "center", "right"]


def rx(deg):
    t = np.radians(deg)
    c, s = np.cos(t), np.sin(t)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


class RimFit:
    def __init__(self, counter_z=None, wall_height=None):
        self.zc = SC.counter_z if counter_z is None else counter_z
        self.wh = float(SC.cfg["bins"]["wall_height"]) if wall_height is None else wall_height
        self.zrim = self.zc + self.wh
        self.far_x = {n: NOM[n][0] + HALF[1] for n in NAMES}
        self.yspan = {n: (NOM[n][1] - HALF[0], NOM[n][1] + HALF[0]) for n in NAMES}

    def intersect(self, u, v, deg, shift):
        ray_opt = np.c_[(u - K[0, 2]) / K[0, 0], (v - K[1, 2]) / K[1, 1], np.ones(len(u))]
        ray = ray_opt @ (R @ rx(deg)).T
        o = O + np.asarray(shift, float)
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (self.zrim - o[2]) / ray[:, 2]
        ok = (ray[:, 2] < -1e-6) & (s > 0) & np.isfinite(s)
        Q = o + s[:, None] * ray
        return Q, ok

    def residuals(self, bins, deg, shift=(0, 0, 0)):
        """{name: trimmed rms point-to-far-rim distance (m)}; bins = {name: (u, v) envelope pixels}."""
        out = {}
        for n, (u, v) in bins.items():
            Q, ok = self.intersect(u, v, deg, shift)
            if ok.sum() < 20:
                out[n] = np.inf
                continue
            dx = Q[ok, 0] - self.far_x[n]
            dy = Q[ok, 1] - np.clip(Q[ok, 1], *self.yspan[n])
            r = np.sort(np.hypot(dx, dy))
            k = max(20, int(len(r) * 0.9))          # trim worst 10% (occlusions, merged objects)
            out[n] = float(np.sqrt(np.mean(r[:k] ** 2)))
        return out

    def cost(self, x, bins, model):
        deg = x[0]
        shift = (0, 0, 0) if model == "pitch" else (0, 0, x[1]) if model == "pitch_z" else (x[1], 0, x[2])
        if not (-20 <= deg <= 20) or np.max(np.abs(shift)) > 0.20:
            return 1e6
        res = self.residuals(bins, deg, shift)
        vals = [v for v in res.values() if np.isfinite(v)]
        return float(np.mean(vals)) if vals else 1e6

    def fit(self, bins, model="pitch"):
        """(deg, shift, residuals). Models: pitch | pitch_z (+dz) | pitch_xz (+dx,+dz). 3 starts, Nelder-Mead."""
        x0 = {"pitch": [0.0], "pitch_z": [0.0, 0.0], "pitch_xz": [0.0, 0.0, 0.0]}[model]
        best = None
        for start in (-5.0, 0.0, 5.0):             # 3 starts: the cost is mildly non-convex in pitch
            x0[0] = start
            r = minimize(self.cost, x0, args=(bins, model), method="Nelder-Mead",
                         options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 400})
            if best is None or r.fun < best.fun:
                best = r
        deg = float(best.x[0])
        shift = (0, 0, 0) if model == "pitch" else (0, 0, float(best.x[1])) if model == "pitch_z" else (float(best.x[1]), 0, float(best.x[2]))
        return deg, shift, self.residuals(bins, deg, shift)


def far_rim_bins(rgb, tight=True):
    """{name: (u, v) far-rim envelope}, n_bins. The three bins merge into one orange component (tape + shared
    hue), so connected components cannot separate them. Instead take the TOP orange edge per column (the far
    rim, which sits well above the objects and the counter tape in v), filter to wall-like columns, and split
    by u into the three nominal bins. tight = real-bin hue (H<=9 or >=170); False for synthetic renders."""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    m = cv2.inRange(hsv, (0, 60, 40), (26, 255, 255)) | cv2.inRange(hsv, (165, 60, 40), (180, 255, 255))
    if tight:
        m &= ((hsv[..., 0] <= 9) | (hsv[..., 0] >= 170)).astype(np.uint8) * 255
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
    Hh, Ww = m.shape
    top = m.argmax(0)                      # first orange pixel per column = far rim
    ext = m.sum(0) // 255                  # vertical orange extent (a wall, not a thin tape line)
    has = m.any(0)
    v_cut = 0.42 * Hh                      # rims live in the top ~40%; objects + tape are lower
    cols = np.flatnonzero(has & (top <= v_cut) & (ext >= 15))
    if len(cols) < 60:
        return {}, 0
    # u boundaries from the nominal projected bin centres (bins are well separated in u)
    pu = {n: CAL.project("head", SC.mat_to_pelvis([*SC.cfg["bins"]["centers"][n], 0.06]))[0] for n in NAMES}
    b1 = (pu["left"] + pu["center"]) / 2
    b2 = (pu["center"] + pu["right"]) / 2
    bins = {}
    for nm, lo, hi in (("left", -1, b1), ("center", b1, b2), ("right", b2, Ww + 1)):
        sel = cols[(cols >= lo) & (cols < hi)]
        if len(sel) >= 30:
            bins[nm] = (sel.astype(float), top[sel].astype(float))
    return bins, len(bins)


def synth_case(deg_true, fit: RimFit):
    """Render bins with a KNOWN pitch offset, fit blind. Returns (true, fitted, mean rms m) or None."""
    T_true = np.eye(4)
    T_true[:3, :3] = R @ rx(deg_true)
    T_true[:3, 3] = O
    rend = Renderer(SC, CAL, T_pelvis_optical=T_true)
    rgb, _ = rend.render([])                        # bins only, no objects
    bins, _ = far_rim_bins(rgb, tight=False)
    if len(bins) < 3:
        return None
    deg, shift, res = fit.fit(bins, "pitch")
    return deg_true, deg, float(np.mean(list(res.values())))


def real_frame(path, fit: RimFit):
    bgr = cv2.imread(path)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    bins, n_bins = far_rim_bins(rgb, tight=True)
    row = {"frame": Path(path).name, "n_bins": len(bins)}
    if len(bins) >= 2:
        res0 = fit.residuals(bins, 0.0)
        row["rms0_cm"] = {k: round(100 * v, 1) for k, v in res0.items()}
        for model in ("pitch", "pitch_z", "pitch_xz"):
            deg, shift, res = fit.fit(bins, model)
            row[model] = {"deg": round(deg, 2),
                          "shift_cm": [round(100 * s, 1) for s in shift],
                          "rms_cm": {k: round(100 * v, 1) for k, v in res.items()}}
    return row


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--synth-only", action="store_true", help="only the synthetic validation")
    ap.add_argument("--real-only", action="store_true", help="skip the synthetic validation")
    ap.add_argument("--frames", default="/var/tmp/hf_g1/frames", help="directory of head RGB pngs")
    ap.add_argument("--out", default="pitch_fit_results.json", help="where to write the JSON results")
    ap.add_argument("--counter-z", type=float, default=None, help="override scene counter_z (sensitivity)")
    ap.add_argument("--wall-height", type=float, default=None, help="override bin wall_height (sensitivity)")
    args = ap.parse_args(argv)

    out = {}
    fit = RimFit(counter_z=args.counter_z, wall_height=args.wall_height)
    if not args.real_only:
        print("== synthetic validation (known pitch, fitted blind) ==")
        rows = []
        for d in (-6.0, -3.0, 0.0, 3.0, 6.0):
            r = synth_case(d, fit)
            if r:
                rows.append({"true_deg": r[0], "fitted_deg": round(r[1], 2), "rms_cm": round(100 * r[2], 2)})
                print(f"  true {r[0]:+5.1f} deg -> fitted {r[1]:+6.2f} deg, rim rms {100 * r[2]:5.2f} cm")
        out["synthetic"] = rows
    if not args.synth_only:
        print(f"== real frames from {args.frames} (counter_z={fit.zc:.3f} wall={fit.wh:.3f}) ==")
        paths = sorted(glob.glob(str(Path(args.frames) / "*.png")))
        rows = [real_frame(p, fit) for p in paths]
        for r in rows:
            if "pitch" in r:
                p, pz = r["pitch"], r["pitch_z"]
                print(f"  {r['frame']:10} bins={r['n_bins']}  pitch={p['deg']:+6.2f} deg rms={np.mean(list(p['rms_cm'].values())):5.1f} cm"
                      f"  | +dz: {pz['deg']:+6.2f} deg dz={pz['shift_cm'][2]:+5.1f} cm rms={np.mean(list(pz['rms_cm'].values())):4.1f} cm"
                      f"  | no-pitch rms={np.mean(list(r['rms0_cm'].values())):5.1f} cm")
            else:
                print(f"  {r['frame']:10} bins={r['n_bins']} (not fittable)")
        out["real"] = rows
        ok = [r for r in rows if "pitch" in r]
        if ok:
            degs = [r["pitch"]["deg"] for r in ok]
            out["summary"] = {"n": len(ok), "pitch_mean_deg": round(float(np.mean(degs)), 2),
                              "pitch_std_deg": round(float(np.std(degs)), 2),
                              "pitch_min": min(degs), "pitch_max": max(degs)}
            print(f"  SUMMARY: pitch {np.mean(degs):+.2f} +/- {np.std(degs):.2f} deg over {len(ok)} frames")
    with open(args.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
