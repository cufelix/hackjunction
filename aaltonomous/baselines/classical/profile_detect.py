"""Auto-detect the config profile ("sim" or "none" = real robot) from the first head RGB-D frames.

Gate: flag profile_auto.auto_detect (default OFF; ClassicalServerPolicy). Two signals per frame, combined:
  depth   invalid px (0 / NaN / inf) over the whole image, and the std (mm) of a trimmed plane fit on the lower-central
          counter region. Sim renders are exact (~0 holes, ~0 noise); a D435i has holes and mm-level noise.
          A missing depth array is "not sim".
  palette hue (OpenCV H 0-180) of saturated px ABOVE the counter (plane residual > RAISED_MM). Sim objects are pure
          red / blue / yellow, real ones orange / purple / yellow. Bands from config.yaml perception.hsv (sim DR red H
          0-7 / 170-179, blue H 100-116; real purple H 108-127). Yellow (22-35) is shared and orange (9-21) is neutral:
          sim bins are H ~12 as well, so only purple counts as a real vote. Raised, object-sized blobs only.
Per-frame vote: sim iff the depth signal says sim (palette is computed for the log only: real purple measures H110,
inside the sim blue band, so a palette veto never fired on real frames). Missing depth votes none.
Decision: sim iff EVERY inspected frame votes sim (unanimity, N >= 1), else none.
"""
from __future__ import annotations

import numpy as np

# ponytail: thresholds are tuned on the mock renderer and config.yaml notes, not on real D435i frames (none on this
# machine). Re-check with tools/profile_detect_eval.py on drop_2 ep<40 before trusting it on the robot.
HOLE_SIM_MAX = 0.01          # sim: ~0 invalid px; real D435i 2-5 %. Above 1 % is not sim.
PLANE_STD_SIM_MAX_MM = 1.0   # sim: uint16-mm quantisation only (mock: 0.43 mm incl. edge bleed); D435i 1-3 mm at 0.5-1 m
TRIM_FLOOR_MM = 3.0          # residual clip floor for the trimmed fit (object edges out, real noise in)
PLANE_MIN_PX = 500           # counter-region px needed for a plane fit
RAISED_MM = 10.0             # px at least this far in front of the counter plane count as objects
OBJ_AREA_MIN, OBJ_AREA_MAX = 300, 15000   # blob px: a 45 mm cube is ~4.5-5.5k px at 640 px wide; mock bin walls 14-40k
S_MIN, V_MIN = 70, 60        # saturated px (config.yaml hsv: S min 70)
PALETTE_MIN_PX = 150         # fewer saturated raised px -> palette inconclusive
PALETTE_SIM_FRAC = 0.8       # sim-hued share of (sim+real-hued) px to call the palette "sim"
PALETTE_REAL_FRAC = 0.2      # ... and "real" at or below this share
SIM_HUE = ((0, 8), (170, 179), (100, 116))    # red (wraps at 0), blue
REAL_HUE = ((117, 130),)                      # purple / violet. Orange (9-21) is NOT a real vote: sim bins are H ~12 too
INF = float("inf")


def _plane(depth):
    """Trimmed plane fit on rows 55-95 %, cols 20-80 %. For a plane 1/z is affine in pixel (u, v) exactly, so the fit
    is linear in 1/z and residuals are reported in mm. Returns (coef, normalise, std_mm) or (None, None, inf)."""
    h, w = depth.shape
    r0, r1, c0, c1 = int(0.55 * h), int(0.95 * h), int(0.2 * w), int(0.8 * w)
    vv, uu = np.mgrid[r0:r1, c0:c1]
    z = depth[r0:r1, c0:c1].ravel()
    ok = np.isfinite(z) & (z > 0)
    if ok.sum() < PLANE_MIN_PX:
        return None, None, INF
    cu, cv = uu.mean(), vv.mean()
    su, sv = max(1.0, uu.std()), max(1.0, vv.std())
    A = np.stack([np.ones(ok.sum()), (uu.ravel()[ok] - cu) / su, (vv.ravel()[ok] - cv) / sv], 1)
    zz = z[ok]
    res = np.abs(zz - 1.0 / np.maximum(A @ np.linalg.lstsq(A, 1.0 / zz, rcond=None)[0], 1e-12))
    keep = res <= np.quantile(res, 0.5)          # start from the closer half: objects in the region pull the fit
    for _ in range(5):
        coef = np.linalg.lstsq(A[keep], 1.0 / zz[keep], rcond=None)[0]
        res_s = zz - 1.0 / np.maximum(A @ coef, 1e-12)
        keep = np.abs(res_s) <= max(3.0 * 1.4826 * np.median(np.abs(res_s[keep])), TRIM_FLOOR_MM)
        if keep.sum() < PLANE_MIN_PX:
            return None, None, INF
    return coef, (cu, cv, su, sv), float(np.std(res_s[keep]))


def _holes(depth):
    return float(np.mean(~np.isfinite(depth) | (depth == 0)))


def _palette(rgb, depth, coef, norm):
    import cv2
    hsv = cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb, np.uint8)), cv2.COLOR_RGB2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    mask = (s >= S_MIN) & (v >= V_MIN)
    if coef is not None:   # keep only px raised above the counter plane (bins' floors and counter drop out)
        vv, uu = np.mgrid[0:depth.shape[0], 0:depth.shape[1]]
        cu, cv, su, sv = norm
        plane = 1.0 / np.maximum(coef[0] + coef[1] * (uu - cu) / su + coef[2] * (vv - cv) / sv, 1e-12)
        with np.errstate(invalid="ignore"):
            mask &= (plane - depth) > RAISED_MM
    # object-sized blobs only (bins are big blobs)
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    area = stats[1:, cv2.CC_STAT_AREA]
    keep = np.flatnonzero((area >= OBJ_AREA_MIN) & (area <= OBJ_AREA_MAX)) + 1
    mask = np.isin(lab, keep)
    hh = h[mask]
    sim = int(np.count_nonzero(_in(hh, SIM_HUE)))
    real = int(np.count_nonzero(_in(hh, REAL_HUE)))
    if sim + real < PALETTE_MIN_PX:
        return "inconclusive", sim, real
    frac = sim / (sim + real)
    return ("sim" if frac >= PALETTE_SIM_FRAC else "real" if frac <= PALETTE_REAL_FRAC else "inconclusive"), sim, real


def _in(h, bands):
    return np.any([(h >= lo) & (h <= hi) for lo, hi in bands], axis=0)


def rgb_noise(rgb):
    """Whole-frame grey noise: median |grey - gaussian_blur(grey)| (px). Median so object edges do not count.
    Log-only metric for now; the sim/real noise-floor threshold is still to be set from drop_2 data."""
    import cv2
    g = cv2.cvtColor(np.ascontiguousarray(np.asarray(rgb, np.uint8)), cv2.COLOR_RGB2GRAY).astype(np.float32)
    return float(np.median(np.abs(g - cv2.GaussianBlur(g, (5, 5), 0))))


def _frame(obs):
    rgb = obs.get("head_rgb")
    depth = obs.get("head_depth")
    out = {"holes": 1.0, "plane_std_mm": INF, "depth_sim": False, "palette": "inconclusive", "sim_px": 0, "real_px": 0,
           "rgb_noise": None}
    if rgb is not None and np.ndim(rgb) == 3:
        out["rgb_noise"] = round(rgb_noise(rgb), 3)
    coef = norm = None
    if depth is not None and np.ndim(depth) == 2:
        depth = np.asarray(depth, float)
        out["holes"] = _holes(depth)
        coef, norm, out["plane_std_mm"] = _plane(depth)
        out["depth_sim"] = out["holes"] <= HOLE_SIM_MAX and out["plane_std_mm"] <= PLANE_STD_SIM_MAX_MM
    if rgb is not None and np.ndim(rgb) == 3 and (depth is None or np.ndim(depth) != 2 or np.shape(rgb)[:2] == np.shape(depth)):
        out["palette"], out["sim_px"], out["real_px"] = _palette(
            rgb, depth if depth is not None and np.ndim(depth) == 2 else None, coef, norm)
    out["sim"] = out["depth_sim"]   # palette is log-only (see module docstring)
    return out


def detect_profile(frames):
    """frames: list of obs dicts (head_rgb, head_depth). Returns (profile in {"sim", "none"}, info dict)."""
    per = [_frame(o) for o in frames]
    if not per:
        return "none", {"frames": 0, "votes": 0, "depth_holes": 1.0, "plane_std_mm": INF, "palette": "inconclusive",
                        "rgb_noise": [], "per_frame": []}
    labels = [p["palette"] for p in per]
    palette = "real" if "real" in labels else ("sim" if all(x == "sim" for x in labels) else "inconclusive")
    votes = sum(p["sim"] for p in per)
    info = {"frames": len(per), "votes": votes, "depth_holes": max(p["holes"] for p in per),
            "plane_std_mm": max(p["plane_std_mm"] for p in per), "palette": palette,
            "rgb_noise": [p["rgb_noise"] for p in per], "per_frame": per}   # rgb_noise: log-only, no vote
    return ("sim" if votes == len(per) > 0 else "none"), info   # empty list -> none   # unanimity: any non-sim frame -> none
