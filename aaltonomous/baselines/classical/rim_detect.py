"""Image-only detection of the three orange bins and their front rims (perception.bins_from_image).

The real bins are orange cardboard boxes with orange tape along the rims and, in the organiser recordings, there is
no depth.  Nominal bin geometry projects the bins to the wrong image rows (the real ones are 5-14 cm farther), so
objects just in front of them were gated away.  This module finds the bins in the picture itself:

* bin mass  = warm (orange / red hue), saturated, not-object-bright pixels, large connected components;
* tape      = thin, elongated warm components (never a target; returned as a mask so the caller can drop them);
* per bin   = a lateral extent (runs of supported columns, merged neighbours split by the expected bin width) and the
              FRONT-RIM LINE v = a*u + b fitted robustly to the lowest bin/tape pixel of every column.

Pure numpy / cv2, no scene or camera knowledge except the optional ``expect_w_px`` hint.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

DEFAULTS = {
    "h_max": 17, "h_min_wrap": 170, "s_min": 110, "tape_s_min": 110,
    "object_s_min": 140, "object_v_min": 150,
    "row_frac": 0.66,            # only the upper part of the image can hold bins / rim tape
    "open_px": 5, "min_component_px": 2500,
    "tape_min_len_px": 30, "tape_max_thick_px": 24, "tape_min_aspect": 3.5, "tape_min_fill": 0.45,
    "tape_dilate_px": 3, "tape_reach_px": 25,
    "min_col_px": 6, "gap_px": 10, "min_bin_w_px": 40,
    "max_bins": 3, "min_bins": 2, "lateral_margin_px": 6, "rim_margin_px": 4, "max_slope": 0.30,
    "fit_min_cols": 12, "spike_px": 12,
}


@dataclass
class RimBin:
    u0: int
    u1: int                 # inclusive pixel columns
    a: float                # front-rim line: v = a * u + b (image rows grow downwards)
    b: float
    n_cols: int
    clipped: bool           # touches the image border (lateral extent is a lower bound)
    resid_px: float         # median |v - line(u)| over the supported columns
    name: str | None = None

    def v_at(self, u):
        return self.a * np.asarray(u, float) + self.b


@dataclass
class RimResult:
    bins: list = field(default_factory=list)
    tape: np.ndarray | None = None            # uint8 (H, W), 255 on tape pixels (dilated), else 0
    tape_blobs: list = field(default_factory=list)   # [(area, (cx, cy))]
    row_limit: np.ndarray | None = None       # float (W,): rows <= limit are bin / above; -1e6 = no bin in the column


def _warm(hsv, p):
    h, s = hsv[..., 0], hsv[..., 1]
    return ((h <= int(p["h_max"])) | (h >= int(p["h_min_wrap"]))) & (s >= int(p["s_min"]))


def find_tape(hsv_top, p):
    """Thin elongated warm components: (mask uint8, [(area, centroid)], [(x, y, w, h, bool mask)])."""
    warm = _warm(hsv_top, p) & (hsv_top[..., 1] >= int(p["tape_s_min"]))
    m = cv2.morphologyEx(warm.astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    n, lab, stats, cent = cv2.connectedComponentsWithStats(m, connectivity=8)
    tape = np.zeros(m.shape, np.uint8)
    blobs = []
    comps = []
    for i in range(1, n):
        x, y, w, h, area = (int(v) for v in stats[i])
        long, short = max(w, h), min(w, h)
        if (long >= int(p["tape_min_len_px"]) and short <= int(p["tape_max_thick_px"])
                and long / max(short, 1) >= float(p["tape_min_aspect"])
                and area / float(w * h) >= float(p["tape_min_fill"])):
            tape[lab == i] = 255
            blobs.append((area, (float(cent[i][0]), float(cent[i][1]))))
            comps.append((x, y, w, h, lab == i))
    return tape, blobs, comps


def detect_bin_rims(hsv, params=None, expect_w_px=None):
    """-> RimResult.  ``hsv`` is a uint8 HSV image (cv2 convention); ``expect_w_px`` the nominal projected bin width."""
    p = dict(DEFAULTS)
    p.update(params or {})
    H, W = hsv.shape[:2]
    R = int(H * float(p["row_frac"]))
    top = hsv[:R]
    res = RimResult()
    tape, blobs, tcomps = find_tape(top, p)
    # Mass: warm and not an object-bright pixel.  Bright, saturated tape stays out of the mass but joins the rim below.
    bright = (top[..., 1] >= int(p["object_s_min"])) & (top[..., 2] >= int(p["object_v_min"]))
    mass = (_warm(top, p) & ~bright).astype(np.uint8)
    k = int(p["open_px"])
    if k > 0:
        mass = cv2.morphologyEx(mass, cv2.MORPH_OPEN, np.ones((k, k), np.uint8))
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mass, connectivity=8)
    big = np.zeros(mass.shape, bool)
    for i in range(1, n):
        if int(stats[i, cv2.CC_STAT_AREA]) >= int(p["min_component_px"]):
            big |= lab == i
    if not big.any():
        return res
    mass_cols = big.sum(0) >= int(p["min_col_px"])
    # close small gaps, then runs
    gap = int(p["gap_px"])
    runs, start = [], None
    last = -10 ** 9
    for u in range(W):
        if mass_cols[u]:
            if start is None:
                start = u
            last = u
        elif start is not None and u - last > gap:
            runs.append((start, last))
            start = None
    if start is not None:
        runs.append((start, last))
    runs = [(a, b) for a, b in runs if b - a + 1 >= int(p["min_bin_w_px"])]
    if not runs:
        return res
    # lowest bin pixel per column; detached bright tape just under the wall extends it (<= tape_reach_px)
    rows = np.arange(R)[:, None]
    bot_mass = np.where(big.any(0), R - 1 - np.argmax(big[::-1], axis=0), -1)
    tp = tape.astype(bool)
    bot_tape = np.where(tp.any(0), R - 1 - np.argmax(tp[::-1], axis=0), -1)
    bottom = np.where((bot_tape > bot_mass) & (bot_tape - bot_mass <= int(p["tape_reach_px"])), bot_tape, bot_mass)
    # bright warm pixels (rim tape) hanging directly under the wall, contiguous within a 2 px gap, belong to the rim
    bw = _warm(top, p) & bright
    cur = bottom.copy()
    cols = np.arange(W)
    for _ in range(int(p["tape_reach_px"])):
        nxt = cur + 1
        ok = (cur >= 0) & (nxt < R)
        step = ok & bw[np.minimum(nxt, R - 1), cols]
        step2 = ok & ~step & (nxt + 1 < R) & bw[np.minimum(nxt + 1, R - 1), cols]
        cur = np.where(step, nxt, np.where(step2, nxt + 1, cur))
    bottom = cur
    del rows
    w_nom = float(expect_w_px) if expect_w_px else None
    nmax = int(p["max_bins"])
    segs = []
    for u0, u1 in runs:
        n_parts = 1
        if w_nom:
            n_parts = max(1, int(round((u1 - u0 + 1) / (0.9 * w_nom))))
        n_parts = min(n_parts, nmax)
        edges = np.linspace(u0, u1 + 1, n_parts + 1)
        for j in range(n_parts):
            segs.append((int(round(edges[j])), int(round(edges[j + 1])) - 1))
    segs = segs[:nmax] if len(segs) > nmax else segs
    rl = np.full(W, -1e6)
    for u0, u1 in segs:
        cols = np.arange(u0, u1 + 1)
        ok = (bottom[cols] >= 0) & mass_cols[cols]
        cols = cols[ok]
        if len(cols) < int(p["fit_min_cols"]):
            continue
        pts = np.c_[cols, bottom[cols]].astype(np.float32)
        vx, vy, x0, y0 = (float(t) for t in cv2.fitLine(pts, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel())
        if abs(vx) < 1e-6:
            continue
        a = vy / vx
        a = float(np.clip(a, -float(p["max_slope"]), float(p["max_slope"])))
        b = y0 - a * x0
        # the clamped slope may need a re-anchored intercept: median of the residuals keeps the line on the data
        b = float(np.median(pts[:, 1] - a * pts[:, 0]))
        # Hands / shadows only RAISE the visible bottom, a touching object only drags single columns far below it:
        # refit on the lower envelope (at or below the line, spikes beyond spike_px dropped).
        d = pts[:, 1] - (a * pts[:, 0] + b)
        sel = (d >= -3.0) & (d <= float(p["spike_px"]))
        if int(sel.sum()) >= int(p["fit_min_cols"]):
            b = float(np.median(pts[sel, 1] - a * pts[sel, 0]))
            pts = pts[sel]
        resid = float(np.median(np.abs(pts[:, 1] - (a * pts[:, 0] + b))))
        res.bins.append(RimBin(u0, u1, a, b, int(len(cols)), clipped=(u0 <= 0 or u1 >= W - 1), resid_px=resid))
    if not res.bins:
        return res
    # Tape belongs to a bin: inside a bin's columns, not farther below its rim than tape_reach_px.  A thin strip
    # elsewhere (e.g. the top face of an object) is NOT tape.
    full_tape = np.zeros((H, W), np.uint8)
    reach = float(p["tape_reach_px"])
    for x, y, w, h, m in tcomps:
        for rb in res.bins:
            cu = x + w / 2.0
            if rb.u0 - 2 <= cu <= rb.u1 + 2 and y <= rb.v_at(cu) + reach:
                full_tape[:R][m] = 255
                res.tape_blobs.append((int(m.sum()), (cu, y + h / 2.0)))
                break
    td = int(p["tape_dilate_px"])
    if td > 0 and full_tape.any():
        full_tape = cv2.dilate(full_tape, np.ones((2 * td + 1, 2 * td + 1), np.uint8))
    res.tape = full_tape
    lm = int(p["lateral_margin_px"])
    for rb in res.bins:
        c0, c1 = max(0, rb.u0 - lm), min(W - 1, rb.u1 + lm)
        cols = np.arange(c0, c1 + 1)
        rl[cols] = np.maximum(rl[cols], rb.v_at(cols) + float(p["rim_margin_px"]))
    res.row_limit = rl
    return res
