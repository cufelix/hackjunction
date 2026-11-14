"""perception.hue_guard: judge orange-vs-yellow candidates near the red|yellow hue boundary.

Pure functions, no image work. perception.py builds one Cand per red/yellow component (layer 3: top-face H, S, V)
and applies the verdict. Verdict label None = ambiguous (abstain, color_conf 0.3, not placed).

Layer 1 (the robust_v2 HSV set, config.yaml): red H 0-22, yellow H 24-52. The open band (22, 24) is the abstain band.
Layer 2 (here, per frame): adjacent red|yellow candidates (sorted by hue) closer than min_gap_deg abstain. Pairs of the
same label never abstain each other, and the threshold labels are otherwise kept.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

RED, YELLOW = "red", "yellow"
AMBIGUOUS_CONF = 0.3   # Detection.color_conf for an abstained candidate


@dataclass(frozen=True)
class Cand:
    label: str    # active-set class: red (orange in the bin vocabulary) | yellow | other
    hue: float    # OpenCV H, 0-179
    sat: float    # OpenCV S, 0-255 (top face)
    val: float    # OpenCV V, 0-255 (top face)


def boundaries(ranges):
    """(band_lo, band_hi, yellow_hi): the open hue band between the red and yellow ranges of the active set."""
    red_his = [int(hi[0]) for lo, hi in ranges["red"] if int(lo[0]) < 90]   # low-hue red ranges only (not the wrap)
    yel = ranges["yellow"]
    return max(red_his), min(int(lo[0]) for lo, _ in yel), max(int(hi[0]) for _, hi in yel)


def judge(cands, ranges, min_gap_deg=6.0, red_min_s=0.0):
    """Return a list of (label | None, reason | None), aligned with cands. reason None = accepted as label.

    band: a red/yellow candidate with band_lo < H < band_hi -> ambiguous.
    low_s_red: a red-labelled candidate with top-face S < red_min_s -> ambiguous (absolute floor; real orange/red tops
        S 144-200, sim red 217; a yellow top drifting to H 20 at S 73 is labelled red and must not pass).
    gap: in the red..yellow window (H in [0, yellow_hi]) sorted by hue, two neighbours with different labels that are
        less than min_gap_deg apart -> both ambiguous. Same-label neighbours never abstain each other.
    """
    band_lo, band_hi, yellow_hi = boundaries(ranges)
    out = [(c.label, None) for c in cands]
    win = sorted((i for i, c in enumerate(cands) if c.label in (RED, YELLOW) and 0.0 <= c.hue <= yellow_hi),
                 key=lambda i: cands[i].hue)
    for a, b in zip(win, win[1:]):
        if cands[a].label != cands[b].label and cands[b].hue - cands[a].hue < min_gap_deg:
            out[a] = (None, "gap")
            out[b] = (None, "gap")
    for i, c in enumerate(cands):
        if c.label in (RED, YELLOW) and band_lo < c.hue < band_hi:
            out[i] = (None, "band")
    for i, c in enumerate(cands):
        if c.label == RED and c.sat < red_min_s:
            out[i] = (None, "low_s_red")
    return out


def classify_hue(h, ranges):
    """Class of one OpenCV hue under the active set: red | yellow | None (outside both, e.g. the band)."""
    for name in ("red", "yellow"):
        for lo, hi in ranges[name]:
            if int(lo[0]) <= h <= int(hi[0]):
                return name
    return None


def top_face_hsv(px, frac=0.35):
    """Median (H, S, V) of the brightest `frac` of the pixels by V (the top face; sides are darker)."""
    px = np.asarray(px, float).reshape(-1, 3)
    k = max(1, int(np.ceil(frac * len(px))))
    top = px[np.argsort(-px[:, 2], kind="stable")[:k]]
    med = np.median(top, axis=0)
    return (float(med[0]), float(med[1]), float(med[2]))
