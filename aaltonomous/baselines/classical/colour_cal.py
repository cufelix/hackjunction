"""Online colour calibration: hue histogram of the object pixels -> peaks -> shifted HSV class ranges.

Pure numpy, no scene needed. Used by Perception._calibrate_colours (flag perception.online_colour_cal)."""
import numpy as np

MAX_SHIFT = 8          # H units: centre of a class range moves at most this far from the configured centre
MIN_S = 60             # only saturated pixels vote
MIN_PEAK_PX = 120      # pixels within +-4 H of a peak
MIN_PEAK_REL = 0.08    # peak weight relative to the strongest peak
MAX_PEAKS = 4


def hue_histogram(hsv, sel):
    """180-bin hue histogram weighted by S*V/255^2 over the selected saturated pixels, plus the unweighted counts."""
    px = hsv[sel]
    px = px[px[:, 1] >= MIN_S]
    w = px[:, 1].astype(float) * px[:, 2] / 65025.0
    hist = np.bincount(px[:, 0].astype(int) % 180, weights=w, minlength=180)[:180]
    cnt = np.bincount(px[:, 0].astype(int) % 180, minlength=180)[:180]
    return np.stack([hist, cnt]), len(px)


def _circ_sum(a, r):
    return sum(np.roll(a, k) for k in range(-r, r + 1))


def find_peaks(hist2):
    """[(h, weight)] up to MAX_PEAKS, strongest first. Peak = local max of the +-2 smoothed weighted histogram."""
    hist, cnt = hist2
    sm = _circ_sum(hist, 2)
    mass = _circ_sum(cnt, 4)
    peaks = []
    for h in np.argsort(-sm):
        if sm[h] <= 0 or len(peaks) >= MAX_PEAKS:
            break
        if mass[h] < MIN_PEAK_PX or (peaks and sm[h] < MIN_PEAK_REL * peaks[0][1]):
            continue
        if any(min((h - p) % 180, (p - h) % 180) < 8 for p, _ in peaks):
            continue
        peaks.append((int(h), float(sm[h])))
    return peaks


def _intervals(hsv_cfg):
    """class -> (lo, hi, [(slo, shi, vlo, vhi)]) in unwrapped H (red wrap: lo < 0); None if the shape is unusual."""
    out = {}
    for col, ranges in hsv_cfg.items():
        rs = [(list(a), list(b)) for a, b in ranges]
        if len(rs) == 1:
            out[col] = (rs[0][0][0], rs[0][1][0], rs[0])
        elif len(rs) == 2:
            lo_r = min(rs, key=lambda r: r[0][0])
            hi_r = max(rs, key=lambda r: r[0][0])
            out[col] = (hi_r[0][0] - 180, lo_r[1][0], lo_r)
        else:
            return None
    return out


def _emit(lo, hi, base):
    (_, s0, v0), (_, s1, v1) = base
    if lo < 0:
        return [[[0, s0, v0], [int(hi), s1, v1]], [[int(180 + lo), s0, v0], [180, s1, v1]]]
    return [[[int(lo), s0, v0], [int(hi), s1, v1]]]


def adapt_ranges(hsv_cfg, hist2, n_px=0):
    """New hsv config (same structure) with each class range shifted onto its measured peak. Returns (cfg, info)."""
    iv = _intervals(hsv_cfg)
    peaks = find_peaks(hist2)
    info = {"peaks": peaks, "assigned": {}, "ranges": {}}
    if iv is None or not peaks:
        info["ranges"] = {k: [list(map(list, r)) for r in v] for k, v in hsv_cfg.items()}
        return hsv_cfg, info
    new = {}
    for col, (lo, hi, _) in iv.items():
        new[col] = [lo, hi]
    cen = {col: (lo + hi) / 2.0 for col, (lo, hi, _) in iv.items()}
    cand = {}                                    # class -> (unwrapped peak, weight): strongest peak nearest to it
    for h, w in peaks:
        dist, col, hh = min((abs(hh - cen[k]), k, hh) for k in cen for hh in (h, h - 180, h + 180))
        lo, hi, _ = iv[col]
        if lo - MAX_SHIFT <= hh <= hi + MAX_SHIFT and (col not in cand or w > cand[col][1]):
            cand[col] = (hh, w)
    for col, (hh, _) in cand.items():
        lo, hi, _ = iv[col]
        shift = int(round(np.clip(hh - cen[col], -MAX_SHIFT, MAX_SHIFT)))
        new[col] = [lo + shift, hi + shift]
        info["assigned"][col] = (int(hh % 180), shift)
    # overlap protection: sort by lower bound; overlapping neighbours meet at the histogram valley between them
    sm = _circ_sum(hist2[0], 2)
    order = sorted(new, key=lambda k: new[k][0])
    for a, b in zip(order, order[1:]):
        if new[a][1] >= new[b][0] and (a in cand) != (b in cand):   # one class keeps its robust range: trim the other
            if a in cand:
                new[a][1] = new[b][0] - 1
            else:
                new[b][0] = new[a][1] + 1
        elif new[a][1] >= new[b][0]:
            lo_h, hi_h = int(new[b][0]), int(new[a][1])      # the contested band
            cand = range(lo_h, hi_h + 1)
            v = min(cand, key=lambda h: (sm[h % 180], abs(h - (lo_h + hi_h) / 2))) if hi_h >= lo_h else lo_h
            new[a][1] = min(new[a][1], v - 1)
            new[b][0] = max(new[b][0], v)
            if new[a][1] < new[a][0] or new[b][1] < new[b][0]:    # degenerate: give up on this class pair
                orig = _intervals(hsv_cfg)
                new[a], new[b] = list(orig[a][:2]), list(orig[b][:2])
    out = {}
    for col, (lo, hi, base) in iv.items():
        out[col] = _emit(new[col][0], new[col][1], base)
        info["ranges"][col] = [int(new[col][0]), int(new[col][1])]
    return out, info
