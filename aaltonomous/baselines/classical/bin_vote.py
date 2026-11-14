"""bin.vote (default OFF): per-bin position from up to 3 votes (depth rim, orange tape, nominal prior).

Pure combine() + per-frame vote extractors + vote_bins() driver. Never raises: any error -> the prior (shift 0)."""
from __future__ import annotations

import numpy as np

TAPE_V_MAX_FRAC = 0.65   # tape only in the upper part of the image (the bins' far rims)


def _agree_mean(pts, agree_m):
    """(mean of the closest pair, n_agree) over pts; n_agree 3 if all pairs agree, 2 if only a pair, 0 if none."""
    best = None
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.linalg.norm(pts[i] - pts[j]))
            if best is None or d < best[0]:
                best = (d, i, j)
    if best is None or best[0] > agree_m:
        return None, 0
    if len(pts) == 3 and all(np.linalg.norm(pts[i] - pts[j]) <= agree_m for i, j in ((0, 1), (0, 2), (1, 2))):
        return np.mean(pts, axis=0), 3
    return (pts[best[1]] + pts[best[2]]) / 2.0, 2


def combine(prior, rim, tape, v):
    """-> (xy, n_agree, drop_extra) for one bin. Votes = prior (always) + rim/tape (None = missing)."""
    prior = np.asarray(prior, float)
    votes = [prior] + [np.asarray(x, float) for x in (rim, tape) if x is not None and np.all(np.isfinite(x))]
    extra = float(v["low_conf_extra_m"])
    if len(votes) == 1:
        return prior, 0, extra
    xy, n = _agree_mean(votes, float(v["agree_m"]))
    if xy is not None:
        extra_out = 0.0
    else:
        xy, n, extra_out = np.median(votes, axis=0), 0, extra
    if np.linalg.norm(xy - prior) > float(v["max_shift_m"]):
        return prior, 0, extra
    return xy, n, extra_out


def resolve(priors, rims, tapes, v):
    """priors/rims/tapes: {bin: xy|None}. -> (shift {bin: xy}, drop_extra {bin: m}, info {bin: (xy, n_agree)}).
    Left/right ordering in y must survive, else the prior for all bins."""
    res = {n: combine(p, rims.get(n), tapes.get(n), v) for n, p in priors.items()}
    order = sorted(priors, key=lambda n: priors[n][1])
    ys = [res[n][0][1] for n in order]
    if any(b <= a for a, b in zip(ys, ys[1:])):
        res = {n: (np.asarray(p, float), 0, float(v["low_conf_extra_m"])) for n, p in priors.items()}
    shift = {n: res[n][0] - np.asarray(priors[n], float) for n in priors}
    return shift, {n: res[n][2] for n in priors}, {n: (res[n][0], res[n][1]) for n in priors}


def _nearest_bin_points(P, centers, half, search):
    """{name: points (N,2) within the bin's window and nearer to it than to any other centre}."""
    names = list(centers)
    C = np.array([centers[n] for n in names])
    near = np.argmin(np.linalg.norm(P[:, None, :2] - C[None], axis=2), axis=1)
    out = {}
    for k, n in enumerate(names):
        q = P[near == k]
        out[n] = q[np.all(np.abs(q[:, :2] - C[k]) <= half + search, axis=1)][:, :2]
    return out


def _box_edge_dist(Q, ctr, half):
    q = np.abs(Q - ctr) - half
    sd = np.linalg.norm(np.maximum(q, 0), axis=1) + np.minimum(q.max(axis=1), 0)
    return np.abs(sd)


def rim_votes(perception, depth, centers, v):
    """{bin: xy|None}: per-bin fit of the outer+inner rim rectangles to elevated depth points, +-search_m window."""
    sc, cfg = perception.scene, perception.cfg
    c = cfg.get("self_calibration") or {}
    b = sc.cfg["bins"]
    out = {n: None for n in centers}
    if depth is None or np.ndim(depth) != 2:
        return out
    P, coef, why = perception._counter_plane(np.nan_to_num(np.asarray(depth, float), nan=0.0), c)
    if why:
        return out
    height = P[:, 2] - (np.c_[P[:, :2], np.ones(len(P))] @ coef)
    keep = (height >= float(c.get("min_wall_height_m", 0.008))) & (height <= float(b["wall_height"]) + 0.015)
    P = P[keep]
    outer = np.asarray(b["outer_size"], float)[::-1] / 2
    inner = outer - float(b["wall_thickness"])
    search = float(v["search_m"])
    trunc = 0.015
    for n, Q in _nearest_bin_points(P, centers, outer, search).items():
        if len(Q) < int(v["min_rim_points"]):
            continue
        Q = Q[np.linspace(0, len(Q) - 1, min(len(Q), 1500)).astype(int)]

        def score(off):
            d = np.minimum(_box_edge_dist(Q + off, centers[n], outer), _box_edge_dist(Q + off, centers[n], inner))
            return float(np.mean(np.minimum(d, trunc) ** 2))

        def search_grid(ctr, span, step):
            vals = np.arange(-span, span + step / 2, step)
            cand = [(score(ctr + [dx, dy]), ctr + [dx, dy]) for dx in vals for dy in vals
                    if max(abs(ctr[0] + dx), abs(ctr[1] + dy)) <= search + 1e-9]
            return min(cand, key=lambda t: t[0])[1]

        off = search_grid(np.zeros(2), search, 0.01)
        off = search_grid(off, 0.01, 0.002)
        off = search_grid(off, 0.002, 0.0005)
        d = np.minimum(_box_edge_dist(Q + off, centers[n], outer), _box_edge_dist(Q + off, centers[n], inner))
        if float(np.mean(d <= 0.006)) < 0.35:
            continue
        out[n] = np.asarray(centers[n], float) + off
    return out


def tape_votes(perception, rgb, depth, centers, v):
    """{bin: xy|None}: centroid (pelvis xy) of orange-tape pixels near the bin's rim, lifted via depth, or via the
    rim-height plane where depth is missing. # ponytail: the centroid stands for the bin centre (tape position on
    the rim is not modelled); upgrade to a per-edge model once the real tape layout is known."""
    import cv2
    sc = perception.scene
    out = {n: None for n in centers}
    if rgb is None or np.ndim(rgb) != 3:
        return out
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    lo, hi = v["tape_hsv"]
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask[int(mask.shape[0] * TAPE_V_MAX_FRAC):] = 0
    vv, uu = np.nonzero(mask)
    if len(vv) < int(v["min_tape_px"]):
        return out
    rim_z = sc.counter_z + float(sc.cfg["bins"]["wall_height"])
    d = np.zeros(len(vv))
    if depth is not None and np.ndim(depth) == 2:
        d = np.nan_to_num(np.asarray(depth, float)[vv, uu], nan=0.0, posinf=0.0, neginf=0.0)
    lo_d, hi_d = perception.cfg["depth_range_mm"]
    ok = (d >= lo_d) & (d <= hi_d)
    pts = np.zeros((len(vv), 3))
    if ok.any():
        pts[ok] = perception.cal.pixel_to_pelvis(perception.cam, uu[ok], vv[ok], d[ok])
    if (~ok).any():  # pelvis z is linear in depth along a ray: solve z(d) = rim_z from two samples
        u, w = uu[~ok], vv[~ok]
        p1 = perception.cal.pixel_to_pelvis(perception.cam, u, w, np.full(len(u), 1000.0))
        p2 = perception.cal.pixel_to_pelvis(perception.cam, u, w, np.full(len(u), 2000.0))
        dz = p2[:, 2] - p1[:, 2]
        t = np.where(np.abs(dz) > 1e-9, (rim_z - p1[:, 2]) / np.where(np.abs(dz) > 1e-9, dz, 1.0), np.nan)
        pts[~ok] = p1 + (p2 - p1) * t[:, None]
        ok2 = np.isfinite(pts).all(axis=1)
    else:
        ok2 = np.ones(len(vv), bool)
    near_rim = np.abs(pts[:, 2] - rim_z) <= 0.03          # blobs not near rim height are not tape
    pts = pts[ok2 & near_rim]
    if len(pts) < int(v["min_tape_px"]):
        return out
    outer = np.asarray(sc.cfg["bins"]["outer_size"], float)[::-1] / 2
    for n, Q in _nearest_bin_points(pts, centers, outer, float(v["search_m"])).items():
        if len(Q) >= int(v["min_tape_px"]):
            out[n] = np.median(Q, axis=0)
    return out


def _med(frames, n):
    xs = [f[n] for f in frames if f.get(n) is not None]
    return np.median(xs, axis=0) if xs else None


def vote_bins(perception, planner, frames, cfg, log=print):
    """frames: [{"head_rgb", "head_depth"}]. -> (shift, drop_extra). Never raises (prior on any error)."""
    v = cfg["bin"]["vote"]
    names = list(planner.scene.bin_names)
    priors = {n: np.asarray(planner.bin_xy(n), float) for n in names}
    try:
        rims, tapes = [], []
        for f in frames:
            rims.append(rim_votes(perception, f.get("head_depth"), priors, v))
            tapes.append(tape_votes(perception, f.get("head_rgb"), f.get("head_depth"), priors, v))
        rim = {n: _med(rims, n) for n in names}
        tape = {n: _med(tapes, n) for n in names}
        shift, extra, info = resolve(priors, rim, tape, v)
    except Exception as exc:  # optional feature: never terminate the control loop
        log(f"[bin.vote] failed ({type(exc).__name__}: {exc}); prior positions")
        return {}, {n: float(v["low_conf_extra_m"]) for n in names}
    log("[bin.vote] " + " ".join(f"{n}: shift=({1000 * shift[n][0]:+.0f},{1000 * shift[n][1]:+.0f})mm "
                                  f"agree={info[n][1]} extra={1000 * extra[n]:.0f}mm" for n in names))
    return shift, extra
