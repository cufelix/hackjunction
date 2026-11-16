"""bin.vote (default OFF): per-bin position from up to 3 votes (depth rim, orange tape, nominal prior).

Pure combine() + per-frame vote extractors + vote_bins() driver. Never raises: any error -> the prior (shift 0)."""
from __future__ import annotations

import numpy as np

TAPE_V_MAX_FRAC = 0.65   # tape only in the upper part of the image (the bins' far rims)
TAPE_AMBIG_RATIO = 0.5   # two tape blobs on one bin, the 2nd at least half the size of the 1st -> ambiguous, no vote
CONF_AGREE, CONF_PRIOR, CONF_DISAGREE = 1.0, 0.5, 0.3   # bin confidence: 2-of-3 / 3 agree, prior only, full disagreement


def bin_geometry(scene, shift=None, name=None):
    """THE runtime bin geometry (flag bin.geometry_runtime): scene.yaml nominal + a per-bin pelvis xy `shift`
    ({bin: (dx, dy)}, e.g. planner.bin_shift from bin.vote; None/{} = nominal). -> {bin: g} (or g if `name`), g keys:
    shift (pelvis xy | None), centre_mat / centre_pelvis (xy), outer_lo / outer_hi (mat xy), inner_lo / inner_hi (mat xyz),
    inner_pelvis (xlo, xhi, ylo, yhi). Mat <-> pelvis: pelvis (x, y) = mat (y, x) + origin, so shift (dx, dy) -> mat (dy, dx)."""
    shift = shift or {}
    b = scene.cfg["bins"]
    half = np.asarray(b["outer_size"], float) / 2
    out = {}
    for n in ([name] if name is not None else scene.bin_names):
        sh = np.asarray(shift[n], float) if n in shift else None
        c = np.asarray(b["centers"][n], float)
        cp = scene.mat_to_pelvis(np.r_[c, 0.0])[:2]
        lo, hi = scene.bin_inner_box(n)
        corners = scene.mat_to_pelvis(np.array([[x, y, 0.0] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])]))
        s2 = sh if sh is not None else np.zeros(2)
        if sh is not None:   # the whole bin moves: centre, outer + inner walls
            c, cp = c + sh[::-1], cp + sh
            sm = np.r_[sh[::-1], 0.0]
            lo, hi = lo + sm, hi + sm
        out[n] = {"shift": sh, "centre_mat": c, "centre_pelvis": cp, "outer_lo": c - half, "outer_hi": c + half,
                  "inner_lo": lo, "inner_hi": hi,
                  "inner_pelvis": (corners[:, 0].min() + s2[0], corners[:, 0].max() + s2[0],
                                   corners[:, 1].min() + s2[1], corners[:, 1].max() + s2[1])}
    return out if name is None else out[name]


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
    """-> (xy, n_agree, drop_extra, conf) for one bin. Votes = prior (always) + rim/tape (None = missing).
    Full disagreement keeps the prior (shift 0) and raises the release; the max_shift cap only guards agreed results."""
    prior = np.asarray(prior, float)
    votes = [prior] + [np.asarray(x, float) for x in (rim, tape) if x is not None and np.all(np.isfinite(x))]
    extra = float(v["low_conf_extra_m"])
    if len(votes) == 1:
        return prior, 0, extra, CONF_PRIOR
    xy, n = _agree_mean(votes, float(v["agree_m"]))
    if xy is None:
        return prior, 0, extra, CONF_DISAGREE
    if np.linalg.norm(xy - prior) > float(v["max_shift_m"]):
        return prior, 0, extra, CONF_PRIOR
    return xy, n, 0.0, CONF_AGREE


def resolve(priors, rims, tapes, v):
    """priors/rims/tapes: {bin: xy|None}. -> (shift {bin: xy}, drop_extra {bin: m}, info {bin: (xy, n_agree, conf)}).
    Left/right ordering in y must survive, else the prior for all bins."""
    res = {n: combine(p, rims.get(n), tapes.get(n), v) for n, p in priors.items()}
    order = sorted(priors, key=lambda n: priors[n][1])
    ys = [res[n][0][1] for n in order]
    if any(b <= a for a, b in zip(ys, ys[1:])):
        res = {n: (np.asarray(p, float), 0, float(v["low_conf_extra_m"]), CONF_PRIOR) for n, p in priors.items()}
    shift = {n: res[n][0] - np.asarray(priors[n], float) for n in priors}
    return shift, {n: res[n][2] for n in priors}, {n: (res[n][0], res[n][1], res[n][3]) for n in priors}


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


def rim_votes(perception, depth, centers, v, transform_snapshot=None, diag=None):
    """{bin: xy|None}: per-bin fit of the outer+inner rim rectangles to elevated depth points, +-search_m window."""
    sc, cfg = perception.scene, perception.cfg
    c = cfg.get("self_calibration") or {}
    b = sc.cfg["bins"]
    out = {n: None for n in centers}
    diag = {} if diag is None else diag
    if depth is None or np.ndim(depth) != 2:
        diag["why"] = "no depth"
        return out
    clean_depth = np.nan_to_num(np.asarray(depth, float), nan=0.0)
    if transform_snapshot is None:
        P, coef, why = perception._counter_plane(clean_depth, c)
    else:
        P, coef, why = perception._counter_plane(clean_depth, c, transform_snapshot=transform_snapshot)
    if why:
        diag["why"] = f"plane: {why}"
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
            diag[n] = f"{len(Q)} pts < min {int(v['min_rim_points'])}"
            continue
        Q = Q[np.linspace(0, len(Q) - 1, min(len(Q), 1500)).astype(int)]
        signed = bool(v.get("signed_rim_fit", False))

        def score(off):
            fit = Q - off if signed else Q + off
            d = np.minimum(_box_edge_dist(fit, centers[n], outer), _box_edge_dist(fit, centers[n], inner))
            return float(np.mean(np.minimum(d, trunc) ** 2))

        def search_grid(ctr, span, step):
            vals = np.arange(-span, span + step / 2, step)
            cand = [(score(ctr + [dx, dy]), ctr + [dx, dy]) for dx in vals for dy in vals
                    if max(abs(ctr[0] + dx), abs(ctr[1] + dy)) <= search + 1e-9]
            return min(cand, key=lambda t: t[0])[1]

        off = search_grid(np.zeros(2), search, 0.01)
        off = search_grid(off, 0.01, 0.002)
        off = search_grid(off, 0.002, 0.0005)
        fit = Q - off if signed else Q + off
        d = np.minimum(_box_edge_dist(fit, centers[n], outer), _box_edge_dist(fit, centers[n], inner))
        inl = float(np.mean(d <= 0.006))
        diag[n] = f"{len(Q)} pts, inlier={inl:.2f}, off=({1000 * off[0]:+.0f},{1000 * off[1]:+.0f})mm"
        if inl < 0.35:
            diag[n] += " REJECTED (inlier < 0.35)"
            continue
        out[n] = np.asarray(centers[n], float) + off
    return out


def _blob_vote(Q, elong):
    """Tape blob (pelvis xy points) -> vote xy|None. Round blob: its centroid (median). Elongated (PCA std-dev
    aspect > elong): the inner end, i.e. the end nearer the robot (pelvis origin). Aspect not computable -> None."""
    Q = np.asarray(Q, float)[:, :2]
    Q = Q[np.all(np.isfinite(Q), axis=1)]
    if len(Q) < 3:
        return None
    X = Q - Q.mean(axis=0)
    w, V = np.linalg.eigh(X.T @ X / len(Q))            # ascending eigenvalues
    if not np.all(np.isfinite(w)) or w[1] <= 1e-12:    # all points on one spot: no shape
        return None
    aspect = float(np.sqrt(w[1] / max(w[0], 1e-12)))
    if aspect <= elong:
        return np.median(Q, axis=0)
    t = X @ V[:, 1]
    ends = [Q[t <= np.quantile(t, 0.2)], Q[t >= np.quantile(t, 0.8)]]
    inner = min(ends, key=lambda e: np.linalg.norm(np.median(e, axis=0)))
    return np.median(inner, axis=0)


def _bin_blob_vote(blobs, elong):
    """Blobs (pelvis xy arrays) assigned to one bin -> vote|None. Two comparable blobs on one bin are ambiguous."""
    blobs = sorted(blobs, key=len, reverse=True)
    if not blobs:
        return None
    if len(blobs) > 1 and len(blobs[1]) >= TAPE_AMBIG_RATIO * len(blobs[0]):
        return None
    return _blob_vote(blobs[0], elong)


def tape_votes(perception, rgb, depth, centers, v, transform_snapshot=None, diag=None):
    """{bin: xy|None}: per orange-tape blob near the bin's rim (8-connected in the image), lifted via depth or via the
    rim-height plane where depth is missing; the blob votes its centroid, or its inner end when elongated (see
    _blob_vote). Ambiguous or missing -> None. # ponytail: no per-edge tape layout model; upgrade once the real
    tape layout is known."""
    import cv2
    sc = perception.scene
    out = {n: None for n in centers}
    diag = {} if diag is None else diag
    if rgb is None or np.ndim(rgb) != 3:
        diag["why"] = "no rgb"
        return out
    hsv = cv2.cvtColor(np.ascontiguousarray(rgb), cv2.COLOR_RGB2HSV)
    lo, hi = v["tape_hsv"]
    mask = cv2.inRange(hsv, np.array(lo, np.uint8), np.array(hi, np.uint8))
    mask[int(mask.shape[0] * TAPE_V_MAX_FRAC):] = 0
    vv, uu = np.nonzero(mask)
    if len(vv) < int(v["min_tape_px"]):
        diag["why"] = f"{len(vv)} tape-coloured px < min"
        return out
    labels = cv2.connectedComponents(mask, connectivity=8)[1]
    lab = labels[vv, uu]
    sizes = np.bincount(lab)
    diag["blob_px"] = sorted((int(x) for x in sizes[1:] if x >= int(v["min_tape_px"])), reverse=True)[:8]
    max_px = int(v.get("tape_max_px", 0) or 0) if v.get("tape_body_reject", False) else 0
    if max_px > 0:   # venue: the red-orange paper-wrapped bin body (8-19k px) matches tape_hsv; real tape is a thin strip
        big = sizes[lab] > max_px
        diag["rejected_body_px"] = sorted((int(x) for x in np.unique(sizes[lab[big]])), reverse=True)
        vv, uu, lab = vv[~big], uu[~big], lab[~big]
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
        pts[ok] = (perception._pixel_to_pelvis(uu[ok], vv[ok], d[ok]) if transform_snapshot is None else
                   perception._pixel_to_pelvis(uu[ok], vv[ok], d[ok], transform_snapshot=transform_snapshot))
    if (~ok).any():  # pelvis z is linear in depth along a ray: solve z(d) = rim_z from two samples
        u, w = uu[~ok], vv[~ok]
        if transform_snapshot is None:
            p1 = perception._pixel_to_pelvis(u, w, np.full(len(u), 1000.0))
            p2 = perception._pixel_to_pelvis(u, w, np.full(len(u), 2000.0))
        else:
            p1 = perception._pixel_to_pelvis(u, w, np.full(len(u), 1000.0), transform_snapshot=transform_snapshot)
            p2 = perception._pixel_to_pelvis(u, w, np.full(len(u), 2000.0), transform_snapshot=transform_snapshot)
        dz = p2[:, 2] - p1[:, 2]
        t = np.where(np.abs(dz) > 1e-9, (rim_z - p1[:, 2]) / np.where(np.abs(dz) > 1e-9, dz, 1.0), np.nan)
        pts[~ok] = p1 + (p2 - p1) * t[:, None]
        ok2 = np.isfinite(pts).all(axis=1)
    else:
        ok2 = np.ones(len(vv), bool)
    near_rim = np.abs(pts[:, 2] - rim_z) <= 0.03          # blobs not near rim height are not tape
    keep = ok2 & near_rim
    pts, lab = pts[keep], lab[keep]
    if len(pts) < int(v["min_tape_px"]):
        return out
    outer = np.asarray(sc.cfg["bins"]["outer_size"], float)[::-1] / 2
    search = float(v["search_m"])
    names = list(centers)
    C = np.array([centers[n] for n in names], float)
    per_bin = {n: [] for n in names}
    for l in np.unique(lab):
        B = pts[lab == l][:, :2]
        if len(B) < int(v["min_tape_px"]):
            continue
        med = np.median(B, axis=0)
        k = int(np.argmin(np.linalg.norm(C - med, axis=1)))
        if np.all(np.abs(med - C[k]) <= outer + search):   # blob belongs to its nearest bin's window
            per_bin[names[k]].append(B)
    for n in names:
        out[n] = _bin_blob_vote(per_bin[n], float(v["tape_elong_aspect"]))
    diag["blobs_per_bin"] = {n: [len(b) for b in per_bin[n]] for n in names}
    return out


def _med(frames, n):
    xs = [f[n] for f in frames if f.get(n) is not None]
    return np.median(xs, axis=0) if xs else None


def _fmt(x):
    return "none" if x is None else f"({x[0]:.3f},{x[1]:.3f})"


def _log_voters(log, names, priors, rim, tape, diags):
    """[bin.vote.voters]: every voter's estimate (pelvis xy, m) + distance to the prior + why a voter was missing, so a
    disagreement (conf 0.3) is explained by the next real run's log. Never raises."""
    try:
        for n in names:
            parts = []
            for k, est in (("rim", rim[n]), ("tape", tape[n])):
                dist = "" if est is None else f" d_prior={1000 * float(np.linalg.norm(np.asarray(est) - priors[n])):.0f}mm"
                parts.append(f"{k}={_fmt(est)}{dist}")
            log(f"[bin.vote.voters] {n}: prior={_fmt(priors[n])} " + " ".join(parts))
        rd, td = diags[0]
        log(f"[bin.vote.voters] frame0 rim: {dict(rd)}")
        log(f"[bin.vote.voters] frame0 tape: {dict(td)}")
    except Exception as exc:
        log(f"[bin.vote.voters] log failed ({type(exc).__name__})")


def vote_bins(perception, planner, frames, cfg, log=print):
    """frames: [{"head_rgb", "head_depth"}]. -> (shift, drop_extra, conf). Never raises (prior on any error)."""
    v = cfg["bin"]["vote"]
    names = list(planner.scene.bin_names)
    priors = {n: np.asarray(planner.bin_xy(n), float) for n in names}
    try:
        rims, tapes, diags = [], [], []
        for f in frames:
            rd, td = {}, {}
            diags.append((rd, td))
            if v.get("frame_transform_snapshot", False):
                T = f["transform_snapshot"]
                rims.append(rim_votes(perception, f.get("head_depth"), priors, v, T, diag=rd))
                tapes.append(tape_votes(perception, f.get("head_rgb"), f.get("head_depth"), priors, v, T, diag=td))
            else:
                frame = f.get("transform") or perception.frame_transform()
                with perception.use_frame_transform(frame):
                    rims.append(rim_votes(perception, f.get("head_depth"), priors, v, diag=rd))
                    tapes.append(tape_votes(perception, f.get("head_rgb"), f.get("head_depth"), priors, v, diag=td))
        rim = {n: _med(rims, n) for n in names}
        tape = {n: _med(tapes, n) for n in names}
        shift, extra, info = resolve(priors, rim, tape, v)
        _log_voters(log, names, priors, rim, tape, diags)
    except Exception as exc:  # optional feature: never terminate the control loop
        log(f"[bin.vote] failed ({type(exc).__name__}: {exc}); prior positions")
        return {}, {n: float(v["low_conf_extra_m"]) for n in names}, {n: CONF_PRIOR for n in names}
    log("[bin.vote] " + " ".join(f"{n}: shift=({1000 * shift[n][0]:+.0f},{1000 * shift[n][1]:+.0f})mm "
                                  f"agree={info[n][1]} conf={info[n][2]:.1f} extra={1000 * extra[n]:.0f}mm"
                                  for n in names))
    return shift, extra, {n: info[n][2] for n in names}
