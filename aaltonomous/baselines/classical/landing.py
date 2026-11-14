"""policy.verify_landing (default OFF): classify where a released object ended up, from fresh head frames.

Pure helpers, no policy state. The old policy.verify_placement (#6) judged a release by colour blobs in the bin and
stopped the episode on any unplannable correction (Modal A/B 65/124 vs 97/111). This one only OBSERVES and
re-queues:

  ok          the target bin gained volume (depth occupancy: works for a purple object in a shadowed bin) or a
              same-colour blob sits in the target bin (colour fallback when depth is inconclusive)
  wrong_bin   another bin gained volume: -1, logged (the object is rarely reachable inside a bin)
  bounce      a same-colour object appeared on the counter away from where it was picked -> pick it again
  grasp_failed  the object is still where it was picked (normal attempt accounting)
  lost        nothing anywhere: fell off the table / not visible -> logged, forgotten
  unknown     the target bin was not seen -> no action

Occupancy is the summed height above the bin's own floor (25th percentile of the interior points, so a constant
head-camera pitch/height error cancels) of every depth point inside the runtime inner footprint, shrunk so the
walls never count. Frames are compared before / after the release, so objects already in the bin do not matter.
"""
from __future__ import annotations

import numpy as np

PARAMS = {"stride": 2, "shrink_m": 0.015, "min_h_m": 0.02, "h_cap_m": 0.12, "floor_pct": 25.0, "min_points": 40}


def bin_occupancy(perception, depth, stride=None):
    """{bin: summed occupied height (m x sampled px)} over the runtime inner footprints; None = bin not seen."""
    p = PARAMS
    stride = int(stride or p["stride"])
    out = {name: None for name in perception.scene.bin_names}
    if depth is None:
        return out
    d = np.asarray(depth)[::stride, ::stride].astype(float)
    dlo, dhi = perception.cfg["depth_range_mm"]
    vv, uu = np.nonzero((d >= dlo) & (d <= dhi))
    if len(vv) == 0:
        return out
    P = perception._pixel_to_pelvis(uu * stride, vv * stride, d[vv, uu])
    sc = perception.scene
    pm = sc.pelvis_to_mat(P)
    zrel = P[:, 2] - sc.counter_z
    for name, g in perception.bin_geometry().items():
        lo, hi = g["inner_lo"][:2] + p["shrink_m"], g["inner_hi"][:2] - p["shrink_m"]
        sel = np.all((pm[:, :2] >= lo) & (pm[:, :2] <= hi), axis=1) & (zrel < 0.30)
        if int(sel.sum()) < p["min_points"]:
            continue
        z = zrel[sel]
        floor = np.percentile(z, p["floor_pct"])
        out[name] = float(np.clip(z - floor - p["min_h_m"], 0.0, p["h_cap_m"]).sum())
    return out


def classify_landing(pending, occ_prev, occ_now, counter_dets, min_delta, colour_bin=None, near_m=0.05):
    """-> (kind, info). pending: {color, bin, src_xy, pre_xy}. counter_dets: counter detections of this look."""
    target = pending["bin"]
    delta = {}
    for name, now in (occ_now or {}).items():
        before = (occ_prev or {}).get(name)
        if now is not None and before is not None:
            delta[name] = now - before
    info = {"delta": delta}
    if delta.get(target, -1.0) >= min_delta or colour_bin == target:
        return "ok", info
    others = {n: v for n, v in delta.items() if n != target and v >= min_delta}
    if others:
        info["bin"] = max(others, key=others.get)
        return "wrong_bin", info
    same = [d for d in counter_dets if d.color == pending["color"]]
    pre = [np.asarray(x, float) for x in pending.get("pre_xy", [])]
    new = [d for d in same if all(np.linalg.norm(d.top[:2] - x) > near_m for x in pre)]
    if new:
        info["det"] = new[0]
        return "bounce", info
    src = np.asarray(pending["src_xy"], float)
    if any(np.linalg.norm(d.top[:2] - src) <= near_m for d in same):
        return "grasp_failed", info
    if delta.get(target) is None:
        return "unknown", info
    return "lost", info


def inside_bin(perception, xy, margin_m=0.0):
    """Name of the bin whose runtime OUTER footprint (+margin) contains the pelvis-frame xy, else None."""
    pm = perception.scene.pelvis_to_mat(np.r_[np.asarray(xy, float)[:2], 0.0])[:2]
    for name, g in perception.bin_geometry().items():
        if np.all(pm >= g["outer_lo"] - margin_m) and np.all(pm <= g["outer_hi"] + margin_m):
            return name
    return None


def jitter(n, radius_m, xy, toward=(0.30, 0.0)):
    """Deterministic retry jitter (pelvis xy, m): radius_m towards `toward` (the middle of the easy reach, away from
    the bins), rotated 0 / +60 / -60 deg on the 1st / 2nd / 3rd bounce of a colour. Always inward-ish, so a jittered
    point never moves towards a bin rim the planner already struggled with."""
    v = np.asarray(toward, float) - np.asarray(xy, float)[:2]
    n_ = np.linalg.norm(v)
    base = np.arctan2(v[1], v[0]) if n_ > 1e-6 else np.pi
    a = base + np.deg2rad([0.0, 60.0, -60.0][int(n) % 3])
    return float(radius_m) * np.array([np.cos(a), np.sin(a)])
