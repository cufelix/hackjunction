"""Decision layer (D1-D4): pure functions + small state objects, no I/O, no perception/planner imports.

Everything here is only reached from policy.py behind four flags (all default off):
  policy.confidence_order   D1/D2  priority = 2*p/est_time, cube > cylinder > prism, never the hardest first
  policy.object_time_cap    D2     per-object cap, T-20 s no new object, T-8 s release over the bin / retract
  policy.speed_tiers        D3     first object 0.7 + clearance, clean -> 0.85/0.9, failure -> 0.6 + centre bin
  policy.confidence_tiers   D4     hold still, calibration confidence C -> CONFIDENT / DEGRADED / PRIOR-ONLY
  decision.isolated_first   --     candidates within isolated_min_m of another detection (touching) go after the isolated ones
Numbers live in config.yaml `decision:` (architecture v4 sections 2 and 3). Tested with synthetic lists / clocks in
test_decision.py.
"""
from __future__ import annotations

import numpy as np

CONFIDENT, DEGRADED, PRIOR_ONLY = "CONFIDENT", "DEGRADED", "PRIOR-ONLY"


def _clip(x, lo, hi):
    return float(min(max(x, lo), hi))


# ---------------------------------------------------------------------------------------------------- D1 confidence
def hue_margin_deg(ranges, h):
    """Distance (degrees, OpenCV H 0..180 -> x2) of hue `h` from the nearest edge of the colour's hue ranges
    [[[hlo, ., .], [hhi, ., .]], ...]; 0 when outside. Ranges touching 0 and 180 (red) form one circular interval."""
    iv = sorted((float(lo[0]), float(hi[0])) for lo, hi in ranges)
    if len(iv) > 1 and iv[0][0] <= 0.0 and iv[-1][1] >= 180.0:      # wrap: [170,180] + [0,16] -> [170, 196]
        first, last = iv[0], iv[-1]
        iv = iv[1:-1] + [(last[0], first[1] + 180.0)]
    best = 0.0
    for x in (float(h), float(h) + 180.0):
        for lo, hi in iv:
            if lo <= x <= hi:
                best = max(best, min(x - lo, hi - x))
    return 2.0 * best


def p_colour(n_px, hue_margin=None, stability=None, d=None):
    """p_colour in (0, 1]: hue margin (>= 15 deg full), blob area (>= 60 px usable, ~300 px full), temporal stability
    (fraction of the last frames that saw this object). Unknown inputs (None) are neutral (1.0)."""
    d = d or DEFAULTS
    ph = 1.0 if hue_margin is None else 0.3 + 0.7 * _clip(hue_margin / d["hue_full_margin_deg"], 0.0, 1.0)
    lo, full = float(d["area_min_px"]), float(d["area_full_px"])
    pa = 0.3 if n_px < lo else 0.6 + 0.4 * _clip((n_px - lo) / max(full - lo, 1.0), 0.0, 1.0)
    ps = 1.0 if stability is None else 0.7 + 0.3 * _clip(stability, 0.0, 1.0)
    return ph * pa * ps


def p_bin(bin_is_centre, C=None, d=None):
    """p_bin: the centre bin is the prior (1.0); a side bin trusts the calibration: C >= 0.6 full, 0.4 + C below.
    C None (not measured) = 1.0."""
    if C is None or bin_is_centre:
        return 1.0
    return _clip(0.4 + C, 0.0, 1.0)


def ik_margin(xy, bin_y, d=None):
    """Cheap IK-margin proxy (no IK per object, it costs 0.3-6 s): 1 within the comfortable reach, falling to 0.5 at the
    reach limit, x cross_body_factor when the object and the target bin are on opposite sides (relocation needed)."""
    d = d or DEFAULTS
    r = float(np.hypot(xy[0], xy[1]))
    lo, hi = float(d["reach_comfort_m"]), float(d["reach_max_m"])
    m = 1.0 - 0.5 * _clip((r - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if abs(bin_y) > 0.1 and abs(xy[1]) > 0.05 and bin_y * xy[1] < 0:
        m *= float(d["cross_body_factor"])
    return m


def p_grasp(shape, xy, bin_y, depth_cov=1.0, d=None):
    """p_grasp = shape prior (cube 0.9 / cylinder 0.8 / prism 0.6, unknown 0.75) x IK margin x depth coverage."""
    d = d or DEFAULTS
    pri = d["shape_prior"]
    prior = float(pri.get(shape if shape is not None else "unknown", pri["unknown"]))
    return prior * ik_margin(xy, bin_y, d) * _clip(0.5 + 0.5 * depth_cov, 0.5, 1.0)


def est_time(xy, bin_xy, d=None):
    """Cheap pick+place duration estimate (s): base + per-metre of the object -> bin distance."""
    d = d or DEFAULTS
    return float(d["est_base_s"]) + float(d["est_per_m"]) * float(np.hypot(xy[0] - bin_xy[0], xy[1] - bin_xy[1]))


def priority(p, t_est):
    """D2 order key (higher = earlier): 2 * p / est_time."""
    return 2.0 * p / max(t_est, 1e-6)


def rank(items, d=None):
    """items: dicts with keys p, t (est s), shape, centre_first (bool, DEGRADED). Returns them best first.
    Order: priority 2p/t descending; shape ties cube < cylinder < prism; then 'never the hardest first': when there is
    a strictly hardest item (lowest p) and another item exists, the hardest never leads."""
    d = d or DEFAULTS
    cost = {"cube": 0, "cylinder": 1, "triangle": 2}
    out = sorted(items, key=lambda it: (not it.get("centre_first", False),      # DEGRADED: centre bin first
                                        -round(priority(it["p"], it["t"]), 9), cost.get(it["shape"], 1)))
    # the swap stays inside the leading centre_first group: it never moves a side object ahead of a centre one
    grp = [it for it in out if bool(it.get("centre_first", False)) == bool(out[0].get("centre_first", False))] if out else []
    if len(grp) > 1:
        lowest = min(it["p"] for it in grp)
        hardest = [it for it in grp if it["p"] <= lowest + 1e-9]
        if len(hardest) == 1 and out[0] is hardest[0] and lowest < max(it["p"] for it in grp) - 1e-9:
            out[0], out[1] = out[1], out[0]
    return out


# ---------------------------------------------------------------------------------------------------- D2 clocks
def object_cap(remaining, n_left, d=None):
    """Per-object time cap (s) = (remaining - 15) / objects left, clamped to 15..35."""
    d = d or DEFAULTS
    n = max(int(n_left), 1)
    return _clip((float(remaining) - float(d["cap_reserve_s"])) / n, float(d["cap_min_s"]), float(d["cap_max_s"]))


def no_new_object(t, time_limit, d=None):
    """True from T-20 s on: nothing new is started."""
    return float(t) >= float(time_limit) - float((d or DEFAULTS)["no_new_s"])


def endgame(t, time_limit, d=None):
    """True from T-8 s on (release over the bin / retract)."""
    return float(t) >= float(time_limit) - float((d or DEFAULTS)["release_s"])


def phase_at(phases, tick):
    """Name of the plan phase containing `tick` ((name, start, end) list), or None past the end."""
    for name, s, e in phases:
        if s <= tick < e:
            return name
    return None


def endgame_action(phase, over_target_bin):
    """T-8 s rule. 'release' = open the hand where it is (only while it carries and is over the target bin),
    'retract' = abandon the plan and go home, None = nothing to do (already releasing / retreating / home)."""
    if phase in (None, "open", "retreat", "home"):
        return None
    if phase == "carry" and over_target_bin:
        return "release"
    return "retract"


# ---------------------------------------------------------------------------------------------------- D3 speed tiers
class SpeedTiers:
    """first object 0.7 (+1.5 cm approach clearance); after a clean success 0.85 (then 0.9 from the second clean one);
    after a failure 0.6 and centre-bin objects only until the next clean success. Never above 1.0 (stock speed)."""

    def __init__(self, d=None):
        self.d = d or DEFAULTS
        self.n_clean = 0
        self.failed = False
        self.n_done = 0            # outcomes observed

    def scale(self):
        t = self.d
        if self.failed:
            return float(t["fail_scale"])
        if self.n_done == 0:
            return float(t["first_scale"])
        lo, hi = t["clean_scale"]
        return float(lo if self.n_clean < 2 else hi)

    def clearance(self):
        return float(self.d["first_clearance_m"]) if self.n_done == 0 and not self.failed else 0.0

    def centre_only(self, has_centre=True):
        """Centre-bin only after a failure, but only while a centre-bin candidate exists (else: no deadlock)."""
        return self.failed and bool(has_centre)

    def observe(self, clean):
        self.n_done += 1
        if clean:
            self.n_clean += 1
            self.failed = False
        else:
            self.failed = True
            self.n_clean = 0


# ---------------------------------------------------------------------------------------------------- D4 tiers
def calib_confidence(pitch=None, bin_agree=None, depth_valid_frac=None, d=None):
    """Calibration confidence C in 0..1 from what exists; missing components (None) drop out and the weights are
    renormalised. pitch: online pitch-cal result (0..1), bin_agree: depth bin-wall agreement with the prior (0..1),
    depth_valid_frac: fraction of valid depth pixels (0.2 -> 0, 0.6+ -> 1). None if nothing is known."""
    d = d or DEFAULTS
    w = d["c_weights"]
    parts = {}
    if pitch is not None:
        parts["pitch"] = _clip(pitch, 0.0, 1.0)
    if bin_agree is not None:
        parts["bin"] = _clip(bin_agree, 0.0, 1.0)
    if depth_valid_frac is not None:
        lo, hi = float(d["depth_lo"]), float(d["depth_hi"])
        parts["depth"] = _clip((depth_valid_frac - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    if not parts:
        return None
    tot = sum(float(w[k]) for k in parts)
    return sum(float(w[k]) * v for k, v in parts.items()) / tot


def tier_of(C, d=None):
    """CONFIDENT C >= 0.6; DEGRADED 0.3 <= C < 0.6; PRIOR-ONLY C < 0.3 (C None = nothing measurable = PRIOR-ONLY)."""
    d = d or DEFAULTS
    if C is None or C < float(d["degraded_c"]):
        return PRIOR_ONLY
    return CONFIDENT if C >= float(d["confident_c"]) else DEGRADED


# ---------------------------------------------------------------------------------------------------- isolated first
def defer_touching(cands, dets, min_m):
    """decision.isolated_first: a candidate whose xy is closer than min_m to any OTHER detection (touching, pushed
    together) moves after every isolated candidate. Each group keeps its order; nothing is dropped. dets = all current
    detections (not only candidates)."""
    def touching(d):
        xy = np.asarray(d.top[:2], float)
        return any(o is not d and float(np.hypot(*(xy - np.asarray(o.top[:2], float)))) < float(min_m) for o in dets)
    flags = [touching(d) for d in cands]
    return [d for d, t in zip(cands, flags) if not t] + [d for d, t in zip(cands, flags) if t]


# defaults mirror config.yaml `decision:` (used when a caller passes no cfg; load_config always supplies it)
DEFAULTS = {
    "shape_prior": {"cube": 0.9, "cylinder": 0.8, "triangle": 0.6, "unknown": 0.75},
    "hue_full_margin_deg": 15.0, "area_min_px": 60, "area_full_px": 300,
    "est_base_s": 10.0, "est_per_m": 20.0,
    "reach_comfort_m": 0.35, "reach_max_m": 0.45, "cross_body_factor": 0.75,
    "stab_frames": 5,
    "cap_reserve_s": 15.0, "cap_min_s": 15.0, "cap_max_s": 35.0, "no_new_s": 20.0, "release_s": 8.0,
    "first_scale": 0.7, "first_clearance_m": 0.015, "clean_scale": [0.85, 0.9], "fail_scale": 0.6,
    "hold_s": 4.0, "hold_frames": 4, "confident_c": 0.6, "degraded_c": 0.3,
    "degraded_scale": 0.6, "degraded_min_p": 0.85,
    "c_weights": {"pitch": 0.4, "bin": 0.35, "depth": 0.25}, "depth_lo": 0.2, "depth_hi": 0.6,
    "isolated_min_m": 0.06,
}
