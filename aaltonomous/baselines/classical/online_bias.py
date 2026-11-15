"""grasp.online_bias (FLAG, default OFF): the robot learns its own per-hand grasp xy bias from its misses.

State per hand: b (2-D, pelvis frame) = the grasp-point offset ADDED to the detection (it starts at the configured
grasp.arm_xy_offset, or 0, and replaces it in the planner). Only the grasp point is ever biased, never release/drop.

Update rule (pure, no I/O, so it is unit-simulated):
  success            keep b; n_succ += 1 (alpha = alpha0 / (1 + n_succ / ALPHA_DECAY_N), floor ALPHA_MIN); the
                     air-probe streak ends.
  miss, object moved (|delta| >= MIN_MOVE_M; delta = re-detected - pre-grasp position)
                     closing fingers push the object toward the grasp centre, so delta points from the object toward
                     where the fingers were: b -= alpha * delta_hat * STEP_M.   (air-probe streak ends)
  miss, object did not move (closed on air)
                     probe along the closing axis (pelvis y): positions streak_base + k * s0 * STEP_M with
                     k = 1, 2, -1, 3, -2, 4, -3, -4 on consecutive misses (prior side first, sign then alternates,
                     magnitude grows); s0 is the first-probe sign per hand (left -1, right +1, as in the lab notes
                     and the sim sweep). Starting from a wrong-side offset still reaches the true one in <= 6 misses.
Clamps: |b| <= CLAMP_M, and per episode |b - b_at_episode_start| <= EPISODE_CAP_M. State lives in the policy
instance and survives reset(); only begin_episode() re-bases the per-episode cap.
"""
import numpy as np

STEP_M = 0.01
CLAMP_M = 0.04
EPISODE_CAP_M = 0.03
MIN_MOVE_M = 0.005
ALPHA0 = 1.0
ALPHA_DECAY_N = 4.0
ALPHA_MIN = 0.25
PROBE_K = (1, 2, -1, 3, -2, 4, -3, -4)   # prior side first, then zig-zag back (sign alternates after the 2nd)
PROBE_AXIS = (0.0, 1.0)
PROBE_SIGN0 = {"left": -1.0, "right": 1.0}


def _clip_norm(v, lim):
    n = float(np.linalg.norm(v))
    return v if n <= lim or n == 0.0 else v * (lim / n)


class OnlineBias:
    HANDS = ("left", "right")

    def __init__(self, init=None, log=print):
        init = init or {}
        self.log = log
        self.b = {h: np.asarray(init.get(h, (0.0, 0.0)), float).copy() for h in self.HANDS}
        self.n_succ = {h: 0 for h in self.HANDS}
        self.n_miss = {h: 0 for h in self.HANDS}
        self.streak = {h: 0 for h in self.HANDS}              # consecutive air-misses
        self.base = {h: self.b[h].copy() for h in self.HANDS}  # streak start bias
        self.ep0 = {h: self.b[h].copy() for h in self.HANDS}   # bias at episode start

    def get(self, hand):
        return self.b[hand].copy()

    def begin_episode(self):
        """reset(): keep everything learned, only re-base the per-episode change cap."""
        self.ep0 = {h: self.b[h].copy() for h in self.HANDS}

    def alpha(self, hand):
        return max(ALPHA_MIN, ALPHA0 / (1.0 + self.n_succ[hand] / ALPHA_DECAY_N))

    def _set(self, hand, new):
        new = np.asarray(new, float)
        new = self.ep0[hand] + _clip_norm(new - self.ep0[hand], EPISODE_CAP_M)
        self.b[hand] = _clip_norm(new, CLAMP_M)

    def _emit(self, hand, reason):
        b = self.b[hand]
        self.log(f"[online_bias] hand={hand} b=({b[0]:+.3f},{b[1]:+.3f}) reason={reason}")

    def success(self, hand):
        self.n_succ[hand] += 1
        self.streak[hand] = 0
        self._emit(hand, f"success n_succ={self.n_succ[hand]} alpha={self.alpha(hand):.2f}")

    def miss(self, hand, delta=None):
        """delta: re-detected minus pre-grasp object xy (2-vector), or None (unknown: treated as did-not-move)."""
        self.n_miss[hand] += 1
        d = None if delta is None else np.asarray(delta, float)[:2]
        if d is not None and np.all(np.isfinite(d)) and np.linalg.norm(d) >= MIN_MOVE_M:
            self.streak[hand] = 0
            step = self.alpha(hand) * STEP_M * d / np.linalg.norm(d)
            self._set(hand, self.b[hand] - step)
            self._emit(hand, f"miss_moved delta=({d[0]:+.3f},{d[1]:+.3f})")
            return
        if self.streak[hand] == 0:
            self.base[hand] = self.b[hand].copy()
        k = PROBE_K[self.streak[hand] % len(PROBE_K)]
        self.streak[hand] += 1
        axis = np.asarray(PROBE_AXIS, float)
        self._set(hand, self.base[hand] + k * PROBE_SIGN0[hand] * STEP_M * axis)
        self._emit(hand, f"miss_air probe k={k:+d} streak={self.streak[hand]}")
