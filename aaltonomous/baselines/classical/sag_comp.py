"""motion.sag_comp: adaptive position-offset gravity-sag compensation for the REAL arm PD.

The organiser runner's arm PD (kp 80 shoulders/elbow, 40 wrists, target clipped to measured +-0.12 rad) sags by
tau_g(q)/kp when it does not run with --arm-gravity-ff (and ignores our arm_tau_ff). We command
    q_cmd = q_plan + kappa * tau_g(q_plan) / kp
with kappa estimated online from STILL ticks: at rest the PD balances gravity, cmd - q_meas = tau_g(q)/kp, independent
of what offset we send, so the regression ratio of (cmd - q_meas) on tau_g(q_meas)/kp is kappa.
  * runner already compensates (ff ON): the error is ~0 -> kappa = 0 -> the offset is exactly 0.0 and the output is
    bit-identical to the flag-off stream (until a still window proves otherwise kappa is 0 anyway).
  * runner does not: ratio ~ 1 (measured 0.9-1.1) -> kappa 1.0 on the first confirmation, EMA afterwards, clamp [0, 1.2].
The offset is slew-limited (<= 0.5 rad/s), bounded (< the runner's 0.12 rad clip) and is added BEFORE the stateful
output limiter, so the per-tick change of the emitted command stays inside motion.output_caps.
"""
from __future__ import annotations

from collections import deque

import numpy as np

KP = np.array([80.0, 80.0, 80.0, 80.0, 40.0, 40.0, 40.0] * 2)   # organiser runner arm PD
KAPPA_MAX = 1.2
OFF_MAX = 0.09              # rad, < runner target clip 0.12 (leaves headroom for the tracking error itself)
SLEW = 0.5                  # rad/s offset change
STILL_Q = 0.004             # rad, measured ptp over the window
STILL_CMD = 0.003           # rad, commanded ptp over the window
STILL_S = 0.5               # s, window span
MIN_SIGNAL = 0.012          # rad: sqrt(sum pred^2) needed for a ratio to mean anything
CONFIRM_MIN = 0.35          # first-estimate ratio that confirms "runner does not compensate" -> kappa prior 1.0
EMA = 0.4


class SagComp:
    def __init__(self, gff, dt=1.0 / 30.0, log=None):
        self.gff = gff
        self.dt = float(dt)
        self.log = log or (lambda m: None)
        self.reset()

    def reset(self):
        self.kappa = 0.0
        self.estimated = False
        self.off = np.zeros(14)
        self._hist = deque(maxlen=64)   # (t, q_meas, cmd_prev, pred)
        self._last_cmd = None
        self._k_first = []

    def soft_reset(self):
        self.off = np.zeros(14)
        self._hist.clear()
        self._last_cmd = None

    def _g(self, q, h=None):
        return self.gff.raw(q, h)

    def target_offset(self, q_plan, h=None):
        """kappa-free per-joint sag q-offset for one planned pose (rad), bounded."""
        return np.clip(self._g(q_plan, h) / KP, -OFF_MAX, OFF_MAX)

    # ------------------------------------------------------------------ estimation
    def observe(self, t, q_meas, h_meas=None):
        """Call once per act() with the measured encoders, BEFORE process(): cmd_prev = last emitted row."""
        q = np.asarray(q_meas, float).reshape(-1)
        if q.shape != (14,) or not np.all(np.isfinite(q)) or self._last_cmd is None:
            return
        h = None
        if h_meas is not None:
            hh = np.asarray(h_meas, float).reshape(-1)
            h = hh if hh.shape == (14,) and np.all(np.isfinite(hh)) else None
        self._hist.append((float(t), q.copy(), self._last_cmd.copy(), self._g(q, h) / KP))
        self._maybe_update()

    def _maybe_update(self):
        if len(self._hist) < 4:
            return
        ts = np.array([x[0] for x in self._hist])
        sel = np.where(ts >= ts[-1] - STILL_S)[0]
        if len(sel) < 4 or ts[-1] - ts[sel[0]] < STILL_S * 0.8:
            return
        Q = np.array([self._hist[i][1] for i in sel]); C = np.array([self._hist[i][2] for i in sel])
        if np.ptp(Q, axis=0).max() > STILL_Q or np.ptp(C, axis=0).max() > STILL_CMD:
            return
        e = (C - Q).mean(0)
        p = np.mean([self._hist[i][3] for i in sel], axis=0)
        den = float(p @ p)
        if np.sqrt(den) < MIN_SIGNAL:
            return
        k = float(np.clip(e @ p / den, 0.0, KAPPA_MAX)) if (e @ p) > 0 else 0.0
        self._hist.clear()   # one sample per fresh still window
        if not self.estimated:
            self._k_first.append(k)
            if len(self._k_first) < 2:
                return
            k = float(np.mean(self._k_first))
            # sag sign + size confirmed: the measured 0.8-1.2 (organiser runs: 0.9-1.1) snaps to the prior 1.0, else the ratio
            self.kappa = 0.0 if k < CONFIRM_MIN else (1.0 if 0.8 <= k <= KAPPA_MAX else k)
            self.estimated = True
            self.log(f"[sag_comp] first estimate ratio {k:.2f} -> kappa {self.kappa:.2f}")
            return
        kn = (1 - EMA) * self.kappa + EMA * k
        self.kappa = 0.0 if kn < 0.1 else float(np.clip(kn, 0.0, KAPPA_MAX))

    # ------------------------------------------------------------------ command
    def process(self, arm, hand=None):
        """(n,14) planned rows -> (n,14) compensated rows (offset slew-limited). Exactly the input while kappa == 0
        and the offset has decayed to 0."""
        arm = np.asarray(arm, float)
        if arm.ndim != 2 or arm.shape[1] != 14 or not len(arm):
            return arm
        if self.kappa == 0.0 and not self.off.any():
            self._last_cmd = arm[-1].copy()
            return arm
        out = arm.copy()
        step = SLEW * self.dt
        for i in range(len(arm)):
            h = None if hand is None or np.asarray(hand).ndim != 2 or len(hand) != len(arm) else hand[i]
            tgt = self.kappa * self.target_offset(arm[i], h)
            self.off = self.off + np.clip(tgt - self.off, -step, step)
            out[i] = arm[i] + self.off
        self.off[np.abs(self.off) < 1e-12] = 0.0
        self._last_cmd = out[-1].copy()
        return out

    def note_emitted(self, last_row):
        """The final (limiter-capped, joint-clipped) command actually sent: this is what the PD sees."""
        self._last_cmd = np.asarray(last_row, float).copy()
