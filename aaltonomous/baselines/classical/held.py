"""Held-object detection for the Dex3 hands."""
from __future__ import annotations

import math

import numpy as np


class HeldDetector:
    """Legacy max-error detector or the real-hand, bias-corrected finger sum."""

    REFERENCE_S = 1.0
    # Real Dex3 fingers lag a close command by 300--380 ms.  The close sample is
    # therefore no earlier than 0.8 s from close start: lag plus >=0.4 s held.
    SETTLE_S = 0.8
    _FINGERS = {
        # Each hand stays in its native DDS order. Left is middle-before-index;
        # right is index-before-middle (joint_spec.yaml).
        "left": {"middle": (3, 4), "index": (5, 6)},
        "right": {"index": (3, 4), "middle": (5, 6)},
    }
    _CLOSING_SIGN = {"left": -1.0, "right": 1.0}

    def __init__(self, mode="stall", threshold_rad=0.045):
        self.mode = str(mode)
        self.threshold_rad = float(threshold_rad)
        if self.mode not in ("stall", "finger_sum"):
            raise ValueError(f"unknown held detector {self.mode!r}")
        if self.threshold_rad < 0.0:
            raise ValueError("held threshold must be non-negative")
        self.soft = None    # held.soft_detector: {rest, margin, lift_margin, prior} once configure_soft() ran
        self.reset()

    def configure_soft(self, rest, margin, lift_margin, prior_ratio, read_delay_s=0.0):
        """Soft-finger mode (needs finger_sum). `rest` = the 14-vector the fingers close from (preshape). Held then
        means: the fingers stalled further open than an EMPTY close of the same command would leave them."""
        if self.mode != "finger_sum":
            raise ValueError("the soft detector needs finger_sum")
        self.soft = {"rest": np.asarray(rest, float).copy(), "margin": float(margin),
                     "lift_margin": float(lift_margin), "prior": float(prior_ratio),
                     "read_delay_s": float(read_delay_s)}

    def reset(self):
        self._soft_ratio = {}
        self._reference_start = None
        self._reference_sum = np.zeros(14, float)
        self._reference_count = 0
        self._calibration_required = False
        self._calibration_invalid = False
        self._calibration_reason = None
        self._calibration_params = None
        self._calibration_prev = None
        self._calibration_steady_since = None

    def start(self, t):
        if self.mode == "finger_sum" and self._reference_start is None:
            self._reference_start = float(t)

    def observe(self, t, command, measured):
        """Accumulate native-order ``command - measured`` during episode second one."""
        if self.mode != "finger_sum":
            return
        self.start(t)
        if float(t) - self._reference_start > self.REFERENCE_S:
            return
        command, measured = np.asarray(command, float), np.asarray(measured, float)
        self._reference_sum += command - measured
        self._reference_count += 1

    @property
    def ready(self):
        """finger_sum has a valid bias reference; without one the detector is unsure."""
        return (self.mode == "finger_sum" and self._reference_count > 0
                and not self._calibration_invalid)

    @property
    def calibration_reason(self):
        return self._calibration_reason

    def require_empty_close_calibration(self, t, *, steady_samples, max_velocity_rad_s,
                                        hold_after_settle_s, timeout_s):
        """Start M3 calibration at home, immediately before an intentional empty close.

        ``e0`` is never borrowed from home/open tracking.  It is the mean
        command-minus-measured error of a settled empty close, and remains
        unavailable unless every validity and steady-motion gate passes.
        """
        if self.mode != "finger_sum":
            return
        n = int(steady_samples)
        vmax, held, timeout = map(float, (max_velocity_rad_s, hold_after_settle_s, timeout_s))
        if n <= 0 or vmax <= 0.0 or held < 0.4 or timeout <= held:
            self._calibration_invalid = True
            self._calibration_reason = "invalid calibration parameters"
            return
        self._calibration_required = True
        self._reference_start = float(t)
        self._reference_sum[:] = 0.0
        self._reference_count = 0
        self._calibration_params = (n, vmax, held, timeout)
        self._calibration_prev = None
        self._calibration_steady_since = None

    @classmethod
    def _matching_finger_commands(cls, command):
        """The real controller sends identical index/middle commands per hand."""
        for side, fingers in cls._FINGERS.items():
            off = 0 if side == "left" else 7
            a = command[off + np.asarray(fingers["index"])]
            b = command[off + np.asarray(fingers["middle"])]
            if not np.allclose(a, b, atol=1e-6, rtol=0.0):
                return False
        return True

    def sample_empty_close(self, t, command, measured):
        """Collect one steady empty-close sample; return pending/ready/invalid."""
        if self.mode != "finger_sum" or not self._calibration_required:
            return "invalid"
        if self._calibration_invalid:
            return "invalid"
        now = float(t)
        n, vmax, held, timeout = self._calibration_params
        if now - self._reference_start > timeout:
            self._calibration_invalid = True
            self._calibration_reason = "empty-close calibration timed out"
            self._reference_count = 0
            return "invalid"
        try:
            command, measured = np.asarray(command, float), np.asarray(measured, float)
        except (TypeError, ValueError):
            return "pending"
        if command.shape != (14,) or measured.shape != (14,) or not (
                np.all(np.isfinite(command)) and np.all(np.isfinite(measured))):
            self._calibration_prev = None
            self._calibration_steady_since = None
            self._reference_sum[:] = 0.0
            self._reference_count = 0
            return "pending"
        if not self._matching_finger_commands(command):
            self._calibration_invalid = True
            self._calibration_reason = "index/middle close commands differ"
            self._reference_count = 0
            return "invalid"
        prev = self._calibration_prev
        self._calibration_prev = (now, command.copy(), measured.copy())
        if prev is None or now <= prev[0]:
            return "pending"
        dt = now - prev[0]
        cmd_v = float(np.max(np.abs(command - prev[1])) / dt)
        meas_v = float(np.max(np.abs(measured - prev[2])) / dt)
        if cmd_v > vmax or meas_v > vmax:
            self._calibration_steady_since = None
            self._reference_sum[:] = 0.0
            self._reference_count = 0
            return "pending"
        if self._calibration_steady_since is None:
            self._calibration_steady_since = now
            return "pending"
        if now - self._calibration_steady_since < held:
            return "pending"
        self._reference_sum += command - measured
        self._reference_count += 1
        if self._reference_count < n:
            return "pending"
        self._calibration_reason = "steady empty close"
        return "ready"

    @property
    def reference(self):
        if not self._reference_count:
            return np.zeros(14, float)
        return self._reference_sum / self._reference_count

    def evaluation_tick(self, close_start, close_end, hz, hold_start=None):
        """First tick at which this detector may classify the close. Soft mode: at least read_delay_s (lag + settle +
        read-after) after the close ramp ended (`hold_start`), else at the end of the hold."""
        if self.mode == "stall":
            return int(close_end)
        if self.soft is not None and hold_start is not None:
            return max(int(close_end), int(hold_start) + int(math.ceil(self.soft["read_delay_s"] * float(hz))))
        return max(int(close_end), int(close_start) + int(math.ceil(self.SETTLE_S * float(hz))))

    def empty(self, side, command, measured):
        """Open hand: no finger of `side` is still closed past the commanded (open) posture by threshold_rad
        (bias-corrected like held(), but on measured - command, the direction an open command cannot hide)."""
        off = 0 if side == "left" else 7
        native = np.asarray(measured, float) - np.asarray(command, float) + self.reference
        sign = self._CLOSING_SIGN[side]
        return all(sign * native[off + np.asarray(pair)].sum() < self.threshold_rad
                   for pair in self._FINGERS[side].values())

    def _travel(self, side, command):
        """Closing-positive finger travel (sum of the pair) of `command` from the soft rest posture."""
        off = 0 if side == "left" else 7
        d = np.asarray(command, float) - self.soft["rest"]
        sign = self._CLOSING_SIGN[side]
        return {f: float(sign * d[off + np.asarray(pair)].sum()) for f, pair in self._FINGERS[side].items()}

    def soft_calibrate(self, side, samples):
        """Empty-close baseline from N (command, measured) pairs read late in an EMPTY close hold: per finger the
        bias-corrected error as a fraction of the commanded travel. A hand that lags (soft) leaves a constant fraction."""
        if self.soft is None or not samples:
            return False
        ratios = {}
        for finger in self._FINGERS[side]:
            vals = []
            for command, measured in samples:
                _, scores = self._finger_scores(side, command, measured)
                travel = self._travel(side, command)[finger]
                if travel > 1e-6:
                    vals.append(scores[finger] / travel)
            if not vals:
                return False
            ratios[finger] = float(np.clip(np.mean(vals), 0.0, 1.0))
        self._soft_ratio[side] = ratios
        return True

    def soft_ratio(self, side):
        return self._soft_ratio.get(side) or {f: self.soft["prior"] for f in self._FINGERS[side]}

    def _finger_scores(self, side, command, measured):
        off = 0 if side == "left" else 7
        native_err = np.asarray(command, float) - np.asarray(measured, float) - self.reference
        sign = self._CLOSING_SIGN[side]
        scores = {f: float(sign * native_err[off + np.asarray(pair)].sum()) for f, pair in self._FINGERS[side].items()}
        return native_err, scores

    def held(self, side, command, measured, margin=None):
        command, measured = np.asarray(command, float), np.asarray(measured, float)
        if self.soft is not None and self.mode == "finger_sum":
            _, scores = self._finger_scores(side, command, measured)
            travel, ratio = self._travel(side, command), self.soft_ratio(side)
            excess = {f: scores[f] - ratio[f] * max(travel[f], 0.0) for f in scores}
            out = {"index": scores["index"], "middle": scores["middle"], "raw_finger_sum": max(scores.values()),
                   "excess_index": excess["index"], "excess_middle": excess["middle"]}
            out["finger_sum"] = max(excess.values())     # = the stall beyond what an empty (lagging) close leaves
            thr = self.soft["margin"] if margin is None else float(margin)
            return bool(out["finger_sum"] > thr), out

        off = 0 if side == "left" else 7
        sl = slice(off, off + 7)
        if self.mode == "stall":
            # This is intentionally the original expression: the default path must
            # retain the max-|cmd-q| >= 0.010 behaviour exactly.
            err = np.abs(command[sl] - measured[sl])
            return bool(err.max() >= 0.010), {"track_err_max": float(err.max())}

        native_err = command - measured - self.reference
        sign = self._CLOSING_SIGN[side]
        scores = {}
        for finger, pair in self._FINGERS[side].items():
            idx = off + np.asarray(pair)
            scores[finger] = float(sign * native_err[idx].sum())
        scores["finger_sum"] = max(scores["index"], scores["middle"])
        return bool(scores["finger_sum"] >= self.threshold_rad), scores

    def held_max(self, side, samples, margin=None):
        """Classify N ``(command, measured)`` samples by their maximum grip signal."""
        results = [self.held(side, command, measured, margin) if self.soft is not None
                   else self.held(side, command, measured) for command, measured in samples]
        if not results:
            raise ValueError("held_max needs at least one sample")
        key = "finger_sum" if self.mode == "finger_sum" else "track_err_max"
        return max(results, key=lambda item: item[1][key])
