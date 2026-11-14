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
        self.reset()

    def reset(self):
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

    def evaluation_tick(self, close_start, close_end, hz):
        """First tick at which this detector may classify the close."""
        if self.mode == "stall":
            return int(close_end)
        return max(int(close_end), int(close_start) + int(math.ceil(self.SETTLE_S * float(hz))))

    def empty(self, side, command, measured):
        """Open hand: no finger of `side` is still closed past the commanded (open) posture by threshold_rad
        (bias-corrected like held(), but on measured - command, the direction an open command cannot hide)."""
        off = 0 if side == "left" else 7
        native = np.asarray(measured, float) - np.asarray(command, float) + self.reference
        sign = self._CLOSING_SIGN[side]
        return all(sign * native[off + np.asarray(pair)].sum() < self.threshold_rad
                   for pair in self._FINGERS[side].values())

    def held(self, side, command, measured):
        command, measured = np.asarray(command, float), np.asarray(measured, float)
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
