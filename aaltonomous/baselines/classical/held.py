"""Held-object detection for the Dex3 hands."""
from __future__ import annotations

import math

import numpy as np


class HeldDetector:
    """Legacy max-error detector or the real-hand, bias-corrected finger sum."""

    REFERENCE_S = 1.0
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
        """finger_sum has a bias reference from episode second one; without it the detector is unsure."""
        return self.mode == "finger_sum" and self._reference_count > 0

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
