"""Tiny pure-Python inference for the optional learned pick-success model.

The training tool writes a deliberately small JSON model.  This module keeps
runtime inference dependency-free (apart from the policy's existing Python
runtime) and fail-open: a missing, malformed, or incompatible model leaves the
hand-written prior unchanged.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "learned_p.json"


def _sigmoid(x):
    if x >= 0.0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


class LearnedSuccessModel:
    """Load once, then predict from a feature dict in well under a millisecond."""

    def __init__(self, path=None, weight=0.5, warn=None):
        self.path = Path(path).expanduser() if path else DEFAULT_MODEL
        self.weight = min(max(float(weight), 0.0), 1.0)
        self.warn = warn or (lambda _msg: None)
        self._warned = False
        self.model = None
        try:
            data = json.loads(self.path.read_text())
            self._validate(data)
            self.model = data
        except Exception as exc:
            self._warning(f"cannot load {self.path}: {exc}")

    def _warning(self, message):
        if not self._warned:
            self._warned = True
            self.warn(f"[learned_p] {message}; using hand-set prior")

    @staticmethod
    def _validate(model):
        if model.get("version") != 1:
            raise ValueError(f"unsupported model version {model.get('version')!r}")
        names = model["feature_names"]
        means, stds, weights = model["means"], model["stds"], model["weights"]
        if not isinstance(names, list) or not names or len({str(n) for n in names}) != len(names):
            raise ValueError("feature_names must be a non-empty unique list")
        if not (len(names) == len(means) == len(stds) == len(weights)):
            raise ValueError("feature_names/means/stds/weights length mismatch")
        values = [*means, *stds, *weights, model["bias"]]
        if not all(math.isfinite(float(v)) for v in values) or any(float(s) <= 0.0 for s in stds):
            raise ValueError("non-finite coefficient or non-positive std")
        cal = model.get("calibration") or {"a": 1.0, "b": 0.0}
        if not all(math.isfinite(float(cal[k])) for k in ("a", "b")):
            raise ValueError("invalid calibration")
        duration = model.get("duration")
        if duration is not None:
            if len(duration.get("weights", ())) != len(names):
                raise ValueError("duration weight length mismatch")
            if not all(math.isfinite(float(v)) for v in [*duration["weights"], duration["bias"]]):
                raise ValueError("invalid duration model")

    @staticmethod
    def _raw_value(name, features):
        if "=" in name:
            key, value = name.split("=", 1)
            return 1.0 if str(features.get(key, "unknown")) == value else 0.0
        value = features.get(name)
        if value in (None, ""):
            return None
        return float(value)

    def _vector(self, features):
        out = []
        for name, mean, std in zip(self.model["feature_names"], self.model["means"], self.model["stds"]):
            value = self._raw_value(str(name), features)
            # Missing numeric values are mean-imputed, hence zero after scaling.
            out.append(0.0 if value is None else (value - float(mean)) / float(std))
        return out

    def predict(self, features):
        """Calibrated model probability, clamped to the operational range."""
        if self.model is None:
            raise RuntimeError("model unavailable")
        x = self._vector(features)
        logit = float(self.model["bias"]) + sum(float(w) * v for w, v in zip(self.model["weights"], x))
        cal = self.model.get("calibration") or {"a": 1.0, "b": 0.0}
        p = _sigmoid(float(cal["a"]) * logit + float(cal["b"]))
        if not math.isfinite(p):
            raise ValueError("non-finite prediction")
        return min(max(p, 0.05), 0.98)

    def probability(self, features, prior):
        """Blend calibrated model probability with the hand-set prior, fail-open."""
        if self.model is None:
            return float(prior)
        try:
            p_model = self.predict(features)
            return self.weight * p_model + (1.0 - self.weight) * float(prior)
        except Exception as exc:
            self._warning(f"prediction failed: {exc}")
            self.model = None
            return float(prior)

    def duration(self, features, prior):
        """Expected seconds from the optional linear head, or the prior estimate."""
        if self.model is None or not self.model.get("duration"):
            return float(prior)
        try:
            x = self._vector(features)
            head = self.model["duration"]
            seconds = float(head["bias"]) + sum(float(w) * v for w, v in zip(head["weights"], x))
            if not math.isfinite(seconds):
                raise ValueError("non-finite duration")
            return min(max(seconds, 1.0), 120.0)
        except Exception as exc:
            self._warning(f"duration prediction failed: {exc}")
            self.model = None
            return float(prior)
