"""Learned pick-success score for policy.learned_order: a plain-JSON logistic model, pure Python, fail-open.

The score only SORTS candidates (policy._ordered_cands). It never picks a bin, never drops a candidate, and never
touches timing. A missing, malformed or non-finite model logs one warning and disables the key (stock order).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

DEFAULT_MODEL = Path(__file__).resolve().parent / "models" / "learned_p.json"


def _sigmoid(x):
    if x >= 0.0:
        return 1.0 / (1.0 + math.exp(-x))
    z = math.exp(x)
    return z / (1.0 + z)


class LearnedSuccessModel:
    def __init__(self, path=None, warn=None):
        self.path = Path(path) if path else DEFAULT_MODEL
        self.warn = warn or (lambda _msg: None)
        self.model = None
        try:
            data = json.loads(self.path.read_text())
            self._validate(data)
            self.model = data
        except Exception as exc:
            self.warn(f"[learned_order] cannot load {self.path} ({exc}); stock pick order")

    @staticmethod
    def _validate(m):
        if m.get("version") != 1:
            raise ValueError(f"unsupported model version {m.get('version')!r}")
        names, means, stds, weights = m["feature_names"], m["means"], m["stds"], m["weights"]
        if not names or len({str(n) for n in names}) != len(names):
            raise ValueError("feature_names must be a non-empty unique list")
        if not len(names) == len(means) == len(stds) == len(weights):
            raise ValueError("feature_names/means/stds/weights length mismatch")
        cal = m.get("calibration") or {"a": 1.0, "b": 0.0}
        vals = [*means, *stds, *weights, m["bias"], cal["a"], cal["b"]]
        if not all(math.isfinite(float(v)) for v in vals) or any(float(s) <= 0.0 for s in stds):
            raise ValueError("non-finite coefficient or non-positive std")

    @staticmethod
    def _raw(name, features):
        if "=" in name:   # one-hot: "shape=cube" is 1 when features["shape"] == "cube"
            key, value = name.split("=", 1)
            return 1.0 if str(features.get(key, "unknown")) == value else 0.0
        value = features.get(name)
        return None if value in (None, "") else float(value)

    def _logit(self, features):
        m = self.model
        total = float(m["bias"])
        for name, mean, std, w in zip(m["feature_names"], m["means"], m["stds"], m["weights"]):
            value = self._raw(str(name), features)
            if value is not None:          # a missing numeric feature is mean-imputed: contributes 0 after scaling
                total += float(w) * (value - float(mean)) / float(std)
        return total

    def predict(self, features):
        """Calibrated success probability clamped to [0.05, 0.98], or None when the model is unavailable."""
        if self.model is None:
            return None
        try:
            cal = self.model.get("calibration") or {"a": 1.0, "b": 0.0}
            p = _sigmoid(float(cal["a"]) * self._logit(features) + float(cal["b"]))
            if not math.isfinite(p):
                raise ValueError("non-finite prediction")
            return min(max(p, 0.05), 0.98)
        except Exception as exc:
            self.warn(f"[learned_order] prediction failed ({exc}); stock pick order")
            self.model = None
            return None
