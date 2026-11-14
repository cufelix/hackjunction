"""Decision-AI learned probability integration tests (CPU only)."""
import json
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

from . import decision as dec
from .config import load_config
from .learned_p import LearnedSuccessModel
from .test_decision import det, obs
from .policy import ClassicalPolicy


def write_model(path, red_weight=5.0, blue_weight=-5.0):
    model = {
        "version": 1,
        "feature_names": ["colour=red", "colour=blue"], "means": [0.0, 0.0], "stds": [1.0, 1.0],
        "weights": [red_weight, blue_weight], "bias": 0.0, "calibration": {"a": 1.0, "b": 0.0},
        "duration": {"weights": [0.0, 0.0], "bias": 10.0},
    }
    path.write_text(json.dumps(model))


class TestLearnedSuccessModel(unittest.TestCase):
    def test_missing_and_corrupt_model_fail_open_with_one_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            warnings = []
            missing = LearnedSuccessModel(Path(tmp) / "missing.json", warn=warnings.append)
            self.assertEqual(missing.probability({"colour": "red"}, 0.37), 0.37)
            self.assertEqual(missing.probability({"colour": "red"}, 0.42), 0.42)
            self.assertEqual(len(warnings), 1)
            bad_path = Path(tmp) / "bad.json"
            bad_path.write_text("{not-json")
            bad = LearnedSuccessModel(bad_path, warn=warnings.append)
            self.assertEqual(bad.probability({}, 0.61), 0.61)
            self.assertEqual(len(warnings), 2)

    def test_unsupported_version_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            write_model(path)
            data = json.loads(path.read_text())
            data["version"] = 2
            path.write_text(json.dumps(data))
            warnings = []
            model = LearnedSuccessModel(path, warn=warnings.append)
            self.assertEqual(model.probability({}, 0.44), 0.44)
            self.assertEqual(len(warnings), 1)

    def test_prediction_is_clamped_and_fast(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            write_model(path, 100.0, -100.0)
            model = LearnedSuccessModel(path, weight=1.0)
            self.assertEqual(model.predict({"colour": "red"}), 0.98)
            self.assertEqual(model.predict({"colour": "blue"}), 0.05)
            start = time.perf_counter()
            for _ in range(1000):
                model.predict({"colour": "red"})
            self.assertLess((time.perf_counter() - start) / 1000.0, 0.0001)

    def test_non_finite_feature_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            write_model(path)
            warnings = []
            model = LearnedSuccessModel(path, weight=1.0, warn=warnings.append)
            self.assertEqual(model.probability({"colour": "red", "unused": float("nan")}, 0.4),
                             model.predict({"colour": "red"}))
            # Exercise a numeric model feature, not an ignored value.
            data = json.loads(path.read_text())
            data.update(feature_names=["pelvis_x"], means=[0.0], stds=[1.0], weights=[1.0],
                        duration={"weights": [0.0], "bias": 10.0})
            path.write_text(json.dumps(data))
            model = LearnedSuccessModel(path, weight=1.0, warn=warnings.append)
            self.assertEqual(model.probability({"pelvis_x": float("nan")}, 0.4), 0.4)
            self.assertEqual(len(warnings), 1)


class TestLearnedPolicy(unittest.TestCase):
    def _policy(self, path, enabled=True, weight=1.0, confidence_tiers=False):
        overrides = {"policy.confidence_order": True, "policy.confidence_tiers": confidence_tiers,
                     "decision.learned_p": enabled, "decision.learned_p_model": str(path),
                     "decision.learned_p_weight": weight}
        pol = object.__new__(ClassicalPolicy)  # these tests exercise decision wiring, not IK/planning
        pol.cfg = load_config(overrides=overrides, overlay="none")
        pol.mapping = {"red": "left", "yellow": "center", "blue": "right"}
        bins = {"left": (0.45, 0.21), "center": (0.45, 0.0), "right": (0.45, -0.21)}
        pol.planner = SimpleNamespace(bin_xy=lambda name: bins[name])
        pol.scene = SimpleNamespace(bin_names=("left", "center", "right"))
        pol.perception = SimpleNamespace(hsv={})
        pol.time_limit, pol.C, pol.tier = 120.0, None, None
        pol.attempts, pol.attempt_tracks = [], []
        pol._stab, pol._stab_last = [], None
        pol._prior_side, pol._dec_warned = set(), set()
        pol.tiers = dec.SpeedTiers(pol._dcfg())
        pol.log = lambda _message: None
        ld = pol.cfg["decision"]
        pol.learned_p = (LearnedSuccessModel(ld["learned_p_model"], ld["learned_p_weight"], pol.log)
                         if enabled else None)
        return pol

    def test_flag_off_keeps_identical_ordering(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            write_model(path)
            detections = [det("blue", shape="cube"), det("red", shape="cube")]
            stock = self._policy(path, enabled=False)
            off = self._policy(path, enabled=False)
            stock_order = [d.color for d in stock._conf_rank(detections, detections, obs(0), lambda _d: ())]
            off_order = [d.color for d in off._conf_rank(detections, detections, obs(0), lambda _d: ())]
            self.assertEqual(stock_order, ["blue", "red"])  # equal legacy scores retain input order
            self.assertEqual(off_order, stock_order)
            self.assertIsNone(off.learned_p)

    def test_higher_model_probability_orders_first_at_equal_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            write_model(path)
            red, blue = det("red", y=0.0, shape="cube"), det("blue", y=0.0, shape="cube")
            pol = self._policy(path)
            ranked = pol._conf_rank([blue, red], [blue, red], obs(0), lambda _d: ())
            self.assertEqual([d.color for d in ranked], ["red", "blue"])

    def test_wrong_bin_gate_never_uses_learned_probability(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            write_model(path, red_weight=20.0)
            red = det("red", n_px=500)
            stock = self._policy(path, enabled=False, confidence_tiers=True)
            learned = self._policy(path, confidence_tiers=True)
            stock.C = learned.C = 0.4
            stock.tier = learned.tier = dec.DEGRADED
            before, after = stock._conf_info(red, obs(0)), learned._conf_info(red, obs(0))
            self.assertEqual(after["gate"], before["gate"])
            self.assertNotEqual(after["p"], before["p"])
            # Configure the existing perception-only threshold around this example; both make the same safety choice.
            threshold = before["gate"] + 0.01
            stock.cfg["decision"]["degraded_min_p"] = threshold
            learned.cfg["decision"]["degraded_min_p"] = threshold
            self.assertEqual(stock._decision_filter([red], obs(0)), [])
            self.assertEqual(learned._decision_filter([red], obs(0)), [])

    def test_learned_duration_can_only_reject_earlier(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            write_model(path)
            pol = self._policy(path)
            pol.time_limit = 20.0
            short = SimpleNamespace(duration=5.0, corrective=False)
            self.assertTrue(pol._has_time(short, 10.0))
            short.learned_expected_s = 15.0
            self.assertFalse(pol._has_time(short, 10.0))
            short.duration, short.learned_expected_s = 15.0, 5.0
            self.assertFalse(pol._has_time(short, 10.0))

    def test_feature_log_is_deferred_and_uses_execution_time(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.json"
            write_model(path)
            pol = self._policy(path)
            messages = []
            pol.log = messages.append
            red = det("red", shape="cube")
            plan = SimpleNamespace(side="left", target_bin="left")
            pol._learned_attempt(red, [red], obs(1.0), plan)
            self.assertEqual(messages, [])
            pol._log_learned_attempt(plan, 42.0)
            self.assertEqual(len(messages), 1)
            self.assertIn("[learned_p] t=42.00", messages[0])
            self.assertIn("actual_target=left", messages[0])


if __name__ == "__main__":
    unittest.main()
