"""policy.learned_order: learned pick-success score as a candidate ORDER key, decision layer OFF (CPU only)."""
import json
import math
import tempfile
import time
import unittest
from pathlib import Path

from .planner import lookup_bin
from .learned_p import LearnedSuccessModel
from .test_decision import DUR, det, first_color, make, obs

FEATS = {"shape": "cube", "colour": "red", "hand": "left", "target_bin": "left", "attempt_index": 1,
         "pelvis_x": 0.3, "pelvis_y": 0.0, "distance": 0.3, "nearest_neighbor_distance": 0.1, "time_remaining": 100.0}


def write_model(path, red_weight=5.0, blue_weight=-5.0, feature_names=("colour=red", "colour=blue")):
    n = len(feature_names)
    weights = [red_weight, blue_weight] if n == 2 else [1.0]
    path.write_text(json.dumps({"version": 1, "feature_names": list(feature_names), "means": [0.0] * n,
                                "stds": [1.0] * n, "weights": weights, "bias": 0.0,
                                "calibration": {"a": 1.0, "b": 0.0}}))


class TestModel(unittest.TestCase):
    def test_missing_model_fails_open_with_one_warning(self):
        with tempfile.TemporaryDirectory() as tmp:
            warns = []
            m = LearnedSuccessModel(Path(tmp) / "missing.json", warn=warns.append)
            self.assertIsNone(m.predict(FEATS))
            self.assertIsNone(m.predict(FEATS))
            self.assertEqual(len(warns), 1)

    def test_non_finite_feature_fails_open(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            write_model(path, feature_names=("pelvis_x",))
            m = LearnedSuccessModel(path)
            self.assertIsNone(m.predict({"pelvis_x": float("nan")}))
            self.assertIsNone(m.model)

    def test_bundled_model_is_plain_json_in_range_and_fast(self):
        m = LearnedSuccessModel()
        self.assertIsNotNone(m.model)
        p = m.predict(FEATS)
        self.assertTrue(0.05 <= p <= 0.98)
        start = time.perf_counter()
        for _ in range(1000):
            m.predict(FEATS)
        self.assertLess((time.perf_counter() - start) / 1000.0, 1e-3)   # per call, loose bound for the shared box


class TestLearnedOrder(unittest.TestCase):
    DETS = lambda self: [det("blue", n_px=900), det("red", n_px=500)]   # stock key (-n_px): blue first

    def test_flag_off_has_no_model_and_keeps_stock_order(self):
        dets = self.DETS()
        pol = make(DUR, dets)
        self.assertIsNone(pol.learned)
        cands, _ = pol._ordered_cands(dets, obs(0), False)
        self.assertEqual([d.color for d in cands], ["blue", "red"])

    def test_flag_on_orders_by_learned_score_drops_nothing_and_keeps_decision_off(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            write_model(path)                                   # red up, blue down
            dets = self.DETS()
            pol = make(DUR, dets, flags=("learned_order",))
            pol.learned = LearnedSuccessModel(path)
            stock, _ = make(DUR, dets)._ordered_cands(dets, obs(0), False)
            ranked, _ = pol._ordered_cands(dets, obs(0), False)
            self.assertEqual([d.color for d in stock], ["blue", "red"])
            self.assertEqual([d.color for d in ranked], ["red", "blue"])          # predicted order
            self.assertEqual({id(d) for d in ranked}, {id(d) for d in stock})     # no object skipped
            self.assertFalse(pol._decision_on())

    def test_pick_order_changes_but_the_bin_and_the_plan_do_not(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            write_model(path)
            off = make(DUR, self.DETS())
            on = make(DUR, self.DETS(), flags=("learned_order",))
            on.learned = LearnedSuccessModel(path)
            self.assertEqual(first_color(off), "blue")
            self.assertEqual(first_color(on), "red")
            self.assertEqual(on.plans[-1].target_bin, lookup_bin(on.cfg, on.mapping, "red", "cube"))   # bin from the mapping
            self.assertEqual(on.plans[-1].target_bin, off.plans[-1].target_bin if off.plans[-1].color == "red"
                             else lookup_bin(off.cfg, off.mapping, "red", "cube"))

    def test_model_unavailable_falls_back_to_stock_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            dets = self.DETS()
            pol = make(DUR, dets, flags=("learned_order",))
            pol.learned = LearnedSuccessModel(Path(tmp) / "missing.json")
            cands, _ = pol._ordered_cands(dets, obs(0), False)
            self.assertEqual([d.color for d in cands], ["blue", "red"])

    def test_feature_extraction_error_logs_once_and_keeps_stock_order(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "m.json"
            write_model(path)
            dets = self.DETS()
            pol = make(DUR, dets, flags=("learned_order",))
            pol.learned = LearnedSuccessModel(path)
            lines = []
            pol.log = lines.append

            def boom(*_a, **_k):
                raise RuntimeError("bad feature")
            pol._learned_features = boom
            for _ in range(2):
                cands, _ = pol._ordered_cands(dets, obs(0), False)
                self.assertEqual([d.color for d in cands], ["blue", "red"])
            self.assertEqual(len([l for l in lines if l.startswith("[learned_order]")]), 1)


if __name__ == "__main__":
    unittest.main()
