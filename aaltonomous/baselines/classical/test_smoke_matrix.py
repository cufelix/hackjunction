"""Smoke-matrix wiring: tools/smoke_matrix.py on 3 overlays x 1 short mock episode (no fault variants).

Asserts only what is deterministic: every overlay loads, one episode runs, and no act() raises or is swallowed by
the safety shell. Latency is reported on stdout, not asserted: p99 depends on machine load and on the first-look
calibration spike, so the < 33 ms requirement is judged from the full matrix run, not from this test.
"""
import importlib.util
import sys
import unittest
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
OVERLAYS = ["submission", "rc1", "cand_r18"]


def _load_matrix():
    if "smoke_matrix" in sys.modules:
        return sys.modules["smoke_matrix"]
    spec = importlib.util.spec_from_file_location("smoke_matrix", REPO / "tools" / "smoke_matrix.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["smoke_matrix"] = mod  # needed so ProcessPoolExecutor can pickle the worker functions
    spec.loader.exec_module(mod)
    return mod


class TestSmokeMatrix(unittest.TestCase):
    def test_three_overlays_one_short_episode(self):
        sm = _load_matrix()
        rows, wall = sm.run_matrix(OVERLAYS, seeds=(0,), time_limit=20.0, workers=1, variants=())
        self.assertEqual(sorted(r["overlay"] for r in rows), sorted(OVERLAYS))
        for r in rows:
            self.assertTrue(r["load_ok"], f"{r['overlay']}: {r['load_error']}")
            self.assertEqual(len(r["episodes"]), 1, r["overlay"])
            self.assertGreater(r["episodes"][0]["n"] or 0, 0, r["overlay"])
            self.assertEqual(r["exc_n"], 0, f"{r['overlay']}: {r['exc_head']} @ {r['exc_loc']}")
            print(f"\n{r['overlay']}: p50 {r['p50']} ms, p99 {r['p99']} ms, score {r['score']}, {wall:.0f} s wall")


if __name__ == "__main__":
    unittest.main()
