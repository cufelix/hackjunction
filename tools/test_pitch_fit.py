"""tools/pitch_fit.py — the fitter must recover KNOWN pitch offsets before anyone trusts it on real frames.
Run:  python -m pytest tools/test_pitch_fit.py -q   (CPU only, ~30 s)
"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))          # tools/
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))     # repo root

import numpy as np

import pitch_fit as pf


class TestSyntheticRecovery(unittest.TestCase):
    """Blind fit of renders with a known camera pitch: error <= 0.15 deg, rim rms <= 1.5 cm."""

    @classmethod
    def setUpClass(cls):
        cls.fit = pf.RimFit()
        cls.cases = {d: pf.synth_case(d, cls.fit) for d in (-6.0, -3.0, 0.0, 3.0, 6.0)}

    def test_all_cases_rendered(self):
        for d, r in self.cases.items():
            self.assertIsNotNone(r, f"synthetic case {d} did not yield 3 bins")

    def test_recovers_known_pitch(self):
        for d, (true, fitted, rms) in self.cases.items():
            self.assertLessEqual(abs(fitted - true), 0.15, f"true {d}: fitted {fitted}")
            self.assertLess(rms, 0.015, f"true {d}: rim rms {rms * 100:.1f} cm")

    def test_zero_pitch_residual_small(self):
        _, _, rms = self.cases[0.0]
        self.assertLess(rms, 0.01)


class TestExtraction(unittest.TestCase):
    def test_three_bins_on_clean_render(self):
        rend = pf.Renderer(pf.SC, pf.CAL)
        rgb, _ = rend.render([])
        bins, n = pf.far_rim_bins(rgb, tight=False)
        self.assertEqual(n, 3)
        self.assertEqual(set(bins), set(pf.NAMES))
        for u, v in bins.values():
            self.assertGreater(len(u), 30)
            self.assertTrue(np.all(v < 0.42 * rgb.shape[0]))     # rims in the top band

    def test_blank_frame_is_not_fittable(self):
        blank = np.zeros((480, 640, 3), np.uint8)
        bins, n = pf.far_rim_bins(blank, tight=True)
        self.assertEqual(n, 0)
        self.assertEqual(bins, {})


class TestFitGuards(unittest.TestCase):
    def test_empty_bins_does_not_raise(self):
        fit = pf.RimFit()
        deg, shift, res = fit.fit({}, "pitch")
        self.assertEqual(res, {})
        self.assertTrue(np.isfinite(deg))

    def test_cost_discriminates_pitch(self):
        rend = pf.Renderer(pf.SC, pf.CAL)
        rgb, _ = rend.render([])
        bins, _ = pf.far_rim_bins(rgb, tight=False)
        fit = pf.RimFit()
        self.assertLess(fit.cost([0.0], bins, "pitch"), fit.cost([6.0], bins, "pitch"))

    def test_out_of_bounds_cost_is_penalty(self):
        fit = pf.RimFit()
        self.assertEqual(fit.cost([25.0], {}, "pitch"), 1e6)
        self.assertEqual(fit.cost([0.0, 0.5], {}, "pitch_z"), 1e6)

    def test_rx_is_rotation(self):
        for d in (-7.5, 0.0, 12.0):
            m = pf.rx(d)
            np.testing.assert_allclose(m @ m.T, np.eye(3), atol=1e-12)
            self.assertAlmostEqual(np.linalg.det(m), 1.0)


if __name__ == "__main__":
    unittest.main()
