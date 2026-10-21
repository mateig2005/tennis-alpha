"""Phase 12 audit utilities: favourite orientation, calibration bands, and the independent scoring-model check."""
import unittest

import numpy as np

from src.v2 import audit_tools as T
from src.v2 import markov


class TestCalibration(unittest.TestCase):
    def test_favourite_view(self):
        f, w = T.favourite_view([1, 0, 0, 1], [0.7, 0.7, 0.2, 0.4])
        np.testing.assert_allclose(f, [0.7, 0.7, 0.8, 0.6])
        np.testing.assert_array_equal(w, [1, 0, 1, 0])

    def test_calibrated_model_has_no_gap_and_overconfident_one_does(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(0.05, 0.95, 40000)
        y = (rng.random(40000) < p).astype(int)
        groups = np.arange(40000) // 20  # 2,000 'tournaments'
        good = T.calibration_table(y, p, groups)
        self.assertTrue(((good.low <= 0) & (good.high >= 0)).all())   # every band's interval covers 0
        self.assertEqual(good.n.sum(), 40000)
        over = 1 / (1 + np.exp(-2 * np.log(p / (1 - p))))  # far too extreme
        s = T.summary(y, over, groups)
        self.assertLess(s["high"], 0); self.assertLess(s["cal_slope"], 0.7)


class TestScoringModel(unittest.TestCase):
    def test_independent_solver_matches_production(self):
        for pa, pb in np.random.default_rng(1).uniform(0.45, 0.8, (15, 2)):
            for bo in (3, 5):
                self.assertAlmostEqual(T.exact_match(pa, pb, bo), markov.match_prob(pa, pb, bo), places=12)

    def test_simulation_agrees_within_its_error(self):
        for pa, pb, bo in ((0.66, 0.60, 3), (0.62, 0.64, 5)):
            e = T.exact_match(pa, pb, bo)
            s = T.simulate(pa, pb, bo, n=3000, seed=4)
            self.assertLess(abs(s - e), 4 * np.sqrt(e * (1 - e) / 3000))

    def test_final_set_tiebreak_length_matters_little_and_is_symmetric(self):
        d = T.exact_match(0.66, 0.60, 5, final_tb=10) - T.exact_match(0.66, 0.60, 5)
        self.assertGreater(d, 0); self.assertLess(d, 0.005)   # longer tiebreak favours the better player, slightly
        self.assertAlmostEqual(T.exact_match(0.6, 0.6, 5, 10), 0.5)
        self.assertAlmostEqual(T.exact_match(0.66, 0.6, 3, 10) + T.exact_match(0.6, 0.66, 3, 10), 1.0, places=12)
