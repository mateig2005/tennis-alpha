import os
import unittest

import numpy as np
import pandas as pd

from src import canonical, evaluate
from src.canonical import CANON, DEV_PATH, DEV_SPLITS, load_dev, validate
from src.pairs import assign_ab
from tests.helpers import synthetic

POST_MATCH = {"score", "minutes", "w_ace", "l_ace", "w_svpt", "l_svpt", "y", "winner_name", "loser_name"}


class TestPairs(unittest.TestCase):
    def setUp(self):
        self.df = synthetic(2000, seed=7)

    def test_reproducible_and_row_order_independent(self):
        a = assign_ab(self.df).set_index("match_id")
        b = assign_ab(self.df.sample(frac=1, random_state=1)).set_index("match_id").loc[a.index]
        pd.testing.assert_frame_equal(a, b)

    def test_balanced_and_players_preserved(self):
        ab = assign_ab(self.df)
        self.assertTrue(0.45 < ab.y.mean() < 0.55)
        self.assertTrue((ab.a_id != ab.b_id).all())
        self.assertTrue(((ab.a_id.isin(self.df.winner_id)) | (ab.a_id.isin(self.df.loser_id))).all())
        won = ab.a_id.where(ab.y == 1, ab.b_id)
        self.assertTrue((won.to_numpy() == self.df.winner_id.to_numpy()).all())          # y really encodes the winner

    def test_label_and_player_slots_not_predictable_from_ids(self):
        ab = assign_ab(self.df)
        stronger_is_a = (ab.a_id.str[1:].astype(int) < ab.b_id.str[1:].astype(int))      # low id = strong in the synthetic data
        self.assertTrue(0.45 < stronger_is_a.mean() < 0.55)                              # slot A carries no hidden signal


class TestSchemaAndLocks(unittest.TestCase):
    def test_canonical_has_no_post_match_or_winner_columns(self):
        self.assertFalse(POST_MATCH & set(CANON))

    def test_validate_rejects_bad_data(self):
        d = synthetic(10)
        with self.assertRaises(AssertionError): validate(pd.concat([d, d]))
        with self.assertRaises(AssertionError): validate(d.assign(loser_id=d.winner_id))

    def test_dev_splits_exclude_locked(self):
        self.assertFalse({"test", "forward_oos"} & set(DEV_SPLITS))
        self.assertIn("dev", DEV_PATH); self.assertNotIn("locked", DEV_PATH)

    @unittest.skipUnless(os.path.exists(DEV_PATH), "dev data not built; run src/prepare_data.py")
    def test_real_dev_file_is_pre_2025_and_chronology_sorts(self):
        df = load_dev()
        self.assertLess(df.event_date.max(), pd.Timestamp("2025-01-01"))
        self.assertEqual(set(df.split), {"warmup", "train", "validation"})
        # tournaments are never split across splits, so a split boundary is a clean date boundary
        last = {s: df[df.split == s].event_date.max() for s in DEV_SPLITS}
        first = {s: df[df.split == s].event_date.min() for s in DEV_SPLITS}
        self.assertLess(last["warmup"], first["train"]); self.assertLess(last["train"], first["validation"])


class TestMetrics(unittest.TestCase):
    def test_known_values(self):
        y = np.array([1, 0]); p = np.array([0.5, 0.5])
        self.assertAlmostEqual(evaluate.log_loss(y, p), np.log(2))
        self.assertAlmostEqual(evaluate.brier(y, p), 0.25)
        self.assertAlmostEqual(evaluate.accuracy(y, p), 0.5)
        self.assertAlmostEqual(evaluate.log_loss([1], [1.0]), 0.0, places=6)             # clipped, no infinity

    def test_ece_zero_when_calibrated(self):
        y = np.array([1] * 7 + [0] * 3); p = np.full(10, 0.7)
        self.assertAlmostEqual(evaluate.ece(y, p), 0.0)

    def test_rank_baseline_is_symmetric_and_positive(self):
        d = synthetic(3000, seed=9); ab = assign_ab(d)
        b = evaluate.fit_rank_baseline(ab.rank_a, ab.rank_b, ab.y)
        pa = evaluate.rank_prob(b, ab.rank_a, ab.rank_b); pb = evaluate.rank_prob(b, ab.rank_b, ab.rank_a)
        np.testing.assert_allclose(pa + pb, 1.0)
        self.assertTrue(np.isfinite(b))

    def test_bootstrap_identical_models_give_zero(self):
        l = np.random.default_rng(0).random(200)
        m, lo, hi = evaluate.paired_bootstrap(l, l, np.arange(200) % 20)
        self.assertEqual((m, lo, hi), (0.0, 0.0, 0.0))


if __name__ == "__main__":
    unittest.main()


class TestCalibrationFit(unittest.TestCase):
    def test_perfect_and_overconfident(self):
        rng = np.random.default_rng(0); p = rng.uniform(0.05, 0.95, 40000); y = (rng.random(40000) < p).astype(int)
        a, b = evaluate.calibration_fit(y, p); self.assertAlmostEqual(a, 0, delta=0.06); self.assertAlmostEqual(b, 1, delta=0.06)
        z = np.log(p / (1 - p)); over = 1 / (1 + np.exp(-2 * z))                              # same signal, twice as extreme
        a2, b2 = evaluate.calibration_fit(y, over); self.assertAlmostEqual(b2, 0.5, delta=0.05)
