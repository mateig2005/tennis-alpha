import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import diagnostics, gbm, models
from src.canonical import DEV_PATH


def toy(n=4000, seed=0):
    rng = np.random.default_rng(seed)
    cols = models.columns(["elo", "rank", "last"], 30)
    X = pd.DataFrame(rng.normal(size=(n, len(cols))), columns=cols)
    y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-X["elo"] - 0.5 * X["wr5"] * X["lrank"]))).astype(int))
    year = pd.Series(rng.choice(np.arange(2015, 2025), n))
    return X, y, year, pd.Series(True, index=X.index), cols


class TestXGBoostSymmetry(unittest.TestCase):
    def test_probabilities_are_complementary(self):
        X, y, *_ = toy()
        m = gbm.fit(X, y, gbm.GRID[-1], 60)
        np.testing.assert_allclose(gbm.predict(m, X) + gbm.predict(m, -X), 1.0, atol=1e-12)

    def test_mirror_doubles_rows_and_flips_labels(self):
        Xm, ym = gbm.mirror(np.array([[1.0, -2.0]]), np.array([1]))
        np.testing.assert_array_equal(Xm, [[1, -2], [-1, 2]]); np.testing.assert_array_equal(ym, [1, 0])

    def test_reproducible(self):
        X, y, *_ = toy()
        a, b = (gbm.predict(gbm.fit(X, y, gbm.GRID[3], 40), X) for _ in range(2))
        np.testing.assert_array_equal(a, b)


class TestNoLeakageInTuning(unittest.TestCase):
    def test_fold_year_and_later_never_influence_the_fold_model(self):
        X, y, year, mask, cols = toy(6000, seed=1)
        base, _, trees = gbm.oof(X, y, year, mask, cols, gbm.GRID[2])
        y2 = y.copy(); y2[year >= 2022] = 1 - y2[year >= 2022]                 # corrupt labels from 2022 on
        alt, _, trees2 = gbm.oof(X, y2, year, mask, cols, gbm.GRID[2])
        early = year.isin([2020, 2021])                                        # models trained on <= 2019 / <= 2020
        pd.testing.assert_series_equal(base[early], alt[early])
        self.assertEqual({f: trees[f] for f in (2020, 2021)}, {f: trees2[f] for f in (2020, 2021)})  # early stopping too

    def test_early_stopping_uses_the_last_training_year_only(self):
        X, y, year, mask, cols = toy(6000, seed=2)
        _, n = gbm.train(X, y, year, mask, cols, gbm.GRID[2], 2021)
        y2 = y.copy(); y2[year == 2022] = 1 - y2[year == 2022]                 # the year to be predicted
        _, n2 = gbm.train(X, y2, year, mask, cols, gbm.GRID[2], 2021)
        self.assertEqual(n, n2)

    def test_simplest_within_tolerance(self):
        tab = pd.DataFrame({"max_depth": [3, 1, 2], "min_child_weight": [5, 25, 5], "reg_lambda": [1, 10, 1],
                            "mean_logloss": [0.6100, 0.6102, 0.6101]})
        self.assertEqual(int(gbm.simplest_within(tab, 0.0003).max_depth), 1)   # near-tie -> shallowest
        self.assertEqual(int(gbm.simplest_within(tab, 0.00005).max_depth), 3)  # clear winner -> best
        self.assertLessEqual(max(g["max_depth"] for g in gbm.GRID), 3)         # complexity cap


class TestDiagnostics(unittest.TestCase):
    def test_vif(self):
        rng = np.random.default_rng(0); a, b = rng.normal(size=5000), rng.normal(size=5000)
        X = pd.DataFrame({"a": a, "b": b, "c": a + 0.05 * rng.normal(size=5000)})
        v = diagnostics.vif(X)
        self.assertAlmostEqual(v["b"], 1.0, delta=0.05); self.assertGreater(v["a"], 50); self.assertGreater(v["c"], 50)
        self.assertEqual(diagnostics.top_correlations(X).iloc[0][["feature 1", "feature 2"]].tolist(), ["a", "c"])

    def test_group_permutation_finds_the_useful_group(self):
        X, y, _, _, cols = toy(6000, seed=3)
        m = models.fit(X[:4000], y[:4000], 1.0)
        imp = diagnostics.group_permutation([(lambda Z: models.predict(m, Z), X[4000:], y[4000:].to_numpy())],
                                            {"elo": ["elo", "surf_elo"], "noise": ["lpts", "rank_miss"]})
        self.assertGreater(imp["elo"], 0.02); self.assertAlmostEqual(imp["noise"], 0.0, delta=0.005)

    def test_by_group(self):
        y = np.array([1, 0, 1, 0]); p = pd.Series([0.5] * 4)
        t = diagnostics.by_group(y, {"m": p}, pd.Series(["a", "a", "b", "b"]))
        self.assertEqual(t.n.tolist(), [2, 2]); np.testing.assert_allclose(t.m, np.log(2))


class TestPhase5Guards(unittest.TestCase):
    def test_rank_independent_columns_have_no_ranking(self):
        from src.run_phase5 import NORANK
        cols = models.columns(NORANK, 60)
        self.assertFalse({"lrank", "lpts", "rank_miss", "pts_miss"} & set(cols)); self.assertIn("adj10", cols); self.assertIn("elo", cols)

    def test_phase5_reads_only_the_dev_file(self):
        src = "".join(Path(f).read_text() for f in ("src/run_phase5.py", "src/dataset.py", "src/gbm.py"))
        self.assertNotIn("locked", src.replace("locked 2025", "")); self.assertNotIn("atp_matches_clean", src)
        self.assertIn("dev", DEV_PATH)


if __name__ == "__main__":
    unittest.main()
