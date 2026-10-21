import unittest

import numpy as np
import pandas as pd

from src import models
from src.canonical import MAX_DAYS
from src.elo import expected
from src.features import PRIOR, build_features, design, elo_ratings
from src.pairs import assign_ab
from tests.helpers import synthetic


def pipeline(ab, release=MAX_DAYS, k=24):
    ab = ab.sort_values(["event_date", "round_rank", "match_id"], kind="stable")
    rt = elo_ratings(ab, k, k, release)
    fa, fb = build_features(ab, rt, release)
    return ab, rt, fa, fb


def weekly_x(wins, start="2016-01-04", opp_elo=None, gap_days=7):
    """Player 'x' plays one level-A hard-court match per week against a new opponent each time."""
    n = len(wins)
    d = synthetic(n); d["level"] = "A"; d["surface"] = "Hard"
    d["winner_id"] = ["x" if w else f"o{i}" for i, w in enumerate(wins)]
    d["loser_id"] = [f"o{i}" if w else "x" for i, w in enumerate(wins)]
    d["event_date"] = [pd.Timestamp(start) + pd.Timedelta(days=gap_days * i) for i in range(n)]
    d["round_rank"] = 1; d["tourney_id"] = [f"t:{i}" for i in range(n)]
    return assign_ab(d)


def flat_ratings(ab, x_elo=1500.0, opp=1500.0):
    r = pd.DataFrame(index=ab.match_id)
    xa = (ab.a_id == "x").to_numpy()
    r["elo_a"] = np.where(xa, x_elo, opp); r["elo_b"] = np.where(xa, opp, x_elo)
    r["surf_a"], r["surf_b"] = r.elo_a, r.elo_b
    return r


def x_row(ab, fa, fb, i):
    m = ab.sort_values(["event_date", "round_rank", "match_id"]).iloc[i]
    return (fa if m.a_id == "x" else fb).loc[m.match_id]


class TestRollingWindows(unittest.TestCase):
    wins = [1, 0, 1, 1, 0, 1, 0, 0, 1, 1]

    def test_hand_computed_windows(self):
        ab = weekly_x(self.wins)
        fa, fb = build_features(ab, flat_ratings(ab), MAX_DAYS)
        i = 9                                   # 10th match; prior results = wins[:9], ages 7,14,...,63 days
        row = x_row(ab, fa, fb, i); prior = self.wins[:9]
        rate = lambda w, n: (w + PRIOR / 2) / (n + PRIOR)
        self.assertAlmostEqual(row.wr5, rate(sum(prior[-5:]), 5))
        self.assertAlmostEqual(row.wr10, rate(sum(prior), 9))                              # only 9 matches exist
        self.assertAlmostEqual(row.wr30, rate(sum(prior[-4:]), 4))                         # ages 7,14,21,28 <= 30
        self.assertAlmostEqual(row.wr90, rate(sum(prior), 9))                              # all ages <= 63
        self.assertAlmostEqual(row.ln30, np.log1p(4)); self.assertAlmostEqual(row.ln365, np.log1p(9))
        self.assertAlmostEqual(row.swr365, rate(sum(prior), 9))                            # all on hard
        self.assertAlmostEqual(row.rest, 0.0)                                              # last tournament released exactly today

    def test_365_day_window_drops_old_matches(self):
        ab = weekly_x([1, 1, 1, 0], gap_days=200)                                          # ages: 600, 400, 200 days
        fa, fb = build_features(ab, flat_ratings(ab), MAX_DAYS)
        row = x_row(ab, fa, fb, 3)
        self.assertAlmostEqual(row.ln365, np.log1p(1)); self.assertAlmostEqual(row.wr365, (1 + PRIOR / 2) / (1 + PRIOR))

    def test_decayed_form_matches_formula(self):
        ab = weekly_x(self.wins)
        fa, fb = build_features(ab, flat_ratings(ab), MAX_DAYS, half_lives=(30,))
        row = x_row(ab, fa, fb, 9)
        ages = np.arange(9, 0, -1) * 7.0; w = 0.5 ** (ages / 30)
        self.assertAlmostEqual(row.dec_wr_30, (np.dot(w, self.wins[:9]) + PRIOR / 2) / (w.sum() + PRIOR))


class TestOpponentAdjustment(unittest.TestCase):
    def residual_after_one_win(self, opp):
        ab = weekly_x([1, 1])
        fa, fb = build_features(ab, flat_ratings(ab, 1500.0, opp), MAX_DAYS)
        return x_row(ab, fa, fb, 1)

    def test_beating_a_strong_opponent_counts_more(self):
        strong, weak = self.residual_after_one_win(1700.0), self.residual_after_one_win(1300.0)
        p_strong = float(expected(1500, 1700))                                             # 0.2403
        self.assertAlmostEqual(strong.adj10, (1 - p_strong) / (1 + PRIOR))                 # win residual = 1 - expectation
        self.assertAlmostEqual(weak.adj10, (1 - float(expected(1500, 1300))) / (1 + PRIOR))
        self.assertGreater(strong.adj10, weak.adj10)
        self.assertAlmostEqual(strong.wr10, weak.wr10)                                     # raw win rate cannot tell them apart
        self.assertAlmostEqual(strong.opp10, 200.0 / (1 + PRIOR))


class TestDebutsAndMissing(unittest.TestCase):
    def test_debut_is_neutral(self):
        ab = weekly_x([1, 0])
        fa, fb = build_features(ab, flat_ratings(ab), MAX_DAYS)
        row = x_row(ab, fa, fb, 0)
        for c in ("wr5", "wr10", "wr30", "wr365", "swr10", "dec_wr_30"):
            self.assertAlmostEqual(row[c], 0.5, msg=c)
        for c in ("adj10", "adj90", "opp10", "adj_dec_30", "ln30", "has_hist"):
            self.assertAlmostEqual(row[c], 0.0, msg=c)
        self.assertAlmostEqual(row.rest, np.log1p(365))
        self.assertFalse(row.isna().any())

    def test_missing_ranks_and_points_are_finite_and_flagged(self):
        d = synthetic(60, seed=2)
        d.loc[:9, "winner_rank"] = np.nan; d.loc[5:14, "loser_points"] = np.nan; d.loc[20:29, ["winner_rank", "loser_rank"]] = np.nan
        ab, rt, fa, fb = pipeline(assign_ab(d))
        D = design(ab, rt, fa, fb)
        self.assertTrue(np.isfinite(D.to_numpy()).all())
        both = ab.set_index("match_id").loc[D.index]
        both_missing = both.rank_a.isna() & both.rank_b.isna()
        self.assertTrue((D.loc[both_missing, "rank_miss"] == 0).all()); self.assertTrue((D.loc[both_missing, "lrank"] == 0).all())
        one = both.rank_a.isna() ^ both.rank_b.isna()
        self.assertTrue((D.loc[one, "rank_miss"].abs() == 1).all())


class TestChronologyAndLeakage(unittest.TestCase):
    def setUp(self):
        self.ab = assign_ab(synthetic(500, seed=21))

    def _flip_future(self, release):
        cut = self.ab.event_date.quantile(0.5)
        f = self.ab.copy(); late = f.event_date > cut; f.loc[late, "y"] = 1 - f.loc[late, "y"]
        _, _, fa, fb = pipeline(self.ab, release); _, _, ga, gb = pipeline(f, release)
        keep = self.ab.loc[self.ab.event_date <= cut, "match_id"]
        pd.testing.assert_frame_equal(fa.loc[keep], ga.loc[keep]); pd.testing.assert_frame_equal(fb.loc[keep], gb.loc[keep])

    def test_future_results_cannot_change_past_features_strict(self):
        self._flip_future(MAX_DAYS)

    def test_future_results_cannot_change_past_features_group_rule(self):
        self._flip_future(None)

    def test_strict_hides_own_tournament_group_rule_does_not(self):
        # two rounds of one tournament: round 2 features vs flipping the round-1 result
        d = synthetic(2); d["level"] = "A"; d["surface"] = "Hard"; d["event_date"] = pd.Timestamp("2016-01-04")
        d["winner_id"] = ["x", "x"]; d["loser_id"] = ["y", "z"]; d["round_rank"] = [1, 2]; d["tourney_id"] = "t:0"
        for release, expect_change in ((MAX_DAYS, False), (None, True)):
            ab = assign_ab(d); flipped = ab.copy(); flipped.loc[flipped.round_rank == 1, "y"] = 1 - flipped.y
            r2 = ab.match_id[ab.round_rank == 2].iloc[0]
            out = []
            for df in (ab, flipped):
                _, _, fa, fb = pipeline(df, release); row = fa.loc[r2] if df.set_index("match_id").loc[r2, "a_id"] == "x" else fb.loc[r2]
                out.append(row)
            self.assertEqual(not out[0].equals(out[1]), expect_change, msg=str(release))

    def test_results_become_visible_only_after_max_tournament_length(self):
        for gap, visible in ((7, True), (4, False)):                                       # level A lasts <= 7 days
            ab = weekly_x([1, 1], gap_days=gap); fa, fb = build_features(ab, flat_ratings(ab), MAX_DAYS)
            self.assertEqual(x_row(ab, fa, fb, 1).has_hist == 1.0, visible, msg=gap)

    def test_row_order_independent_and_reproducible(self):
        _, _, fa, fb = pipeline(self.ab)
        _, _, ga, gb = pipeline(self.ab.sample(frac=1, random_state=3))
        _, _, ha, hb = pipeline(self.ab)
        pd.testing.assert_frame_equal(fa.sort_index(), ga.sort_index()); pd.testing.assert_frame_equal(fb.sort_index(), gb.sort_index())
        pd.testing.assert_frame_equal(fa, ha); pd.testing.assert_frame_equal(fb, hb)


class TestSymmetry(unittest.TestCase):
    def test_swapping_A_and_B_flips_every_feature_sign(self):
        base = synthetic(400, seed=22)
        out = []
        for seed in (1, 99):
            ab, rt, fa, fb = pipeline(assign_ab(base, seed=seed)); out.append((ab.set_index("match_id"), design(ab, rt, fa, fb)))
        (ab1, D1), (ab2, D2) = out
        D2 = D2.loc[D1.index]; same_slot = (ab1.a_id == ab2.loc[D1.index].a_id).to_numpy()
        sign = np.where(same_slot, 1.0, -1.0)[:, None]
        np.testing.assert_allclose(D1.to_numpy(), sign * D2.to_numpy(), atol=1e-9)

    def test_model_probabilities_are_complementary(self):
        rng = np.random.default_rng(0); X = pd.DataFrame(rng.normal(size=(500, 4)), columns=list("abcd"))
        y = (rng.random(500) < 1 / (1 + np.exp(-X.a))).astype(int)
        m = models.fit(X, y, 1.0)
        np.testing.assert_allclose(models.predict(m, X) + models.predict(m, -X), 1.0, atol=1e-12)


class TestModelTraining(unittest.TestCase):
    def test_scaler_uses_training_rows_only_and_does_not_centre(self):
        rng = np.random.default_rng(1); tr = pd.DataFrame(rng.normal(size=(300, 3)) * [1, 5, 10], columns=list("abc"))
        te = tr * 100 + 7
        y = rng.integers(0, 2, 300)
        sc, lr = models.fit(tr, y, 1.0)
        np.testing.assert_allclose(sc.scale_, tr.std(ddof=0).to_numpy(), rtol=1e-9)
        np.testing.assert_allclose(sc.transform(tr), tr.to_numpy() / sc.scale_)               # pure scaling: no centring
        self.assertEqual(lr.intercept_.tolist(), [0.0])
        models.predict((sc, lr), te); np.testing.assert_allclose(sc.scale_, tr.std(ddof=0).to_numpy(), rtol=1e-9)   # unchanged by test data

    def test_walk_forward_folds_never_train_on_the_future(self):
        n = 3000; rng = np.random.default_rng(2)
        year = pd.Series(rng.choice(np.arange(2015, 2025), n)); X = pd.DataFrame(rng.normal(size=(n, len(models.columns(["elo", "rank"], 30)))),
                                                                              columns=models.columns(["elo", "rank"], 30))
        y = pd.Series((rng.random(n) < 1 / (1 + np.exp(-X["elo"]))).astype(int)); mask = pd.Series(True, index=X.index)
        base = models.oof_predictions(X, y, year, mask, ["elo", "rank"], 30, 1.0)
        y2 = y.copy(); y2[year >= 2022] = 1 - y2[year >= 2022]                                # corrupt labels from 2022 on
        alt = models.oof_predictions(X, y2, year, mask, ["elo", "rank"], 30, 1.0)
        early = year < 2022                                                                    # folds 2020, 2021 only trained on <2022
        pd.testing.assert_series_equal(base[early & year.isin(models.FOLDS)], alt[early & year.isin(models.FOLDS)])


if __name__ == "__main__":
    unittest.main()
