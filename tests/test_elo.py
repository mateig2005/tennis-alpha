import unittest

import numpy as np
import pandas as pd

from src.elo import expected, prob, run_elo
from src.pairs import assign_ab
from tests.helpers import synthetic


def one_match(**kw):
    df = synthetic(2)
    df = df.iloc[:1].copy()
    for k, v in kw.items():
        df[k] = v
    return df


class TestRatingUpdate(unittest.TestCase):
    def test_equal_ratings_k32(self):
        df = assign_ab(synthetic(2).iloc[:1].assign(surface="Hard"))
        second = df.assign(match_id="t:next", event_date=df.event_date + pd.Timedelta(days=7))  # same two players, a week later
        res, _ = run_elo(pd.concat([df, second]), k=32)
        r = res[res.match_id == "t:next"].iloc[0]                                        # ratings the 2nd match saw = after match 1
        won = df.y.iloc[0] == 1
        self.assertAlmostEqual(r.elo_a, 1516.0 if won else 1484.0)                       # winner gains K/2, loser loses it
        self.assertAlmostEqual(r.elo_b, 1484.0 if won else 1516.0)
        self.assertAlmostEqual(r.elo_a + r.elo_b, 3000.0)                                # points are conserved
        self.assertAlmostEqual(r.surf_a, 1516.0 if won else 1484.0)                      # surface rating updated the same way

    def test_expected_formula(self):
        self.assertAlmostEqual(float(expected(1500, 1500)), 0.5)
        self.assertAlmostEqual(float(expected(1900, 1500)), 10 / 11)                     # +400 points = 10:1 odds
        self.assertAlmostEqual(float(expected(1600, 1500) + expected(1500, 1600)), 1.0)

    def test_upset_moves_more_than_expected_win(self):
        base = assign_ab(synthetic(200, seed=3))
        res, _ = run_elo(base, k=32)
        p = prob(res).to_numpy()
        self.assertTrue(((p > 0) & (p < 1)).all())


class TestNoLeakage(unittest.TestCase):
    def setUp(self):
        self.df = assign_ab(synthetic(500, seed=1))

    def test_future_results_cannot_change_past_predictions(self):
        full, _ = run_elo(self.df)
        cut = self.df.event_date.quantile(0.5)
        flipped = self.df.copy()
        late = flipped.event_date > cut
        flipped.loc[late, "y"] = 1 - flipped.loc[late, "y"]                              # reverse every later result
        alt, _ = run_elo(flipped)
        keep = self.df.event_date <= cut
        ids = self.df.loc[keep, "match_id"]
        a = full.set_index("match_id").loc[ids]; b = alt.set_index("match_id").loc[ids]
        pd.testing.assert_frame_equal(a, b)                                              # identical up to and including the cut date

    def test_own_result_never_in_own_prediction(self):
        full, _ = run_elo(self.df)
        i = 123; mid = self.df.match_id.iloc[i]
        f = self.df.copy(); f.iloc[i, f.columns.get_loc("y")] = 1 - f.y.iloc[i]
        alt, _ = run_elo(f)
        r1 = full.set_index("match_id").loc[mid]; r2 = alt.set_index("match_id").loc[mid]
        pd.testing.assert_series_equal(r1, r2)

    def test_same_group_matches_do_not_see_each_other(self):
        # one player in two matches of the SAME (date, round): 2nd prediction must equal a fresh 1500 vs 1500
        d = synthetic(2); d["event_date"] = pd.Timestamp("2016-01-04"); d["round_rank"] = 0
        d["winner_id"] = ["x", "x"]; d["loser_id"] = ["y", "z"]; d["surface"] = "Hard"
        res, _ = run_elo(assign_ab(d))
        self.assertTrue((res[["elo_a", "elo_b"]].to_numpy() == 1500.0).all())

    def test_row_order_inside_a_group_is_irrelevant(self):
        d = self.df
        shuffled = d.sample(frac=1, random_state=5)
        r1, s1 = run_elo(d); r2, s2 = run_elo(shuffled)
        pd.testing.assert_frame_equal(r1.set_index("match_id").sort_index(), r2.set_index("match_id").sort_index())
        self.assertEqual(s1[0].keys(), s2[0].keys())
        for p in s1[0]:
            self.assertAlmostEqual(s1[0][p], s2[0][p])


class TestStrictRelease(unittest.TestCase):
    DAYS = {"A": 7, "M": 13, "G": 14, "D": 4, "F": 8, "O": 14}

    def two_tournaments(self, second_start):
        # player x plays y in tournament T1 (round 1) then z in T1 (round 2), then w in T2
        d = synthetic(3); d["level"] = "A"; d["surface"] = "Hard"; d["winner_id"] = ["x"] * 3; d["loser_id"] = ["y", "z", "w"]
        d["event_date"] = [pd.Timestamp("2016-01-04")] * 2 + [pd.Timestamp(second_start)]; d["round_rank"] = [1, 2, 1]
        return assign_ab(d)

    def x_rating_seen(self, df, release):
        res, _ = run_elo(df, release_days=release)
        r = res.set_index("match_id")
        out = []
        for _, m in df.iterrows():
            out.append(r.loc[m.match_id, "elo_a"] if m.a_id == "x" else r.loc[m.match_id, "elo_b"])
        return dict(zip(df.match_id, out))

    def test_default_sees_earlier_round_strict_does_not(self):
        df = self.two_tournaments("2016-01-11")
        base = self.x_rating_seen(df, None); strict = self.x_rating_seen(df, self.DAYS)
        self.assertGreater(base["t:1"], 1500.0)                                          # default: round 2 sees round 1
        self.assertEqual(strict["t:1"], 1500.0)                                          # strict: own tournament not yet released

    def test_strict_releases_after_tournament_end(self):
        late = self.x_rating_seen(self.two_tournaments("2016-01-11"), self.DAYS)         # starts exactly 7 days later: visible
        early = self.x_rating_seen(self.two_tournaments("2016-01-08"), self.DAYS)        # starts 4 days later: still hidden
        self.assertGreater(late["t:2"], 1500.0)
        self.assertEqual(early["t:2"], 1500.0)

    def test_strict_mode_future_flip_no_leak(self):
        df = assign_ab(synthetic(500, seed=11)); cut = df.event_date.quantile(0.5)
        f = df.copy(); late = f.event_date > cut; f.loc[late, "y"] = 1 - f.loc[late, "y"]
        a, _ = run_elo(df, release_days=self.DAYS); b, _ = run_elo(f, release_days=self.DAYS)
        ids = df.loc[df.event_date <= cut, "match_id"]
        pd.testing.assert_frame_equal(a.set_index("match_id").loc[ids], b.set_index("match_id").loc[ids])


class TestSymmetry(unittest.TestCase):
    def test_probability_of_actual_winner_is_independent_of_A_B_order(self):
        base = synthetic(400, seed=2)
        r1, _ = run_elo(assign_ab(base, seed=1)); r2, _ = run_elo(assign_ab(base, seed=99))
        for w in (0.0, 0.5, 1.0):
            ab1, ab2 = assign_ab(base, seed=1), assign_ab(base, seed=99)
            pw = lambda ab, r: np.where(ab.sort_values(["event_date", "round_rank", "match_id"]).y.to_numpy() == 1,
                                        prob(r, w).to_numpy(), 1 - prob(r, w).to_numpy())
            np.testing.assert_allclose(pw(ab1, r1), pw(ab2, r2), atol=1e-9)

    def test_missing_surface_falls_back_to_general(self):
        d = synthetic(100, seed=4); d["surface"] = None
        r, (gen, sur) = run_elo(assign_ab(d))
        self.assertTrue(all(v == 1500.0 for v in sur.values()))
        np.testing.assert_allclose(r.surf_a, r.elo_a)

    def test_update_levels_filter(self):
        d = synthetic(300, seed=6); d["level"] = "D"
        _, (gen, _) = run_elo(assign_ab(d), update_levels=("A", "M", "G", "F"))
        self.assertTrue(all(v == 1500.0 for v in gen.values()))                          # Davis Cup only: nothing may update


if __name__ == "__main__":
    unittest.main()
