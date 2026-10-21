"""Phase 12: data-readiness checks, odds matching and the frozen forward evaluation (mechanics on synthetic data)."""
import tempfile
import unittest

import numpy as np
import pandas as pd

from src.v2 import forward_eval as FE
from src.v2.data import STATS
from src.v2.odds import match_odds
from src.v2.sources import find_duplicates, link_players, validate_source
from tests.test_v21 import synth_q


class TestSources(unittest.TestCase):
    def test_valid_table_and_soft_report(self):
        d = synth_q(300, seed=1)
        r = validate_source(d)
        self.assertEqual(r["rows"], 300); self.assertEqual(r["rows_with_valid_statistics"], 300)
        idx = d.set_index("match_id")
        for a, b in find_duplicates(d):   # random data can repeat a pair within one tournament; anything flagged must be exactly that
            self.assertEqual(idx.tourney_id[a], idx.tourney_id[b])
            self.assertEqual({idx.winner_id[a], idx.loser_id[a]}, {idx.winner_id[b], idx.loser_id[b]})
        results_only = d.drop(columns=[c for c in d.columns if c[:2] in ("w_", "l_")] + ["has_stats", "venue"])
        self.assertEqual(validate_source(results_only)["rows_with_valid_statistics"], 0)  # V2 would be unavailable

    def test_hard_errors(self):
        d = synth_q(50, seed=2)
        for col, val in (("best_of", 4), ("surface", "Carpet"), ("level", "X")):
            bad = d.copy(); bad.loc[0, col] = val
            with self.assertRaises(ValueError):
                validate_source(bad)
        with self.assertRaises(ValueError):
            validate_source(d.drop(columns=["score"]))
        bad = d.copy(); bad.loc[1, "loser_id"] = bad.loc[1, "winner_id"]
        with self.assertRaises(ValueError):
            validate_source(bad)

    def test_inconsistent_scores_are_counted(self):
        d = synth_q(50, seed=3)
        d.loc[0, "score"] = "6-4 nonsense"; d.loc[1, "score"] = "6-3 2-1 RET"; d.loc[1, "retired"] = False
        r = validate_source(d)
        self.assertGreaterEqual(r["unparseable_scores"], 1); self.assertGreaterEqual(r["retired_flag_mismatch"], 1)

    def test_duplicates_vs_rematches(self):
        d = synth_q(2, seed=4)
        d["winner_id"], d["loser_id"], d["round_rank"] = ["x", "y"], ["y", "x"], 5
        d["event_date"] = [pd.Timestamp("2021-05-03"), pd.Timestamp("2021-05-10")]
        d["tourney_id"] = ["t:madrid", "t:rome"]
        self.assertEqual(find_duplicates(d), [])                       # a real rematch a week later
        d.loc[1, "source"] = "other"                                   # same match recorded by two sources
        self.assertEqual(len(find_duplicates(d)), 1)

    def test_player_linking_never_guesses(self):
        mapping, unmatched, ambiguous = link_players({"n1": "Jannik Sinner", "n2": "Nobody Known"}, {"sackmann:206173": "Jannik Sinner"})
        self.assertEqual(mapping, {"n1": "sackmann:206173"}); self.assertEqual(unmatched, ["n2"]); self.assertEqual(ambiguous, [])


class TestOddsMatching(unittest.TestCase):
    def test_orientation_window_unmatched_ambiguous(self):
        matches = pd.DataFrame({"match_id": ["m1", "m2", "m3", "m4"], "a_id": ["p", "q", "r", "r"], "b_id": ["q2", "p", "s", "s"],
                                "event_date": pd.to_datetime(["2026-06-01", "2026-06-01", "2026-06-08", "2026-06-10"])})
        odds = pd.DataFrame({"p1_id": ["p", "p", "z", "r"], "p2_id": ["q", "q2", "w", "s"], "odds_p1": [1.5, 1.8, 2.0, 1.4], "odds_p2": [2.6, 2.0, 1.8, 3.0],
                             "odds_time": pd.to_datetime(["2026-06-02", "2026-06-25", "2026-06-02", "2026-06-09"])})
        got, unmatched, ambiguous = match_odds(odds, matches)
        self.assertEqual(got.match_id.tolist(), ["m2"])
        self.assertEqual((got.odds_a[0], got.odds_b[0]), (2.6, 1.5))     # m2 has A = q, so the prices are flipped
        self.assertEqual(len(unmatched), 2)                              # outside the window, unknown players
        self.assertEqual(len(ambiguous), 1)                              # two r-s matches in range: not guessed


class TestForwardEvaluation(unittest.TestCase):
    def setUp(self):
        self.d = synth_q(900, seed=6)
        self.d["level"] = "A"
        self.cut = pd.Timestamp("2015-06-01")

    def test_evaluation_outcomes_never_reach_a_fit(self):
        # Player state may legitimately absorb evaluation results once released; model parameters may not. So predictions
        # made before any evaluation result is released (first week after the cutoff) must not change when those results do.
        base = FE.evaluate(self.d, self.cut)
        d2 = self.d.copy()
        late = d2.event_date > self.cut
        pairs = [("winner_id", "loser_id"), ("winner_rank", "loser_rank"), ("winner_points", "loser_points")] + [(f"w_{x}", f"l_{x}") for x in STATS]
        for w, l in pairs:  # flip every evaluation result, moving each player's own columns with them
            d2.loc[late, [w, l]] = d2.loc[late, [l, w]].to_numpy()
        alt = FE.evaluate(d2, self.cut)
        self.assertTrue((base.event_date > self.cut).all())
        first = base.index[base.event_date <= self.cut + pd.Timedelta(days=7)]
        self.assertGreater(len(first), 5)
        cols = ["D", "C", "V2", "Elo", "v2_available"]
        # swapping winner and loser also swaps which player is A (the hash picks a side), so compare distance from 50%
        np.testing.assert_allclose((base.loc[first, cols[:4]] - 0.5).abs().to_numpy(), (alt.loc[first, cols[:4]] - 0.5).abs().to_numpy(), atol=1e-12)
        pd.testing.assert_series_equal(base.loc[first, "v2_available"], alt.loc[first, "v2_available"])
        self.assertFalse(base[cols[:3]].equals(alt[cols[:3]]))   # later predictions do use the released results (state updates)

    def test_summary_scores_identical_rows(self):
        summ, d = FE.summarise(FE.evaluate(self.d, self.cut))
        self.assertEqual(summ.n.nunique(), 1); self.assertTrue(np.isfinite(d).all())
        self.assertIn("inconclusive", FE.verdict(500, (-0.01, -0.02, -0.001)))
        self.assertEqual(FE.verdict(3000, (-0.01, -0.02, -0.001)), "V2 better than Model D")

    def test_refusals(self):
        with self.assertRaises(SystemExit):
            FE.check_frozen({"models/frozen_v2.json": "0000000000000000"})
        FE.check_frozen()   # the committed files match the protocol
        with tempfile.NamedTemporaryFile(suffix=".md") as t:
            with self.assertRaises(SystemExit):
                FE.guard(t.name)
            FE.guard(t.name, approved=True)

    def test_period_a_boundary_quarantines_unfinished_tournaments(self):
        new = pd.DataFrame({"tourney_id": ["t1", "t1", "t2", "t3"], "match_id": ["a", "b", "c", "d"], "level": ["G", "G", "M", "A"],
                            "event_date": pd.to_datetime(["2026-09-01", "2026-09-01", "2026-10-07", "2026-10-12"])})
        keep, out, t = FE.period_a_split(new, pd.Timestamp("2026-10-19"))
        self.assertEqual(sorted(keep.match_id), ["a", "b", "d"])        # 1 Sep + 14 and 12 Oct + 7 finish by 19 Oct
        self.assertEqual(out.match_id.tolist(), ["c"])                  # 7 Oct + 13 = 20 Oct: may run past the cutoff
        self.assertEqual(t.set_index("tourney_id").status.to_dict(), {"t1": "period A", "t2": "quarantined", "t3": "period A"})

