"""V2.1: score parsing, match-quality records, rolling momentum, leakage, symmetry, odds and calibration (PLAN_V2.md V2.1)."""
import unittest

import numpy as np
import pandas as pd

from src.canonical import MAX_DAYS
from src.features import elo_ratings
from src.pairs import assign_ab
from src.v2 import calibration as cal
from src.v2 import features as F
from src.v2 import markov, odds
from src.v2 import quality as Q
from src.v2.score import parse
from tests.test_v2 import synth_v2

QCOLS = Q.GROUPS["quality"] + Q.GROUPS["qmomentum"]


def synth_q(n=700, seed=0):
    d = synth_v2(n, seed=seed)
    rng = np.random.default_rng(seed + 2)
    d["score"] = rng.choice(["6-4 6-3", "7-6(4) 3-6 6-2", "6-1 6-0", "4-6 7-5 7-6(8)", "6-3 2-1 RET"], n)
    d.loc[d.score.str.endswith("RET"), "retired"] = True
    return d


def run_q(d, seed=2024):
    ab = assign_ab(d, seed).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    rt = elo_ratings(ab, 24, 32, MAX_DAYS)
    *_, ctx = F.build(ab, rt, MAX_DAYS)
    qa, qb = Q.build(ab, ctx, MAX_DAYS)
    return ab, qa, qb


class TestScore(unittest.TestCase):
    def test_valid_scores(self):
        self.assertEqual(parse("6-4 6-3"), dict(games_w=12, games_l=7, sets_w=2, sets_l=0, tiebreaks=0, complete=True))
        self.assertEqual(parse("6-7(5) 7-6(2) [10-7]"), dict(games_w=14, games_l=13, sets_w=2, sets_l=1, tiebreaks=3, complete=True))
        self.assertEqual(parse("4-6 6-3 13-11"), dict(games_w=23, games_l=20, sets_w=2, sets_l=1, tiebreaks=0, complete=True))

    def test_retirement_and_abandoned(self):
        r = parse("6-2 3-1 RET")
        self.assertEqual((r["sets_w"], r["sets_l"], r["games_w"], r["games_l"], r["complete"]), (1, 0, 9, 3, False))
        self.assertFalse(parse("6-2 6-1 0-0 RET")["complete"])
        self.assertFalse(parse("6-4 ABD")["complete"])

    def test_malformed(self):
        for s in (None, "", "RET", "W/O", "6-4 x", "6-4 3-1", "3-4(3) 4-2 4-3(1)", "6-4 RET 6-2", "[10-10]", "4-6 3-6", "9-3 6-2"):
            self.assertIsNone(parse(s), s)


class TestRecords(unittest.TestCase):
    sa = {"ace": 5, "df": 2, "svpt": 60, "1stIn": 36, "1stWon": 28, "2ndWon": 12, "SvGms": 10, "bpSaved": 3, "bpFaced": 5}
    sb = {"ace": 1, "df": 4, "svpt": 70, "1stIn": 40, "1stWon": 26, "2ndWon": 14, "SvGms": 11, "bpSaved": 2, "bpFaced": 6}

    def test_hand_example_and_mirror(self):
        ra, rb, sets = Q.match_records(1, parse("6-4 7-5"), True, self.sa, self.sb, 0.64, 0.62, 3)
        self.assertEqual(sets, 2)
        self.assertAlmostEqual(ra["tpw"], (40 + 30) / 130 - (60 * 0.64 + 70 * 0.38) / 130)   # A won 40 serve + 30 return points
        self.assertAlmostEqual(ra["srv"], 40 / 60 - 0.64)
        self.assertAlmostEqual(ra["ret"], 0.62 - 40 / 70)
        self.assertAlmostEqual(ra["dom"], 13 / 22 - (markov.game(0.64) + 1 - markov.game(0.62)) / 2)
        self.assertAlmostEqual(ra["luck"], 1 - markov.match_prob(40 / 60, 40 / 70, 3))
        for k in ("tpw", "dom", "luck"):
            self.assertAlmostEqual(ra[k], -rb[k])

    def test_score_view_follows_winner(self):
        ra, _, _ = Q.match_records(0, parse("6-4 7-5"), False, None, None, 0.62, 0.62, 3)   # A lost 4-6 5-7
        self.assertAlmostEqual(ra["dom"], 9 / 22 - 0.5)
        self.assertNotIn("tpw", ra)  # no statistics: score record only

    def test_retirement_gives_workload_only(self):
        self.assertEqual(Q.match_records(1, parse("6-3 2-1 RET"), True, self.sa, self.sb, 0.6, 0.6, 3), (None, None, 1))
        self.assertEqual(Q.match_records(1, None, True, self.sa, self.sb, 0.6, 0.6, 3), (None, None, 0))


class TestRolling(unittest.TestCase):
    def test_shrunk_windows(self):
        st = Q._Q()
        for i, t in enumerate([0.1, -0.05, 0.02, 0.04, 0.0, 0.03]):
            r = {"tpw": t, "srv": t, "ret": 0.0, "dom": t, "luck": 0.1, "s": 0}
            st.all.append((i, 1, i, r)); st.stat.append((i, 1, i, r)); st.load.append((i, 1, i, 2))
        f = Q.player_features(st, 10, 0, 0.0025)
        self.assertAlmostEqual(f["pq3"], (0.04 + 0.0 + 0.03) / 6)
        self.assertAlmostEqual(f["pq20"], 0.14 / 9)
        self.assertAlmostEqual(f["pq_trend"], (-0.05 + 0.02 + 0.04 + 0.0 + 0.03) / 8 - 0.14 / 9)
        self.assertAlmostEqual(f["load30"], np.log1p(12))
        self.assertAlmostEqual(f["pq_vol10"], np.sqrt((sum(t * t for t in [0.1, -0.05, 0.02, 0.04, 0.0, 0.03]) + 3 * 0.0025) / 9))
        self.assertEqual(Q.player_features(None, 10, 0, 0.0025)["pq5"], 0.0)  # new player: neutral


class TestLeakageAndSymmetry(unittest.TestCase):
    def test_future_changes_nothing_before(self):
        d = synth_q(seed=1)
        cut = pd.Timestamp("2015-06-01")
        ab, qa, qb = run_q(d)
        d2 = d.copy(); fut = d2.event_date >= cut
        d2.loc[fut, ["winner_id", "loser_id"]] = d2.loc[fut, ["loser_id", "winner_id"]].to_numpy()
        d2.loc[fut, "score"] = "6-0 6-0"
        _, qa2, qb2 = run_q(d2)
        past = ab.match_id[ab.event_date < cut]
        pd.testing.assert_frame_equal(qa.loc[past], qa2.loc[past]); pd.testing.assert_frame_equal(qb.loc[past], qb2.loc[past])

    def test_own_tournament_invisible(self):
        d = synth_q(seed=2)
        t = d.tourney_id.value_counts().index[0]
        day = d.loc[d.tourney_id == t, "event_date"].min()
        ab, qa, _ = run_q(d)
        d2 = d.copy(); d2.loc[d2.event_date == day, "score"] = "6-0 6-0"   # every match starting that day, all rounds
        _, qa2, _ = run_q(d2)
        same = ab.match_id[ab.event_date <= day]
        pd.testing.assert_frame_equal(qa.loc[same], qa2.loc[same])

    def test_swap_flips_differences(self):
        d = synth_q(seed=3)
        ab1, qa1, qb1 = run_q(d, 2024)
        _, qa2, qb2 = run_q(d, 99)
        x1, x2 = (qa1 - qb1)[QCOLS], (qa2 - qb2)[QCOLS].loc[qa1.index]
        a1 = ab1.set_index("match_id").a_id.loc[qa1.index]
        a2 = assign_ab(d, 99).set_index("match_id").a_id.loc[qa1.index]
        sign = np.where(a1 == a2, 1.0, -1.0)
        self.assertTrue((sign < 0).any())
        np.testing.assert_allclose(x1.to_numpy(), x2.to_numpy() * sign[:, None], atol=1e-12)

    def test_missing_statistics_and_finite(self):
        d = synth_q(seed=4)
        d.loc[d.index % 4 == 0, "has_stats"] = False
        _, qa, qb = run_q(d)
        self.assertTrue(np.isfinite(qa[QCOLS].to_numpy()).all() and np.isfinite(qb[QCOLS].to_numpy()).all())


class TestOdds(unittest.TestCase):
    def test_conversions(self):
        self.assertAlmostEqual(float(odds.fair_odds(0.6)), 1 / 0.6)
        self.assertAlmostEqual(float(odds.overround(1.8, 2.1)), 1 / 1.8 + 1 / 2.1 - 1)
        p = odds.no_vig(1.8, 2.1)
        self.assertAlmostEqual(float(p), (1 / 1.8) / (1 / 1.8 + 1 / 2.1))
        pw = odds.no_vig(np.array([1.25]), np.array([4.0]), "power")[0]
        k = np.log(pw) / np.log(1 / 1.25)
        self.assertAlmostEqual((1 / 1.25) ** k + 0.25 ** k, 1.0, places=9)
        self.assertGreater(pw, float(odds.no_vig(1.25, 4.0)))   # power takes more margin from the longshot
        self.assertAlmostEqual(float(odds.no_vig(2.0, 2.0)), 0.5)
        with self.assertRaises(ValueError):
            odds.fair_odds(1.0)
        with self.assertRaises(ValueError):
            odds.no_vig(2.5, 2.5)  # implied probabilities sum below 1: not real bookmaker odds

    def test_benchmark_refuses_late_odds(self):
        df = pd.DataFrame({"match_id": ["m1", "m2"], "y": [1, 0], "odds_a": [1.5, 2.2], "odds_b": [2.7, 1.7], "p_model": [0.6, 0.45],
                           "odds_time": pd.to_datetime(["2024-01-01", "2024-01-03"]), "match_start": pd.to_datetime(["2024-01-02", "2024-01-02"])})
        with self.assertRaises(ValueError):
            odds.benchmark(df, ["p_model"])
        df.loc[1, "odds_time"] = pd.Timestamp("2024-01-01")
        tab, by = odds.benchmark(df, ["p_model"])
        self.assertEqual(list(tab.source), ["market (no-vig)", "p_model"])


class TestCalibration(unittest.TestCase):
    def test_platt_detects_overconfidence_and_keeps_symmetry(self):
        rng = np.random.default_rng(0)
        true = rng.uniform(0.2, 0.8, 20000)
        y = (rng.random(20000) < true).astype(int)
        over = 1 / (1 + np.exp(-1.5 * np.log(true / (1 - true))))   # 50% too extreme
        a = cal.platt_fit(over, y)
        self.assertAlmostEqual(a, 1 / 1.5, delta=0.05)
        q = np.array([0.1, 0.3, 0.7])
        np.testing.assert_allclose(cal.platt_apply(a, q) + cal.platt_apply(a, 1 - q), 1.0)
        f = cal.isotonic_fit(over, y)
        np.testing.assert_allclose(f(q) + f(1 - q), 1.0)
        self.assertTrue(np.all(np.diff(f(np.linspace(0.05, 0.95, 50))) >= -1e-12))   # monotone
