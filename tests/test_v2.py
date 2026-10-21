"""V2 research pipeline: chronology, leakage, symmetry, feature correctness and missing values (PLAN_V2.md)."""
import hashlib
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src.canonical import MAX_DAYS
from src.features import design as design_v1, build_features, elo_ratings
from src.pairs import assign_ab
from src.v2 import features as F
from src.v2 import markov
from src.v2.data import STATS
from tests.helpers import synthetic


def synth_v2(n=700, seed=0, players=30):
    """Synthetic V2 table: canonical rows plus consistent random statistics, best_of and venue."""
    d = synthetic(n, players=players, seed=seed)
    rng = np.random.default_rng(seed + 1)
    for p in ("w", "l"):
        svpt = rng.integers(40, 100, n).astype(float)
        fin = np.round(svpt * rng.uniform(0.5, 0.7, n))
        bpf = rng.integers(0, 10, n).astype(float)
        cols = {"svpt": svpt, "1stIn": fin, "1stWon": np.round(fin * rng.uniform(0.6, 0.8, n)),
                "2ndWon": np.round((svpt - fin) * rng.uniform(0.4, 0.6, n)), "SvGms": rng.integers(8, 16, n).astype(float),
                "bpFaced": bpf, "bpSaved": np.floor(bpf * rng.uniform(0, 1, n)), "ace": rng.integers(0, 15, n).astype(float),
                "df": rng.integers(0, 6, n).astype(float)}
        for s in STATS:
            d[f"{p}_{s}"] = cols[s]
    d["best_of"] = np.where(d.level == "G", 5, 3)
    d["venue"] = "v" + d.tourney_id.str[2:]
    d["has_stats"] = True
    return d


def run(d, seed=2024):
    ab = assign_ab(d, seed).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    rt = elo_ratings(ab, 24, 32, MAX_DAYS)
    fa, fb, ctx = F.build(ab, rt, MAX_DAYS)
    X = F.design(design_v1(ab, rt, *build_features(ab, rt, MAX_DAYS)), fa, fb, ctx)
    return ab, fa, fb, ctx, X


V2_COLS = [c for g in F.GROUPS.values() for c in g]


class TestMarkov(unittest.TestCase):
    def test_symmetry_and_known_values(self):
        rng = np.random.default_rng(0)
        for pa, pb in rng.uniform(0.45, 0.8, (25, 2)):
            for bo in (3, 5):
                self.assertAlmostEqual(markov.match_prob(pa, pb, bo) + markov.match_prob(pb, pa, bo), 1.0, places=12)
        self.assertAlmostEqual(markov.game(0.5), 0.5)
        self.assertAlmostEqual(markov.match_prob(0.62, 0.62), 0.5)
        self.assertGreater(markov.match_prob(0.66, 0.62, 5), markov.match_prob(0.66, 0.62, 3))  # longer format favours the stronger

    def test_game_formula_matches_recursion(self):
        def rec(a, b, p):  # brute force with deuce closed form
            if a >= 4 and a - b >= 2: return 1.0
            if b >= 4 and b - a >= 2: return 0.0
            if a == b == 3: return p * p / (p * p + (1 - p) ** 2)
            return p * rec(a + 1, b, p) + (1 - p) * rec(a, b + 1, p)
        for p in (0.55, 0.62, 0.7):
            self.assertAlmostEqual(markov.game(p), rec(0, 0, p))


class TestCounts(unittest.TestCase):
    def test_hand_example(self):
        me = dict(ace=5, df=2, svpt=60, **{"1stIn": 36, "1stWon": 28, "2ndWon": 12}, SvGms=10, bpSaved=3, bpFaced=5)
        op = dict(ace=1, df=4, svpt=70, **{"1stIn": 40, "1stWon": 26, "2ndWon": 14}, SvGms=11, bpSaved=2, bpFaced=6)
        c = F.counts(me, op)
        self.assertEqual(c["held"], 8)            # 10 service games, 2 breaks conceded
        self.assertEqual(c["svwon"], 40)
        self.assertEqual(c["rwon"], 30)           # 70 - 26 - 14
        self.assertEqual(c["r1won"], 14); self.assertEqual(c["r2pts"], 30); self.assertEqual(c["r2won"], 16)
        self.assertEqual(c["brk"], 4); self.assertEqual(c["bpc"], 6); self.assertEqual(c["rg"], 11)


class TestChronology(unittest.TestCase):
    def test_future_results_and_statistics_change_nothing_before(self):
        d = synth_v2()
        cut = pd.Timestamp("2015-06-01")
        base = run(d)
        d2 = d.copy()
        fut = d2.event_date >= cut
        d2.loc[fut, ["winner_id", "loser_id"]] = d2.loc[fut, ["loser_id", "winner_id"]].to_numpy()  # flip future results
        for s in STATS:  # and scramble future statistics
            d2.loc[fut, f"w_{s}"] = d2.loc[fut, f"l_{s}"].to_numpy()
        alt = run(d2)
        past = base[0].match_id[base[0].event_date < cut]
        for i in range(1, 5):
            pd.testing.assert_frame_equal(base[i].loc[past], alt[i].loc[past])

    def test_own_tournament_is_never_visible(self):
        d = synth_v2(seed=3)
        t = d.tourney_id.value_counts().index[0]
        first_day = d.loc[d.tourney_id == t, "event_date"].min()
        rows = (d.tourney_id == t) & (d.event_date == first_day)
        base = run(d)
        d2 = d.copy()
        d2.loc[rows, ["winner_id", "loser_id"]] = d2.loc[rows, ["loser_id", "winner_id"]].to_numpy()
        d2.loc[rows, "w_ace"] = 50.0
        alt = run(d2)
        same_day = base[0].match_id[base[0].event_date <= first_day]   # includes later rounds of that tournament
        for i in range(1, 5):
            pd.testing.assert_frame_equal(base[i].loc[same_day], alt[i].loc[same_day])

    def test_pace_uses_previous_editions_only(self):
        d = synth_v2(seed=4)
        d["venue"] = np.where(d.event_date.dt.year == 2015, "same", d.venue)  # one venue with many 'editions'
        d["level"] = "A"
        days = sorted(d.loc[d.venue == "same", "event_date"].unique())
        edition = d.event_date == days[5]
        base = run(d)
        d2 = d.copy(); d2.loc[edition & (d2.venue == "same"), "w_ace"] = 60.0
        alt = run(d2)
        cur = base[0].match_id[base[0].event_date <= days[5]]
        pd.testing.assert_series_equal(base[3].loc[cur, "pace_ace"], alt[3].loc[cur, "pace_ace"])
        later = base[0].match_id[(base[0].event_date > days[5] + pd.Timedelta(days=7)) & (base[0].venue == "same")]
        self.assertFalse(np.allclose(base[3].loc[later, "pace_ace"], alt[3].loc[later, "pace_ace"]))


    def test_pace_index_stays_plausible(self):
        # regression: statistics from before any tour average existed once pushed early venues' pace far off
        ctx = run(synth_v2(seed=11))[3]
        self.assertLess(ctx.pace_ace.abs().max(), 0.1)
        self.assertLess(ctx.pace_spw.abs().max(), 0.1)


class TestSymmetry(unittest.TestCase):
    def test_swapping_players_flips_every_v2_input(self):
        d = synth_v2(seed=7)
        ab, *_, X1 = run(d, seed=2024)
        _, *_, X2 = run(d, seed=99)  # a different hash seed gives a different A/B order
        a1 = ab.set_index("match_id").a_id
        a2 = assign_ab(d, 99).set_index("match_id").a_id.loc[a1.index]
        sign = np.where(a1 == a2, 1.0, -1.0)
        self.assertTrue((sign < 0).any() and (sign > 0).any())
        np.testing.assert_allclose(X1.loc[a1.index, V2_COLS].to_numpy(), X2.loc[a1.index, V2_COLS].to_numpy() * sign[:, None], atol=1e-9)


class TestValuesAndMissing(unittest.TestCase):
    def test_no_missing_or_infinite_inputs(self):
        d = synth_v2(seed=8)
        miss = d.index % 7 == 0
        for p in ("w", "l"):
            for s in STATS:
                d.loc[miss, f"{p}_{s}"] = np.nan
        d.loc[miss, "has_stats"] = False
        X = run(d)[4]
        self.assertTrue(np.isfinite(X[V2_COLS].to_numpy()).all())

    def test_missing_statistics_add_nothing(self):
        d = synth_v2(seed=9)
        base = run(d)
        d2 = d.copy()
        rows = d2.index % 5 == 0
        d2.loc[rows, "has_stats"] = False
        d2.loc[rows, "w_ace"] = 99.0  # ignored because the row is flagged as missing
        d3 = d2.copy(); d3.loc[rows, "w_ace"] = 0.0
        pd.testing.assert_frame_equal(run(d2)[1], run(d3)[1])
        self.assertFalse(base[1].equals(run(d2)[1]))

    def test_hand_computed_serve_rating_and_h2h(self):
        # m0 (other players) creates the tour average; x beats y in m1; they meet again 30 days later in m2
        d = synth_v2(3, seed=10)
        d["level"] = "A"; d["surface"] = "Hard"; d["round_rank"] = 1; d["tourney_id"] = ["t:0", "t:1", "t:2"]; d["venue"] = ["v0", "v1", "v2"]
        d["event_date"] = [pd.Timestamp("2015-01-05"), pd.Timestamp("2015-01-19"), pd.Timestamp("2015-02-18")]
        d["winner_id"] = ["z", "x", "x"]; d["loser_id"] = ["w", "y", "y"]
        ab, fa, fb, ctx, _ = run(d)
        m2 = ab.iloc[2]
        fx = (fa if m2.a_id == "x" else fb).loc[m2.match_id]
        m0, m1 = d.iloc[0], d.iloc[1]
        mu = (m0.w_1stWon + m0.w_2ndWon + m0.l_1stWon + m0.l_2ndWon) / (m0.w_svpt + m0.l_svpt)  # tour average after m0
        spw = (m1.w_1stWon + m1.w_2ndWon) / m1.w_svpt
        g = 2.0 ** (-30 / 240) * m1.w_svpt
        self.assertAlmostEqual(fx.srv, (spw - mu) * g / (g + 300))   # y had no return rating yet
        h = ctx.loc[m2.match_id]
        self.assertAlmostEqual(h.h2h * (1 if m2.a_id == "x" else -1), 0.5 / (1 + 5))   # first meeting: Elo expected 50%
        self.assertEqual(h.h2h_n, 1)
        first = ctx.loc[ab.iloc[1].match_id]
        self.assertEqual((first.h2h, first.h2h_n), (0.0, 0))


class TestV1Untouched(unittest.TestCase):
    def test_frozen_specification_unchanged(self):
        h = hashlib.sha256(Path("models/frozen_phase5.json").read_bytes()).hexdigest()[:16]
        self.assertEqual(h, FROZEN_SHA)


FROZEN_SHA = "16fe577b12eb96d3"  # models/frozen_phase5.json as committed in Phase 5
