"""Phase 14: TennisMyLife adapter (synthetic rows only; no real or protected data is read)."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import tml
from src.v2.data import STATS


def row(tid, date, rnd, w, l, wn, ln, score="6-4 6-4", level="250", num=1, **kw):
    r = {"tourney_id": tid, "tourney_name": "T", "surface": "Hard", "draw_size": 32, "tourney_level": level, "indoor": "O",
         "tourney_date": date, "match_num": num, "winner_id": w, "winner_seed": np.nan, "winner_entry": np.nan, "winner_name": wn,
         "winner_hand": "R", "winner_ht": 185, "winner_ioc": "ITA", "winner_age": 24, "winner_rank": 1, "winner_rank_points": 9000,
         "loser_id": l, "loser_seed": np.nan, "loser_entry": np.nan, "loser_name": ln, "loser_hand": "R", "loser_ht": 183,
         "loser_ioc": "ESP", "loser_age": 22, "loser_rank": 2, "loser_rank_points": 8000, "score": score, "best_of": 3, "round": rnd, "minutes": 90}
    for p in "wl":
        r |= {f"{p}_ace": 5, f"{p}_df": 2, f"{p}_svpt": 60, f"{p}_1stIn": 36, f"{p}_1stWon": 28, f"{p}_2ndWon": 12,
              f"{p}_SvGms": 10, f"{p}_bpSaved": 3, f"{p}_bpFaced": 5}
    return r | kw


def frame(rows):
    df = pd.DataFrame(rows)
    df["tourney_date"] = pd.to_datetime(df.tourney_date)
    return df


class TestSplitAndClean(unittest.TestCase):
    def test_split_uses_tournament_start_not_row_date(self):
        rows = [row("2026-520", "20260524", "R128", "A1", "B1", "Ann One", "Bob One"),
                row("2026-520", "20260601", "F", "A1", "B2", "Ann One", "Bob Two"),        # same tournament, later row date
                row("2026-0500", "20260601", "R32", "A2", "B3", "Ann Two", "Bob Three")]  # tournament starting after the cutoff
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / "2026.csv"
            pd.DataFrame(rows).to_csv(p, index=False)
            known, protected = tml.read_split(p, pd.Timestamp("2026-05-25"))
        self.assertEqual(len(known), 2); self.assertEqual(len(protected), 1)
        self.assertTrue((known.tourney_date == pd.Timestamp("2026-05-24")).all())   # rows carry the tournament start

    def test_clean_levels_ids_and_filters(self):
        df = frame([row("2025-339", "20250101", "R32", "A1", "B1", "a", "b", level="500"),
                    row("2025-339", "20250101", "R16", "A1", "B2", "a", "c", level="500", num=np.nan),
                    row("2025-339", "20250101", "QF", "A1", "B3", "a", "d", score="W/O"),
                    row("2025-339", "20250101", np.nan, "A1", "B4", "a", "e"),
                    row("2025-339", "20250101", "SF", "A1", "B5", "a", "f", score="6-3 2-1 RET", level="M")])
        c = tml.clean(df)
        self.assertEqual(len(c), 3)                                    # walkover and no-round rows dropped
        self.assertEqual(set(c.tourney_level), {"A", "M"})
        self.assertEqual(c.match_id.iloc[1], "2025-339_R16_A1-B2")    # stable without match_num
        self.assertTrue(c.retired.iloc[2])
        with self.assertRaises(AssertionError):
            tml.clean(frame([row("2025-1", "20250101", "R32", "A", "B", "a", "b", level="X")]))

    def test_tournament_id_padding(self):
        self.assertEqual(tml.norm_tid(pd.Series(["2024-0301", "2024-301", "2024-M020"])).tolist(), ["2024-301", "2024-301", "2024-M020"])


class TestLinkAndValidate(unittest.TestCase):
    def setUp(self):
        self.t = tml.clean(frame([row("2025-339", "20250106", "R32", "A1", "B1", "Jannik Sinner", "Carlos Alcaraz"),
                                  row("2025-339", "20250106", "R16", "A1", "B2", "Jannik Sinner", "Novak Djokovic"),
                                  row("2025-339", "20250106", "QF", "B2", "B3", "Novak Djokovic", "Taylor Fritz")]))
        s = self.t.copy()
        s["tourney_id"] = "2025-0339"; s["tourney_level"] = "A"
        s["winner_id"] = s.winner_id.map({"A1": 206173, "B2": 104925}); s["loser_id"] = s.loser_id.map({"B1": 207989, "B2": 104925, "B3": 126203})
        s["match_id"] = ["s1", "s2", "s3"]
        self.s = s

    def test_perfect_overlap_passes_and_maps_players(self):
        met, ok, m = tml.validate_overlap(self.t, self.s)
        self.assertTrue(ok["linked_share"] and ok["winner_agreement"] and ok["stats_equal_share"], met)
        self.assertFalse(ok["stats_coverage_2026"])   # no 2026 rows here: coverage cannot pass by default
        pm, bad = tml.player_map(m)
        self.assertEqual(pm, {"A1": "206173", "B1": "207989", "B2": "104925", "B3": "126203"}); self.assertEqual(bad, [])

    def test_disagreements_are_measured(self):
        s = self.s.copy()
        s.loc[0, ["winner_name", "loser_name"]] = ["Carlos Alcaraz", "Jannik Sinner"]   # the sources disagree on a winner
        s.loc[1, "w_ace"] = 20                                                           # and on a statistic
        met, ok, _ = tml.validate_overlap(self.t, s)
        self.assertAlmostEqual(met["winner_agreement"], 2 / 3); self.assertFalse(ok["winner_agreement"])
        self.assertLess(met["stats_equal_share"], 1)

    def test_conflicting_ids_are_not_mapped(self):
        m = tml.link(self.t, self.s)[0]
        m.loc[m.index[1], "winner_id_s"] = 999   # A1 now points at two different Sackmann ids
        pm, bad = tml.player_map(m)
        self.assertIn("A1", bad); self.assertNotIn("A1", pm)

    def test_v2_table_uses_our_ids_and_history_venue(self):
        v2 = tml.to_v2_table(self.t, {"A1": "206173"})
        self.assertEqual(v2.winner_id.iloc[0], "sackmann:206173"); self.assertEqual(v2.loser_id.iloc[0], "tml:B1")
        self.assertTrue(v2.match_id.str.startswith("tml:").all()); self.assertEqual(set(v2.source), {"tml"})
        self.assertEqual(v2.venue.iloc[0], "sackmann:339")                 # same key as the history, so pace history continues
        self.assertTrue(v2.has_stats.all())
        self.assertEqual(v2.w_ace.iloc[0], 5)
