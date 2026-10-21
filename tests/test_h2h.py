import json
import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src import h2h
from src.api import ARTIFACT, H2H_ARTIFACT, create_app
from src.build_artifacts import build
from src.predictor import Predictor
from tests.helpers import synthetic
JUNE = Path("artifacts/snapshot_2026-06-08")  # the preserved June 2026 artifacts: frozen results must keep reproducing from them

FROZEN = json.loads(Path("models/frozen_phase5.json").read_text())
A, B, C, D = "sackmann:1", "sackmann:2", "sackmann:3", "sackmann:4"
SINNER, ALCARAZ = "sackmann:206173", "sackmann:207989"


def match(date, rr, winner, loser, surface="Hard", score="6-4 6-4", retired=False, name="Test Open", level="A", rnd="R32"):
    return dict(tourney_name=name, tourney_level=level, tourney_date=date, round=rnd, round_rank=rr, surface=surface, score=score,
                retired=retired, winner_id=winner, loser_id=loser)


def frame():
    """Hand-made history, deliberately out of order. Player 1 v player 2: six meetings."""
    return pd.DataFrame([
        match("2024-06-10", 7, 2, 1, "Grass", "7-6(3) 6-4", name="Queens", rnd="F"),
        match("2021-02-01", 3, 1, 2, "Hard"),
        match("2023-11-12", 7, 1, 2, "Hard", "6-3 6-3", name="Tour Finals", level="F", rnd="F"),      # same tournament, later round
        match("2023-11-12", 0, 2, 1, "Hard", "7-5 6-7(4) 7-6(2)", name="Tour Finals", level="F", rnd="RR"),
        match("2022-05-08", 5, 1, 2, "Clay", "6-2 2-1 RET", retired=True, name="Rome Masters", level="M", rnd="QF"),
        match("2019-09-13", 0, 2, 1, np.nan, "6-4 6-4", name="Davis Cup", level="D", rnd="RR"),         # surface not recorded
        match("2022-01-10", 4, 1, 3, "Hard"),                                                       # other opponents: must not leak in
        match("2022-01-17", 4, 3, 2, "Hard"),
    ])


class TestHeadToHeadRecords(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.index = h2h.build_index(frame())

    def test_totals_and_wins(self):
        r = h2h.summary(self.index, A, B)
        self.assertEqual((r["total"], r["a_wins"], r["b_wins"]), (6, 3, 3))
        self.assertAlmostEqual(r["a_win_pct"], 0.5)
        self.assertEqual(r["a_wins"] + r["b_wins"], r["total"])
        self.assertTrue(all({m["winner_id"]} <= {A, B} for m in r["matches"]))               # only these two players

    def test_swapping_players_swaps_only_the_win_counts(self):
        x, y = h2h.summary(self.index, A, B), h2h.summary(self.index, B, A)
        self.assertEqual((x["a_wins"], x["b_wins"]), (y["b_wins"], y["a_wins"]))
        self.assertEqual(x["matches"], y["matches"]); self.assertEqual(x["total"], y["total"])
        self.assertAlmostEqual(x["a_win_pct"] + y["a_win_pct"], 1.0)
        for s in h2h.SURFACES:
            self.assertEqual((x["by_surface"][s]["a_wins"], x["by_surface"][s]["b_wins"]), (y["by_surface"][s]["b_wins"], y["by_surface"][s]["a_wins"]))

    def test_surface_counts(self):
        r = h2h.summary(self.index, A, B)
        self.assertEqual(r["by_surface"], {"Hard": {"matches": 3, "a_wins": 2, "b_wins": 1}, "Clay": {"matches": 1, "a_wins": 1, "b_wins": 0},
                                           "Grass": {"matches": 1, "a_wins": 0, "b_wins": 1}})
        self.assertEqual(r["other_surface"], 1)                                              # the Davis Cup tie with no surface
        self.assertEqual(sum(s["matches"] for s in r["by_surface"].values()) + r["other_surface"], r["total"])

    def test_history_is_most_recent_first_and_rounds_order_a_tournament(self):
        ms = h2h.summary(self.index, A, B)["matches"]
        self.assertEqual([str(m["date"]) for m in ms], ["2024-06-10", "2023-11-12", "2023-11-12", "2022-05-08", "2021-02-01", "2019-09-13"])
        self.assertEqual([m["round"] for m in ms[1:3]], ["F", "RR"])                           # the final was played after the round robin
        r = h2h.summary(self.index, A, B)
        self.assertEqual(r["last_meeting"], ms[0]); self.assertEqual(r["last_meeting"]["winner_id"], B)
        self.assertEqual(r["last5"], ms[:5]); self.assertEqual(len(r["last5"]), 5)

    def test_no_previous_meetings(self):
        r = h2h.summary(self.index, C, D)
        self.assertEqual((r["total"], r["a_wins"], r["b_wins"], r["matches"], r["last5"]), (0, 0, 0, [], []))
        self.assertIsNone(r["a_win_pct"]); self.assertIsNone(r["last_meeting"])                # no percentage is invented
        self.assertEqual(h2h.summary(self.index, A, "sackmann:999")["total"], 0)

    def test_retirements_count_and_are_flagged(self):
        r = h2h.summary(self.index, A, B)
        ret = [m for m in r["matches"] if m["retired"]]
        self.assertEqual(r["retirements"], 1); self.assertEqual(len(ret), 1)
        self.assertEqual((ret[0]["winner_id"], ret[0]["score"]), (A, "6-2 2-1 RET"))           # counted as a win, score kept as recorded
        self.assertEqual(r["a_wins"], 3)

    def test_missing_surface_stays_missing(self):
        dc = next(m for m in h2h.summary(self.index, A, B)["matches"] if m["tournament"] == "Davis Cup")
        self.assertIsNone(dc["surface"]); self.assertEqual(dc["level"], "Davis Cup")

    def test_every_match_is_indexed_once(self):
        self.assertEqual(sum(len(v) for v in self.index.values()), len(frame()))


class TestHeadToHeadApi(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = synthetic(300, players=12, seed=3); d["level"] = np.where(d.level == "D", "A", d.level)
        cls.P = Predictor(build(d, FROZEN)); cls.ids = list(cls.P.art["players"].index)
        a, b = cls.ids[0], cls.ids[1]
        rows = frame().replace({"winner_id": {1: 101, 2: 102}, "loser_id": {1: 101, 2: 102}})
        cls.index = {(min(a, b), max(a, b)): [dict(m, winner_id=a if m["winner_id"] == "sackmann:101" else b)
                                              for m in h2h.build_index(rows)[("sackmann:101", "sackmann:102")]]}
        cls.ctx = TestClient(create_app(cls.P, h2h=cls.index)); cls.c = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_response_schema_and_values(self):
        a, b = self.ids[0], self.ids[1]
        r = self.c.get("/h2h", params={"a": a, "b": b}); self.assertEqual(r.status_code, 200, r.text); r = r.json()
        self.assertEqual(set(r), {"a_id", "b_id", "total", "a_wins", "b_wins", "a_win_pct", "by_surface", "other_surface", "retirements",
                                  "last_meeting", "last5", "matches", "data_start", "data_end", "note"})
        self.assertEqual((r["a_id"], r["b_id"], r["total"], r["a_wins"], r["b_wins"], r["retirements"]), (a, b, 6, 3, 3, 1))
        self.assertEqual(set(r["matches"][0]), {"date", "tournament", "level", "round", "surface", "score", "winner_id", "retired"})
        self.assertEqual(r["matches"][0]["date"], "2024-06-10"); self.assertIsNone(r["matches"][-1]["surface"])
        self.assertIn("not an input to any model", r["note"]); self.assertIn("before the data starts and after the data cutoff are missing", r["note"])
        self.assertEqual((r["data_start"], r["data_end"]), (str(self.P.art["data_start"]), str(self.P.art["data_end"])))

    def test_symmetry_and_never_met(self):
        a, b, c = self.ids[0], self.ids[1], self.ids[2]
        x, y = self.c.get("/h2h", params={"a": a, "b": b}).json(), self.c.get("/h2h", params={"a": b, "b": a}).json()
        self.assertEqual((x["a_wins"], x["b_wins"], x["matches"]), (y["b_wins"], y["a_wins"], y["matches"]))
        z = self.c.get("/h2h", params={"a": a, "b": c}).json()
        self.assertEqual((z["total"], z["a_win_pct"], z["last_meeting"], z["matches"]), (0, None, None, []))

    def test_bad_requests(self):
        a = self.ids[0]
        self.assertEqual(self.c.get("/h2h", params={"a": a, "b": a}).status_code, 400)
        self.assertEqual(self.c.get("/h2h", params={"a": a, "b": "nobody"}).status_code, 404)
        self.assertEqual(self.c.get("/h2h", params={"a": a}).status_code, 422)

    def test_missing_index_is_a_clear_503_and_predictions_still_work(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with TestClient(create_app(self.P, h2h_artifact=Path("does/not/exist.pkl"))) as c:
                r = c.get("/h2h", params={"a": self.ids[0], "b": self.ids[1]})
                self.assertEqual(r.status_code, 503); self.assertIn("python -m src.h2h", r.json()["detail"])
                self.assertEqual(c.get("/predict", params={"a": self.ids[0], "b": self.ids[1]}).status_code, 200)

    def test_head_to_head_does_not_change_predictions(self):
        a, b = self.ids[0], self.ids[1]
        with TestClient(create_app(self.P, h2h={})) as bare:
            self.assertEqual(bare.get("/predict", params={"a": a, "b": b}).json()["p"], self.c.get("/predict", params={"a": a, "b": b}).json()["p"])
        self.assertEqual(self.c.get("/predict", params={"a": a, "b": b}).json()["p"], self.P.predict(a, b, "Hard")["p"])


@unittest.skipUnless(os.path.exists(JUNE / "h2h.pkl"), "June 2026 artifacts not present")
class TestHeadToHeadOnRealData(unittest.TestCase):
    def test_real_matchup_equals_an_independent_count_from_the_match_files(self):
        df = h2h.load_matches()
        pair = df[((df.winner_id == 206173) & (df.loser_id == 207989)) | ((df.winner_id == 207989) & (df.loser_id == 206173))]
        with TestClient(create_app(artifact=JUNE / "dashboard.pkl", h2h_artifact=JUNE / "h2h.pkl", v2_artifact=JUNE / "v2.pkl")) as c:
            r = c.get("/h2h", params={"a": SINNER, "b": ALCARAZ}).json()
        self.assertEqual(r["total"], len(pair)); self.assertEqual(r["a_wins"], int((pair.winner_id == 206173).sum()))
        self.assertEqual(r["b_wins"], int((pair.winner_id == 207989).sum())); self.assertEqual(r["retirements"], int(pair.retired.sum()))
        for s in h2h.SURFACES:
            self.assertEqual(r["by_surface"][s]["matches"], int((pair.surface == s).sum()))
        self.assertEqual(r["matches"][0]["date"], pair.tourney_date.max()); self.assertGreater(r["total"], 0)
        self.assertFalse(pair.score.str.contains("W/O|DEF|Walkover", case=False).any())      # excluded upstream, as in the research data


if __name__ == "__main__":
    unittest.main()
