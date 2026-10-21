import json
import os
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate, identity, models
from src.build_artifacts import build
from src.canonical import MAX_DAYS, SCORED_LEVELS
from src.dataset import ELO_K_GEN, ELO_K_SURF
from src.features import build_features, design, elo_ratings
from src.pairs import assign_ab
from src.predictor import ARTIFACT, SURFACES, Predictor
from tests.helpers import synthetic

FROZEN = json.loads(Path("models/frozen_phase5.json").read_text())


def raw_data(seed=0):
    d = synthetic(900, players=30, seed=seed)
    d["level"] = np.where(d.level == "D", "A", d.level)
    return d


class TestPredictionInterface(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = raw_data(); cls.P = Predictor(build(cls.raw, FROZEN)); cls.ids = list(cls.P.players(active_only=False).index)

    def test_probabilities_are_symmetric_for_every_model_and_surface(self):
        for s in SURFACES:
            for a, b in ((self.ids[0], self.ids[5]), (self.ids[3], "new:B"), ("new:A", "new:B")):
                r, q = self.P.predict(a, b, s), self.P.predict(b, a, s)
                for k in ("D", "C", "Elo"):
                    self.assertAlmostEqual(r["p"][k] + q["p"][k], 1.0, places=12); self.assertTrue(0 < r["p"][k] < 1)

    def test_invalid_requests_are_rejected(self):
        with self.assertRaises(ValueError): self.P.predict(self.ids[0], self.ids[0], "Hard")
        with self.assertRaises(ValueError): self.P.predict(self.ids[0], self.ids[1], "Carpet")

    def test_new_players_are_neutral_and_flagged(self):
        r = self.P.predict("new:A", "new:B", "Clay")
        self.assertEqual({k: round(v, 12) for k, v in r["p"].items()}, {"D": 0.5, "C": 0.5, "Elo": 0.5})
        self.assertFalse(r["reliable"]); self.assertEqual(len(r["notes"]), 2); self.assertIn("no match history", r["notes"][0])
        self.assertEqual(r["A"]["features"]["wr10"], 0.5); self.assertEqual(r["A"]["elo"], 1500.0); self.assertTrue(np.isnan(r["A"]["rank"]))

    def test_features_equal_the_research_pipeline_for_the_same_hypothetical_match(self):
        art = self.P.art
        for a, b, s in ((self.ids[0], self.ids[7], "Hard"), (self.ids[12], self.ids[2], "Grass"), (self.ids[4], self.ids[20], "Clay")):
            r = self.P.predict(a, b, s)
            fake = pd.DataFrame([{**self.raw.iloc[0].to_dict(), "match_id": "t:fake", "tourney_id": "t:fake", "event_date": pd.Timestamp(art["snapshot_date"]),
                                  "round_rank": 1, "surface": s, "level": "A", "winner_id": a, "loser_id": b, "winner_rank": r["A"]["rank"],
                                  "loser_rank": r["B"]["rank"], "winner_points": r["A"]["points"], "loser_points": r["B"]["points"]}])
            ab = assign_ab(pd.concat([self.raw, fake], ignore_index=True)).sort_values(["event_date", "round_rank", "match_id"], kind="stable")
            rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
            D = design(ab, rt, *build_features(ab, rt, MAX_DAYS))
            sign = 1.0 if ab.set_index("match_id").loc["t:fake", "a_id"] == a else -1.0
            np.testing.assert_allclose(r["diff"][D.columns].to_numpy(float), sign * D.loc["t:fake"].to_numpy(float), atol=1e-9)
            # and the stored coefficients reproduce scikit-learn's own prediction
            meta = ab.set_index("match_id").drop(index="t:fake"); Dtr = D.drop(index="t:fake").loc[meta.index]
            tr = (meta.level.isin(SCORED_LEVELS) & (meta.event_date.dt.year >= models.TRAIN_FROM)).to_numpy()
            for key, name in (("D", "D LR"), ("C", "C LR")):
                spec = FROZEN["models"][name]
                m = models.fit(Dtr.loc[tr, spec["features"]], meta.y[tr], spec["C"])
                self.assertAlmostEqual(r["p"][key], float(models.predict(m, r["diff"][spec["features"]].to_frame().T.astype(float))[0]), places=9)

    def test_snapshot_is_after_all_data_and_nothing_later_is_used(self):
        art = self.P.art
        self.assertGreater(pd.Timestamp(art["snapshot_date"]), pd.Timestamp(art["data_end"]))
        last_seen = max(rec[0] for st in art["state"].values() for rec in st.hist)
        self.assertLessEqual(last_seen, art["snapshot_day"])
        self.assertEqual(set(art["models"]), {"D", "C"})

    def test_missing_rank_is_flagged_not_invented(self):
        raw = raw_data(1); pid = raw.winner_id.iloc[0]
        raw.loc[raw.winner_id == pid, ["winner_rank", "winner_points"]] = np.nan; raw.loc[raw.loser_id == pid, ["loser_rank", "loser_points"]] = np.nan
        P = Predictor(build(raw, FROZEN)); other = next(i for i in P.players(False).index if i != pid)
        r = P.predict(pid, other, "Hard")
        self.assertTrue(np.isnan(r["A"]["rank"])); self.assertEqual(r["diff"]["rank_miss"], 1.0)
        self.assertTrue(any("no ranking" in n for n in r["notes"]))

    def test_inactive_player_is_marked_unreliable_and_hidden_by_default(self):
        raw = raw_data(2); pid = "p_old"
        old = pd.DataFrame([{**raw.iloc[0].to_dict(), "match_id": "t:old", "event_date": pd.Timestamp("2012-01-02"), "winner_id": pid, "loser_id": "p1"}])
        P = Predictor(build(pd.concat([old, raw], ignore_index=True), FROZEN))
        self.assertNotIn(pid, P.players().index); self.assertIn(pid, P.players(active_only=False).index)
        r = P.predict(pid, "p2", "Hard")
        self.assertFalse(r["reliable"]); self.assertTrue(any("not an honest current estimate" in n for n in r["notes"]))

    def test_dashboard_code_has_no_betting_odds_or_xgboost(self):
        src = "".join(Path(f).read_text().lower() for f in ("src/predictor.py", "src/build_artifacts.py", "app.py"))
        for word in ("xgboost", "gbm", "cote", "pinnacle", "valuebet"):
            self.assertNotIn(word, src, msg=word)


class TestHelpers(unittest.TestCase):
    def test_markdown_table_round_trip(self):
        df = pd.DataFrame({"model": ["A Elo", "D LR"], "log_loss": [0.6291, 0.6207]})
        back = evaluate.read_md_table("## X\n\n" + evaluate.md(df) + "\n\nmore", "## X")
        self.assertEqual(back.model.tolist(), ["A Elo", "D LR"]); self.assertEqual(back.log_loss.astype(float).tolist(), [0.6291, 0.6207])

    def test_player_identity_mapping_reports_instead_of_guessing(self):
        known = {"k1": "Carlos Alcaraz", "k2": "Felix Auger-Aliassime", "k3": "John Smith", "k4": "John Smith"}
        ext = {"e1": "Carlos Alcaraz", "e2": "Félix Auger Aliassime", "e3": "John Smith", "e4": "Nobody Known", "e5": "carlos alcaraz"}
        mapping, unmatched, ambiguous = identity.map_players(ext, known)
        self.assertEqual(mapping, {"e2": "k2"}); self.assertEqual(unmatched, ["e4"]); self.assertEqual(ambiguous, ["e1", "e3", "e5"])

    def test_extension_validation(self):
        base = synthetic(50, seed=1); new = synthetic(20, seed=2)
        new["match_id"] = "n:" + new.match_id; new["source"] = "new"; new["event_date"] = new.event_date + pd.Timedelta(days=3650)
        self.assertEqual(identity.validate_extension(new, base)["matches"], 20)
        with self.assertRaises(AssertionError): identity.validate_extension(new.assign(event_date=base.event_date.min()), base)
        with self.assertRaises(AssertionError): identity.validate_extension(new.assign(source="t"), base)


@unittest.skipUnless(os.path.exists(ARTIFACT), "dashboard artifact not built; run python -m src.build_artifacts")
class TestStreamlitApp(unittest.TestCase):
    def test_app_runs_shows_the_data_date_and_predicts(self):
        from streamlit.testing.v1 import AppTest
        at = AppTest.from_file(os.path.abspath("app.py"), default_timeout=120).run()
        self.assertFalse(at.exception, msg=str(at.exception))
        art = Predictor.load().art
        banner = " ".join(w.value for w in at.warning)
        self.assertIn(str(art["data_end"]), banner); self.assertIn(str(art["snapshot_date"]), banner); self.assertIn("not a current forecast", banner.lower())
        self.assertEqual(len(at.metric), 3)                                   # Model D, Model C, Elo
        at.selectbox[2].set_value("Clay").run()
        self.assertFalse(at.exception); self.assertEqual(len(at.metric), 3)


if __name__ == "__main__":
    unittest.main()
