import json
import math
import os
import unittest
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import ARTIFACT, GROUP_LABELS, create_app
from src.build_artifacts import build
from src.features import PRIOR
from src.predictor import SURFACES, Predictor
from tests.helpers import synthetic
JUNE = Path("artifacts/snapshot_2026-06-08")  # the preserved June 2026 artifacts: frozen results must keep reproducing from them

FROZEN = json.loads(Path("models/frozen_phase5.json").read_text())
SINNER, ALCARAZ = "sackmann:206173", "sackmann:207989"


def synthetic_predictor():
    d = synthetic(900, players=30, seed=0)
    d["level"] = np.where(d.level == "D", "A", d.level)
    old = pd.DataFrame([{**d.iloc[0].to_dict(), "match_id": "t:old", "event_date": pd.Timestamp("2012-01-02"), "winner_id": "p_old", "loser_id": "p1"}])
    return Predictor(build(pd.concat([old, d], ignore_index=True), FROZEN))


class TestApiOnSyntheticData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = synthetic_predictor()
        cls.ctx = TestClient(create_app(cls.P)); cls.c = cls.ctx.__enter__()
        cls.ids = [p["id"] for p in cls.c.get("/players").json()["players"]]

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def predict(self, a, b, surface="Hard"):
        r = self.c.get("/predict", params={"a": a, "b": b, "surface": surface}); self.assertEqual(r.status_code, 200, r.text); return r.json()

    def test_health_meta_surfaces(self):
        self.assertEqual(self.c.get("/health").json(), {"status": "ok", "artifact_loaded": True, "v2_loaded": False, "detail": "ready"})
        self.assertEqual(self.c.get("/surfaces").json(), ["Hard", "Clay", "Grass"])
        m = self.c.get("/meta").json()
        self.assertFalse(m["live"]); self.assertIn("not a live forecast", m["disclaimer"]); self.assertGreater(m["snapshot_date"], m["data_end"])
        self.assertEqual([(x["key"], x["role"]) for x in m["models"]], [("D", "primary"), ("C", "challenger"), ("Elo", "benchmark")])
        self.assertEqual(m["n_players"], len(self.P.art["players"]))

    def test_player_list_schema_search_and_order(self):
        r = self.c.get("/players").json()
        self.assertEqual(r["total"], len(r["players"]))
        self.assertEqual(set(r["players"][0]), {"id", "name", "rank", "elo", "matches", "last_event", "days_since_last_event", "active", "stale"})
        active = [p["active"] for p in r["players"]]
        self.assertEqual(active, sorted(active, reverse=True))                                   # active players first
        elo = [p["elo"] for p in r["players"] if p["active"]]; self.assertEqual(elo, sorted(elo, reverse=True))
        self.assertEqual(len({p["id"] for p in r["players"]}), r["total"])                        # ids are unique
        self.assertEqual([p["id"] for p in self.c.get("/players", params={"q": "P_OLD"}).json()["players"]], ["p_old"])   # case-insensitive
        self.assertEqual(self.c.get("/players", params={"q": "nobody-by-this-name"}).json(), {"total": 0, "players": []})
        self.assertNotIn("p_old", [p["id"] for p in self.c.get("/players", params={"active_only": True}).json()["players"]])
        self.assertEqual(len(self.c.get("/players", params={"limit": 3}).json()["players"]), 3)

    def test_prediction_equals_the_python_predictor(self):
        for s in SURFACES:
            a, b = self.ids[0], self.ids[4]
            api, ref = self.predict(a, b, s), self.P.predict(a, b, s)
            self.assertEqual(api["p"], ref["p"])                                                 # exact: same code path, lossless JSON floats
            self.assertEqual(api["notes"], ref["notes"]); self.assertEqual(api["reliable"], ref["reliable"]); self.assertEqual(api["surface"], s)
            self.assertEqual({c["group"]: c["value"] for c in api["contributions"]}, ref["contributions"]["D"])

    def test_contributions_are_the_trained_coefficients_times_real_inputs(self):
        a, b = self.ids[2], self.ids[9]
        api, ref, m = self.predict(a, b, "Clay"), self.P.predict(a, b, "Clay"), self.P.art["models"]["D"]
        terms = np.array(m["coef"]) * ref["diff"][m["features"]].to_numpy(float) / np.array(m["scale"])
        self.assertAlmostEqual(api["logit"], float(terms.sum()), places=12)
        self.assertAlmostEqual(1 / (1 + math.exp(-api["logit"])), api["p"]["D"], places=12)      # bars sum to the shown probability
        self.assertEqual([c["label"] for c in api["contributions"]], [GROUP_LABELS[c["group"]] for c in api["contributions"]])
        self.assertEqual(len(api["contributions"]), 9)

    def test_swapping_players_gives_complementary_probabilities(self):
        for s in SURFACES:
            x, y = self.predict(self.ids[1], self.ids[7], s), self.predict(self.ids[7], self.ids[1], s)
            for k in ("D", "C", "Elo"):
                self.assertAlmostEqual(x["p"][k] + y["p"][k], 1.0, places=12); self.assertTrue(0 < x["p"][k] < 1)
            self.assertAlmostEqual(x["logit"], -y["logit"], places=12)
            self.assertEqual(x["a"]["id"], y["b"]["id"]); self.assertEqual(x["a"]["form"], y["b"]["form"])

    def test_surface_changes_the_inputs(self):
        a, b = self.ids[0], self.ids[3]
        out = {s: self.predict(a, b, s) for s in SURFACES}
        self.assertEqual(len({out[s]["a"]["inputs"]["swr365"] for s in SURFACES} | {out[s]["p"]["D"] for s in SURFACES}) > 2, True)
        self.assertEqual(len({out[s]["p"]["Elo"] for s in SURFACES}), 1)                          # the Elo benchmark ignores surface
        self.assertEqual(self.c.get("/predict", params={"a": a, "b": b, "surface": "Carpet"}).status_code, 422)

    def test_bad_requests(self):
        a = self.ids[0]
        r = self.c.get("/predict", params={"a": a, "b": a}); self.assertEqual(r.status_code, 400); self.assertIn("different", r.json()["detail"])
        r = self.c.get("/predict", params={"a": a, "b": "nobody"}); self.assertEqual(r.status_code, 404); self.assertIn("Unknown player", r.json()["detail"])
        self.assertEqual(self.c.get("/players/nobody").status_code, 404)
        self.assertEqual(self.c.get("/predict", params={"a": a}).status_code, 422)

    def test_inactive_player_is_flagged_not_hidden(self):
        r = self.predict("p_old", self.ids[0])
        self.assertFalse(r["reliable"]); self.assertFalse(r["a"]["active"]); self.assertTrue(r["a"]["stale"])
        self.assertTrue(any("not an honest current estimate" in n for n in r["notes"]))
        self.assertEqual(r["a"]["form"]["d365"], {"wins": 0, "matches": 0, "rate": None})          # no recent matches: nothing invented

    def test_observed_statistics_agree_with_the_model_inputs(self):
        for pid in self.ids[:8]:
            d = self.c.get(f"/players/{pid}", params={"surface": "Grass"}).json()
            shrunk = lambda rec: (rec["wins"] + PRIOR / 2) / (rec["matches"] + PRIOR)             # the model's own shrinkage
            for key, feature in (("last10", "wr10"), ("d30", "wr30"), ("d90", "wr90"), ("d365", "wr365")):
                self.assertAlmostEqual(shrunk(d["form"][key]), d["inputs"][feature], places=12, msg=f"{pid} {key}")
            self.assertAlmostEqual(shrunk(d["surface_form"]["Grass"]), d["inputs"]["swr365"], places=12)
            self.assertAlmostEqual(math.log1p(d["form"]["d90"]["matches"]), d["inputs"]["ln90"], places=12)
            self.assertEqual(d["adj_form"], d["inputs"]["adj10"]); self.assertEqual(d["surface"], "Grass")
            self.assertEqual(len(d["last5"]), min(5, d["matches"])); self.assertLessEqual(set(d["last5"]), {"W", "L"})
            if d["form"]["d365"]["matches"]:
                self.assertAlmostEqual(d["form"]["d365"]["rate"], d["form"]["d365"]["wins"] / d["form"]["d365"]["matches"])

    def test_rating_history_is_real_and_ends_at_the_current_rating(self):
        for pid in self.ids[:10]:
            d = self.c.get(f"/players/{pid}").json(); h = d["elo_history"]
            self.assertEqual(h[0]["elo"], 1500.0)                                                 # everyone starts at 1500
            self.assertAlmostEqual(h[-1]["elo"], d["elo"], places=6)                              # the path ends at the engine's rating
            dates = [p["date"] for p in h]; self.assertEqual(dates, sorted(set(dates)))            # strictly increasing dates
            self.assertLessEqual(dates[-1], self.c.get("/meta").json()["snapshot_date"])
        total = sum(self.P.art["elo"].values()); self.assertAlmostEqual(total / len(self.P.art["elo"]), 1500.0, places=6)   # Elo is zero-sum

    def test_unrated_surface_is_null_not_invented(self):
        d = self.c.get("/players/p_old").json()
        self.assertIn(None, d["surface_elo"].values())

    def test_missing_artifact_gives_a_clear_503(self):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with TestClient(create_app(artifact=Path("does/not/exist.pkl"))) as c:
                self.assertEqual(c.get("/health").json()["status"], "unavailable")
                r = c.get("/players"); self.assertEqual(r.status_code, 503); self.assertIn("build_artifacts", r.json()["detail"])


@unittest.skipUnless(os.path.exists(JUNE / "dashboard.pkl"), "June 2026 artifacts not present")
class TestApiOnRealSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = Predictor.load(JUNE / "dashboard.pkl")
        cls.ctx = TestClient(create_app(artifact=JUNE / "dashboard.pkl", h2h_artifact=JUNE / "h2h.pkl", v2_artifact=JUNE / "v2.pkl")); cls.c = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_real_players_are_searchable_by_name_with_stable_ids(self):
        for q, pid in (("sinner", SINNER), ("Carlos Alcaraz", ALCARAZ), ("alcaraz carlos", ALCARAZ)):
            hits = self.c.get("/players", params={"q": q}).json()["players"]
            self.assertIn(pid, [p["id"] for p in hits], msg=q)
        self.assertEqual(self.c.get(f"/players/{SINNER}").json()["name"], "Jannik Sinner")

    def test_api_matches_the_streamlit_predictor_on_every_surface(self):
        for s in SURFACES:
            api = self.c.get("/predict", params={"a": SINNER, "b": ALCARAZ, "surface": s}).json()
            self.assertEqual(api["p"], self.P.predict(SINNER, ALCARAZ, s)["p"])
            back = self.c.get("/predict", params={"a": ALCARAZ, "b": SINNER, "surface": s}).json()
            for k in ("D", "C", "Elo"):
                self.assertAlmostEqual(api["p"][k] + back["p"][k], 1.0, places=12)

    def test_snapshot_metadata(self):
        m = self.c.get("/meta").json()
        self.assertEqual((m["data_end"], m["snapshot_date"], m["live"]), ("2026-05-25", "2026-06-08", False))
        self.assertEqual(m["n_players"], len(self.P.art["players"]))

    def test_long_retired_player_is_marked_unreliable(self):
        fed = next(p for p in self.c.get("/players", params={"q": "roger federer"}).json()["players"])
        self.assertFalse(fed["active"])
        r = self.c.get("/predict", params={"a": fed["id"], "b": SINNER}).json()
        self.assertFalse(r["reliable"]); self.assertTrue(r["notes"])


if __name__ == "__main__":
    unittest.main()
