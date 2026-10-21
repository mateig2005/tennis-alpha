"""Phase 11: V2 artifact, V2 predictor and API integration (reproducibility, symmetry, surfaces, Markov inputs, missing data, V1 preserved)."""
import json
import math
import os
import pickle
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from src.api import ARTIFACT, V2_ARTIFACT, V2_MISSING, create_app
from src.build_artifacts import build as build_v1
from src.predictor import SURFACES, Predictor
from src.v2 import markov
from src.v2.build_artifact import build as build_v2
from src.v2.data import STATS
from src.v2.predictor import GROUP_LABELS, V2Predictor
from tests.test_v21 import synth_q
JUNE = Path("artifacts/snapshot_2026-06-08")  # the preserved June 2026 artifacts: frozen results must keep reproducing from them

FROZEN_V1 = json.loads(Path("models/frozen_phase5.json").read_text())
FROZEN_V2 = json.loads(Path("models/frozen_v2.json").read_text())
SINNER, ALCARAZ = "sackmann:206173", "sackmann:207989"


def synthetic_models():
    d = synth_q(900, seed=5)
    d["level"] = np.where(d.level == "D", "A", d.level)
    d["best_of"] = np.where(d.level == "G", 5, 3)
    # 'p_nostats' plays a few matches that have no statistics at all
    ns = d.iloc[:3].copy()
    ns["match_id"] = ["t:ns0", "t:ns1", "t:ns2"]; ns["winner_id"] = "p_nostats"; ns["loser_id"] = ["p1", "p2", "p3"]
    ns["has_stats"] = False
    for p in ("w", "l"):
        for s in STATS:
            ns[f"{p}_{s}"] = np.nan
    d = pd.concat([d, ns], ignore_index=True)
    P = Predictor(build_v1(d, FROZEN_V1))
    return d, P, V2Predictor(build_v2(d, FROZEN_V2), P)


class TestV2OnSyntheticData(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d, cls.P, cls.V = synthetic_models()
        cls.ctx = TestClient(create_app(cls.P, h2h={}, v2=cls.V)); cls.c = cls.ctx.__enter__()
        cls.ids = [p for p in cls.P.art["players"].index if p != "p_nostats"]

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def api(self, a, b, surface="Hard", best_of=3):
        r = self.c.get("/predict", params={"a": a, "b": b, "surface": surface, "best_of": best_of}); self.assertEqual(r.status_code, 200, r.text); return r.json()

    def test_artifact_is_reproducible(self):
        again = build_v2(self.d, FROZEN_V2)
        self.assertEqual(again["coef"], self.V.art["coef"]); self.assertEqual(again["scale"], self.V.art["scale"])
        self.assertEqual(pickle.dumps(again["stat_matches"]), pickle.dumps(self.V.art["stat_matches"]))
        a, b = self.ids[:2]
        self.assertEqual(V2Predictor(again, self.P).predict(a, b, "Clay")["p"], self.V.predict(a, b, "Clay")["p"])

    def test_api_matches_standalone_and_symmetry(self):
        for a, b in ((self.ids[0], self.ids[1]), (self.ids[2], self.ids[5])):
            for s in SURFACES:
                for bo in (3, 5):
                    v = self.api(a, b, s, bo)["v2"]
                    self.assertEqual(v["status"], "ok")
                    self.assertAlmostEqual(v["p"], self.V.predict(a, b, s, bo)["p"], places=12)
                    self.assertAlmostEqual(v["p"] + self.api(b, a, s, bo)["v2"]["p"], 1.0, places=12)

    def test_surface_and_format_change_the_estimate(self):
        a, b = self.ids[:2]
        ps = {s: self.api(a, b, s)["v2"]["p"] for s in SURFACES}
        self.assertEqual(len(set(ps.values())), 3)
        p3, p5 = self.api(a, b, "Hard", 3)["v2"], self.api(a, b, "Hard", 5)["v2"]
        self.assertEqual((p3["best_of"], p5["best_of"]), (3, 5))
        self.assertGreaterEqual(abs(p5["matchup"]["p_points_model"] - 0.5), abs(p3["matchup"]["p_points_model"] - 0.5))
        self.assertEqual(self.c.get("/predict", params={"a": a, "b": b, "best_of": 4}).status_code, 422)

    def test_service_return_inputs(self):
        a, b = self.ids[:2]
        for s in SURFACES:
            v = self.api(a, b, s, 5)["v2"]
            pa_, pb_ = v["players"]["a"], v["players"]["b"]
            m = v["matchup"]
            self.assertAlmostEqual(m["pa"], m["tour_serve_avg"] + pa_["serve_surface"] - pb_["ret_surface"])
            self.assertAlmostEqual(m["pb"], m["tour_serve_avg"] + pb_["serve_surface"] - pa_["ret_surface"])
            self.assertAlmostEqual(m["p_points_model"], markov.match_prob(m["pa"], m["pb"], 5))

    def test_contributions_add_up(self):
        v = self.api(self.ids[3], self.ids[4], "Grass")["v2"]
        total = sum(c["value"] for c in v["contributions"])
        self.assertAlmostEqual(total, v["logit"], places=9)
        self.assertAlmostEqual(1 / (1 + math.exp(-total)), v["p"], places=12)
        self.assertTrue({c["group"] for c in v["contributions"]} <= set(GROUP_LABELS))
        self.assertIn("Experimental", v["validation"]); self.assertEqual(v["version"], FROZEN_V2["version"])

    def test_player_without_statistics_is_unavailable(self):
        r = self.api("p_nostats", self.ids[0])
        self.assertEqual(r["v2"]["status"], "unavailable"); self.assertIsNone(r["v2"]["p"]); self.assertIn("p_nostats", r["v2"]["reason"])
        self.assertEqual(r["p"], self.P.predict("p_nostats", self.ids[0], "Hard")["p"])   # V1 still answers
        self.assertIsNone(r["v2"]["players"]["a"]["quality"])

    def test_v1_numbers_unchanged_by_v2(self):
        a, b = self.ids[:2]
        with TestClient(create_app(self.P, h2h={}, v2_artifact=Path("does/not/exist.pkl"))) as c:
            bare = c.get("/predict", params={"a": a, "b": b, "surface": "Clay"}).json()
        self.assertEqual(bare["v2"], {**bare["v2"], "status": "unavailable", "reason": V2_MISSING, "p": None})
        withv2 = self.api(a, b, "Clay")
        for k in ("p", "logit", "contributions", "notes", "reliable", "a", "b"):
            self.assertEqual(bare[k], withv2[k])
        self.assertEqual([m["key"] for m in self.c.get("/meta").json()["models"]], ["D", "C", "Elo", "V2"])

    def test_snapshot_mismatch_is_refused(self):
        art = dict(self.V.art, snapshot_day=self.V.art["snapshot_day"] - 1)
        with self.assertRaises(ValueError):
            V2Predictor(art, self.P)


@unittest.skipUnless(os.path.exists(JUNE / "v2.pkl"), "June 2026 artifacts not present")
class TestV2OnRealSnapshot(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.P = Predictor.load(JUNE / "dashboard.pkl"); cls.V = V2Predictor.load(cls.P, JUNE / "v2.pkl")
        cls.ctx = TestClient(create_app(cls.P, h2h={}, v2=cls.V)); cls.c = cls.ctx.__enter__()

    @classmethod
    def tearDownClass(cls):
        cls.ctx.__exit__(None, None, None)

    def test_sinner_alcaraz_all_surfaces(self):
        v1 = {"Hard": 0.6878, "Clay": 0.6069, "Grass": 0.5977}  # Model D at the snapshot (Phase 9/10): must not move
        for s in SURFACES:
            r = self.c.get("/predict", params={"a": SINNER, "b": ALCARAZ, "surface": s}).json()
            self.assertAlmostEqual(r["p"]["D"], v1[s], places=4)
            self.assertEqual(r["v2"]["status"], "ok")
            self.assertAlmostEqual(r["v2"]["p"], self.V.predict(SINNER, ALCARAZ, s)["p"], places=12)
            self.assertNotAlmostEqual(r["v2"]["p"], r["p"]["D"], places=3)   # V2 is its own calculation
        from src.v2.build_artifact import FROZEN, sha
        self.assertEqual(self.V.art["provenance"]["frozen_v2_sha256_16"], sha(FROZEN))   # artifact built from the frozen file on disk
        self.assertEqual(self.V.art["cols"], FROZEN_V2["model"]["features"]); self.assertEqual(self.V.art["C"], FROZEN_V2["model"]["C"])
