"""Phase 15: prospective prediction log, safe artifact refresh, sync failure recovery and data freshness (synthetic data only)."""
import json
import pickle
import tempfile
import unittest
from datetime import datetime, timezone
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd

from src import api, predlog, refresh, sync
from src.build_artifacts import build as build_v1
from src.v2.build_artifact import build as build_v2
from src.v2.predictor import V2Predictor
from tests.test_v2_integration import FROZEN_V1, FROZEN_V2, synthetic_models


def utc(s):
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc)


class TestPredictionLog(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(); self.log = Path(self.dir.name) / "log.jsonl"
        self.entry = {"a_id": "x", "b_id": "y", "a_name": "X", "b_name": "Y", "surface": "Hard", "best_of": 3, "tournament": "T",
                      "tournament_start": "2026-10-26", "match_start_utc": None, "p": {"D": 0.6, "C": 0.6, "Elo": 0.55, "V2": 0.62}}

    def tearDown(self):
        self.dir.cleanup()

    def test_timing_classes(self):
        self.assertEqual(predlog.timing_class("2026-10-25T23:59:00+00:00", "2026-10-26"), "validated")
        self.assertEqual(predlog.timing_class("2026-10-27T10:00:00+00:00", "2026-10-26", "2026-10-27T14:00:00+00:00"), "declared")
        self.assertEqual(predlog.timing_class("2026-10-27T15:00:00+00:00", "2026-10-26", "2026-10-27T14:00:00+00:00"), "uncertain")
        self.assertEqual(predlog.timing_class("2026-10-27T10:00:00+00:00", "2026-10-26"), "uncertain")   # no start time: not prospective

    def test_append_only_and_tamper_evident(self):
        r1 = predlog.append(self.entry, self.log, now=utc("2026-10-20T10:00:00"))
        first_line = self.log.read_text().splitlines()[0]
        r2 = predlog.append(self.entry | {"a_id": "z"}, self.log, now=utc("2026-10-21T10:00:00"))
        self.assertEqual(self.log.read_text().splitlines()[0], first_line)        # never rewritten
        self.assertEqual(r2["prev_hash"], r1["hash"]); self.assertEqual(r1["timing"], "validated")
        self.assertTrue(predlog.verify(self.log))
        lines = self.log.read_text().splitlines()
        self.log.write_text(lines[0].replace('"D": 0.6', '"D": 0.7') + "\n" + lines[1] + "\n")   # edit an old probability
        with self.assertRaises(ValueError):
            predlog.verify(self.log)
        self.log.write_text(lines[1] + "\n")                                     # delete the first record
        with self.assertRaises(ValueError):
            predlog.verify(self.log)

    def test_join_results(self):
        recs = [predlog.append(self.entry, self.log, now=utc("2026-10-20T10:00:00")),
                predlog.append(self.entry | {"a_id": "q", "b_id": "r"}, self.log, now=utc("2026-10-20T11:00:00")),
                predlog.append(self.entry | {"a_id": "m", "b_id": "n"}, self.log, now=utc("2026-10-20T12:00:00"))]
        results = pd.DataFrame({"winner_id": ["y", "m", "n"], "loser_id": ["x", "n", "m"],
                                "event_date": pd.to_datetime(["2026-10-26", "2026-10-26", "2026-10-27"])})
        scored, unmatched, ambiguous = predlog.join_results(recs, results)
        self.assertEqual(scored.y.tolist(), [0])                   # B ('y') won: y = 0 for A
        self.assertEqual(scored.p_D.tolist(), [0.6])               # the probabilities logged at the time, not recomputed
        self.assertEqual(unmatched, [recs[1]["id"]]); self.assertEqual(ambiguous, [recs[2]["id"]])

    def test_already_played_matches_are_refused(self):
        from datetime import date
        index = {("x", "y"): [{"date": date(2026, 9, 30), "winner_id": "x"}]}
        self.assertTrue(predlog.already_played(index, "y", "x", "2026-10-02"))
        self.assertFalse(predlog.already_played(index, "x", "y", "2026-11-02"))
        self.assertFalse(predlog.already_played(index, "x", "z", "2026-10-02"))

    def test_entry_from_real_predictors(self):
        _, P, V = synthetic_models()
        ids = [p for p in P.art["players"].index if p != "p_nostats"]
        e = predlog.make_entry(P, V, ids[0], ids[1], "Clay", 5, "T", "2016-01-04")
        self.assertAlmostEqual(e["p"]["D"], P.predict(ids[0], ids[1], "Clay")["p"]["D"])
        self.assertAlmostEqual(e["p"]["V2"], V.predict(ids[0], ids[1], "Clay", 5)["p"])
        self.assertEqual(set(e["model_versions"]), {"models/frozen_phase5.json", "models/frozen_v2.json", "models/v2_config.json"})
        e2 = predlog.make_entry(P, V, "p_nostats", ids[0], "Hard", 3, "T", "2016-01-04")
        self.assertIsNone(e2["p"]["V2"]); self.assertEqual(e2["v2_status"], "unavailable")


class TestRefresh(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.d, cls.P, cls.V = synthetic_models()

    def test_completion_uses_file_content_date(self):
        old = {"last_modified": "Mon, 19 Oct 2026 07:44:37 GMT", "fetched_utc": "2026-10-21T14:43:00+00:00"}
        self.assertEqual(refresh.content_day(old), pd.Timestamp("2026-10-19"))   # downloaded today, content from the 19th
        self.assertEqual(refresh.content_day({"last_modified": None, "fetched_utc": "2026-10-21T14:43:00+00:00"}), pd.Timestamp("2026-10-21"))

    def test_duplicates_against_history(self):
        hist = pd.DataFrame({"tourney_id": ["sackmann:2026-0520"], "round_rank": [7], "winner_id": ["a"], "loser_id": ["b"]})
        new = pd.DataFrame({"tourney_id": ["tml:2026-520", "tml:2026-520", "tml:2026-560"], "round_rank": [7, 6, 7],
                            "winner_id": ["b", "a", "a"], "loser_id": ["a", "b", "b"]})
        np.testing.assert_array_equal(refresh.known_duplicates(new, hist), [True, False, False])   # same match with the winner flipped still counts

    def test_fixed_coefficients_are_copied_and_state_moves(self):
        later = self.d.copy()
        later["event_date"] = later.event_date + pd.Timedelta(days=400)
        later["match_id"] = "late:" + later.match_id
        table = pd.concat([self.d, later], ignore_index=True)
        v1 = build_v1(table, FROZEN_V1, fixed=self.P.art["models"])
        v2 = build_v2(table, FROZEN_V2, fixed=(self.V.art["coef"], self.V.art["scale"]))
        self.assertEqual(v1["models"], self.P.art["models"]); self.assertEqual(v2["coef"], self.V.art["coef"])
        self.assertGreater(v1["snapshot_day"], self.P.art["snapshot_day"])
        self.assertNotEqual(v1["elo"], self.P.art["elo"])            # ratings absorbed the later results

    def _write(self, d, P, V):
        (d / "dashboard.pkl").write_bytes(pickle.dumps(P.art)); (d / "v2.pkl").write_bytes(pickle.dumps(V.art))
        (d / "h2h.pkl").write_bytes(pickle.dumps({})); (d / "status.json").write_text("{}")

    def test_validation_catches_changed_coefficients(self):
        ids = tuple(p for p in self.P.art["players"].index if p != "p_nostats")[:2]
        with tempfile.TemporaryDirectory() as ref, tempfile.TemporaryDirectory() as stage:
            ref, stage = Path(ref), Path(stage)
            self._write(ref, self.P, self.V); self._write(stage, self.P, self.V)
            self.assertEqual(refresh.validate(stage, ref, ids), [])
            bad = dict(self.P.art, models={k: dict(v) for k, v in self.P.art["models"].items()})
            bad["models"]["D"]["coef"] = [c * 1.01 for c in bad["models"]["D"]["coef"]]
            (stage / "dashboard.pkl").write_bytes(pickle.dumps(bad))
            self.assertIn("Model D coefficients differ from the frozen reference", refresh.validate(stage, ref, ids))

    def test_activation_backs_up_and_failure_restores(self):
        with tempfile.TemporaryDirectory() as root:
            art, stage = Path(root) / "art", Path(root) / "stage"
            art.mkdir(); stage.mkdir()
            for f in refresh.FILES:
                (art / f).write_text("old " + f); (stage / f).write_text("new " + f)
            with mock.patch.object(refresh, "ART", art), mock.patch.object(refresh, "BACKUPS", art / "backups"):
                backup = refresh.activate(stage)
                self.assertTrue(all((art / f).read_text() == "new " + f for f in refresh.FILES))
                self.assertTrue(all((backup / f).read_text() == "old " + f for f in refresh.FILES))
                for f in refresh.FILES:
                    (stage / f).write_text("newer " + f)
                (stage / "h2h.pkl").unlink()   # staged set incomplete: activation must fail and restore
                with self.assertRaises(FileNotFoundError):
                    refresh.activate(stage)
                self.assertTrue(all((art / f).read_text() == "new " + f for f in refresh.FILES))
                refresh.rollback()             # the latest backup holds the 'new' set; rollback restores it
                self.assertEqual((art / "dashboard.pkl").read_text(), "new dashboard.pkl")


class TestCoverage(unittest.TestCase):
    def test_missing_events_and_short_draws(self):
        from src import tml
        from tests.test_tml import frame, row
        prev = [row("2025-0500", "20250616", "R32", "A", "B", "a", "b", draw_size=32), row("2025-0321", "20250609", "R32", "A", "B", "a", "b", draw_size=28)]
        with tempfile.TemporaryDirectory() as d:
            pd.DataFrame(prev).to_csv(Path(d) / "2025.csv", index=False)
            now = tml.clean(frame([row("2026-500", "20260615", f"R{r}", f"A{i}", f"B{i}", "a", "b", draw_size=32) for i, r in enumerate(["32", "16"])]))
            cov = refresh.coverage(now, now.iloc[:0], Path(d))
        self.assertEqual(cov["same_weeks_last_year"], 2)
        self.assertEqual(cov["missing_vs_last_year"], ["T"])   # code 321 held a year earlier is absent now
        self.assertEqual(cov["short_draws"], ["T"])            # 2 rows for a 32-draw


class TestSync(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory(); root = Path(self.dir.name)
        self.patches = [mock.patch.object(sync, "STATE", root / "state.json"), mock.patch.object(sync, "LOG", root / "sync.log"),
                        mock.patch.object(refresh, "ART", root / "art")]
        for p in self.patches:
            p.start()
        (root / "art").mkdir()

    def tearDown(self):
        for p in self.patches:
            p.stop()
        self.dir.cleanup()

    def test_download_failure_keeps_previous_state(self):
        with mock.patch.object(sync, "download_changes", side_effect=OSError("network down")), mock.patch.object(refresh, "run") as run:
            self.assertEqual(sync.run(), 1)
            run.assert_not_called()
        st = sync.state()
        self.assertEqual(st["last_result"], "failed"); self.assertIn("network down", st["last_error"])
        self.assertIn("FAILED", sync.LOG.read_text())

    def test_validation_failure_is_reported_and_nothing_activated(self):
        with mock.patch.object(sync, "download_changes", return_value=["2026.csv"]), \
                mock.patch.object(refresh, "run", return_value=(False, ["V2 coefficients differ"])):
            self.assertEqual(sync.run(), 1)
        self.assertIn("V2 coefficients differ", sync.state()["last_error"])

    def test_no_change_skips_rebuild(self):
        today = str(pd.Timestamp.now(tz="UTC").date())
        (refresh.ART / "status.json").write_text(json.dumps({"pending_tournaments": ["Shanghai"], "download_date": today}))
        with mock.patch.object(sync, "download_changes", return_value=[]), mock.patch.object(refresh, "run") as run:
            self.assertEqual(sync.run(), 0)
            run.assert_not_called()
        self.assertEqual(sync.state()["last_result"], "no change")

    def test_at_most_one_check_per_day_unless_forced(self):
        sync.save_state(last_success_utc=sync.now())
        with mock.patch.object(sync, "download_changes") as dl:
            self.assertEqual(sync.run(), 0)
            dl.assert_not_called()                       # a second run the same day does not download
        with mock.patch.object(sync, "download_changes", return_value=[]), mock.patch.object(refresh, "run", return_value=(True, {"latest_complete_through": "x", "tml_matches_added": 0})):
            self.assertEqual(sync.run(force=True), 0)   # ./update.sh --force still works

    def test_download_detects_changes_and_rejects_bad_format(self):
        with tempfile.TemporaryDirectory() as raw:
            raw = Path(raw)
            cols = sorted(sync.REQUIRED_COLUMNS)
            (raw / "2026.csv").write_text(",".join(cols) + "\n")
            (raw / "manifest.json").write_text(json.dumps({"files": {"2026.csv": {"sha256": "old"}}}))

            def fake_fetch(names, out):
                for n in names:
                    (out / n).write_text(",".join(cols) + "\n" + ",".join("1" for _ in cols) + "\n")
                (out / "manifest.json").write_text(json.dumps({"files": {n: {"sha256": "new", "fetched_utc": "x"} for n in names}}))
            with mock.patch.object(sync.tml, "fetch", fake_fetch):
                self.assertEqual(sync.download_changes(["2026.csv"], raw), ["2026.csv"])
            self.assertEqual(json.loads((raw / "manifest.json").read_text())["files"]["2026.csv"]["sha256"], "new")

            def broken_fetch(names, out):
                for n in names:
                    (out / n).write_text("unexpected,columns\n1,2\n")
                (out / "manifest.json").write_text(json.dumps({"files": {n: {"sha256": "newer"} for n in names}}))
            with mock.patch.object(sync.tml, "fetch", broken_fetch), self.assertRaises(ValueError):
                sync.download_changes(["2026.csv"], raw)
            self.assertIn("1,1", (raw / "2026.csv").read_text())   # the previous good file is untouched


class TestFreshness(unittest.TestCase):
    def test_age_and_warnings(self):
        with tempfile.TemporaryDirectory() as d:
            d = Path(d)
            status = {"source": "S", "attribution": "A", "latest_tournament_start": "2026-09-30", "latest_complete_through": "2026-10-07",
                      "pending_tournaments": ["Shanghai"], "model_coefficients": "frozen"}
            (d / "status.json").write_text(json.dumps(status))
            (d / "sync.json").write_text(json.dumps({"last_success_utc": "2026-10-20T15:00:00+00:00", "last_result": "artifacts rebuilt"}))
            with mock.patch.object(api, "STATUS_FILE", d / "status.json"), mock.patch.object(api, "SYNC_STATE", d / "sync.json"):
                fresh = api.data_status(today=pd.Timestamp("2026-10-20").date())
                self.assertEqual(fresh["data_age_days"], 13); self.assertEqual(fresh["warnings"], [])
                old = api.data_status(today=pd.Timestamp("2026-11-30").date())
                self.assertEqual(len(old["warnings"]), 2)                  # stale results and no recent successful update
                (d / "sync.json").write_text(json.dumps({"last_success_utc": "2026-10-20T15:00:00+00:00", "last_result": "failed", "last_error": "boom"}))
                self.assertTrue(any("failed" in w for w in api.data_status(today=pd.Timestamp("2026-10-21").date())["warnings"]))
            with mock.patch.object(api, "STATUS_FILE", d / "missing.json"):
                self.assertIsNone(api.data_status())
