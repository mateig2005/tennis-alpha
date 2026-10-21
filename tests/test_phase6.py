import copy
import glob
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from src import run_phase6 as p6
from tests.helpers import synthetic


class TestFreezeAndGuards(unittest.TestCase):
    def setUp(self):
        self.fz = json.loads(p6.FROZEN.read_text())

    def test_committed_frozen_file_matches_the_code(self):
        p6.check_frozen(copy.deepcopy(self.fz))

    def test_tampering_is_detected(self):
        for change in (lambda f: f["models"]["D LR"]["features"].pop(), lambda f: f["elo"].update(k_general=32),
                       lambda f: f["models"]["D XGB"]["params"].update(learning_rate=0.3), lambda f: f.update(primary="D XGB"),
                       lambda f: f["data"]["release_days"].update(A=0)):
            f = copy.deepcopy(self.fz); change(f)
            with self.assertRaises(AssertionError):
                p6.check_frozen(f)

    def test_second_run_is_refused(self):
        with tempfile.NamedTemporaryFile(suffix=".md") as t:
            with self.assertRaises(SystemExit):
                p6.guard(t.name, approved=False)
            p6.guard(t.name, approved=True)
        p6.guard("does/not/exist.md")

    def test_only_approved_modules_open_the_locked_file(self):
        users = [f for f in glob.glob("src/**/*.py", recursive=True) if "load_with_locked" in Path(f).read_text() or "LOCKED_PATH" in Path(f).read_text()]
        # build_artifacts joined in Phase 7: the evaluation is finished, and it only refits the frozen specifications
        # h2h joined in Phase 10: it lists recorded meetings for display and feeds no model
        # v2/audit joined in V2: it rebuilds history to the snapshot for one diagnostic case; it fits and selects nothing on 2025/2026
        # v2/build_artifact joined in Phase 11: production refit of the frozen V2 settings, like build_artifacts; it selects nothing
        # refresh joined in Phase 15: it rebuilds the app's player state from all history, copying the frozen coefficients
        # tml joined in Phase 14: it compares TennisMyLife with already-known 2024-2026 rows to validate the source; it fits nothing
        # v2/forward_eval joined in Phase 12: the frozen forward evaluation needs history up to the snapshot; it fits only on rows up to 2026-05-25
        self.assertEqual(sorted(users), ["src/build_artifacts.py", "src/canonical.py", "src/h2h.py", "src/refresh.py", "src/run_phase6.py", "src/tml.py",
                                         "src/v2/audit.py", "src/v2/build_artifact.py", "src/v2/forward_eval.py"])

    def test_tiers_are_the_pre_registered_ones(self):
        self.assertEqual(p6.TIERS["Primary"], [("D LR", "A Elo")])
        self.assertEqual(p6.TIERS["Secondary"], [("D LR", "B+ LR"), ("D XGB", "D LR"), ("D-norank LR", "A Elo")])


class TestPeriodPrediction(unittest.TestCase):
    def test_test_year_labels_never_reach_the_models(self):
        d = synthetic(1500, players=40, seed=5); d["level"] = "A"
        d["event_date"] = pd.Timestamp("2015-01-05") + pd.to_timedelta(7 * np.random.default_rng(5).integers(0, 300, len(d)), unit="D")
        S, fz = p6.build(d), json.loads(p6.FROZEN.read_text())
        base, n_train = p6.predict_period(S, fz, 2018, 2019)
        S2 = p6.SimpleNamespace(**vars(S)); S2.y = S.y.copy(); S2.y[S.year >= 2019] = 1 - S2.y[S.year >= 2019]   # flip labels from the test year on
        alt, _ = p6.predict_period(S2, fz, 2018, 2019)
        pd.testing.assert_frame_equal(base, alt)
        self.assertEqual(n_train, int((S.year <= 2018).sum()))
        np.testing.assert_allclose(base.to_numpy().min() >= 0, True); self.assertEqual(list(base.columns), p6.MODELS)


if __name__ == "__main__":
    unittest.main()
