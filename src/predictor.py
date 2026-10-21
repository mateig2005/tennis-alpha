"""Prediction interface behind the dashboard: pre-match probabilities for a hypothetical match on the snapshot date.

It reuses the research pipeline's own feature functions (features.player_features, features.design), so the numbers
shown are produced exactly the way the models were trained. Nothing here knows anything after the snapshot date.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import models
from src.elo import expected
from src.features import design, player_features

ARTIFACT = Path("artifacts/dashboard.pkl")
SURFACES = ("Hard", "Clay", "Grass")
NEW = ("new:A", "new:B")  # stand-ins for a player with no history at all
STALE_DAYS = 180       # warn: ratings and ranking may be stale
UNRELIABLE_DAYS = 365  # beyond this the prediction is marked not reliable (ratings do not decay with inactivity)


class Predictor:
    def __init__(self, art):
        self.art = art

    @classmethod
    def load(cls, path=ARTIFACT):
        return cls(pickle.loads(Path(path).read_bytes()))  # built locally by src/build_artifacts.py, never downloaded

    def players(self, active_only=True):
        """Known players, strongest (general Elo) first. active_only keeps those who played within UNRELIABLE_DAYS of the snapshot."""
        p = self.art["players"].sort_values("elo", ascending=False)
        gap = (pd.Timestamp(self.art["snapshot_date"]) - p.last_date).dt.days
        return p[gap <= UNRELIABLE_DAYS] if active_only else p

    def _profile(self, pid, surface):
        a, known = self.art, pid in self.art["players"].index
        f = player_features(a["state"], pid, a["snapshot_day"], surface)
        row = a["players"].loc[pid] if known else None
        rank = float(row["rank"]) if known and pd.notna(row["rank"]) else np.nan
        pts = float(row["points"]) if known and pd.notna(row["points"]) else np.nan
        st = a["state"].get(pid)
        return {"id": pid, "name": row["name"] if known else "New player (no history)", "known": known, "features": f,
                "elo": a["elo"].get(pid, 1500.0), "surface_elo": a["surface_elo"].get((surface, pid), 1500.0), "rank": rank, "points": pts,
                "last_date": row["last_date"].date() if known else None, "matches": int(row["matches"]) if known else 0,
                "days_since_release": None if st is None or st.last_rel is None else a["snapshot_day"] - st.last_rel}

    def predict(self, a_id, b_id, surface):
        """Returns probabilities that A beats B for Model D, Model C and Elo, plus the inputs and plain-language notes."""
        if surface not in SURFACES:
            raise ValueError(f"surface must be one of {SURFACES}")
        if a_id == b_id:
            raise ValueError("Player A and Player B must be different")
        A, B = self._profile(a_id, surface), self._profile(b_id, surface)
        ab = pd.DataFrame({"match_id": ["hypothetical"], "rank_a": [A["rank"]], "rank_b": [B["rank"]], "pts_a": [A["points"]], "pts_b": [B["points"]]})
        rt = pd.DataFrame({"elo_a": [A["elo"]], "elo_b": [B["elo"]], "surf_a": [A["surface_elo"]], "surf_b": [B["surface_elo"]]}, index=["hypothetical"])
        fa, fb = (pd.DataFrame([p["features"]], index=["hypothetical"]) for p in (A, B))
        d = design(ab, rt, fa, fb).iloc[0]  # A-minus-B differences, same function as training
        probs, contrib = {}, {}
        for key, m in self.art["models"].items():
            terms = np.array(m["coef"]) * d[m["features"]].to_numpy(float) / np.array(m["scale"])
            probs[key] = float(1.0 / (1.0 + np.exp(-terms.sum())))
            groups = {g: c for g, c in models.group_columns(self.art["half_life"]).items() if set(c) <= set(m["features"])}
            contrib[key] = {g: float(sum(terms[m["features"].index(c)] for c in cols)) for g, cols in groups.items()}
        probs["Elo"] = float(expected(A["elo"], B["elo"]))
        notes, reliable = self._notes(A, B)
        return {"p": probs, "A": A, "B": B, "surface": surface, "diff": d, "contributions": contrib, "notes": notes, "reliable": reliable}

    def _notes(self, A, B):
        """Plain-language limitations. reliable=False when an input is missing or too old for an honest prediction."""
        notes, reliable = [], True
        for p in (A, B):
            if not p["known"] or p["matches"] == 0:
                notes.append(f"{p['name']}: no match history in the data. Form features take neutral starting values and Elo starts at 1500.")
                reliable = False
                continue
            if np.isnan(p["rank"]):
                notes.append(f"{p['name']}: no ranking in the data. Treated as unranked, as in training (no ranking was invented).")
            gap = (self.art["snapshot_date"] - p["last_date"]).days
            if gap > UNRELIABLE_DAYS:
                notes.append(f"{p['name']}: last match in the data started {p['last_date']}, {gap} days before the snapshot. Ratings do not decay with "
                             "inactivity and the ranking shown is that old, so this is not an honest current estimate.")
                reliable = False
            elif gap > STALE_DAYS:
                notes.append(f"{p['name']}: last match in the data started {p['last_date']} ({gap} days before the snapshot). Ratings and ranking may be stale.")
        return notes, reliable
