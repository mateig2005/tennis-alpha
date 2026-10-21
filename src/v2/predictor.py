"""V2 prediction interface (Phase 11): Tennis Alpha V2 probabilities for a hypothetical match on the V1 snapshot date.

It reuses the research functions (v2.features.player_features / match_context / design) on the artifact's player
states, and the V1 Predictor for the Model D inputs V2 builds on, so every number is computed exactly as in training.
A player with no released match carrying serve/return statistics gets status 'unavailable': nothing is filled in.
"""
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import models
from src.features import SURF
from src.v2 import features as F
from src.v2 import quality as Q

ARTIFACT = Path("artifacts/v2.pkl")
FEW_STAT_MATCHES = 10
VALIDATION = ("Experimental. Better than Model D on development folds 2023-2024 (held out from V2 choices but not independent); "
              "no independent prospective test yet.")
GROUP_LABELS = {"elo": "Elo ratings", "rank": "Ranking", "last": "Last 5 / 10", "win": "Win-rate windows", "surf": "Surface form",
                "rest": "Rest / workload", "dec": "Decayed form", "adj": "Opponent-adjusted form", "adjdec": "Opp-adj. decayed",
                "sr_adj": "Serve & return ratings", "sr_raw": "Serve & return rates (last 20)", "pressure": "Break points",
                "matchup": "Serve-vs-return matchup", "pace": "Pace interactions", "h2h": "Head-to-head (shrunk)",
                "context": "Best-of-5 × Elo", "momentum": "Recent serve/return & Elo trend"}


class V2Predictor:
    def __init__(self, art, v1):
        if art["snapshot_day"] != v1.art["snapshot_day"]:
            raise ValueError("V2 and V1 artifacts were built from different data snapshots; rebuild both")
        self.art, self.v1 = art, v1
        groups = {**models.group_columns(v1.art["half_life"]), **F.GROUPS}
        self.group_of = {c: g for g, cols in groups.items() for c in cols}

    @classmethod
    def load(cls, v1, path=ARTIFACT):
        return cls(pickle.loads(Path(path).read_bytes()), v1)  # built locally by src/v2/build_artifact.py, never downloaded

    def _profile(self, pid, s, elo_now):
        a = self.art
        f = F.player_features(a["state"].get(pid) or F._Player(), a["tour"], a["snapshot_day"], s, elo_now)
        q = Q.player_features(a["qstate"].get(pid), a["snapshot_day"], s, a["vol0"])
        return f, q, a["stat_matches"].get(pid, 0)

    def predict(self, a_id, b_id, surface, best_of=3, v1_result=None):
        """P(A beats B) from V2, its inputs and group contributions; status 'unavailable' when a player has no statistics."""
        if best_of not in (3, 5):
            raise ValueError("best_of must be 3 or 5")
        r = v1_result or self.v1.predict(a_id, b_id, surface)  # Model D inputs (V1, unchanged)
        s, art = SURF[surface], self.art
        fa, qa, na = self._profile(a_id, s, self.v1.art["elo"].get(a_id, 1500.0))
        fb, qb, nb = self._profile(b_id, s, self.v1.art["elo"].get(b_id, 1500.0))
        ctx = F.match_context(fa, fb, art["tour"], s, best_of, None, a_id, b_id)  # no venue: pace uses the surface average
        out = {"version": art["version"], "best_of": best_of, "validation": VALIDATION, "data_end": art["data_end"],
               "players": {"a": self._public(fa, qa, na), "b": self._public(fb, qb, nb)},
               "matchup": {"pa": float(ctx["pa"]), "pb": float(ctx["pb"]), "tour_serve_avg": float(ctx["mu"]), "p_points_model": float(ctx["p_mk"])},
               "h2h": {"residual": float(ctx["h2h"]), "meetings": int(ctx["h2h_n"])}}
        missing = [r[k]["name"] for k, n in (("A", na), ("B", nb)) if n == 0]
        if missing:
            return out | {"status": "unavailable", "p": None, "logit": None, "contributions": [], "notes": [],
                          "reason": f"No match with serve/return statistics in the data for {' and '.join(missing)}: V2 needs them and does not invent them."}
        idx = ["hypothetical"]
        X = F.design(pd.DataFrame([r["diff"]], index=idx), pd.DataFrame([fa], index=idx), pd.DataFrame([fb], index=idx), pd.DataFrame([ctx], index=idx)).iloc[0]
        terms = np.array(art["coef"]) * X[art["cols"]].to_numpy(float) / np.array(art["scale"])
        contrib = {}
        for c, t in zip(art["cols"], terms):
            contrib[self.group_of[c]] = contrib.get(self.group_of[c], 0.0) + float(t)
        logit = float(terms.sum())
        notes = [f"{r[k]['name']}: only {n} matches with serve/return statistics; the ratings lean on the tour average."
                 for k, n in (("A", na), ("B", nb)) if n < FEW_STAT_MATCHES]
        notes += r["notes"]  # V1 staleness and missing-ranking notes apply to V2's inputs too
        return out | {"status": "ok", "reason": None, "p": float(1 / (1 + np.exp(-logit))), "logit": logit, "notes": notes,
                      "contributions": [{"group": g, "label": GROUP_LABELS.get(g, g), "value": v} for g, v in contrib.items()]}

    @staticmethod
    def _public(f, q, n):
        return {"stat_matches": n, "serve": f["srv"], "ret": f["ret"], "serve_surface": f["srv_s"], "ret_surface": f["ret_s"],
                "serve_recent": f["srv60"], "ret_recent": f["ret60"], "spw20": f["spw"], "rpw20": f["rpw"], "hold20": f["hold"],
                "break20": f["brk"], "ace20": f["ace"], "elo_trend": f["elo_trend"],
                "quality": {"points_vs_expected20": q["pq20"], "luck10": q["luck10"]} if n else None}
