"""V2 input table: the V1 canonical table plus match context and serve/return statistics.

A future data source needs one adapter producing the same columns: CANON + best_of + venue + w_<stat>/l_<stat> for STATS.
Statistics are post-match: the feature pass only uses them after the match is released.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.canonical import DEV_PATH, load_dev

STATS = ["ace", "df", "svpt", "1stIn", "1stWon", "2ndWon", "SvGms", "bpSaved", "bpFaced"]
CONFIG = json.loads((Path(__file__).resolve().parents[2] / "models/v2_config.json").read_text())  # works from any working directory (the API)


def stats_ok(df, p):
    """True where one side's statistics are complete and internally consistent (otherwise they are treated as missing)."""
    c = {s: df[f"{p}_{s}"] for s in STATS}
    return (df[[f"{p}_{s}" for s in STATS]].notna().all(axis=1) & (c["svpt"] > 0) & (c["SvGms"] > 0) & (c["1stIn"] <= c["svpt"])
            & (c["1stWon"] <= c["1stIn"]) & (c["2ndWon"] <= c["svpt"] - c["1stIn"]) & (c["bpSaved"] <= c["bpFaced"]))


def from_sackmann_v2(df, canon):
    """Attach best_of, venue (tournament code + surface key for the pace index) and validated statistics to canon rows."""
    extra = pd.DataFrame({"match_id": "sackmann:" + df.match_id.astype(str), "best_of": df.best_of.astype(int),
                          "venue": "sackmann:" + df.tourney_id.astype(str).str.split("-").str[1].str.lstrip("0")})  # '0891' and '891' are one event
    ok = stats_ok(df, "w") & stats_ok(df, "l")  # both sides needed: a player's return statistics are the opponent's serve
    for p in ("w", "l"):
        for s in STATS:
            extra[f"{p}_{s}"] = df[f"{p}_{s}"].where(ok)
    extra["has_stats"] = ok.to_numpy()
    extra["score"] = df.score.to_numpy()  # post-match: used only by the V2.1 quality pass after release
    out = canon.merge(extra, on="match_id", how="left", validate="one_to_one")
    assert len(out) == len(canon)
    return out


def load_dev_v2(path=DEV_PATH):
    """Dev data only (2010-2024), like V1 modelling code. The locked file is never read here."""
    return from_sackmann_v2(pd.read_csv(path), load_dev(path))


def orient(ab):
    """Statistics as A/B columns (a_<stat>, b_<stat>) using the hashed A/B order; returns a dict of numpy arrays."""
    aw = ab.y.to_numpy() == 1
    out = {}
    for s in STATS:
        w, l = ab[f"w_{s}"].to_numpy(float), ab[f"l_{s}"].to_numpy(float)
        out[f"a_{s}"], out[f"b_{s}"] = np.where(aw, w, l), np.where(aw, l, w)
    return out
