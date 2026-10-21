"""Logistic Regression on A-minus-B feature differences, with chronological cross-validation.

Why these choices (PLAN.md M3, M6, M7):
- No intercept and scale-only standardisation: inputs are differences, A/B order is random, so P(A) = 1 - P(B) exactly.
  Centring would add a hidden intercept and break that symmetry.
- L2 regularisation (strength C) keeps many correlated form features from over-fitting. The scaler is fitted on training rows only.
- Folds always train on the past and validate on a later year.
"""
import itertools

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from src import evaluate as ev

C_GRID = (0.01, 0.1, 1.0, 10.0)
FOLDS = (2020, 2021, 2022, 2023)  # validate on each year, train on 2015 .. year-1
TRAIN_FROM = 2015


def group_columns(hl):
    return {
        "elo": ["elo", "surf_elo"],
        "rank": ["lrank", "lpts", "rank_miss", "pts_miss"],
        "last": ["wr5", "wr10"],
        "win": ["wr30", "wr90", "wr365"],
        "surf": ["swr10", "swr365"],
        "rest": ["rest", "ln30", "ln90", "ln365", "has_hist"],
        "dec": [f"dec_wr_{hl}"],
        "adj": ["adj10", "adj90", "adj365", "opp10"],
        "adjdec": [f"adj_dec_{hl}"],
    }


def columns(groups, hl):
    g = group_columns(hl)
    return [c for name in groups for c in g[name]]


def fit(X, y, C):
    sc = StandardScaler(with_mean=False).fit(X)  # training rows only
    lr = LogisticRegression(C=C, fit_intercept=False, max_iter=5000).fit(sc.transform(X), y)
    return sc, lr


def predict(model, X):
    sc, lr = model
    return lr.predict_proba(sc.transform(X))[:, 1]


def oof_predictions(D, y, year, mask, groups, hl, C):
    """Out-of-fold probabilities for the dev folds (each year predicted by a model trained only on earlier years)."""
    cols = columns(groups, hl)
    out = pd.Series(np.nan, index=D.index)
    for f in FOLDS:
        tr = mask & (year >= TRAIN_FROM) & (year < f)
        te = mask & (year == f)
        out[te] = predict(fit(D.loc[tr, cols], y[tr], C), D.loc[te, cols])
    return out


def select(D, y, year, mask, groups, c_grid=C_GRID):
    """Pick (half-life, C) by mean out-of-fold log loss over the dev folds. Returns (best, full table)."""
    uses_hl = any(g in ("dec", "adjdec") for g in groups)
    rows = []
    for hl, C in itertools.product((30, 60, 120, 240) if uses_hl else (None,), c_grid):
        p = oof_predictions(D, y, year, mask, groups, hl or 30, C)
        per = [ev.log_loss(y[mask & (year == f)], p[mask & (year == f)]) for f in FOLDS]
        rows.append((hl, C, float(np.mean(per)), *per))
    tab = pd.DataFrame(rows, columns=["half_life", "C", "mean_logloss"] + [f"fold_{f}" for f in FOLDS])
    best = tab.sort_values(["mean_logloss", "C"]).iloc[0]
    return (None if pd.isna(best.half_life) else int(best.half_life), float(best.C)), tab
