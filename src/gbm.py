"""XGBoost with exact player symmetry and leak-free early stopping (PLAN.md N2, N3).

Gradient boosting adds many small decision trees, each correcting the previous ones. Trees can learn curves and
interactions that Logistic Regression cannot, but they over-fit easily, so depth and tree count are kept small.
Symmetry: trees do not know that swapping A and B must flip the probability, so each training row is also added
mirrored, and predictions average the two views: P(A) = (f(x) + 1 - f(-x)) / 2.
Early stopping never looks at the year being predicted: the tree count is chosen on the last training year.
"""
import itertools
import os
import sys

import numpy as np
import pandas as pd

from src import evaluate as ev
from src.models import FOLDS, TRAIN_FROM


def _load_xgboost():
    try:
        import xgboost
        return xgboost
    except Exception:
        # shortcut: macOS without `brew install libomp` -> restart once using the libomp bundled with scikit-learn;
        # install libomp properly if this project is deployed anywhere else.
        if sys.platform == "darwin" and "TENNIS_LIBOMP_FALLBACK" not in os.environ:
            import sklearn
            lib = os.path.join(os.path.dirname(sklearn.__file__), ".dylibs")
            if os.path.exists(os.path.join(lib, "libomp.dylib")):
                env = dict(os.environ, DYLD_LIBRARY_PATH=lib, TENNIS_LIBOMP_FALLBACK="1")
                os.execve(sys.executable, sys.orig_argv, env)
        raise


xgb = _load_xgboost()

FIXED = dict(learning_rate=0.05, subsample=0.8, colsample_bytree=0.8, tree_method="hist", objective="binary:logistic",
             eval_metric="logloss", base_score=0.5, random_state=0, n_jobs=4)
MAX_TREES, PATIENCE, TOLERANCE = 800, 40, 0.0003
GRID = [dict(max_depth=d, min_child_weight=m, reg_lambda=l) for d, m, l in itertools.product((1, 2, 3), (5, 25), (1, 10))]


def mirror(X, y):
    X, y = np.asarray(X, np.float32), np.asarray(y)
    return np.vstack([X, -X]), np.concatenate([y, 1 - y])


def fit(X, y, cfg, n_trees, monitor=None):
    """Fit on mirrored data. With monitor=(X, y) early stopping picks the tree count (read it from .best_iteration)."""
    m = xgb.XGBClassifier(n_estimators=n_trees, early_stopping_rounds=PATIENCE if monitor else None, **FIXED, **cfg)
    m.fit(*mirror(X, y), eval_set=[mirror(*monitor)] if monitor else None, verbose=False)
    return m


def predict(model, X):
    X = np.asarray(X, np.float32)
    a, b = (model.predict_proba(v)[:, 1].astype(np.float64) for v in (X, -X))  # float64 keeps P(A) + P(B) = 1 exact
    return (a + 1.0 - b) / 2.0


def train(D, y, year, mask, cols, cfg, last_year):
    """Model using data up to last_year. Tree count comes from: fit up to last_year-1, monitor last_year."""
    inner, mon, full = (mask & (year >= TRAIN_FROM) & c for c in (year < last_year, year == last_year, year <= last_year))
    n = fit(D.loc[inner, cols], y[inner], cfg, MAX_TREES, monitor=(D.loc[mon, cols], y[mon])).best_iteration + 1
    return fit(D.loc[full, cols], y[full], cfg, n), n


def oof(D, y, year, mask, cols, cfg):
    """Out-of-fold predictions for the dev folds; each fold year is predicted by a model that never saw it."""
    out, fold_models, trees = pd.Series(np.nan, index=D.index), {}, {}
    for f in FOLDS:
        te = mask & (year == f)
        fold_models[f], trees[f] = train(D, y, year, mask, cols, cfg, f - 1)
        out[te] = predict(fold_models[f], D.loc[te, cols])
    return out, fold_models, trees


def simplest_within(tab, tol=TOLERANCE):
    """Among settings within tol of the best mean log loss, take the simplest (shallow, then more regularised)."""
    ok = tab[tab.mean_logloss <= tab.mean_logloss.min() + tol]
    return ok.sort_values(["max_depth", "min_child_weight", "reg_lambda"], ascending=[True, False, False]).iloc[0]


def select(D, y, year, mask, cols, grid=GRID):
    rows, preds = [], {}
    for i, cfg in enumerate(grid):
        p, _, trees = oof(D, y, year, mask, cols, cfg)
        per = [ev.log_loss(y[mask & (year == f)], p[mask & (year == f)]) for f in FOLDS]
        rows.append(dict(cfg, mean_logloss=float(np.mean(per)), mean_trees=float(np.mean(list(trees.values()))), id=i))
        preds[i] = p
    tab = pd.DataFrame(rows)
    best = simplest_within(tab)
    return {k: (int(best[k]) if k != "reg_lambda" else float(best[k])) for k in ("max_depth", "min_child_weight", "reg_lambda")}, tab, preds
