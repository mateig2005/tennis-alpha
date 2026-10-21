"""Model diagnostics: multicollinearity, permutation importance and subgroup scores."""
import numpy as np
import pandas as pd

from src import evaluate as ev


def vif(X):
    """Variance inflation factor per column: 1 = independent, >10 = largely explained by the other columns."""
    c = np.corrcoef(np.asarray(X, float), rowvar=False)
    return pd.Series(np.diag(np.linalg.pinv(c)), index=X.columns)


def top_correlations(X, threshold=0.8):
    c = X.corr()
    rows = [(a, b, c.loc[a, b]) for i, a in enumerate(c.columns) for b in c.columns[i + 1:] if abs(c.loc[a, b]) >= threshold]
    return pd.DataFrame(rows, columns=["feature 1", "feature 2", "r"]).sort_values("r", key=abs, ascending=False)


def group_permutation(folds, groups, repeats=3, seed=0):
    """Rise in log loss when one feature GROUP is shuffled in held-out data (groups cope with correlated features).
    folds: list of (predict_fn, X_test DataFrame, y_test). Returns {group: mean rise}; ~0 means the model ignores it."""
    rng, n = np.random.default_rng(seed), sum(len(y) for _, _, y in folds)
    base = sum(ev.log_loss(y, f(X)) * len(y) for f, X, y in folds) / n
    out = {}
    for g, cols in groups.items():
        tot = 0.0
        for _ in range(repeats):
            for f, X, y in folds:
                Xp = X.copy()
                Xp[cols] = X[cols].to_numpy()[rng.permutation(len(X))]
                tot += ev.log_loss(y, f(Xp)) * len(y)
        out[g] = tot / (repeats * n) - base
    return out


def by_group(y, preds, labels):
    """Log loss of each model inside each subgroup. preds: {model name: Series}."""
    rows = []
    for lab in pd.unique(labels.dropna()):
        m = (labels == lab).to_numpy()
        rows.append({"group": lab, "n": int(m.sum())} | {k: ev.log_loss(y[m], p[m]) for k, p in preds.items()})
    return pd.DataFrame(rows).sort_values("group")
