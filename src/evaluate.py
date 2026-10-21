"""Metrics, the rank baseline and a tournament-level paired bootstrap.

Log loss punishes confident wrong answers (lower is better; always-50% scores 0.693).
Brier score is the mean squared error of the probability. Calibration asks: of matches given ~70%, did ~70% happen?
"""
import numpy as np
import pandas as pd

UNRANKED = 500  # PLAN.md section 5: assumed rank for players with no ranking


def log_loss(y, p, eps=1e-12):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return float(-np.mean(np.where(np.asarray(y) == 1, np.log(p), np.log(1 - p))))


def brier(y, p):
    return float(np.mean((np.asarray(p, float) - np.asarray(y)) ** 2))


def accuracy(y, p):
    p, y = np.asarray(p, float), np.asarray(y)
    return float(np.mean(np.where(p == 0.5, 0.5, (p > 0.5) == (y == 1))))  # a 50% call counts as half right


def reliability(y, p, bins=10):
    """Table of predicted vs actual frequency in equal-width probability bins."""
    p, y = np.asarray(p, float), np.asarray(y)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    rows = [(i / bins, (i + 1) / bins, int((idx == i).sum()), p[idx == i].mean(), y[idx == i].mean())
            for i in range(bins) if (idx == i).any()]
    return pd.DataFrame(rows, columns=["lo", "hi", "n", "mean_pred", "actual"])


def ece(y, p, bins=10):
    """Expected calibration error: bin-size-weighted gap between predicted and actual frequency."""
    r = reliability(y, p, bins)
    return float((r.n * (r.mean_pred - r.actual).abs()).sum() / r.n.sum())


def _x(rank_a, rank_b):  # positive when A is the better-ranked player
    ra, rb = (np.where(np.isnan(np.asarray(r, float)), UNRANKED, r).astype(float) for r in (rank_a, rank_b))
    return np.log(rb) - np.log(ra)


def fit_rank_baseline(rank_a, rank_b, y, iters=30):
    """One-parameter logistic regression P(A) = sigmoid(b * (ln rank_B - ln rank_A)), fitted by Newton's method."""
    x, y, b = _x(rank_a, rank_b), np.asarray(y, float), 0.0
    for _ in range(iters):
        p = 1 / (1 + np.exp(-b * x))
        b += np.sum((y - p) * x) / np.sum(p * (1 - p) * x * x)
    return float(b)


def rank_prob(b, rank_a, rank_b):
    return 1 / (1 + np.exp(-b * _x(rank_a, rank_b)))


def paired_bootstrap(loss_a, loss_b, groups, reps=2000, seed=0):
    """Mean(loss_a - loss_b) with a 95% interval, resampling whole tournaments (matches inside one are correlated).
    Negative = model A has the lower (better) loss."""
    d = pd.Series(np.asarray(loss_a) - np.asarray(loss_b)).groupby(np.asarray(groups))
    s, c = d.sum().to_numpy(), d.size().to_numpy()
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(s), size=(reps, len(s)))
    boot = s[pick].sum(1) / c[pick].sum(1)
    return float(s.sum() / c.sum()), float(np.percentile(boot, 2.5)), float(np.percentile(boot, 97.5))


def per_match_loss(y, p, eps=1e-12):
    p = np.clip(np.asarray(p, float), eps, 1 - eps)
    return -np.where(np.asarray(y) == 1, np.log(p), np.log(1 - p))


def calibration_fit(y, p, iters=50):
    """Diagnostic only: logistic fit of outcome on logit(p) -> (intercept, slope). Perfect calibration = (0, 1);
    slope < 1 means the probabilities are too extreme (overconfident)."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    X = np.column_stack([np.ones(len(p)), np.log(p / (1 - p))])
    y, beta = np.asarray(y, float), np.zeros(2)
    for _ in range(iters):
        q = 1 / (1 + np.exp(-X @ beta))
        beta += np.linalg.solve(X.T @ (X * (q * (1 - q))[:, None]) + 1e-9 * np.eye(2), X.T @ (y - q))
    return float(beta[0]), float(beta[1])


def md(df, fmt=None):
    """Tiny markdown table writer (avoids another dependency)."""
    fmt, cols = fmt or {}, list(df.columns)
    rows = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for r in df.itertuples(index=False):
        rows.append("| " + " | ".join(fmt.get(c, lambda v: f"{v}")(v) for c, v in zip(cols, r)) + " |")
    return "\n".join(rows)


def read_md_table(text, heading):
    """Read back the first markdown table that follows a line starting with `heading` (inverse of md)."""
    lines = text[text.index(heading):].split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("|"))
    rows = []
    for l in lines[start:]:
        if not l.startswith("|"):
            break
        rows.append([c.strip() for c in l.strip().strip("|").split("|")])
    return pd.DataFrame(rows[2:], columns=rows[0])
