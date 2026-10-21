"""Recalibration that keeps P(A) = 1 - P(B) exact (PLAN_V2.md Q4). Fit on earlier predictions, apply to later ones.

Platt slope-only: p' = sigmoid(a * logit(p)); no intercept, because A/B order is random.
Isotonic: fitted on the predictions AND their mirror (1 - p, 1 - y), then averaged with its mirror, so f(1 - p) = 1 - f(p).
"""
import numpy as np
from sklearn.isotonic import IsotonicRegression


def _logit(p):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def platt_fit(p, y, iters=50):
    """Slope a of the logistic fit of y on logit(p), by Newton's method. a < 1 means the inputs were overconfident."""
    x, y, a = _logit(p), np.asarray(y, float), 1.0
    for _ in range(iters):
        q = 1 / (1 + np.exp(-a * x))
        a += np.sum((y - q) * x) / np.sum(q * (1 - q) * x * x)
    return float(a)


def platt_apply(a, p):
    return 1 / (1 + np.exp(-a * _logit(p)))


def isotonic_fit(p, y):
    p, y = np.asarray(p, float), np.asarray(y, float)
    iso = IsotonicRegression(y_min=1e-4, y_max=1 - 1e-4, out_of_bounds="clip").fit(np.concatenate([p, 1 - p]), np.concatenate([y, 1 - y]))
    return lambda q: (iso.predict(np.asarray(q, float)) + 1 - iso.predict(1 - np.asarray(q, float))) / 2
