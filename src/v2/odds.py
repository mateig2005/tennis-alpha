"""Probability <-> decimal odds, bookmaker margin removal, and a market benchmark (PLAN_V2.md Q6).

No market prices exist in the approved data: these tools are ready for a future, licensed odds source.
Odds are a benchmark only and must never become a model input.
"""
import numpy as np
import pandas as pd
from scipy.optimize import brentq

from src import evaluate as ev


def fair_odds(p):
    """Decimal odds with no margin for probability p (a 60% chance = 1.667)."""
    p = np.asarray(p, float)
    if ((p <= 0) | (p >= 1)).any():
        raise ValueError("probabilities must be strictly between 0 and 1")
    return 1.0 / p


def overround(odds_a, odds_b):
    """Bookmaker margin: implied probabilities summed minus 1 (0.05 = 5%)."""
    return 1.0 / np.asarray(odds_a, float) + 1.0 / np.asarray(odds_b, float) - 1.0


def no_vig(odds_a, odds_b, method="proportional"):
    """Market probability for A with the margin removed. 'proportional' rescales both implied probabilities;
    'power' finds k with (1/oa)^k + (1/ob)^k = 1, which takes more margin from the longshot."""
    ia, ib = 1.0 / np.asarray(odds_a, float), 1.0 / np.asarray(odds_b, float)
    if ((ia <= 0) | (ib <= 0)).any() or (ia + ib < 1 - 1e-9).any():
        raise ValueError("odds must be > 1 and include a non-negative margin")
    if method == "proportional":
        return ia / (ia + ib)
    if method == "power":
        k = np.array([brentq(lambda k: x ** k + y ** k - 1.0, 1e-6, 50.0) if x + y > 1 + 1e-12 else 1.0 for x, y in zip(np.atleast_1d(ia), np.atleast_1d(ib))])
        return np.atleast_1d(ia) ** k
    raise ValueError(f"unknown method {method}")


def benchmark(df, models, method="proportional", bins=(0, 0.05, 0.1, 1.0)):
    """Compare model probabilities with no-vig market probabilities on matched rows.

    df columns: match_id, y (1 if A won), odds_a, odds_b, odds_time, match_start, and one probability column per model.
    Rows whose odds are not recorded strictly before the match start are refused (raises), never silently dropped.
    Returns (overall table, table by size of the model-market disagreement for the first model).
    """
    if not (pd.to_datetime(df.odds_time) < pd.to_datetime(df.match_start)).all():
        raise ValueError("odds recorded at or after the match start: not a pre-match benchmark")
    if df.match_id.duplicated().any():
        raise ValueError("duplicate match_id: one price per match")
    y, pm = df.y.to_numpy(), no_vig(df.odds_a, df.odds_b, method)
    rows = [dict(source="market (no-vig)", n=len(df), log_loss=ev.log_loss(y, pm), brier=ev.brier(y, pm), accuracy=ev.accuracy(y, pm), mean_abs_gap=0.0)]
    for m in models:
        p = df[m].to_numpy()
        rows.append(dict(source=m, n=len(df), log_loss=ev.log_loss(y, p), brier=ev.brier(y, p), accuracy=ev.accuracy(y, p), mean_abs_gap=float(np.mean(np.abs(p - pm)))))
    gap = np.abs(df[models[0]].to_numpy() - pm)
    by = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        k = (gap >= lo) & (gap < hi)
        if k.any():
            by.append(dict(gap=f"{lo:.2f}-{hi:.2f}", n=int(k.sum()), model_log_loss=ev.log_loss(y[k], df[models[0]].to_numpy()[k]),
                           market_log_loss=ev.log_loss(y[k], pm[k])))
    return pd.DataFrame(rows), pd.DataFrame(by)


def match_odds(odds, matches, window_days=7):
    """Attach pre-match odds to matches. odds: p1_id, p2_id (already linked to our ids), odds_p1, odds_p2, odds_time.
    matches: match_id, a_id, b_id, event_date (tournament start or match day). A price matches when the two players are the
    same and odds_time falls within [event_date - 1 day, event_date + window_days]. Prices are re-oriented to player A.
    Returns (matched rows, unmatched odds rows, ambiguous odds rows): nothing is guessed."""
    m = matches.assign(pair=[tuple(sorted(p)) for p in zip(matches.a_id, matches.b_id)])
    o = odds.reset_index(drop=True).assign(pair=[tuple(sorted(p)) for p in zip(odds.p1_id, odds.p2_id)], row=lambda d: d.index)
    j = o.merge(m, on="pair")
    t, start = pd.to_datetime(j.odds_time), pd.to_datetime(j.event_date)
    j = j[(t >= start - pd.Timedelta(days=1)) & (t <= start + pd.Timedelta(days=window_days))]
    count = j.groupby("row").size()
    ambiguous = o[o.row.isin(count[count > 1].index)]
    j = j[j.row.isin(count[count == 1].index)]
    a_is_p1 = j.a_id == j.p1_id
    out = j.assign(odds_a=np.where(a_is_p1, j.odds_p1, j.odds_p2), odds_b=np.where(a_is_p1, j.odds_p2, j.odds_p1))
    unmatched = o[~o.row.isin(count.index)]
    return out[["match_id", "odds_a", "odds_b", "odds_time"]].reset_index(drop=True), unmatched.drop(columns=["pair", "row"]), ambiguous.drop(columns=["pair", "row"])

