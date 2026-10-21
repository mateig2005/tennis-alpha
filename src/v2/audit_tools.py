"""Phase 12 audit utilities: calibration with tournament-bootstrap intervals, and an independent check of the scoring model.

The exact solver here is written separately from src/v2/markov.py (explicit server tracking, generic tiebreak target),
so agreement between the two is a real check; the simulator plays matches point by point by the rules.
"""
from functools import lru_cache

import numpy as np
import pandas as pd

from src import evaluate as ev


# ---- calibration -------------------------------------------------------------------------------------------------
def favourite_view(y, p):
    """Orient every match to the model's favourite: (favourite probability, 1 if the favourite won)."""
    p, y = np.asarray(p, float), np.asarray(y)
    a_fav = p >= 0.5
    return np.where(a_fav, p, 1 - p), np.where(a_fav, y, 1 - y)


def gap_ci(f, w, groups, reps=2000, seed=0):
    """Mean (won - predicted) for favourites with a 95% interval resampling whole tournaments. Negative = overconfident."""
    return ev.paired_bootstrap(np.asarray(w, float) - np.asarray(f, float), np.zeros(len(f)), groups, reps, seed)


def calibration_table(y, p, groups, edges=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)):
    """Favourite-probability bins: n, mean prediction, observed win rate, gap and its tournament-bootstrap interval."""
    f, w = favourite_view(y, p)
    groups = np.asarray(groups)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (f >= lo) & (f < hi)
        if m.sum() < 2:
            continue
        g, l, h = gap_ci(f[m], w[m], groups[m])
        rows.append(dict(band=f"{lo:.0%}-{min(hi, 1):.0%}", n=int(m.sum()), predicted=float(f[m].mean()), observed=float(w[m].mean()), gap=g, low=l, high=h))
    return pd.DataFrame(rows)


def summary(y, p, groups):
    f, w = favourite_view(y, p)
    g, l, h = gap_ci(f, w, groups)
    return dict(n=len(f), predicted=float(f.mean()), observed=float(w.mean()), gap=g, low=l, high=h,
                log_loss=ev.log_loss(y, p), cal_slope=ev.calibration_fit(y, p)[1])


# ---- independent scoring model ---------------------------------------------------------------------------------------
def exact_match(pa, pb, best_of=3, final_tb=7):
    """P(A wins) by explicit recursion over points, games, sets and the match. pa, pb: each player's serve-point win chance.
    final_tb: tiebreak length at 6-6 in the deciding set (7 = classic; 10 = Grand Slams since 2022)."""
    def point(server_is_a):
        return pa if server_is_a else 1 - pb  # chance that A wins the point

    @lru_cache(None)
    def game(server_is_a, a, b):  # A's chance to win a game in progress
        if a >= 4 and a - b >= 2: return 1.0
        if b >= 4 and b - a >= 2: return 0.0
        q = point(server_is_a)
        if a >= 3 and a == b:  # deuce: win two in a row before losing two in a row
            return q * q / (q * q + (1 - q) * (1 - q))
        return q * game(server_is_a, a + 1, b) + (1 - q) * game(server_is_a, a, b + 1)

    def tiebreak(first_is_a, target):
        @lru_cache(None)
        def t(a, b):
            if a >= target and a - b >= 2: return 1.0
            if b >= target and b - a >= 2: return 0.0
            k = a + b  # points played; server: first player for point 0, then alternating every two points
            server_is_a = first_is_a if ((k + 1) // 2) % 2 == 0 else not first_is_a
            q = point(server_is_a)
            if a == b and a >= target - 1:  # level at target-1: the next two points have different servers; win both to win
                q2 = point(not server_is_a)
                return q * q2 / (q * q2 + (1 - q) * (1 - q2))
            return q * t(a + 1, b) + (1 - q) * t(a, b + 1)
        return t(0, 0)

    def set_(first_is_a, target):
        @lru_cache(None)
        def s(a, b):
            if (a >= 6 and a - b >= 2) or a == 7: return 1.0
            if (b >= 6 and b - a >= 2) or b == 7: return 0.0
            if a == b == 6: return tiebreak(first_is_a if (a + b) % 2 == 0 else not first_is_a, target)
            server_is_a = first_is_a if (a + b) % 2 == 0 else not first_is_a
            g = game(server_is_a, 0, 0)
            return g * s(a + 1, b) + (1 - g) * s(a, b + 1)
        return s(0, 0)

    need = best_of // 2 + 1

    @lru_cache(None)
    def m(a, b):
        if a == need: return 1.0
        if b == need: return 0.0
        decider = a == b == need - 1
        sp = set_(True, final_tb if decider else 7)  # who serves first does not change a set's outcome (checked in tests)
        return sp * m(a + 1, b) + (1 - sp) * m(a, b + 1)
    return m(0, 0)


def simulate(pa, pb, best_of=3, n=20000, seed=0, final_tb=7):
    """Monte Carlo share of matches A wins, playing every point by the rules (servers alternate games and tiebreak pairs)."""
    rng = np.random.default_rng(seed)
    need = best_of // 2 + 1
    wins = 0
    for _ in range(n):
        sets_a = sets_b = 0
        a_serves = bool(rng.integers(2))
        while sets_a < need and sets_b < need:
            ga = gb = 0
            decider = sets_a == sets_b == need - 1
            while True:
                if ga == gb == 6:  # tiebreak
                    target, ta, tb_, k, first = (final_tb if decider else 7), 0, 0, 0, a_serves
                    while not ((ta >= target or tb_ >= target) and abs(ta - tb_) >= 2):
                        srv_a = first if ((k + 1) // 2) % 2 == 0 else not first
                        if rng.random() < (pa if srv_a else 1 - pb): ta += 1
                        else: tb_ += 1
                        k += 1
                    a_set = ta > tb_
                    a_serves = not first
                    break
                pa_, pb_ = 0, 0
                while not ((pa_ >= 4 or pb_ >= 4) and abs(pa_ - pb_) >= 2):
                    if rng.random() < (pa if a_serves else 1 - pb): pa_ += 1
                    else: pb_ += 1
                if pa_ > pb_: ga += 1
                else: gb += 1
                a_serves = not a_serves
                if (ga >= 6 and ga - gb >= 2) or ga == 7: a_set = True; break
                if (gb >= 6 and gb - ga >= 2) or gb == 7: a_set = False; break
            sets_a += a_set; sets_b += not a_set
        wins += sets_a == need
    return wins / n
