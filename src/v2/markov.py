"""Service-return matchup model: from each player's chance of winning a point on serve to a match-win probability.

Standard independent-points model (Barnett & Clarke; Newton & Keller): games, tiebreak sets, best of 3 or 5.
The result does not depend on who serves first, which is what keeps P(A) = 1 - P(B) exact (tested).
Final sets are treated as tiebreak sets (true for all Slams since 2022; a small approximation before).
"""
from functools import lru_cache

import numpy as np


def game(p):
    """Chance the server wins a service game when winning each point with probability p."""
    q = 1.0 - p
    return p ** 4 * (1 + 4 * q + 10 * q * q) + 20 * p ** 3 * q ** 3 * p * p / (1 - 2 * p * q)


def tiebreak(pa, pb):
    """Chance A wins a 7-point tiebreak; A serves point 1, then each player serves two points in turn."""
    @lru_cache(None)
    def f(a, b):
        if a == 7 and b <= 5:
            return 1.0
        if b == 7 and a <= 5:
            return 0.0
        if a == b == 6:  # from 6-6 each pair of points has one serve each: win both to win
            x, y = pa * (1 - pb), (1 - pa) * pb
            return x / (x + y)
        k = a + b + 1  # next point number
        p = pa if (k == 1 or (k // 2) % 2 == 0) else 1 - pb
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)
    return f(0, 0)


def set_prob(pa, pb):
    """Chance A wins a tiebreak set, A serving the first game."""
    ga, gb = game(pa), game(pb)

    @lru_cache(None)
    def f(a, b):
        if a == 6 and b <= 4 or a == 7:
            return 1.0
        if b == 6 and a <= 4 or b == 7:
            return 0.0
        if a == b == 6:
            return tiebreak(pa, pb)
        p = ga if (a + b) % 2 == 0 else 1 - gb
        return p * f(a + 1, b) + (1 - p) * f(a, b + 1)
    return f(0, 0)


def match_prob(pa, pb, best_of=3):
    """Chance A wins the match. pa, pb: each player's chance of winning a point on own serve."""
    pa, pb = float(np.clip(pa, 0.3, 0.95)), float(np.clip(pb, 0.3, 0.95))  # outside this range the inputs are not credible
    s = set_prob(pa, pb)
    t = 1 - s
    return s * s * (1 + 2 * t) if best_of == 3 else s ** 3 * (1 + 3 * t + 6 * t * t)
