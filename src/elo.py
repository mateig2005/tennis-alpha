"""Elo ratings (general and per-surface) with leak-free group updates.

Elo in one line: every player has a rating; the expected win chance comes from the rating gap;
after a match the winner takes K * (1 - expected) points from the loser, so upsets move ratings more.

No match dates exist, so matches sharing (event_date, round_rank) form a group treated as simultaneous:
all predictions in a group use the ratings from BEFORE the group, and only then are ratings updated.
"""
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd


def expected(ra, rb):
    """Chance that a player rated ra beats one rated rb (400 points = 10x odds)."""
    return 1.0 / (1.0 + 10.0 ** ((np.asarray(rb) - np.asarray(ra)) / 400.0))


def run_elo(df, k=32.0, init=1500.0, update_levels=None, release_days=None):
    """df: A/B table from pairs.assign_ab. Returns (ratings table with PRE-match ratings, final state).

    update_levels: tournament levels allowed to change ratings (None = all). Other levels are still predicted.
    release_days: optional {level: days}. A result then becomes visible only once the tournament must have ended
    (event_date + days <= the later match's event_date): the strict rule for overlapping tournaments and unknown dates.
    """
    df = df.sort_values(["event_date", "round_rank", "match_id"], kind="stable")  # deterministic order
    a, b, y = df.a_id.to_numpy(), df.b_id.to_numpy(), df.y.to_numpy()
    surf = df.surface.where(df.surface.notna(), None).to_numpy()
    upd = np.ones(len(df), bool) if update_levels is None else df.level.isin(update_levels).to_numpy()
    key = list(zip(df.event_date.to_numpy(), df.round_rank.to_numpy()))
    n = len(df)
    ga, gb, sa, sb = (np.empty(n) for _ in range(4))
    gen, sur = defaultdict(lambda: init), defaultdict(lambda: init)
    when = df.event_date.to_numpy()
    lag = np.zeros(n, "timedelta64[D]") if release_days is None else df.level.map(release_days).to_numpy().astype("timedelta64[D]")
    pending, tick = [], 0  # heap of (release date, tie-breaker, is_surface, rating key, points)

    start = 0
    while start < n:
        end = start + 1
        while end < n and key[end] == key[start]:
            end += 1
        while pending and pending[0][0] <= when[start]:  # results from tournaments that have ended become visible
            _, _, is_s, who, d = heapq.heappop(pending)
            (sur if is_s else gen)[who] += d
        for i in range(start, end):  # 1) predict the whole group with pre-group ratings
            ga[i], gb[i] = gen[a[i]], gen[b[i]]
            sa[i], sb[i] = (sur[(surf[i], a[i])], sur[(surf[i], b[i])]) if surf[i] else (ga[i], gb[i])
        for i in range(start, end):  # 2) queue updates computed from pre-group ratings; none is applied inside the group
            if not upd[i]:
                continue
            win, lose = (a[i], b[i]) if y[i] else (b[i], a[i])
            rel = when[i] + lag[i]
            d = k * (1.0 - float(expected(*((ga[i], gb[i]) if y[i] else (gb[i], ga[i])))))
            for who, pts in ((win, d), (lose, -d)):
                heapq.heappush(pending, (rel, tick, False, who, pts)); tick += 1
            if surf[i]:
                d = k * (1.0 - float(expected(*((sa[i], sb[i]) if y[i] else (sb[i], sa[i])))))
                for who, pts in (((surf[i], win), d), ((surf[i], lose), -d)):
                    heapq.heappush(pending, (rel, tick, True, who, pts)); tick += 1
        start = end
    while pending:  # final state: everything released
        _, _, is_s, who, d = heapq.heappop(pending)
        (sur if is_s else gen)[who] += d

    out = pd.DataFrame({"match_id": df.match_id.to_numpy(), "elo_a": ga, "elo_b": gb, "surf_a": sa, "surf_b": sb})
    return out, (dict(gen), dict(sur))


def prob(ratings, w=0.0):
    """P(A wins). w=0: general Elo; w=1: surface Elo; between: blend of the two ratings."""
    ra = w * ratings.surf_a + (1 - w) * ratings.elo_a
    rb = w * ratings.surf_b + (1 - w) * ratings.elo_b
    return pd.Series(expected(ra, rb), index=ratings.index)
