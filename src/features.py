"""Pre-match form features for each player, built in one chronological pass with no peeking.

How it stays leak-free (PLAN.md M1):
- Matches are processed in groups of equal (event_date, round_rank). A group's features are computed first.
- A finished match is queued and becomes visible only at its 'release day' = event_date + lag(level).
  With release_days=MAX_DAYS (primary) that is after the whole tournament must be over; with None it is the
  Phase 3 group rule (earlier rounds of the same tournament are visible).
- Windows are measured from tournament START dates, so they can only make a match look older than it is.
Rates are shrunk toward 50% (residuals toward 0) with PRIOR pseudo-matches so debut players get a neutral value.
"""
import bisect
import heapq
from collections import defaultdict

import numpy as np
import pandas as pd

from src.elo import expected, run_elo

EPOCH = np.datetime64("2010-01-01")
PRIOR = 3.0  # fixed pseudo-matches of prior information; not tuned (PLAN.md M5)
HALF_LIVES = (30, 60, 120, 240)
SURF = {"Hard": 0, "Clay": 1, "Grass": 2}
# record layout (kept sorted by (day, round_rank, tick)): day, rr, tick, win, surf, resid, opp, release_day
DAY, RR, TICK, WIN, SRF, RES, OPP, REL = range(8)


class _Player:
    __slots__ = ("hist", "dec", "last_rel")

    def __init__(self, n_hl):
        self.hist, self.dec, self.last_rel = [], np.zeros((n_hl, 3)), None


def _days(dates):
    return ((dates.to_numpy("datetime64[D]") - EPOCH).astype(int)).tolist()


def elo_ratings(ab, k_gen, k_surf, release_days=None):
    """Pre-match general ratings (K=k_gen) and surface ratings (K=k_surf), keyed by match_id."""
    g, _ = run_elo(ab, k=k_gen, release_days=release_days)
    s = g if k_surf == k_gen else run_elo(ab, k=k_surf, release_days=release_days)[0]
    return g.set_index("match_id")[["elo_a", "elo_b"]].join(s.set_index("match_id")[["surf_a", "surf_b"]])


def _features(st, day, surf, half_lives):
    h, f = st.hist, {}
    rate = lambda w, n: (w + PRIOR * 0.5) / (n + PRIOR)
    last10 = h[-10:]
    f["wr10"] = rate(sum(r[WIN] for r in last10), len(last10))
    f["wr5"] = rate(sum(r[WIN] for r in last10[-5:]), len(last10[-5:]))
    f["adj10"] = sum(r[RES] for r in last10) / (len(last10) + PRIOR)
    f["opp10"] = sum(r[OPP] for r in last10) / (len(last10) + PRIOR)
    n = dict.fromkeys((30, 90, 365), 0); w = dict.fromkeys((30, 90, 365), 0.0); res = dict.fromkeys((30, 90, 365), 0.0)
    sn = sw = 0
    for r in reversed(h):
        age = day - r[DAY]
        if age > 365:
            break
        for W in (30, 90, 365):
            if age <= W:
                n[W] += 1; w[W] += r[WIN]; res[W] += r[RES]
        if surf >= 0 and r[SRF] == surf:
            sn += 1; sw += r[WIN]
    for W in (30, 90, 365):
        f[f"wr{W}"] = rate(w[W], n[W]); f[f"ln{W}"] = np.log1p(n[W])
    f["adj90"], f["adj365"] = res[90] / (n[90] + PRIOR), res[365] / (n[365] + PRIOR)
    f["swr365"] = rate(sw, sn)
    sl = [r[WIN] for r in reversed(h) if surf >= 0 and r[SRF] == surf][:10]
    f["swr10"] = rate(sum(sl), len(sl))
    f["rest"] = np.log1p(365 if st.last_rel is None else min(day - st.last_rel, 365))  # lower bound on days since last tournament
    f["has_hist"] = float(len(h) > 0)
    for i, hl in enumerate(half_lives):
        sw_, sn_, sr_ = st.dec[i]
        fac = 2.0 ** (-day / hl)
        f[f"dec_wr_{hl}"] = (sw_ * fac + PRIOR * 0.5) / (sn_ * fac + PRIOR)
        f[f"adj_dec_{hl}"] = sr_ * fac / (sn_ * fac + PRIOR)
    return f


def build_features(ab, ratings, release_days=None, half_lives=HALF_LIVES, return_state=False):
    """ab: A/B table from pairs.assign_ab. ratings: elo_ratings() output. Returns (features_A, features_B), indexed by match_id.
    return_state=True also returns every player's final history and the snapshot day (first day when all results are released)."""
    ab = ab.sort_values(["event_date", "round_rank", "match_id"], kind="stable")
    r = ratings.loc[ab.match_id]
    pa = expected(r.elo_a.to_numpy(), r.elo_b.to_numpy())
    ea, eb = r.elo_a.to_numpy(), r.elo_b.to_numpy()
    a, b, y, rr = ab.a_id.to_numpy(), ab.b_id.to_numpy(), ab.y.to_numpy(), ab.round_rank.to_numpy()
    surf = ab.surface.map(SURF).fillna(-1).astype(int).to_numpy()
    day = _days(ab.event_date)
    lag = [0] * len(ab) if release_days is None else ab.level.map(release_days).astype(int).tolist()
    n = len(ab)
    state = defaultdict(lambda: _Player(len(half_lives)))
    pending, tick = [], 0
    rows_a, rows_b = [None] * n, [None] * n

    def release(item):
        _, _, who, rec = item
        st = state[who]
        bisect.insort(st.hist, rec)
        st.last_rel = rec[REL] if st.last_rel is None else max(st.last_rel, rec[REL])
        for i, hl in enumerate(half_lives):
            g = 2.0 ** (rec[DAY] / hl)
            st.dec[i] += (rec[WIN] * g, g, rec[RES] * g)

    start = 0
    while start < n:
        end = start + 1
        while end < n and (day[end], rr[end]) == (day[start], rr[start]):
            end += 1
        while pending and pending[0][0] <= day[start]:  # tournaments that must have ended
            release(heapq.heappop(pending))
        for i in range(start, end):  # 1) features from released history only
            rows_a[i] = _features(state[a[i]], day[i], surf[i], half_lives)
            rows_b[i] = _features(state[b[i]], day[i], surf[i], half_lives)
        for i in range(start, end):  # 2) only now queue this group's results
            rel = day[i] + lag[i]
            for who, win, opp_elo, p in ((a[i], y[i], eb[i], pa[i]), (b[i], 1 - y[i], ea[i], 1 - pa[i])):
                rec = (day[i], rr[i], tick, float(win), surf[i], float(win - p), float(opp_elo - 1500.0), rel)
                heapq.heappush(pending, (rel, tick, who, rec)); tick += 1
        start = end
    idx = ab.match_id.to_numpy()
    fa, fb = pd.DataFrame(rows_a, index=idx), pd.DataFrame(rows_b, index=idx)
    if not return_state:
        return fa, fb
    snapshot_day = max(d + l for d, l in zip(day, lag))
    while pending:
        release(heapq.heappop(pending))
    return fa, fb, dict(state), snapshot_day


def player_features(state, player, day, surface, half_lives=HALF_LIVES):
    """Features of one player for a hypothetical match on `day` (days since EPOCH); unknown players get the debut values."""
    return _features(state.get(player) or _Player(len(half_lives)), day, SURF.get(surface, -1), half_lives)


def design(ab, ratings, fa, fb):
    """All model inputs as differences A minus B, so swapping A and B flips every sign (exact symmetry)."""
    ab = ab.set_index("match_id").loc[fa.index]
    r = ratings.loc[fa.index]
    from src.evaluate import UNRANKED
    d = fa - fb
    d["elo"] = (r.elo_a - r.elo_b) / 400.0
    d["surf_elo"] = (r.surf_a - r.surf_b) / 400.0
    ra, rb = ab.rank_a.isna(), ab.rank_b.isna()
    d["lrank"] = np.log(ab.rank_a.fillna(UNRANKED)) - np.log(ab.rank_b.fillna(UNRANKED))
    d["rank_miss"] = ra.astype(float) - rb.astype(float)
    pa_, pb_ = ab.pts_a.isna(), ab.pts_b.isna()
    d["lpts"] = np.log1p(ab.pts_a.fillna(0)) - np.log1p(ab.pts_b.fillna(0))
    d["pts_miss"] = pa_.astype(float) - pb_.astype(float)
    return d
