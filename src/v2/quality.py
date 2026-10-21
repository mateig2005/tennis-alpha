"""V2.1 match-quality and statistical-momentum features (PLAN_V2.md, V2.1 protocol Q1-Q2).

For every finished match, each player's performance is compared with a PRE-match expectation built from both players'
V2 serve/return ratings on that surface (opponent- and surface-adjusted):
  tpw  = total points won share - expected share        srv/ret = serve / return points won % - expected
  dom  = games won share - share implied by expected holds
  luck = result - match-win chance implied by the points ACTUALLY won (positive = won more than the points deserved)
Records are released with the same tournament-completion rule as results, so no feature sees its own match or later ones.
"""
import bisect
import heapq
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from src.features import SURF, _days
from src.v2 import markov
from src.v2.data import orient
from src.v2.score import parse

CFG = json.loads((Path(__file__).resolve().parents[2] / "models/v21_config.json").read_text())
P = CFG["quality_prior_matches"]
GROUPS = {"quality": ["pq3", "pq5", "pq10", "pq20", "dom5", "dom20", "luck10"],
          "qmomentum": ["pq_trend", "srv_trend", "ret_trend", "pq_vol10", "pq_surf10", "load30"]}
VOL_FALLBACK = 0.05 ** 2  # spread used only before the first released record (warm-up weeks of 2010)


def _clip(p):
    return min(max(p, 0.3), 0.95)


def match_records(y, sc, has, sa, sb, pa, pb, best_of):
    """Performance records (A, B) for one match. sc: parse() result in the winner's view; sa/sb: A's and B's statistics.
    Returns (rec_a, rec_b, sets_played); a record is None when the score is incomplete or unparseable."""
    if sc is None:
        return None, None, 0
    sets = sc["sets_w"] + sc["sets_l"]
    if not sc["complete"]:
        return None, None, sets
    ga, gb = (sc["games_w"], sc["games_l"]) if y == 1 else (sc["games_l"], sc["games_w"])
    dom = ga / (ga + gb) - (markov.game(_clip(pa)) + 1 - markov.game(_clip(pb))) / 2
    ra, rb = {"dom": dom}, {"dom": -dom}
    if has:
        won_a, won_b = sa["1stWon"] + sa["2ndWon"], sb["1stWon"] + sb["2ndWon"]
        spw_a, spw_b = won_a / sa["svpt"], won_b / sb["svpt"]
        pts = sa["svpt"] + sb["svpt"]
        tpw = (won_a + sb["svpt"] - won_b) / pts - (sa["svpt"] * pa + sb["svpt"] * (1 - pb)) / pts
        luck = y - markov.match_prob(spw_a, spw_b, best_of)
        ra.update(tpw=tpw, srv=spw_a - pa, ret=pb - spw_b, luck=luck)
        rb.update(tpw=-tpw, srv=spw_b - pb, ret=pa - spw_a, luck=-luck)
    return ra, rb, sets


class _Q:
    __slots__ = ("all", "stat", "load")

    def __init__(self):
        self.all, self.stat, self.load = [], [], []  # sorted (day, rr, tick, record or sets)


def _shr(vals):
    return sum(vals) / (len(vals) + P)


def player_features(st, day, s, vol0):
    st = st or _Q()
    t = [r[3]["tpw"] for r in st.stat]
    f = {f"pq{k}": _shr(t[-k:]) for k in CFG["quality_windows"]}
    dom = [r[3]["dom"] for r in st.all]
    f["dom5"], f["dom20"] = _shr(dom[-5:]), _shr(dom[-20:])
    f["luck10"] = _shr([r[3]["luck"] for r in st.stat[-10:]])
    s_, l_ = CFG["trend_windows"]
    for key, name in (("tpw", "pq_trend"), ("srv", "srv_trend"), ("ret", "ret_trend")):
        v = [r[3][key] for r in st.stat[-l_:]]
        f[name] = _shr(v[-s_:]) - _shr(v)
    v = t[-CFG["volatility_window"]:]
    f["pq_vol10"] = math.sqrt((sum(x * x for x in v) + P * vol0) / (len(v) + P))
    same = []
    for r in reversed(st.stat):
        if len(same) == CFG["surface_window"]:
            break
        if s >= 0 and r[3]["s"] == s:
            same.append(r[3]["tpw"])
    f["pq_surf10"] = _shr(same)
    sets = 0
    for r in reversed(st.load):
        if day - r[0] > CFG["load_days"]:
            break
        sets += r[3]
    f["load30"] = math.log1p(sets)
    return f


def build(ab, ctx, release_days, return_state=False):
    """ab: assign_ab output of a v2.data table (needs `score`). ctx: V2 match context (pre-match pa, pb). Returns (fa, fb)."""
    ab = ab.sort_values(["event_date", "round_rank", "match_id"], kind="stable")
    c = ctx.loc[ab.match_id]
    pa_, pb_ = c.pa.to_numpy(), c.pb.to_numpy()
    a, b, y, rr = ab.a_id.to_numpy(), ab.b_id.to_numpy(), ab.y.to_numpy(), ab.round_rank.to_numpy()
    surf = ab.surface.map(SURF).fillna(-1).astype(int).to_numpy()
    bo, has, score = ab.best_of.to_numpy(), ab.has_stats.to_numpy(bool), ab.score.to_numpy()
    S = orient(ab)
    stats = [k[2:] for k in S if k.startswith("a_")]
    day = _days(ab.event_date)
    lag = ab.level.map(release_days).astype(int).tolist()
    n = len(ab)
    state, tour = defaultdict(_Q), [0.0, 0]  # tour: sum of squared points residuals, count (volatility prior)
    pending, tick = [], 0
    rows_a, rows_b = [None] * n, [None] * n

    def release(item):
        _, _, recs = item
        for who, key, rec, sets in recs:
            st = state[who]
            if sets:
                bisect.insort(st.load, key + (sets,))
            if rec is not None:
                bisect.insort(st.all, key + (rec,))
                if "tpw" in rec:
                    bisect.insort(st.stat, key + (rec,))
                    tour[0] += rec["tpw"] ** 2; tour[1] += 1

    start = 0
    while start < n:
        end = start + 1
        while end < n and (day[end], rr[end]) == (day[start], rr[start]):
            end += 1
        while pending and pending[0][0] <= day[start]:
            release(heapq.heappop(pending))
        vol0 = tour[0] / tour[1] if tour[1] else VOL_FALLBACK
        for i in range(start, end):
            rows_a[i] = player_features(state.get(a[i]), day[i], surf[i], vol0)
            rows_b[i] = player_features(state.get(b[i]), day[i], surf[i], vol0)
        for i in range(start, end):
            sa = {k: S[f"a_{k}"][i] for k in stats}
            sb = {k: S[f"b_{k}"][i] for k in stats}
            ra, rb, sets = match_records(y[i], parse(score[i]), has[i], sa, sb, pa_[i], pb_[i], bo[i])
            for r in (ra, rb):
                if r is not None:
                    r["s"] = surf[i]
            key = (day[i], rr[i], tick)
            heapq.heappush(pending, (day[i] + lag[i], tick, [(a[i], key, ra, sets), (b[i], key, rb, sets)])); tick += 1
        start = end
    idx = ab.match_id.to_numpy()
    fa, fb = pd.DataFrame(rows_a, index=idx), pd.DataFrame(rows_b, index=idx)
    if not return_state:
        return fa, fb
    while pending:
        release(heapq.heappop(pending))
    return fa, fb, dict(state), (tour[0] / tour[1] if tour[1] else VOL_FALLBACK)


def design(X2, fa, fb):
    """Full V2 inputs plus V2.1 differences (A minus B: swapping players flips every sign)."""
    X = X2.loc[fa.index].copy()
    for c in GROUPS["quality"] + GROUPS["qmomentum"]:
        X[c] = fa[c] - fb[c]
    return X
