"""V2 pre-match features: serve/return profiles, matchup model, tournament pace, H2H and momentum (PLAN_V2.md section 3).

Same leak-free structure as V1 (src/features.py): matches are processed in groups of equal (event_date, round_rank);
a group's features are computed first, then its results AND statistics are queued and only applied on the release
day (tournament start + MAX_DAYS). So statistics, pace and H2H can never include the match being predicted, later
rounds of its tournament, or anything later. Every rate is shrunk toward a prior with the fixed weights in models/v2_config.json.
"""
import bisect
import heapq
from collections import defaultdict
from functools import partial

import numpy as np
import pandas as pd

from src.features import SURF, _days
from src.elo import expected
from src.v2 import markov
from src.v2.data import CONFIG, STATS, orient

PR, HL = CONFIG["priors"], CONFIG["half_lives_days"]
WINDOW, TREND = CONFIG["raw_window_matches"], CONFIG["elo_trend_days"]
FALLBACK = 0.5  # rate used only before the first released match (warm-up weeks of 2010, never scored)
ALL = 3         # index of the all-surfaces tour average
# raw rolling rates: name -> (numerator count, denominator count, prior weight key)
RATES = {"ace": ("ace", "svpt", "raw_points"), "df": ("df", "svpt", "raw_points"), "fin": ("fin", "svpt", "raw_points"),
         "fwon": ("fwon", "fin", "raw_points"), "swon": ("swon", "spts2", "raw_points"), "spw": ("svwon", "svpt", "raw_points"),
         "hold": ("held", "sg", "raw_games"), "rpw": ("rwon", "rpts", "raw_points"), "r1": ("r1won", "r1pts", "raw_points"),
         "r2": ("r2won", "r2pts", "raw_points"), "brk": ("brk", "rg", "raw_games"),
         "bpsv": ("bps", "bpf", "raw_break_points"), "bpcv": ("brk", "bpc", "raw_break_points")}
COUNTS = sorted({k for n, d, _ in RATES.values() for k in (n, d)})
GROUPS = {"sr_adj": ["srv", "ret", "srv_s", "ret_s"],
          "sr_raw": ["ace", "df", "fin", "fwon", "swon", "spw", "hold", "rpw", "r1", "r2", "brk"],
          "pressure": ["bpsv", "bpcv"],
          "matchup": ["mk", "dom_cross"],
          "pace": ["px_srv", "px_ret", "px_ace"],
          "h2h": ["h2h", "h2h_s"],
          "context": ["bo5_elo"],
          "momentum": ["srv60", "ret60", "elo_trend"]}


def counts(me, opp):
    """One player's serve and return counts in one match, from both players' raw statistics (dicts keyed by STATS)."""
    held = max(me["SvGms"] - (me["bpFaced"] - me["bpSaved"]), 0.0)
    return {"ace": me["ace"], "df": me["df"], "svpt": me["svpt"], "fin": me["1stIn"], "fwon": me["1stWon"],
            "spts2": me["svpt"] - me["1stIn"], "swon": me["2ndWon"], "svwon": me["1stWon"] + me["2ndWon"],
            "sg": me["SvGms"], "held": held, "bpf": me["bpFaced"], "bps": me["bpSaved"],
            "rpts": opp["svpt"], "rwon": opp["svpt"] - opp["1stWon"] - opp["2ndWon"], "r1pts": opp["1stIn"],
            "r1won": opp["1stIn"] - opp["1stWon"], "r2pts": opp["svpt"] - opp["1stIn"],
            "r2won": opp["svpt"] - opp["1stIn"] - opp["2ndWon"], "rg": opp["SvGms"],
            "brk": opp["bpFaced"] - opp["bpSaved"], "bpc": opp["bpFaced"]}


class _Player:
    __slots__ = ("raw", "adj", "surf", "elo")

    def __init__(self):
        self.raw = []                      # sorted (day, rr, tick, counts) of released matches with statistics
        self.adj = {h: np.zeros(4) for h in HL.values()}  # decayed sums: srv*w, w_srv, ret*w, w_ret
        self.surf = np.zeros((3, 4))       # same, per surface, main half-life
        self.elo = []                      # sorted (day, rr, tick, pre-match general Elo) of released matches


class _Tour:
    """Tour-wide running totals (released matches only): surface averages, rate priors, pace and H2H."""

    def __init__(self):
        self.sv = np.zeros((4, 2))         # serve points won, serve points: per surface + all
        self.tot = dict.fromkeys(COUNTS, 0.0)
        self.pace = defaultdict(partial(np.zeros, 3))   # (venue, surface) -> sum ace residual, sum spw residual, n
        self.pace_s = defaultdict(partial(np.zeros, 3))  # surface -> same
        self.h2h = defaultdict(partial(np.zeros, 2))     # (low id, high id[, surface]) -> sum residual of low id, n

    def mu(self, s):
        k = ALL if s < 0 else s
        return self.sv[k, 0] / self.sv[k, 1] if self.sv[k, 1] else FALLBACK

    def rate(self, num, den):
        return self.tot[num] / self.tot[den] if self.tot[den] else FALLBACK


def _rating(v, fac, prior):
    return v[0] * fac / (v[1] * fac + prior), v[2] * fac / (v[3] * fac + prior)


def player_features(st, tour, day, s, elo_now):
    """One player's V2 features for a match on `day` (days since EPOCH) on surface index s (-1 = unknown)."""
    f = {}
    fac = 2.0 ** (-day / HL["main"])
    f["srv"], f["ret"] = _rating(st.adj[HL["main"]], fac, PR["adj_points"])
    f["srv60"], f["ret60"] = _rating(st.adj[HL["recent"]], 2.0 ** (-day / HL["recent"]), PR["adj_points"])
    if s >= 0:  # surface rating shrunk toward the player's general rating
        v, k = st.surf[s], PR["surface_points"]
        f["srv_s"] = (v[0] * fac + k * f["srv"]) / (v[1] * fac + k)
        f["ret_s"] = (v[2] * fac + k * f["ret"]) / (v[3] * fac + k)
    else:
        f["srv_s"], f["ret_s"] = f["srv"], f["ret"]
    tot = dict.fromkeys(COUNTS, 0.0)
    for r in st.raw[-WINDOW:]:
        for c in COUNTS:
            tot[c] += r[3][c]
    for name, (num, den, pk) in RATES.items():
        f[name] = (tot[num] + PR[pk] * tour.rate(num, den)) / (tot[den] + PR[pk])
    j = bisect.bisect_left(st.elo, (day - TREND,))  # first released match inside the trend window
    f["elo_trend"] = (elo_now - st.elo[j][3]) / 400.0 if j < len(st.elo) else 0.0
    return f


def match_context(fa, fb, tour, s, best_of, venue, a, b):
    """Match-level V2 values, oriented to A: matchup probability, pace index and head-to-head."""
    mu = tour.mu(s)
    pa, pb = mu + fa["srv_s"] - fb["ret_s"], mu + fb["srv_s"] - fa["ret_s"]
    p = min(max(markov.match_prob(pa, pb, best_of), 1e-6), 1 - 1e-6)
    ps, pv = tour.pace_s[s], tour.pace[(venue, s)]
    mean = ps[:2] / ps[2] if ps[2] else np.zeros(2)
    ace_i, spw_i = (pv[:2] + PR["pace_matches"] * mean) / (pv[2] + PR["pace_matches"])
    sign = 1.0 if a < b else -1.0  # stored residuals belong to the lower id
    key = (min(a, b), max(a, b))
    h, hs = tour.h2h[key], tour.h2h[key + (s,)]
    return {"mk": float(np.log(p / (1 - p))), "p_mk": p, "dom_cross": fa["srv"] * fb["ret"] - fb["srv"] * fa["ret"],
            "pace_ace": ace_i, "pace_spw": spw_i, "pace_n": pv[2],
            "h2h": sign * h[0] / (h[1] + PR["h2h_matches"]), "h2h_s": sign * hs[0] / (hs[1] + PR["h2h_matches"]),
            "h2h_n": h[1], "bo5": float(best_of == 5), "pa": pa, "pb": pb, "mu": mu}


def build(ab, ratings, release_days, return_state=False):
    """ab: assign_ab output of a v2.data table. ratings: V1 elo_ratings() (pre-match). Returns (fa, fb, ctx) by match_id."""
    ab = ab.sort_values(["event_date", "round_rank", "match_id"], kind="stable")
    r = ratings.loc[ab.match_id]
    ea, eb = r.elo_a.to_numpy(), r.elo_b.to_numpy()
    pe = expected(ea, eb)
    a, b, y, rr = ab.a_id.to_numpy(), ab.b_id.to_numpy(), ab.y.to_numpy(), ab.round_rank.to_numpy()
    surf = ab.surface.map(SURF).fillna(-1).astype(int).to_numpy()
    bo, venue, has = ab.best_of.to_numpy(), ab.venue.to_numpy(), ab.has_stats.to_numpy(bool)
    S = orient(ab)
    day = _days(ab.event_date)
    lag = ab.level.map(release_days).astype(int).tolist()
    n = len(ab)
    state, tour = defaultdict(_Player), _Tour()
    pending, tick = [], 0
    rows_a, rows_b, ctx = [None] * n, [None] * n, [None] * n

    def release(item):
        _, _, m = item
        for who, rec in m["elo"]:
            bisect.insort(state[who].elo, rec)
        k = m["key"]
        tour.h2h[k][0] += m["h2h"]; tour.h2h[k][1] += 1
        if m["s"] >= 0:
            ks = k + (m["s"],); tour.h2h[ks][0] += m["h2h"]; tour.h2h[ks][1] += 1
        if "stats" not in m:
            return
        for who, c, xs, xr in m["stats"]:
            st = state[who]
            bisect.insort(st.raw, (m["day"], m["rr"], m["tick"], c))
            for cc in COUNTS:
                tour.tot[cc] += c[cc]
            for kk in (ALL, m["s"]) if m["s"] >= 0 else (ALL,):
                tour.sv[kk] += (c["svwon"], c["svpt"])
            if not m["resid"]:
                continue
            for h in HL.values():
                g = 2.0 ** (m["day"] / h)
                st.adj[h] += (xs * c["svpt"] * g, c["svpt"] * g, xr * c["rpts"] * g, c["rpts"] * g)
            if m["s"] >= 0:
                g = 2.0 ** (m["day"] / HL["main"])
                st.surf[m["s"]] += (xs * c["svpt"] * g, c["svpt"] * g, xr * c["rpts"] * g, c["rpts"] * g)
        if not m["resid"]:
            return
        res = np.array([m["pace"][0], m["pace"][1], 1.0])
        tour.pace[(m["venue"], m["s"])] += res; tour.pace_s[m["s"]] += res

    start = 0
    while start < n:
        end = start + 1
        while end < n and (day[end], rr[end]) == (day[start], rr[start]):
            end += 1
        while pending and pending[0][0] <= day[start]:
            release(heapq.heappop(pending))
        for i in range(start, end):  # 1) features from released information only
            rows_a[i] = player_features(state[a[i]], tour, day[i], surf[i], ea[i])
            rows_b[i] = player_features(state[b[i]], tour, day[i], surf[i], eb[i])
            ctx[i] = match_context(rows_a[i], rows_b[i], tour, surf[i], bo[i], venue[i], a[i], b[i])
        for i in range(start, end):  # 2) only now queue this group's results and statistics
            lo_is_a = a[i] < b[i]
            m = {"day": day[i], "rr": rr[i], "tick": tick, "s": surf[i], "venue": venue[i],
                 "key": (min(a[i], b[i]), max(a[i], b[i])),
                 "h2h": (y[i] - pe[i]) if lo_is_a else ((1 - y[i]) - (1 - pe[i])),
                 "elo": [(a[i], (day[i], rr[i], tick, ea[i])), (b[i], (day[i], rr[i], tick, eb[i]))]}
            if has[i]:
                sa = {s_: S[f"a_{s_}"][i] for s_ in STATS}
                sb = {s_: S[f"b_{s_}"][i] for s_ in sa}
                ca, cb = counts(sa, sb), counts(sb, sa)
                fa_, fb_, mu, mu_all = rows_a[i], rows_b[i], tour.mu(surf[i]), tour.mu(-1)
                spw_a, spw_b = ca["svwon"] / ca["svpt"], cb["svwon"] / cb["svpt"]
                # opponent-adjusted samples: own result minus the surface average plus the opponent's pre-match rating
                m["stats"] = [(a[i], ca, spw_a - mu + fb_["ret"], (1 - spw_b) - (1 - mu) + fb_["srv"]),
                              (b[i], cb, spw_b - mu + fa_["ret"], (1 - spw_a) - (1 - mu) + fa_["srv"])]
                pts = ca["svpt"] + cb["svpt"]
                exp_ace = (fa_["ace"] * ca["svpt"] + fb_["ace"] * cb["svpt"]) / pts
                exp_spw = ((mu_all + fa_["srv"] - fb_["ret"]) * ca["svpt"] + (mu_all + fb_["srv"] - fa_["ret"]) * cb["svpt"]) / pts
                m["pace"] = ((ca["ace"] + cb["ace"]) / pts - exp_ace, (ca["svwon"] + cb["svwon"]) / pts - exp_spw)
                m["resid"] = tour.sv[ALL, 1] > 0  # residuals need a tour average: none exists in the first week of 2010 (warm-up)
            heapq.heappush(pending, (day[i] + lag[i], tick, m)); tick += 1
        start = end
    idx = ab.match_id.to_numpy()
    out = pd.DataFrame(rows_a, index=idx), pd.DataFrame(rows_b, index=idx), pd.DataFrame(ctx, index=idx)
    if not return_state:
        return out
    while pending:
        release(heapq.heappop(pending))
    return (*out, dict(state), tour)


def design(D1, fa, fb, ctx):
    """V1 Model D inputs plus V2 columns, all antisymmetric in A/B (swap A and B and every value changes sign)."""
    X = D1.loc[fa.index].copy()
    for c in GROUPS["sr_adj"] + GROUPS["sr_raw"] + GROUPS["pressure"] + ["srv60", "ret60", "elo_trend"]:
        X[c] = fa[c] - fb[c]
    for c in ("mk", "dom_cross", "h2h", "h2h_s"):
        X[c] = ctx[c]
    X["px_srv"], X["px_ret"] = ctx.pace_spw * X.srv, ctx.pace_spw * X.ret
    X["px_ace"] = ctx.pace_ace * X.ace
    X["bo5_elo"] = ctx.bo5 * X.elo
    return X
