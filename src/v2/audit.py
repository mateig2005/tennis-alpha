"""Diagnostic case study: why does Model D give Sinner 60.7% against Alcaraz on clay at the June 2026 snapshot? (PLAN_V2.md 5)

Run from the project root:  python -m src.v2.audit   (needs artifacts/dashboard.pkl)
Writes reports/v2_sinner_alcaraz_audit.md. It reads the locked 2025/2026 file ONLY to rebuild the players' history up to
the snapshot, exactly as the dashboard does. Nothing here fits, tunes or selects anything with 2025/2026 outcomes;
the diagnostic V2 model is fitted on 2015-2024 only and is not a production prediction.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate as ev
from src import models
from src.canonical import DEV_PATH, LOCKED_PATH, MAX_DAYS, SCORED_LEVELS, load_with_locked
from src.dataset import ELO_K_GEN, ELO_K_SURF
from src.elo import run_elo
from src.features import SURF, elo_ratings
from src.pairs import assign_ab
from src.predictor import Predictor
from src.v2 import features as F
from src.v2 import odds
from src.v2 import quality as Q
from src.v2 import run_v21
from src.v2.data import from_sackmann_v2

OUT = Path("reports/v2_sinner_alcaraz_audit.md")
SIN, ALC = "sackmann:206173", "sackmann:207989"
f3, sg = (lambda v: f"{v:.3f}"), (lambda v: f"{v:+.3f}")


def v2_state():
    """V2 history of every player up to the snapshot (all approved data, everything released)."""
    raw = from_sackmann_v2(pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(LOCKED_PATH)], ignore_index=True), load_with_locked())
    ab = assign_ab(raw).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
    *_, ctx, state, tour = F.build(ab, rt, MAX_DAYS, return_state=True)
    *_, qstate, vol0 = Q.build(ab, ctx, MAX_DAYS, return_state=True)  # V2.1 match-quality history
    return raw, ab, state, tour, qstate, vol0


def contributions(P, surface):
    r = P.predict(SIN, ALC, surface)
    m = P.art["models"]["D"]
    terms = np.array(m["coef"]) * r["diff"][m["features"]].to_numpy(float) / np.array(m["scale"])
    assert abs(1 / (1 + np.exp(-terms.sum())) - r["p"]["D"]) < 1e-12, "contributions do not add up to the prediction"
    return r, pd.DataFrame({"feature": m["features"], "difference": r["diff"][m["features"]].to_numpy(float), "contribution": terms})


def v2_row(P, r, state, tour, surface, best_of):
    a = P.art
    s = SURF[surface]
    fa = F.player_features(state[SIN], tour, a["snapshot_day"], s, a["elo"][SIN])
    fb = F.player_features(state[ALC], tour, a["snapshot_day"], s, a["elo"][ALC])
    ctx = F.match_context(fa, fb, tour, s, best_of, None, SIN, ALC)  # no venue: pace falls back to the surface average
    idx = ["hypothetical"]
    X = F.design(pd.DataFrame([r["diff"]], index=idx), pd.DataFrame([fa], index=idx), pd.DataFrame([fb], index=idx), pd.DataFrame([ctx], index=idx))
    return fa, fb, ctx, X


def _groups(model, cols, x, half_life):
    sc, lr = model
    g = {**{c: k for k, cs in models.group_columns(half_life).items() for c in cs}, **{c: k for k, cs in F.GROUPS.items() for c in cs},
         **{c: k for k, cs in Q.GROUPS.items() for c in cs}}
    t = lr.coef_[0] * x[cols].to_numpy(float).ravel() / sc.scale_
    return pd.Series(t, index=cols).groupby(lambda c: g[c], sort=False).sum()


def decomposition(X, meta, tr, cols, cols21, m, m21, Xclay, a):
    """Why V2 (66.1%) and V2.1 (61.4%) differ on Sinner-Alcaraz clay: regularisation vs added features, by feature group."""
    x = Xclay.iloc[0]
    sig = lambda z: 1 / (1 + np.exp(-z))
    t21 = ev.read_md_table(Path("reports/v21_dev_experiments.md").read_text(), "## 1.").set_index("model")
    c21 = float(t21.loc["V2.1 (V2 + both)", "C"])
    m_c = models.fit(X.loc[tr, cols], meta.y[tr], c21)  # V2 features with V2.1's regularisation strength
    g1, g2, g3 = _groups(m, cols, x, a["half_life"]), _groups(m_c, cols, x, a["half_life"]), _groups(m21, cols21, x, a["half_life"])
    T = pd.DataFrame({"V2 (C=0.03)": g1, f"V2 features, C={c21:g}": g2, f"V2.1 (C={c21:g})": g3}).fillna(0.0)
    T = T.reindex(T.abs().max(axis=1).sort_values(ascending=False).index)
    T.loc["total log-odds"] = T.sum()
    T.loc["probability"] = [sig(v) for v in T.loc["total log-odds"]]
    shown = T.map(lambda v: f"{v:+.3f}")
    shown.loc["probability"] = [f"{v:.1%}" for v in T.loc["probability"]]
    steps = pd.DataFrame([("V2 as fitted (C=0.03)", sig(g1.sum())), (f"1. same features, C={c21:g} (stronger regularisation)", sig(g2.sum())),
                          (f"2. + V2.1 quality and momentum features (C={c21:g})", sig(g3.sum()))], columns=["step", "P(Sinner)"])
    trn = X.loc[tr]
    corr = pd.DataFrame([(q, c, trn[q].corr(trn[c])) for q in Q.GROUPS["quality"] + Q.GROUPS["qmomentum"] for c in cols], columns=["new", "existing", "r"])
    corr = corr.loc[corr.groupby("new").r.apply(lambda v: v.abs().idxmax())].sort_values("r", key=abs, ascending=False)
    return ["## 6. Why V2 says 66.1% and V2.1 says 61.4% (both fitted on 2015-2024)\n",
            ev.md(steps, {"P(Sinner)": lambda v: f"{v:.1%}"}), "",
            "Contributions (log-odds toward Sinner) by feature group in each fit:\n",
            ev.md(shown.reset_index(names="group")), "",
            "Each new V2.1 feature and its most correlated existing input (training rows):\n", ev.md(corr, {"r": lambda v: f"{v:+.2f}"}), "",
            "**Reading.** Most of the gap comes from the features V2.1 adds, not from regularisation. Its workload term (sets in the last 30 days) and "
            "points-vs-expectation terms overlap with inputs V2 already has (match counts, recent serve/return ratings), so the fit splits the "
            "same evidence differently between correlated columns. On thousands of matches the two splits score the same (log loss 0.6097 vs 0.6098); "
            "on one extreme case they can differ by several points. This is ordinary model uncertainty, not an error in either model.\n"]


def markov_and_availability(P, prof, ctx):
    from src.v2.predictor import ARTIFACT as V2A, V2Predictor
    V = V2Predictor.load(P, V2A)
    act = P.players(active_only=True).index
    n = pd.Series({pid: V.art["stat_matches"].get(pid, 0) for pid in act})
    S, A = SIN, ALC
    return ["## 7. Checks before integration\n",
            "**Service-return (Markov) model.** Points are treated as independent with a fixed serve-win chance per player; games, 7-point tiebreak sets "
            "and best of 3 or 5 follow the official scoring. The final set is treated as a normal tiebreak set: since 2022 the Slams play a 10-point "
            "tiebreak at 6-6 in the final set, a small approximation. The format is the user's choice in the app (best of 3 by default, best of 5 for "
            "Slams). The result does not depend on who serves first (tested), and inputs outside 30%-95% are clipped. "
            f"Here: Sinner wins {ctx['pa']:.1%} of his serve points and Alcaraz {ctx['pb']:.1%} (clay tour average {ctx['mu']:.1%}).\n",
            f"**Statistics at the snapshot.** Sinner has {V.art['stat_matches'][S]} and Alcaraz {V.art['stat_matches'][A]} released matches with "
            f"serve/return statistics. Of {len(act):,} active players, {int((n == 0).sum())} have none (V2 returns 'unavailable' for them) and "
            f"{int(((n > 0) & (n < 10)).sum())} have fewer than 10 (V2 shows a warning).\n",
            f"**Production artifact.** artifacts/v2.pkl refits the frozen V2 settings on all approved data (to tournaments starting {V.art['data_end']}); "
            f"for Sinner vs Alcaraz on clay, best of 3, it gives **{V.predict(S, A, 'Clay')['p']:.1%}** (the 66.1% above is the 2015-2024 research fit).\n"]


def main():
    P = Predictor.load()
    a = P.art
    raw, ab, state, tour, qstate, vol0 = v2_state()
    # implementation checks
    sur = run_elo(ab, k=ELO_K_SURF, release_days=MAX_DAYS)[1][1]
    assert abs(sur[("Clay", ALC)] - P.predict(SIN, ALC, "Clay")["B"]["surface_elo"]) < 1e-9, "clay Elo lookup mismatch"
    rc, cc = contributions(P, "Clay")
    rh, ch = contributions(P, "Hard")
    groups = {f: g for g, cols in models.group_columns(a["half_life"]).items() for f in cols}
    gc = cc.assign(group=cc.feature.map(groups)).groupby("group", sort=False).contribution.sum()
    gh = ch.assign(group=ch.feature.map(groups)).groupby("group", sort=False).contribution.sum()
    G = pd.DataFrame({"group": gc.index, "clay": gc.to_numpy(), "hard": gh.loc[gc.index].to_numpy()})
    A, B = rc["A"], rc["B"]
    fa1, fb1 = A["features"], B["features"]
    inputs = pd.DataFrame([
        ("General Elo", f"{A['elo']:.0f}", f"{B['elo']:.0f}"), ("Clay Elo", f"{A['surface_elo']:.0f}", f"{B['surface_elo']:.0f}"),
        ("Hard Elo", f"{rh['A']['surface_elo']:.0f}", f"{rh['B']['surface_elo']:.0f}"),
        ("Last ranking / points", f"{A['rank']:.0f} / {A['points']:.0f}", f"{B['rank']:.0f} / {B['points']:.0f}"),
        ("Last tournament start in data", str(A["last_date"]), str(B["last_date"])),
        ("Days since last result was released", str(A["days_since_release"]), str(B["days_since_release"])),
        ("Matches in last 30 / 90 days", f"{np.expm1(fa1['ln30']):.0f} / {np.expm1(fa1['ln90']):.0f}", f"{np.expm1(fb1['ln30']):.0f} / {np.expm1(fb1['ln90']):.0f}"),
        ("Win rate last 10 (shrunk)", f3(fa1["wr10"]), f3(fb1["wr10"])), ("Decayed win rate, 60-day half-life", f3(fa1["dec_wr_60"]), f3(fb1["dec_wr_60"])),
        ("Clay win rate, last 365 days (shrunk)", f3(fa1["swr365"]), f3(fb1["swr365"])),
        ("Opponent-adjusted form, last 10", sg(fa1["adj10"]), sg(fb1["adj10"]))], columns=["input", "Sinner", "Alcaraz"])
    # recorded matches between the two (completed, walkovers excluded by the cleaning step)
    h = raw[raw.winner_id.isin([SIN, ALC]) & raw.loser_id.isin([SIN, ALC])]
    h2h = h.assign(winner=np.where(h.winner_id == SIN, "Sinner", "Alcaraz")).groupby(["surface", "winner"]).size().unstack(fill_value=0)
    # V2 profiles and the diagnostic V2 model (fitted 2015-2024 only)
    rep = Path("reports/v2_dev_experiments.md").read_text()
    C = float(ev.read_md_table(rep, "## 1.").set_index("model").loc["V2 LR (all groups)", "C"])
    meta, X = run_v21.load()[:2]
    year = meta.event_date.dt.year
    tr = meta.level.isin(SCORED_LEVELS) & year.between(2015, 2024)
    cols = a["models"]["D"]["features"] + [c for g in F.GROUPS.values() for c in g]
    m = models.fit(X.loc[tr, cols], meta.y[tr], C)
    t21 = ev.read_md_table(Path("reports/v21_dev_experiments.md").read_text(), "## 1.").set_index("model")
    cols21 = cols + Q.GROUPS["quality"] + Q.GROUPS["qmomentum"]
    m21 = models.fit(X.loc[tr, cols21], meta.y[tr], float(t21.loc["V2.1 (V2 + both)", "C"]))
    rows, prof = [], None
    for surface, rr in (("Clay", rc), ("Hard", rh)):
        for bo in (3, 5):
            fa, fb, ctx, Xh = v2_row(P, rr, state, tour, surface, bo)
            qa, qb = (Q.player_features(qstate.get(pid), a["snapshot_day"], SURF[surface], vol0) for pid in (SIN, ALC))
            Xq = Q.design(Xh, pd.DataFrame([qa], index=Xh.index), pd.DataFrame([qb], index=Xh.index))
            rows.append((surface, bo, rr["p"]["D"], ctx["p_mk"], float(models.predict(m, Xh[cols])[0]), float(models.predict(m21, Xq[cols21])[0])))
            if surface == "Clay" and bo == 3:
                prof, h2h_ctx, qprof, Xclay = (fa, fb), ctx, (qa, qb), Xq
    V = pd.DataFrame(rows, columns=["surface", "best of", "Model D", "matchup model (raw)", "V2 LR (diagnostic)", "V2.1 LR (diagnostic)"])
    labels = {"srv": "Serve rating, opponent-adjusted (all surfaces)", "ret": "Return rating, opponent-adjusted (all surfaces)",
              "srv_s": "Serve rating on clay", "ret_s": "Return rating on clay", "srv60": "Serve rating, 60-day half-life",
              "ret60": "Return rating, 60-day half-life", "spw": "Serve points won, last 20 matches", "rpw": "Return points won, last 20",
              "hold": "Service games held, last 20", "brk": "Return games won, last 20", "ace": "Aces per serve point, last 20",
              "elo_trend": "General Elo change, last 90 days (/400)"}
    fmt = lambda k, v: f3(v) if k in ("spw", "rpw", "hold", "brk", "ace") else sg(v)  # observed rates plain, ratings signed
    Pf = pd.DataFrame([(lab_, fmt(k, prof[0][k]), fmt(k, prof[1][k])) for k, lab_ in labels.items()], columns=["V2 input", "Sinner", "Alcaraz"])
    top = cc.reindex(cc.contribution.abs().sort_values(ascending=False).index).head(10)
    L = ["# Sinner vs Alcaraz on clay: diagnostic audit (auto-generated by src/v2/audit.py)\n",
         f"Historical snapshot {a['snapshot_date']} (data to tournaments starting {a['data_end']}). Diagnostic only: nothing here changes "
         "Model D or chooses anything in V2, and the prediction is not adjusted.\n",
         "## 1. The prediction and where it comes from\n",
         f"Model D: Sinner **{rc['p']['D']:.1%}** on clay, {rh['p']['D']:.1%} on hard. General Elo alone: {rc['p']['Elo']:.1%}. "
         "Each number below is a feature group's push on the log-odds (positive favours Sinner); they add up exactly to the prediction.\n",
         ev.md(G, {"clay": sg, "hard": sg}),
         f"\nTotal: clay {gc.sum():+.3f} log-odds = {rc['p']['D']:.1%}; hard {gh.sum():+.3f} = {rh['p']['D']:.1%}.\n",
         "Largest single features on clay:\n", ev.md(top, {"difference": sg, "contribution": sg}), "",
         "## 2. The inputs (all observed in the data, nothing estimated)\n", ev.md(inputs), "",
         "Recorded completed meetings by surface (2010 onwards):\n", ev.md(h2h.reset_index()), "",
         "## 3. What V2 sees (serve and return, matchup model)\n",
         "Ratings are serve or return points won above the tour average (0.010 = one extra point in 100).\n",
         ev.md(Pf),
         f"\nV2 head-to-head residual (Sinner's side, shrunk with 5 pseudo-meetings): all surfaces {h2h_ctx['h2h']:+.3f} over {int(h2h_ctx['h2h_n'])} meetings.\n",
         ev.md(V, {c: (lambda v: f"{v:.1%}") for c in V.columns[2:]}),
         "\nThe V2 LR here is fitted on 2015-2024 only, with C from the selection folds; it is a diagnostic, not a forecast and not in the app. "
         "No venue is given, so the pace term uses the clay average.\n"]
    form = gc[["last", "win", "surf", "dec", "adj", "adjdec"]].sum()
    L += ["## 4. Findings\n",
          f"1. **Explainable from valid inputs; no implementation error found.** The feature contributions add up exactly to the prediction, the clay Elo "
          "matches the rating engine, and the most recent results are released on the snapshot day by the same rule used in training.",
          f"2. **Clay Elo does favour Alcaraz** ({B['surface_elo']:.0f} vs {A['surface_elo']:.0f}) and pushes {cc.set_index('feature').contribution['surf_elo']:+.3f} "
          f"toward him. Model D learned (2015-2024) to weigh general Elo more than surface Elo, and general Elo favours Sinner by "
          f"{A['elo'] - B['elo']:.0f} points ({cc.set_index('feature').contribution['elo']:+.3f}). Net Elo group: {gc['elo']:+.3f}.",
          f"3. **Activity is the largest single driver** (rest/workload {gc['rest']:+.3f} of the {gc.sum():+.3f} total). Alcaraz has no recorded match after "
          f"the tournament starting {B['last_date']}: {np.expm1(fb1['ln30']):.0f} matches in the last 30 days and {np.expm1(fb1['ln90']):.0f} in 90, against "
          f"{np.expm1(fa1['ln30']):.0f} and {np.expm1(fa1['ln90']):.0f} for Sinner. Historically, players coming off a gap or a light schedule did worse. "
          "**The data does not record why a player stopped** (injury, rest, scheduling). That is the most important missing information here: "
          "for a precautionary break the model may overstate the effect; for an injury it may understate it.",
          f"4. **Form nearly cancels out in Model D** (all form groups together {form:+.3f}). Decayed and opponent-adjusted form favour Sinner, raw win-rate "
          "windows push back. These features are strongly correlated, so single signs are not meaningful; read the group total.",
          "5. **Head-to-head is not a Model D input.** Alcaraz leads the recorded meetings; V2's shrunk H2H residual is small "
          f"({h2h_ctx['h2h']:+.3f}) because {int(h2h_ctx['h2h_n'])} meetings are weak evidence next to thousands of other matches.",
          "6. **Serve and return favour Sinner.** V2 sees a clearly stronger recent serve (clay serve rating "
          f"{prof[0]['srv_s']:+.3f} vs {prof[1]['srv_s']:+.3f}) and equal return. The raw matchup model says {V.iloc[0, 3]:.1%}, but it is overconfident "
          f"(calibration slope {float(ev.read_md_table(rep, '## 1.').set_index('model').loc['Matchup model alone (raw)', 'cal_slope']):.2f} on the reporting folds); the diagnostic V2 LR says {V.iloc[0, 4]:.1%}. V2 does not move the estimate toward Alcaraz.",
          "7. **Not available:** court conditions for a specific venue, weather, fitness or injury reports, and match dates. None was invented.",
          "\n**Conclusion.** The 60.7% is a consequence of valid inputs, not a bug. The counterintuitive part (a higher clay Elo outweighed) comes from "
          "general Elo and, above all, Alcaraz's absence in the data. The honest caveat is that the model cannot tell an injury from a planned rest. "
          "A future V2 stage should model absences explicitly and use documented withdrawal or injury data if a licensed source provides it.\n"]
    ql = {"pq5": "Points won vs expectation, last 5 (shrunk)", "pq20": "Points won vs expectation, last 20", "dom20": "Games won vs expectation, last 20",
          "luck10": "Luck: results minus what the points deserved, last 10", "pq_trend": "Trend: last 5 minus last 20", "pq_surf10": "Points vs expectation on clay, last 10",
          "pq_vol10": "Volatility of points vs expectation, last 10", "load30": "Sets played in the last 30 days (log1p)"}
    Qp = pd.DataFrame([(v, sg(qprof[0][k]), sg(qprof[1][k])) for k, v in ql.items()], columns=["V2.1 input", "Sinner", "Alcaraz"])
    clay3 = V.iloc[0]
    fo = pd.DataFrame([(n, clay3[c], float(odds.fair_odds(clay3[c])), float(odds.fair_odds(1 - clay3[c])))
                       for n, c in (("Model D", "Model D"), ("V2 LR (diagnostic)", "V2 LR (diagnostic)"), ("V2.1 LR (diagnostic)", "V2.1 LR (diagnostic)"))],
                      columns=["model", "P(Sinner)", "fair odds Sinner", "fair odds Alcaraz"])
    dg = ev.read_md_table(Path("reports/v21_dev_experiments.md").read_text(), "## 3.")
    und = dg[(dg.group == "(post hoc) only the underdog inactive 45+ days") & (dg.model == "D (frozen spec)")].iloc[0]
    L += ["## 5. V2.1: performance beneath the results (clay, best of 3)\n",
          "Each match's points and games are compared with what was expected before it, given the opponent and surface; values are shrunk averages "
          "(0.010 = one point in 100 better than expected). Positive luck = won more than the points deserved.\n",
          ev.md(Qp), "",
          "Fair odds (decimal, no bookmaker margin) implied by each model. **No market prices exist in the approved data**, so there is no market comparison.\n",
          ev.md(fo, {"P(Sinner)": lambda v: f"{v:.1%}", "fair odds Sinner": lambda v: f"{v:.2f}", "fair odds Alcaraz": lambda v: f"{v:.2f}"}), "",
          f"**Calibration evidence relevant to this case (dev 2023-2024, post hoc, descriptive):** in {und.n} matches where only the underdog had been "
          f"inactive for 45+ days, Model D gave the favourite {float(und.fav_predicted):.1%} on average and favourites won {float(und.fav_observed):.1%}. "
          "That suggests the models lean slightly too far toward the active player, which is the Sinner side here. It is a hypothesis for a pre-registered "
          "test, not a correction: no prediction was adjusted.\n"]
    L += decomposition(X, meta, tr, cols, cols21, m, m21, Xclay, a) + markov_and_availability(P, prof, h2h_ctx)
    OUT.write_text("\n".join(L))
    print(f"Wrote {OUT}")
    return G, inputs, Pf, V


if __name__ == "__main__":
    main()
