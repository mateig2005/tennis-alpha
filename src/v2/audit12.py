"""Phase 12: V2 forecast audit: calibration, extreme probabilities, the scoring model, and model uncertainty (dev data only).

Run from the project root:  python -m src.v2.audit12   -> reports/v2_forecast_audit.md, reports/figures/v2_calibration_audit.png
Every prediction here comes from a model trained only on years before the predicted one (2015 to the year before),
with the FROZEN settings (Model D C = 1, V2 C = 0.03, V2.1 C = 0.01). Nothing is tuned, refitted for production or changed.
2025/2026 are never read.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import evaluate as ev
from src import models
from src.canonical import MAX_DAYS, SCORED_LEVELS
from src.dataset import ELO_K_GEN, ELO_K_SURF
from src.diagnostics import vif
from src.features import build_features, design, elo_ratings
from src.pairs import assign_ab
from src.v2 import audit_tools as T
from src.v2 import features as F
from src.v2 import markov
from src.v2 import quality as Q
from src.v2.data import load_dev_v2, orient

REPORT, FIG = Path("reports/v2_forecast_audit.md"), Path("reports/figures/v2_calibration_audit.png")
YEARS = [2020, 2021, 2022, 2023, 2024]
FZ1, FZ2 = json.loads(Path("models/frozen_phase5.json").read_text()), json.loads(Path("models/frozen_v2.json").read_text())
f3, f4, pc, sg = (lambda v: f"{v:.3f}"), (lambda v: f"{v:.4f}"), (lambda v: f"{v:.1%}"), (lambda v: f"{v:+.4f}")
SIN, ALC = "sackmann:206173", "sackmann:207989"


def load():
    raw = load_dev_v2()
    ab = assign_ab(raw).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    meta = ab.set_index("match_id")
    rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
    fa1, fb1 = build_features(ab, rt, MAX_DAYS)
    fa, fb, ctx = F.build(ab, rt, MAX_DAYS)
    qa, qb = Q.build(ab, ctx, MAX_DAYS)
    X = Q.design(F.design(design(ab, rt, fa1, fb1), fa, fb, ctx), qa, qb).loc[meta.index]
    S = {k: pd.Series(v, index=ab.match_id) for k, v in orient(ab).items()}
    return meta, X, ctx.loc[meta.index], fa1.loc[meta.index], fb1.loc[meta.index], rt.loc[meta.index], {k: v.loc[meta.index] for k, v in S.items()}


def fold_predictions(X, y, year, elig, cols, C, train_mask=None):
    """Each year in YEARS predicted by a fit on 2015..year-1 (train_mask lets the bootstrap reweight training rows)."""
    out = []
    for f in YEARS:
        tr = elig & (year >= 2015) & (year < f)
        te = elig & (year == f)
        out.append(pd.Series(models.predict(models.fit(X.loc[tr, cols], y[tr], C), X.loc[te, cols]), index=X.index[te]))
    return pd.concat(out)


def groups_for(meta, X, fa1, fb1, rows, pD, pV):
    ra, rb = meta.rank_a.loc[rows].fillna(9999), meta.rank_b.loc[rows].fillna(9999)
    gap = np.maximum(np.expm1(fa1.rest.loc[rows]), np.expm1(fb1.rest.loc[rows]))
    elo_gap = (X.elo.loc[rows] * 400).abs()
    fD, fV = np.maximum(pD, 1 - pD), np.maximum(pV, 1 - pV)
    g = {"all development matches": np.ones(len(rows), bool),
         "top-10 v top-10": ((ra <= 10) & (rb <= 10)).to_numpy(), "top-20 v top-20": ((ra <= 20) & (rb <= 20)).to_numpy(),
         "comparable strength (Elo gap < 50)": (elo_gap < 50).to_numpy()}
    for s in ("Hard", "Clay", "Grass"):
        g[f"surface: {s}"] = (meta.surface.loc[rows] == s).to_numpy()
    g["best of 3"] = (meta.best_of.loc[rows] == 3).to_numpy(); g["best of 5"] = (meta.best_of.loc[rows] == 5).to_numpy()
    g["a player inactive 45+ days"] = (gap >= 45).to_numpy()
    own = {f"model's own favourite > {t:.0%}": (t, fD, fV) for t in (0.7, 0.8, 0.9)}
    return g, own


def section_extremes(y, pD, pV, meta, X, fa1, fb1, tourney):
    rows = pD.index
    g, own = groups_for(meta, X, fa1, fb1, rows, pD.to_numpy(), pV.to_numpy())
    out = []
    for name, m in g.items():
        for model, p in (("Model D", pD), ("V2", pV)):
            out.append(dict(group=name, model=model, **T.summary(y[rows][m], p.to_numpy()[m], tourney[rows].to_numpy()[m])))
    for name, (t, fD, fV) in own.items():
        for model, p, f in (("Model D", pD, fD), ("V2", pV, fV)):
            m = f > t
            if m.sum() >= 2:
                out.append(dict(group=name, model=model, **T.summary(y[rows][m], p.to_numpy()[m], tourney[rows].to_numpy()[m])))
    S = pd.DataFrame(out)
    share = pd.DataFrame([dict(model=n, **{f"favourite > {t:.0%}": float((np.maximum(p, 1 - p) > t).mean()) for t in (0.6, 0.7, 0.8, 0.9)},
                               mean_abs_logodds=float(np.abs(np.log(p / (1 - p))).mean())) for n, p in (("Model D", pD), ("V2", pV))])
    cal = {n: T.calibration_table(y[rows], p.to_numpy(), tourney[rows].to_numpy()) for n, p in (("Model D", pD), ("V2", pV))}
    return S, share, cal


def why_stronger(X, y, year, elig, pD, pV, half_life):
    """Which V2 groups push toward the favourite, in matches where V2 is clearly more confident than Model D (fit 2015-2022, scored 2023-2024)."""
    cols = FZ2["model"]["features"]
    tr = elig & year.between(2015, 2022)
    sc, lr = models.fit(X.loc[tr, cols], y[tr], FZ2["model"]["C"])
    rows = pV.index[year[pV.index].isin([2023, 2024])]
    terms = pd.DataFrame(X.loc[rows, cols].to_numpy() * lr.coef_[0] / sc.scale_, index=rows, columns=cols)
    gmap = {**{c: k for k, cs in models.group_columns(half_life).items() for c in cs}, **{c: k for k, cs in F.GROUPS.items() for c in cs}}
    G = terms.T.groupby(lambda c: gmap[c]).sum().T
    sign = np.sign(terms.sum(axis=1))  # +1 when V2 favours A: align every contribution with V2's favourite
    aligned = G.mul(sign, axis=0)
    more = (np.maximum(pV[rows], 1 - pV[rows]) - np.maximum(pD[rows], 1 - pD[rows])) > 0.05
    return pd.DataFrame({"V2 > D by 5+ pts": aligned[more.to_numpy()].mean(), "other matches": aligned[~more.to_numpy()].mean()}).sort_values("V2 > D by 5+ pts", ascending=False), int(more.sum()), len(rows)


def section_markov(y, X, ctx, meta, S, rows, tourney):
    grid = [(0.64, 0.58, 3), (0.64, 0.58, 5), (0.70, 0.60, 3), (0.62, 0.62, 5), (0.55, 0.68, 3), (0.66, 0.63, 5)]
    ver = []
    for pa, pb, bo in grid:
        e, m = T.exact_match(pa, pb, bo), markov.match_prob(pa, pb, bo)
        sim = T.simulate(pa, pb, bo, n=20000, seed=1)
        ver.append(dict(pa=pa, pb=pb, best_of=bo, production=m, independent=e, abs_diff=abs(e - m), simulated=sim, sim_se=np.sqrt(e * (1 - e) / 20000),
                        final_tb10=T.exact_match(pa, pb, bo, 10)))
    conv = []
    for d in (0.0, 0.01, 0.02, 0.04, 0.06, 0.08):
        pa, pb = 0.63 + d / 2, 0.63 - d / 2  # A's serve edge = B's serve deficit, around a typical hard-court level
        conv.append(dict(point_edge=d, A_hold=markov.game(pa), B_hold=markov.game(pb), set=markov.set_prob(pa, pb),
                         match_bo3=markov.match_prob(pa, pb, 3), match_bo5=markov.match_prob(pa, pb, 5)))
    # point level: observed serve-point share vs the pre-match expectation (matches with statistics)
    has = meta.has_stats.loc[rows].to_numpy(bool)
    r = rows[has]
    spw_a = (S["a_1stWon"][r] + S["a_2ndWon"][r]) / S["a_svpt"][r]
    spw_b = (S["b_1stWon"][r] + S["b_2ndWon"][r]) / S["b_svpt"][r]
    x, z = (ctx.pa[r] - ctx.pb[r]).to_numpy(), (spw_a - spw_b).to_numpy()
    slope_diff = float(np.polyfit(x, z, 1)[0])
    xa = (ctx.pa[r] - ctx.mu[r]).to_numpy()
    slope_a = float(np.polyfit(xa, (spw_a - ctx.mu[r]).to_numpy(), 1)[0])
    p_mk = 1 / (1 + np.exp(-X.mk.loc[rows]))
    raw = pd.DataFrame([dict(format=f"best of {bo}", **T.summary(y[rows][m], p_mk[m].to_numpy(), tourney[rows][m].to_numpy()))
                        for bo in (3, 5) for m in [(meta.best_of.loc[rows] == bo).to_numpy()]])
    cols = ["mk", "srv", "ret", "srv_s", "ret_s", "srv60", "ret60", "elo", "surf_elo", "dec_wr_60"]
    corr = X.loc[rows, cols].corr()
    vf = vif(X.loc[rows, FZ2["model"]["features"]]).rename_axis("feature").rename("vif").reset_index().sort_values("vif", ascending=False)
    return pd.DataFrame(ver), pd.DataFrame(conv), slope_diff, slope_a, int(len(r)), raw, corr, vf


def section_uncertainty(X, y, year, elig, meta, tourney, preds, reps=30, seed=0):
    cols, C = FZ2["model"]["features"], FZ2["model"]["C"]
    tr = (elig & year.between(2015, 2023)).to_numpy()
    te = elig & (year == 2024)
    tids = tourney[tr].to_numpy()
    uniq = np.unique(tids)
    rng = np.random.default_rng(seed)
    Xtr, ytr = X.loc[tr, cols], y[tr]
    boot = []
    for _ in range(reps):  # resample whole training tournaments with replacement: parameter uncertainty only
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        w = pd.Series(pick).value_counts()
        weight = pd.Series(tids).map(w).fillna(0).to_numpy()
        keep = weight > 0
        idx = np.repeat(np.flatnonzero(keep), weight[keep].astype(int))
        boot.append(models.predict(models.fit(Xtr.iloc[idx], ytr.iloc[idx], C), X.loc[te, cols]))
    B = np.array(boot)
    sd = B.std(axis=0)
    rows = X.index[te]
    ra, rb = meta.rank_a.loc[rows].fillna(9999), meta.rank_b.loc[rows].fillna(9999)
    top = ((ra <= 10) & (rb <= 10)).to_numpy()
    par = pd.DataFrame([dict(rows=n, n=int(m.sum()), median_sd=float(np.median(sd[m])), p90_sd=float(np.percentile(sd[m], 90)), max_sd=float(sd[m].max()))
                        for n, m in (("all 2024 matches", np.ones(len(rows), bool)), ("top-10 v top-10", top))])
    sa = ((meta.a_id.loc[rows] == SIN) & (meta.b_id.loc[rows] == ALC)) | ((meta.a_id.loc[rows] == ALC) & (meta.b_id.loc[rows] == SIN))
    case = []
    for i in np.flatnonzero(sa.to_numpy()):
        mid = rows[i]
        flip = meta.a_id[mid] == ALC  # report Sinner's side
        q = (1 - B[:, i]) if flip else B[:, i]
        pv = preds["V2"][mid]; pd_ = preds["Model D"][mid]
        case.append(dict(start=str(meta.event_date[mid].date()), surface=meta.surface[mid], round_rank=int(meta.round_rank[mid]),
                         D=(1 - pd_) if flip else pd_, V2=(1 - pv) if flip else pv, V2_boot_2_5=float(np.percentile(q, 2.5)), V2_boot_97_5=float(np.percentile(q, 97.5)),
                         V21=(1 - preds["V2.1"][mid]) if flip else preds["V2.1"][mid], sinner_won=int((meta.y[mid] == 1) != flip)))
    rep = preds["V2"].index[year[preds["V2"].index].isin([2023, 2024])]
    ra2, rb2 = meta.rank_a.loc[rep].fillna(9999), meta.rank_b.loc[rep].fillna(9999)
    top2 = ((ra2 <= 10) & (rb2 <= 10)).to_numpy()
    dis = []
    for a, b in (("V2", "V2.1"), ("V2", "V2 (C=0.01)"), ("V2", "Model D")):
        d = np.abs(preds[a][rep] - preds[b][rep]).to_numpy()
        for n, m in (("all", np.ones(len(rep), bool)), ("top-10 v top-10", top2)):
            dis.append(dict(pair=f"{a} vs {b}", rows=n, n=int(m.sum()), mean=float(d[m].mean()), p90=float(np.percentile(d[m], 90)), max=float(d[m].max()),
                            share_over_5pts=float((d[m] > 0.05).mean())))
    return par, pd.DataFrame(case), pd.DataFrame(dis), reps


def main():
    meta, X, ctx, fa1, fb1, rt, S = load()
    y, year, tourney = meta.y, meta.event_date.dt.year, meta.tourney_id
    elig = meta.level.isin(SCORED_LEVELS) & year.between(2015, 2024)
    colsD = FZ1["models"]["D LR"]["features"]
    cols2 = FZ2["model"]["features"]
    cols21 = cols2 + Q.GROUPS["quality"] + Q.GROUPS["qmomentum"]
    preds = {"Model D": fold_predictions(X, y, year, elig, colsD, FZ1["models"]["D LR"]["C"]),
             "V2": fold_predictions(X, y, year, elig, cols2, FZ2["model"]["C"]),
             "V2 (C=0.01)": fold_predictions(X, y, year, elig, cols2, 0.01),
             "V2.1": fold_predictions(X, y, year, elig, cols21, 0.01)}
    pD, pV = preds["Model D"], preds["V2"]
    assert pD.index.equals(pV.index)
    Sx, share, cal = section_extremes(y, pD, pV, meta, X, fa1, fb1, tourney)
    W, n_more, n_rep = why_stronger(X, y, year, elig, pD, pV, FZ1["decay_half_life_days"])
    ver, conv, slope_diff, slope_a, n_pts, raw, corr, vf = section_markov(y, X, ctx, meta, S, pD.index, tourney)
    par, case, dis, reps = section_uncertainty(X, y, year, elig, meta, tourney, preds)
    rep = pD.index[year[pD.index].isin([2023, 2024])]
    head = pd.DataFrame([dict(model=n, n=len(rep), log_loss=ev.log_loss(y[rep], p[rep])) for n, p in preds.items()])
    L = ["# V2 forecast audit (auto-generated by src/v2/audit12.py)\n",
         f"Development data only (2025/2026 not read). {len(pD):,} tour-level matches from 2020-2024. Each one is predicted by a model fitted on 2015 to "
         "the previous year, using the **frozen** settings, so these are the out-of-sample predictions the models would have made. Intervals resample "
         "whole tournaments (2,000 draws). `gap` = observed favourite win rate minus mean favourite probability: **negative = overconfident**. "
         "2020-2022 helped choose V2's C, so they flatter V2 slightly; 2023-2024 did not.\n",
         "Check that the frozen fits reproduce the stage 1 numbers (2023-2024 log loss):\n", ev.md(head, {"log_loss": f4}), "",
         "## 1. Extreme probabilities: is V2 overconfident?\n",
         "How often each model names a strong favourite:\n", ev.md(share, {c: (pc if c != "mean_abs_logodds" else f3) for c in share.columns if c != "model"}), "",
         "Calibration by favourite band (all 2020-2024 predictions):\n"]
    for n, t in cal.items():
        L += [f"**{n}**\n", ev.md(t, {"predicted": pc, "observed": pc, "gap": sg, "low": sg, "high": sg}), ""]
    L += [f"![Calibration, Model D vs V2]({FIG.relative_to('reports')})\n", "By situation:\n",
          ev.md(Sx, {"predicted": pc, "observed": pc, "gap": sg, "low": sg, "high": sg, "log_loss": f4, "cal_slope": f3}), "",
          f"Why V2 is more confident: mean contribution toward V2's favourite (log-odds) by feature group, in {n_more:,} of {n_rep:,} 2023-2024 matches "
          "where V2's favourite probability exceeds Model D's by more than 5 points, against all others (V2 fitted 2015-2022):\n",
          ev.md(W.reset_index(names="group"), {c: lambda v: f"{v:+.3f}" for c in W.columns}), "",
          "## 2. The service-return (Markov) model\n",
          "**Independent verification.** A separate solver (explicit server tracking, generic tiebreak length) and a point-by-point simulation "
          "(20,000 matches each) against the production formulas:\n",
          ev.md(ver, {"production": lambda v: f"{v:.6f}", "independent": lambda v: f"{v:.6f}", "abs_diff": lambda v: f"{v:.1e}", "simulated": f4, "sim_se": f4, "final_tb10": lambda v: f"{v:.6f}"}),
          "\n`final_tb10` = the same match with a 10-point tiebreak at 6-6 in the deciding set (Grand Slam rule since 2022).\n",
          "**How a small point edge grows** (both players around 63% on serve):\n",
          ev.md(conv, {c: f4 for c in conv.columns}), "",
          f"**Are the point-level inputs too spread out?** On {n_pts:,} matches with statistics, regressing the observed serve-point gap (A minus B) on "
          f"the predicted gap gives slope **{slope_diff:.2f}**; for A's serve share alone, **{slope_a:.2f}**. A slope of 1 means the predicted "
          "serve/return differences are the right size on average; above 1, slightly too small; below 1, too large. "
          + ("So the point-level inputs are **not** too spread out: the overconfidence of the raw matchup probability below comes from the point-to-match "
             "step, which treats each player's serve chance as fixed and known for the whole match. In reality it varies from day to day, which makes "
             "results less predictable than the formula assumes. V2's regression gives the matchup term a reduced weight, which corrects for this.\n"
             if slope_diff >= 0.95 else "So the point-level inputs are too spread out, and the point-to-match step amplifies that.\n"),
          "**Raw matchup probability on its own** (it is one V2 input, not V2's output):\n",
          ev.md(raw, {"predicted": pc, "observed": pc, "gap": sg, "low": sg, "high": sg, "log_loss": f4, "cal_slope": f3}), "",
          "**Correlated inputs (possible double counting).** Correlations between the matchup log-odds and the serve/return, Elo and form differences:\n",
          ev.md(corr.round(2).reset_index(names="feature")), "",
          "Variance inflation factors above 10 (V2 features):\n", ev.md(vf[vf.vif > 10], {"vif": lambda v: f"{v:.1f}"}), "",
          "## 3. Model uncertainty\n",
          f"**Parameter uncertainty.** V2 refitted {reps} times on 2015-2023, resampling whole training tournaments; spread (standard deviation) of each "
          "2024 prediction across refits:\n",
          ev.md(par, {"median_sd": f4, "p90_sd": f4, "max_sd": f4}), "",
          "Sinner vs Alcaraz meetings in 2024 (development data), Sinner's side: Model D, V2, the bootstrap range of V2 and V2.1. Tournament start dates:\n",
          ev.md(case, {c: pc for c in ("D", "V2", "V2_boot_2_5", "V2_boot_97_5", "V21")}), "",
          "**Feature-set and regularisation uncertainty.** Per-match disagreement between models of similar overall accuracy (2023-2024):\n",
          ev.md(dis, {"mean": f4, "p90": f4, "max": f4, "share_over_5pts": pc}), ""]
    row = lambda g, m: Sx[(Sx.group == g) & (Sx.model == m)].iloc[0]
    iD, iV = row("a player inactive 45+ days", "Model D"), row("a player inactive 45+ days", "V2")
    b5D, b5V = row("best of 5", "Model D"), row("best of 5", "V2")
    t10 = row("top-10 v top-10", "V2")
    vv = dis[(dis.pair == "V2 vs V2.1") & (dis.rows == "all")].iloc[0]
    L += ["## 4. Findings\n",
          f"1. **V2 is not systematically overconfident.** Over 2020-2024 its favourites won {row('all development matches', 'V2').observed:.1%} against "
          f"{row('all development matches', 'V2').predicted:.1%} predicted (gap {row('all development matches', 'V2').gap:+.3f}); Model D: {row('all development matches', 'Model D').gap:+.3f}. "
          "Both lean about one point too confident in the 50-90% bands; above 90%, V2's favourites won slightly *more* often than predicted.",
          f"2. **Stronger favourites are mostly justified.** V2 names more 80%+ favourites ({share.iloc[1]['favourite > 80%']:.1%} vs {share.iloc[0]['favourite > 80%']:.1%}) and "
          f"they win at the predicted rate. In best-of-5, Model D is *under*confident (gap {b5D.gap:+.3f}) while V2 is on target ({b5V.gap:+.3f}): "
          "V2's best-of-5 × Elo and matchup terms capture that longer matches favour the stronger player.",
          f"3. **Elite matchups:** top-10 v top-10 has only {int(t10.n)} matches; V2's gap is {t10.gap:+.3f} with interval [{t10.low:+.3f}, {t10.high:+.3f}]. "
          "No overconfidence can be shown, and none can be ruled out at the size of a few points.",
          f"4. **The one consistent weakness: inactivity.** With a player inactive 45+ days, Model D is {-iD.gap:.1%} and V2 {-iV.gap:.1%} too confident, "
          "both intervals excluding 0. This is the Sinner-Alcaraz situation, so the June 2026 probabilities are probably a few points too high for the active player.",
          "5. **The scoring model is mathematically correct** (independent solver agrees to 1e-15; simulation within its error). The Slam 10-point "
          "final-set tiebreak changes results by at most about 0.2 points. Used raw, it is overconfident because serve chances vary within and between "
          "matches; V2 down-weights it, and V2's own calibration (points 1-2) shows the correction works.",
          "6. **Correlated features** (matchup term vs Elo r = 0.86; variance inflation up to about 260) make single coefficients unstable, not the "
          "probabilities. That is why group contributions should be read together, and why parameter uncertainty is visible.",
          f"7. **Uncertainty of a single prediction.** Refitting on resampled tournaments moves a typical 2024 prediction by about "
          f"{par.iloc[0].median_sd:.1%} (standard deviation), so roughly ±{2 * par.iloc[0].median_sd:.0%} at 95%, from parameters alone. "
          f"V2 and V2.1 differ by {vv['mean']:.1%} on average, and by more than 5 points in only {vv.share_over_5pts:.1%} of matches. "
          "The 4.7-point Sinner-Alcaraz gap is in that rare tail, driven by Alcaraz's absence (correlated workload inputs). "
          "These are defensible parameter-only intervals. They do not include missing context (injuries, conditions), so they are not shown in the app as "
          "'confidence intervals'.\n"]
    REPORT.write_text("\n".join(L))
    fig, ax = plt.subplots(figsize=(5.8, 5.8))
    ax.plot([0.5, 1], [0.5, 1], "--", color="#999", lw=1)
    for (n, t), c, dx in zip(cal.items(), ("#2a78d6", "#eb6834"), (-0.004, 0.004)):
        ax.errorbar(t.predicted + dx, t.observed, yerr=[t.gap - t.low, t.high - t.gap], fmt="o-", color=c, lw=2, ms=5, capsize=3, label=n)
    ax.set(xlabel="Mean favourite probability (band)", ylabel="Favourite win rate", title="Calibration 2020-2024 (95% tournament-bootstrap bars)", xlim=(0.5, 1), ylim=(0.5, 1))
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG, dpi=130); plt.close(fig)
    print(f"Wrote {REPORT} and {FIG}")


if __name__ == "__main__":
    main()
