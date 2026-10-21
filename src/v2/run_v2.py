"""V2 development experiments (PLAN_V2.md sections 4-5). Reads the dev file only: 2025/2026 are never in memory.

Run from the project root:  python -m src.v2.run_v2
Writes reports/v2_dev_experiments.md and reports/figures/v2_calibration_dev.png.
Every choice (C, the 'best' variant) is made on the selection folds 2020-2022; 2023-2024 are only reported.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src import evaluate as ev
from src import gbm, models
from src.canonical import DEV_PATH, MAX_DAYS, SCORED_LEVELS
from src.dataset import ELO_K_GEN, ELO_K_SURF
from src.elo import expected
from src.features import build_features, design, elo_ratings
from src.pairs import assign_ab
from src.v2 import features as F
from src.v2.data import CONFIG, load_dev_v2

REPORT, FIG = Path("reports/v2_dev_experiments.md"), Path("reports/figures/v2_calibration_dev.png")
FROZEN = json.loads(Path("models/frozen_phase5.json").read_text())
FD = CONFIG["folds"]
SEL, REP, TRAIN_FROM = FD["selection"], FD["reporting"], FD["train_from"]
f4, sg = (lambda v: f"{v:.4f}"), (lambda v: f"{v:+.4f}")


def load():
    raw = load_dev_v2()
    ab = assign_ab(raw).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    meta = ab.set_index("match_id")
    rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
    D1 = design(ab, rt, *build_features(ab, rt, MAX_DAYS)).loc[meta.index]
    fa, fb, ctx = F.build(ab, rt, MAX_DAYS)
    X = F.design(D1, fa, fb, ctx).loc[meta.index]
    assert np.isfinite(X.to_numpy()).all(), "NaN/inf in V2 inputs"
    r = rt.loc[meta.index]
    return meta, X, ctx.loc[meta.index], pd.Series(expected(r.elo_a.to_numpy(), r.elo_b.to_numpy()), index=meta.index)


class Lab:
    def __init__(self, meta, X):
        self.X, self.y, self.year = X, meta.y, meta.event_date.dt.year
        self.elig = meta.level.isin(SCORED_LEVELS) & self.year.between(TRAIN_FROM, 2024)

    def fold(self, cols, C, f):
        tr = self.elig & (self.year >= TRAIN_FROM) & (self.year < f)
        te = self.elig & (self.year == f)
        return pd.Series(models.predict(models.fit(self.X.loc[tr, cols], self.y[tr], C), self.X.loc[te, cols]), index=self.X.index[te])

    def lr(self, cols, C=None):
        """Selection-fold predictions for each C in the grid (unless C is fixed); best C by mean fold log loss; then reporting folds."""
        grid = [C] if C else CONFIG["lr_c_grid"]
        best = None
        for c in grid:
            p = pd.concat([self.fold(cols, c, f) for f in SEL])
            loss = np.mean([ev.log_loss(self.y[p.index][self.year[p.index] == f], p[self.year[p.index] == f]) for f in SEL])
            if best is None or loss < best[1]:
                best = (c, loss, p)
        c, loss, p_sel = best
        return dict(C=c, sel_loss=loss, p=pd.concat([p_sel] + [self.fold(cols, c, f) for f in REP]))

    def xgb(self, cols):
        cfg = {k: FROZEN["models"]["D XGB"]["params"][k] for k in ("max_depth", "min_child_weight", "reg_lambda")}
        out = []
        for f in SEL + REP:
            m, _ = gbm.train(self.X, self.y, self.year, self.elig, cols, cfg, f - 1)
            te = self.elig & (self.year == f)
            out.append(pd.Series(gbm.predict(m, self.X.loc[te, cols]), index=self.X.index[te]))
        p = pd.concat(out)
        return dict(C=None, sel_loss=np.mean([ev.log_loss(self.y[p.index][self.year[p.index] == f], p[self.year[p.index] == f]) for f in SEL]), p=p)


def fixed(p, lab):
    """A model with no fitted parameters (Elo, raw matchup probability) on the same rows."""
    p = p[lab.elig & lab.year.isin(SEL + REP)]
    return dict(C=None, sel_loss=np.mean([ev.log_loss(lab.y[p.index][lab.year[p.index] == f], p[lab.year[p.index] == f]) for f in SEL]), p=p)


def summary(name, r, ref, lab, tourney):
    p = r["p"]; rep = p[lab.year[p.index].isin(REP)]; y = lab.y[rep.index]
    a, s = ev.calibration_fit(y, rep)
    d, lo, hi = ev.paired_bootstrap(ev.per_match_loss(y, rep), ev.per_match_loss(y, ref.loc[rep.index]), tourney[rep.index])
    return dict(model=name, C="" if r["C"] is None else f"{r['C']:g}", sel_logloss=r["sel_loss"], rep_logloss=ev.log_loss(y, rep),
                brier=ev.brier(y, rep), accuracy=ev.accuracy(y, rep), cal_int=a, cal_slope=s, vs_D=d, low=lo, high=hi)


def main():
    meta, X, ctx, p_elo = load()
    lab = Lab(meta, X)
    tourney = meta.tourney_id
    colsD, C_D = FROZEN["models"]["D LR"]["features"], FROZEN["models"]["D LR"]["C"]
    G = F.GROUPS
    allv2 = colsD + [c for g in G.values() for c in g]
    R = {"Elo": fixed(p_elo, lab), "D (frozen spec)": lab.lr(colsD, C_D), "D (C re-selected)": lab.lr(colsD)}
    for g, cols in G.items():
        R[f"D + {g}"] = lab.lr(colsD + cols)
    R["V2 LR (all groups)"] = lab.lr(allv2)
    for g, cols in G.items():
        R[f"V2 LR - {g}"] = lab.lr([c for c in allv2 if c not in cols])
    R["V2 core (D + sr_adj + matchup + h2h)"] = lab.lr(colsD + G["sr_adj"] + G["matchup"] + G["h2h"])
    R["Matchup model alone (raw)"] = fixed(ctx.p_mk, lab)
    R["Matchup model alone (recalibrated LR)"] = lab.lr(["mk"])
    R["V2 XGBoost (all groups)"] = lab.xgb(allv2)
    R["D frozen spec (XGBoost)"] = lab.xgb(colsD)
    ens = (R["D (frozen spec)"]["p"] + R["V2 LR (all groups)"]["p"]) / 2
    R["Ensemble: mean(D, V2 LR)"] = dict(C=None, sel_loss=np.mean([ev.log_loss(lab.y[ens.index][lab.year[ens.index] == f], ens[lab.year[ens.index] == f]) for f in SEL]), p=ens)
    ref = R["D (frozen spec)"]["p"]
    idx = ref.index
    for r in R.values():
        assert r["p"].index.sort_values().equals(idx.sort_values()), "models scored on different rows"
    T = pd.DataFrame([summary(n, r, ref, lab, tourney) for n, r in R.items()])
    sel_D = R["D (frozen spec)"]["sel_loss"]
    T["verdict"] = np.where((T.sel_logloss < sel_D) & (T.high < 0), "improves on D", np.where(T.low > 0, "worse than D", "no clear difference"))
    T.loc[T.model == "D (frozen spec)", "verdict"] = "benchmark"
    lr_rows = T[T.model.str.contains("LR") & ~T.model.str.contains("alone")]
    best = lr_rows.sort_values("sel_logloss").iloc[0].model   # chosen on selection folds only
    write_report(T, R, best, lab, meta, ctx, tourney, X, allv2)


def subgroup_table(R, names, lab, meta, ctx, tourney):
    pD = R["D (frozen spec)"]["p"]
    rep = pD.index[lab.year[pD.index].isin(REP)]
    sel = pD.index[lab.year[pD.index].isin(SEL)]
    q = ctx.pace_ace.loc[sel][ctx.pace_n.loc[sel] > 0].quantile([1 / 3, 2 / 3]).to_numpy()  # thresholds from selection folds
    pace = np.where(ctx.pace_n.loc[rep] == 0, "no pace history",
                    np.where(ctx.pace_ace.loc[rep] < q[0], "slow third", np.where(ctx.pace_ace.loc[rep] < q[1], "middle third", "fast third")))
    conf = np.maximum(pD.loc[rep], 1 - pD.loc[rep])
    ra, rb = meta.rank_a.loc[rep].fillna(9999), meta.rank_b.loc[rep].fillna(9999)
    best_rank = np.minimum(ra, rb)
    groups = {"surface": meta.surface.loc[rep].fillna("unknown").to_numpy(), "pace (estimated)": pace,
              "Model D confidence": np.where(conf < 0.6, "close (D 40-60%)", np.where(conf > 0.75, "strong favourite (D > 75%)", "between")),
              "both players top 10": np.where((ra <= 10) & (rb <= 10), "yes", "no"),
              "better player's rank": np.where(best_rank <= 10, "1-10", np.where(best_rank <= 30, "11-30", np.where(best_rank <= 100, "31-100", "100+/unranked"))),
              "format": np.where(ctx.bo5.loc[rep] == 1, "best of 5", "best of 3"), "season": lab.year.loc[rep].astype(str).to_numpy()}
    rows = []
    y = lab.y.loc[rep]
    for gname, lab_ in groups.items():
        for v in sorted(set(lab_)):
            m = lab_ == v
            row = dict(split=gname, group=v, n=int(m.sum()), D=ev.log_loss(y[m], pD.loc[rep][m]))
            for n in names:
                p = R[n]["p"].loc[rep][m]
                row[n] = ev.log_loss(y[m], p)
            d, lo, hi = ev.paired_bootstrap(ev.per_match_loss(y[m], R[names[0]]["p"].loc[rep][m]), ev.per_match_loss(y[m], pD.loc[rep][m]), tourney.loc[rep][m])
            row["diff (first - D)"] = f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]"
            rows.append(row)
    return pd.DataFrame(rows)


def write_report(T, R, best, lab, meta, ctx, tourney, X, allv2):
    elig_rows = R["D (frozen spec)"]["p"].index
    has = meta.has_stats.loc[elig_rows].mean()
    L = ["# V2 development experiments (auto-generated by src/v2/run_v2.py)\n",
         f"Config `{CONFIG['version']}`. Dev file only (2010-2024); **2025/2026 were not read.** Tour-level matches: selection folds "
         f"{SEL} ({int(lab.year.loc[elig_rows].isin(SEL).sum()):,} matches), reporting folds {REP} ({int(lab.year.loc[elig_rows].isin(REP).sum()):,}). "
         f"Each fold year is predicted by a model trained on {TRAIN_FROM} to the year before. Every model is scored on exactly the same rows. "
         f"{has:.1%} of these matches have complete serve/return statistics (the rest still count; they just add no statistics to anyone's profile).\n",
         "The reporting folds are held out from V2 choices but are **not independent** (PLAN_V2.md section 1).\n",
         "## 1. All models\n",
         "`sel_logloss`: mean over selection folds (used for choices). Everything else: pooled reporting folds. `vs_D`: log loss minus Model D's, "
         "with a 95% interval from resampling whole tournaments (negative = better than D). Verdict uses the rule fixed in PLAN_V2.md.\n",
         ev.md(T, {**{c: f4 for c in ("sel_logloss", "rep_logloss", "brier", "accuracy", "cal_int", "cal_slope")}, "vs_D": sg, "low": sg, "high": sg}),
         f"\nBest LR variant by selection-fold log loss (chosen without looking at the reporting folds): **{best}**.\n"]
    yrs = sorted(SEL + REP)
    S = pd.DataFrame([{"model": n, **{str(f): ev.log_loss(lab.y[R[n]["p"].index][lab.year[R[n]["p"].index] == f], R[n]["p"][lab.year[R[n]["p"].index] == f]) for f in yrs}}
                      for n in ("Elo", "D (frozen spec)", "V2 LR (all groups)", best, "V2 XGBoost (all groups)", "Matchup model alone (recalibrated LR)")
                      if n in R]).drop_duplicates("model")
    L += ["## 2. Log loss by season\n", ev.md(S, {str(f): f4 for f in yrs}), ""]
    names = list(dict.fromkeys([best, "V2 LR (all groups)", "V2 XGBoost (all groups)"]))
    sub = subgroup_table(R, names, lab, meta, ctx, tourney)
    L += [f"## 3. Subgroups (reporting folds, descriptive only)\n",
          f"Log loss per group; the interval is **{names[0]} minus Model D**. Pace thirds use thresholds from the selection folds. "
          "Small groups (for example top-10 matchups) have wide intervals.\n",
          ev.md(sub, {c: f4 for c in ["D"] + names}), ""]
    # coefficients of the full V2 LR fitted on training + selection years (interpretation only)
    C = R["V2 LR (all groups)"]["C"]
    tr = lab.elig & (lab.year <= max(SEL))
    sc, lr = models.fit(X.loc[tr, allv2], lab.y[tr], C)
    coef = pd.DataFrame({"feature": allv2, "std_coef": lr.coef_[0]}).assign(abs_=lambda d: d.std_coef.abs()).sort_values("abs_", ascending=False).drop(columns="abs_")
    L += ["## 4. Full V2 LR: standardised coefficients (fitted 2015-2022)\n",
          f"C = {C:g}. Effect of a one-standard-deviation difference on the log-odds. Correlated features share credit, so read groups, not single signs.\n",
          ev.md(coef.head(25), {"std_coef": sg}), ""]
    # pace index sanity check: the most recent estimate per venue
    names_map = pd.read_csv(DEV_PATH, usecols=["tourney_id", "tourney_name"]).assign(venue=lambda d: "sackmann:" + d.tourney_id.str.split("-").str[1].str.lstrip("0")).groupby("venue").tourney_name.last()
    last = ctx.assign(venue=meta.venue, surface=meta.surface, level=meta.level, date=meta.event_date)
    last = last[last.level.isin(SCORED_LEVELS) & (last.pace_n >= 100)].sort_values("date").groupby("venue").tail(1)
    last = last.assign(tournament=last.venue.map(names_map)).sort_values("pace_ace")
    show = pd.concat([last.tail(8).iloc[::-1], last.head(8)])[["tournament", "surface", "pace_ace", "pace_spw", "pace_n"]]
    L += ["## 5. Tournament pace index: sanity check (latest estimate, venues with 100+ earlier matches)\n",
          "Fastest 8 then slowest 8 by ace-rate residual. A **statistical proxy** built from previous editions, not a measured court speed.\n",
          ev.md(show.assign(pace_n=show.pace_n.astype(int)), {"pace_ace": sg, "pace_spw": sg}), ""]
    REPORT.write_text("\n".join(L))
    rep = R["D (frozen spec)"]["p"].index[lab.year[R["D (frozen spec)"]["p"].index].isin(REP)]
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=1)
    for n, c in (("D (frozen spec)", "#2a78d6"), ("V2 LR (all groups)", "#eb6834")):
        t = ev.reliability(lab.y[rep], R[n]["p"].loc[rep])
        ax.plot(t.mean_pred, t.actual, "o-", color=c, lw=2, ms=5, label=n)
    ax.set(xlabel="Predicted probability (player A)", ylabel="Observed frequency", title=f"Calibration, reporting folds {REP[0]}-{REP[-1]}")
    ax.legend(frameon=False); fig.tight_layout(); fig.savefig(FIG, dpi=130); plt.close(fig)
    print(f"Wrote {REPORT} and {FIG}. Best LR variant on selection folds: {best}")


if __name__ == "__main__":
    main()
