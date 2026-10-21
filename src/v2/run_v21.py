"""V2.1 development experiments: match quality, statistical momentum, calibration (PLAN_V2.md, V2.1 protocol).

Run from the project root:  python -m src.v2.run_v21   -> reports/v21_dev_experiments.md
Dev file only (2025/2026 never in memory). Choices on selection folds 2020-2022; 2023-2024 only reported.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate as ev
from src.canonical import MAX_DAYS
from src.dataset import ELO_K_GEN, ELO_K_SURF
from src.elo import expected
from src.features import build_features, design, elo_ratings
from src.pairs import assign_ab
from src.v2 import calibration as cal
from src.v2 import features as F
from src.v2 import quality as Q
from src.v2.data import load_dev_v2
from src.v2.run_v2 import FROZEN, REP, SEL, Lab, fixed, summary

REPORT = Path("reports/v21_dev_experiments.md")
f4, sg = (lambda v: f"{v:.4f}"), (lambda v: f"{v:+.4f}")
FAMILIES = {"pq windows": ["pq3", "pq5", "pq10", "pq20"], "dominance": ["dom5", "dom20"], "luck": ["luck10"],
            "trends": ["pq_trend", "srv_trend", "ret_trend"], "volatility": ["pq_vol10"], "surface quality": ["pq_surf10"], "workload": ["load30"]}


def load():
    raw = load_dev_v2()
    ab = assign_ab(raw).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    meta = ab.set_index("match_id")
    rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
    fa1, fb1 = build_features(ab, rt, MAX_DAYS)
    fa, fb, ctx = F.build(ab, rt, MAX_DAYS)
    qa, qb = Q.build(ab, ctx, MAX_DAYS)
    X = Q.design(F.design(design(ab, rt, fa1, fb1), fa, fb, ctx), qa, qb).loc[meta.index]
    assert np.isfinite(X.to_numpy()).all(), "NaN/inf in V2.1 inputs"
    r = rt.loc[meta.index]
    return meta, X, fa1.loc[meta.index], fb1.loc[meta.index], fa.loc[meta.index], fb.loc[meta.index], qa.loc[meta.index], qb.loc[meta.index], \
        pd.Series(expected(r.elo_a.to_numpy(), r.elo_b.to_numpy()), index=meta.index), r


def calibration_section(R, names, lab, tourney):
    rows = []
    for n in names:
        p = R[n]["p"]; ys = lab.year[p.index]
        ps, pr = p[ys.isin(SEL)], p[ys.isin(REP)]
        a, iso = cal.platt_fit(ps, lab.y[ps.index]), cal.isotonic_fit(ps, lab.y[ps.index])
        y = lab.y[pr.index]
        pp, pi = cal.platt_apply(a, pr), iso(pr)
        d, lo, hi = ev.paired_bootstrap(ev.per_match_loss(y, pp), ev.per_match_loss(y, pr), tourney[pr.index])
        di, loi, hii = ev.paired_bootstrap(ev.per_match_loss(y, pi), ev.per_match_loss(y, pr), tourney[pr.index])
        rows.append(dict(model=n, platt_slope=a, raw=ev.log_loss(y, pr), platt=ev.log_loss(y, pp), platt_minus_raw=f"{d:+.4f} [{lo:+.4f}, {hi:+.4f}]",
                         isotonic=ev.log_loss(y, pi), iso_minus_raw=f"{di:+.4f} [{loi:+.4f}, {hii:+.4f}]"))
    return pd.DataFrame(rows)


def diagnostics(R, names, lab, meta, X, fa1, fb1):
    rep = R[names[0]]["p"].index[lab.year[R[names[0]]["p"].index].isin(REP)]
    y = lab.y[rep]
    pD = R[names[0]]["p"][rep]
    gap = np.maximum(np.expm1(fa1.rest[rep]), np.expm1(fb1.rest[rep]))
    x = X.loc[rep]
    groups = {"all reporting matches": np.ones(len(rep), bool),
              "strong favourite (Model D > 75%)": np.maximum(pD, 1 - pD).to_numpy() > 0.75,
              "both players top 10": ((meta.rank_a[rep] <= 10) & (meta.rank_b[rep] <= 10)).to_numpy(),
              "a player inactive 45+ days": (gap >= 45).to_numpy(),
              "general and surface Elo disagree": (np.sign(x.elo) != np.sign(x.surf_elo)).to_numpy(),
              "better recent record vs weaker opponents": ((np.sign(x.wr10) == -np.sign(x.opp10)) & (x.wr10.abs() > 0.1)).to_numpy()}
    # added after the first run (post hoc, PLAN_V2.md amendment 3): who is the inactive player, the Model D favourite or the underdog?
    ia = np.expm1(fa1.rest[rep]).to_numpy() >= 45
    ib = np.expm1(fb1.rest[rep]).to_numpy() >= 45
    fav_a = pD.to_numpy() >= 0.5
    groups["(post hoc) only the underdog inactive 45+ days"] = np.where(fav_a, ib & ~ia, ia & ~ib)
    groups["(post hoc) only the favourite inactive 45+ days"] = np.where(fav_a, ia & ~ib, ib & ~ia)
    rows = []
    for g, m in groups.items():
        for n in names:
            p = R[n]["p"][rep][m]; yy = y[m]
            fav = np.maximum(p, 1 - p)
            fav_won = np.where(p >= 0.5, yy, 1 - yy)
            rows.append(dict(group=g, model=n, n=int(m.sum()), log_loss=ev.log_loss(yy, p), fav_predicted=float(fav.mean()), fav_observed=float(fav_won.mean()),
                             cal_slope=ev.calibration_fit(yy, p)[1]))
    return pd.DataFrame(rows)


def elo_lead(meta, fa, fb, qa, qb, r):
    """Does statistical form (pre-match) predict a player's Elo change over the next 90 days, beyond past Elo momentum?"""
    long = pd.concat([pd.DataFrame({"pid": meta.a_id, "day": meta.event_date, "elo": r.elo_a, "pq20": qa.pq20, "luck10": qa.luck10, "pq_trend": qa.pq_trend, "elo_trend": fa.elo_trend}),
                      pd.DataFrame({"pid": meta.b_id, "day": meta.event_date, "elo": r.elo_b, "pq20": qb.pq20, "luck10": qb.luck10, "pq_trend": qb.pq_trend, "elo_trend": fb.elo_trend})])
    long = long.sort_values("day").reset_index(drop=True)
    fut = long[["pid", "day", "elo"]].rename(columns={"day": "fday", "elo": "felo"})
    long["target"] = long.day + pd.Timedelta(days=90)
    m = pd.merge_asof(long.sort_values("target"), fut.sort_values("fday"), left_on="target", right_on="fday", by="pid", direction="forward")
    m = m[m.felo.notna() & m.day.dt.year.between(2015, 2023)]   # 2024 rows would need 2025 ratings
    cols = ["pq20", "luck10", "pq_trend", "elo_trend"]
    Z = (m[cols] - m[cols].mean()) / m[cols].std()
    A = np.column_stack([np.ones(len(m)), Z.to_numpy()])
    yv = ((m.felo - m.elo) / 400).to_numpy()
    beta, *_ = np.linalg.lstsq(A, yv, rcond=None)
    resid = yv - A @ beta
    se = np.sqrt(np.diag(np.linalg.inv(A.T @ A)) * resid.var(ddof=A.shape[1]))
    return pd.DataFrame({"input (standardised)": ["intercept"] + cols, "effect on next-90-day Elo change (/400)": beta, "t": beta / se}), len(m)


def main():
    meta, X, fa1, fb1, fa, fb, qa, qb, p_elo, r = load()
    lab = Lab(meta, X)
    tourney = meta.tourney_id
    colsD, C_D = FROZEN["models"]["D LR"]["features"], FROZEN["models"]["D LR"]["C"]
    v2 = colsD + [c for g in F.GROUPS.values() for c in g]
    QG, MG = Q.GROUPS["quality"], Q.GROUPS["qmomentum"]
    R = {"Elo": fixed(p_elo, lab), "D (frozen spec)": lab.lr(colsD, C_D), "V2 full": lab.lr(v2),
         "V2 + quality": lab.lr(v2 + QG), "V2 + qmomentum": lab.lr(v2 + MG), "V2.1 (V2 + both)": lab.lr(v2 + QG + MG),
         "D + both (no V2 groups)": lab.lr(colsD + QG + MG)}
    for fam, cols in FAMILIES.items():
        R[f"V2.1 - {fam}"] = lab.lr([c for c in v2 + QG + MG if c not in cols])
    R["V2.1 XGBoost"] = lab.xgb(v2 + QG + MG)
    ref = R["V2 full"]["p"]
    T = pd.DataFrame([summary(n, x, ref, lab, tourney) for n, x in R.items()]).rename(columns={"vs_D": "vs_V2"})
    sel_ref = R["V2 full"]["sel_loss"]
    T["verdict"] = np.where((T.sel_logloss < sel_ref) & (T.high < 0), "improves on V2", np.where(T.low > 0, "worse than V2", "no clear difference"))
    T.loc[T.model == "V2 full", "verdict"] = "reference"
    names = ["D (frozen spec)", "V2 full", "V2.1 (V2 + both)"]
    C = calibration_section(R, names, lab, tourney)
    Dg = diagnostics(R, names, lab, meta, X, fa1, fb1)
    E, n_e = elo_lead(meta, fa, fb, qa, qb, r)
    L = ["# V2.1 development experiments (auto-generated by src/v2/run_v21.py)\n",
         "Dev file only; **2025/2026 not read**. Selection folds 2020-2022 (all choices), reporting folds 2023-2024 (held out from V2.1 choices, "
         "**not independent**). Same matches for every model. `vs_V2` = log loss minus full V2's on the reporting folds, 95% tournament-bootstrap interval.\n",
         "## 1. Models\n",
         ev.md(T, {**{c: f4 for c in ("sel_logloss", "rep_logloss", "brier", "accuracy", "cal_int", "cal_slope")}, "vs_V2": sg, "low": sg, "high": sg}), "",
         "## 2. Recalibration (fitted on 2020-2022 predictions, scored on 2023-2024)\n",
         "Platt slope < 1 means the raw model was overconfident on the fitting years. Adopt only if the interval is entirely below 0.\n",
         ev.md(C, {"platt_slope": f4, "raw": f4, "platt": f4, "isotonic": f4}), "",
         "## 3. Calibration diagnostics by situation (reporting folds)\n",
         "`fav_predicted`: average probability given to the favourite; `fav_observed`: how often the favourite won. Predicted above observed = overconfident. "
         "Groups are descriptive and overlap; small groups are noisy.\n",
         ev.md(Dg, {c: f4 for c in ("log_loss", "fav_predicted", "fav_observed", "cal_slope")}), "",
         "## 4. Does statistical form show up before Elo moves? (descriptive, 2015-2023)\n",
         f"Ordinary least squares on {n_e:,} player-match rows: each player's general Elo change over the next 90 days against pre-match inputs. "
         "Rows of one player overlap in time, so t-values are optimistic; read signs and sizes, not significance.\n",
         ev.md(E, {"effect on next-90-day Elo change (/400)": lambda v: f"{v:+.4f}", "t": lambda v: f"{v:+.1f}"}), ""]
    REPORT.write_text("\n".join(L))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
