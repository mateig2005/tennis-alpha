"""Phase 6: independent out-of-sample evaluation of the FROZEN models on 2025 and 2026.

    python -m src.run_phase6 --dry-run   # dev data only: must reproduce Phase 5's 2024 numbers (safe to repeat)
    python -m src.run_phase6             # the one real run: opens the locked 2025/2026 data

Protocol: PLAN.md 'Phase 6 protocol: APPROVED'. Nothing is tuned here: every setting comes from
models/frozen_phase5.json. The real run refuses to start if code differs from the commit, and refuses to run twice.
"""
import argparse
import datetime
import hashlib
import json
import re
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

from src import diagnostics as dg
from src import evaluate as ev
from src import gbm, models
from src.canonical import DEV_PATH, LOCKED_PATH, MAX_DAYS, SCORED_LEVELS, load_dev, load_with_locked
from src.dataset import ELO_K_GEN, ELO_K_SURF
from src.elo import expected
from src.features import PRIOR, build_features, design, elo_ratings
from src.pairs import assign_ab
from src.run_phase4 import GROUPS
from src.run_phase5 import BANDS, FULL, NORANK

FROZEN, REPORT = Path("models/frozen_phase5.json"), Path("reports/phase6_backtest.md")
PRED, FIG = Path("data/processed/phase6_predictions.csv"), Path("reports/figures")
LR_GROUPS = {"B+ LR": GROUPS["B+"], "C LR": GROUPS["+dec"], "D LR": FULL, "D-norank LR": NORANK}
XGB_GROUPS = {"D XGB": FULL, "D-norank XGB": NORANK}
MODELS = ["A Elo", "B+ LR", "C LR", "D LR", "D XGB", "D-norank LR", "D-norank XGB"]
TIERS = {"Primary": [("D LR", "A Elo")],
         "Secondary": [("D LR", "B+ LR"), ("D XGB", "D LR"), ("D-norank LR", "A Elo")],
         "Exploratory": [("B+ LR", "A Elo"), ("C LR", "A Elo"), ("D XGB", "A Elo"), ("D LR", "C LR"), ("C LR", "B+ LR"),
                         ("D-norank LR", "D LR"), ("D-norank XGB", "D-norank LR")]}
PERIODS = [("2025", 2024, 2025), ("2026", 2025, 2026)]  # (label, last training year, test year)
PLOT = {"A Elo": ("#2a78d6", "o"), "B+ LR": ("#eb6834", "s"), "D LR": ("#1baf7a", "^"), "D XGB": ("#eda100", "D")}
f4, f3, sg = (lambda v: f"{v:.4f}"), (lambda v: f"{v:.3f}"), (lambda v: f"{v:+.4f}")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def check_frozen(fz):
    """The frozen file must agree with the code that will run. (Values such as C and tree counts are protected by git.)"""
    assert fz["primary"] == "D LR" and fz["rank_independent_primary"] == "D-norank LR"
    assert fz["elo"] == {"k_general": ELO_K_GEN, "k_surface": ELO_K_SURF, "init": 1500}
    assert fz["data"]["release_days"] == MAX_DAYS and fz["form_prior_matches"] == PRIOR
    assert fz["ab_seed"] == 2024 and fz["train_from_year"] == models.TRAIN_FROM
    hl = fz["decay_half_life_days"]
    for name, groups in (LR_GROUPS | XGB_GROUPS).items():
        m = fz["models"][name]
        assert m["features"] == models.columns(groups, m.get("half_life") or hl), f"feature list changed: {name}"
    for name in XGB_GROUPS:
        m = fz["models"][name]
        assert {k: v for k, v in m["params"].items() if k in gbm.FIXED} == gbm.FIXED, f"fixed XGBoost settings changed: {name}"
        assert m["params"]["max_depth"] <= 3 and 0 < m["n_trees"] <= gbm.MAX_TREES
    assert set(fz["models"]) == set(MODELS)
    return fz


def build(raw):
    """Features, labels and Elo probabilities for every match in raw, under the frozen strict visibility rule."""
    ab = assign_ab(raw).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    meta = ab.set_index("match_id")
    rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
    D = design(ab, rt, *build_features(ab, rt, MAX_DAYS)).loc[meta.index]
    assert np.isfinite(D.to_numpy()).all()
    r = rt.loc[meta.index]
    return SimpleNamespace(meta=meta, D=D, y=meta.y, year=meta.event_date.dt.year, scored=meta.level.isin(SCORED_LEVELS), tourney=meta.tourney_id,
                           p_elo=pd.Series(expected(r.elo_a.to_numpy(), r.elo_b.to_numpy()), index=meta.index))


def predict_period(S, fz, train_to, test_year):
    """Refit every frozen model once on tour-level matches TRAIN_FROM..train_to; predict test_year. Labels of test_year are never used."""
    tr = S.scored & (S.year >= models.TRAIN_FROM) & (S.year <= train_to)
    te = S.scored & (S.year == test_year)
    out = pd.DataFrame({"A Elo": S.p_elo[te]})
    for name in LR_GROUPS:
        m = fz["models"][name]
        out[name] = models.predict(models.fit(S.D.loc[tr, m["features"]], S.y[tr], m["C"]), S.D.loc[te, m["features"]])
    for name in XGB_GROUPS:
        m = fz["models"][name]
        cfg = {k: m["params"][k] for k in ("max_depth", "min_child_weight", "reg_lambda")}
        out[name] = gbm.predict(gbm.fit(S.D.loc[tr, m["features"]], S.y[tr], cfg, m["n_trees"]), S.D.loc[te, m["features"]])
    return out[MODELS], int(tr.sum())


def metrics_table(y, P, groups):
    rows = []
    for n in MODELS:
        ll, lo, hi = ev.paired_bootstrap(ev.per_match_loss(y, P[n]), np.zeros(len(y)), groups)
        a, b = ev.calibration_fit(y, P[n])
        rows.append((n, len(y), ll, f"[{lo:.4f}, {hi:.4f}]", ev.brier(y, P[n]), ev.accuracy(y, P[n]), ev.ece(y, P[n]), a, b))
    return pd.DataFrame(rows, columns=["model", "n", "log_loss", "95% CI", "brier", "accuracy", "ece", "cal. intercept", "cal. slope"])


def diff_table(y, P, groups):
    rows = []
    for tier, pairs in TIERS.items():
        for new, old in pairs:
            m, lo, hi = ev.paired_bootstrap(ev.per_match_loss(y, P[new]), ev.per_match_loss(y, P[old]), groups)
            rows.append((tier, f"{new} minus {old}", m, lo, hi, "yes" if hi < 0 else ("worse" if lo > 0 else "no")))
    return pd.DataFrame(rows, columns=["tier", "comparison", "diff", "low", "high", "first clearly better?"])


def dry_run(fz):
    """Dev file only: the frozen settings, trained on 2015-2023, must reproduce Phase 5's reported 2024 numbers."""
    S = build(load_dev())
    P, _ = predict_period(S, fz, 2023, 2024)
    y, text, bad = S.y[P.index], Path("reports/phase5_xgboost.md").read_text(), []
    for n in MODELS:
        ref = re.search(rf"^\| {re.escape(n)} \| validation 2024 \| (\d+) \| ([\d.]+) \| ([\d.]+) \| ([\d.]+) \|", text, re.M)
        got = (str(len(y)), f4(ev.log_loss(y, P[n])), f4(ev.brier(y, P[n])), f3(ev.accuracy(y, P[n])))
        if got != ref.groups():
            bad.append((n, got, ref.groups()))
    assert not bad, f"dry run does NOT reproduce Phase 5: {bad}"
    return S


def plot_reliability(y, P, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    ink, muted, surface = "#0b0b0b", "#52514e", "#fcfcfb"
    fig, ax = plt.subplots(figsize=(6.4, 6.0), dpi=160, facecolor=surface)
    ax.set_facecolor(surface)
    ax.plot([0, 1], [0, 1], color=muted, lw=1, ls=(0, (4, 3)), label="Perfect calibration", zorder=1)
    for name, (color, marker) in PLOT.items():
        r = ev.reliability(y, P[name])
        ax.plot(r.mean_pred, r.actual, color=color, lw=2, marker=marker, ms=7, mec=surface, mew=1.2, label=name, zorder=3)
    ax.set(xlim=(0, 1), ylim=(0, 1), aspect="equal")
    ax.set_xlabel("Predicted probability that player A wins", color=muted); ax.set_ylabel("Share of those matches A actually won", color=muted)
    ax.set_title(title, color=ink, loc="left", fontsize=11, pad=10)
    ax.grid(color="#e4e3df", lw=0.8); ax.tick_params(colors=muted, length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.legend(frameon=False, loc="upper left", labelcolor=ink, fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", facecolor=surface); plt.close(fig)


def guard(report=REPORT, approved=False):
    if Path(report).exists() and not approved:
        sys.exit(f"REFUSED: {report} already exists. The independent evaluation runs once. A rerun needs the owner's approval (--rerun-approved).")


def git(*a):
    return subprocess.run(["git", *a], capture_output=True, text=True, check=True).stdout.strip()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true"); ap.add_argument("--rerun-approved", action="store_true")
    args = ap.parse_args()
    fz = check_frozen(json.loads(FROZEN.read_text()))
    if args.dry_run:
        dry_run(fz)
        print("DRY RUN OK: frozen settings reproduce Phase 5's 2024 results for all 7 models. Locked data was not opened.")
        return
    guard(approved=args.rerun_approved)
    dirty = git("status", "--porcelain", "--", "src", "models", "tests", "PLAN.md", "requirements.txt")
    assert not dirty, f"uncommitted changes; commit before the independent run:\n{dirty}"
    head = git("rev-parse", "HEAD")
    S_dev = dry_run(fz)  # self-check before the locked file is opened

    S = build(load_with_locked())
    dev_ids = S_dev.meta.index
    pd.testing.assert_frame_equal(S.D.loc[dev_ids], S_dev.D)  # adding 2025/2026 changed no earlier feature: no look-ahead
    assert np.array_equal(S.p_elo[dev_ids].to_numpy(), S_dev.p_elo.to_numpy())

    L = ["# Phase 6: independent out-of-sample evaluation, 2025 and 2026 (auto-generated by src/run_phase6.py)\n",
         f"Run date {datetime.date.today()}. Code commit `{head}`. Nothing was tuned in this phase.\n",
         "## 0. Freeze verification and reproducibility\n",
         f"- Frozen specification `{FROZEN}`: sha256 `{sha(FROZEN)}`, last changed in commit `{git('log', '-1', '--format=%h', '--', str(FROZEN))}`; "
         "it and all code match the run commit (checked).",
         f"- PLAN.md sha256 `{sha('PLAN.md')[:16]}`; dev data `{sha(DEV_PATH)[:16]}`; locked data `{sha(LOCKED_PATH)[:16]}`.",
         "- Self-check passed: with the frozen settings, training on 2015-2023 reproduces Phase 5's 2024 log loss, Brier and accuracy for all 7 models.",
         "- Look-ahead check passed: adding the 2025/2026 rows changed no feature or Elo value of any earlier match.",
         "- Visibility: tournament-completion rule. Scored matches: tour-level (A/M/G/F). Elo and form update through the test period; model coefficients and trees do not.\n"]
    preds = []
    for label, train_to, test_year in PERIODS:
        P, n_train = predict_period(S, fz, train_to, test_year)
        ids = P.index
        y, grp, meta = S.y[ids], S.tourney[ids].to_numpy(), S.meta.loc[ids]
        L += [f"## {label}: {'independent test' if label == '2025' else 'forward out-of-sample (partial season)'}\n",
              f"Trained on {n_train:,} tour-level matches {models.TRAIN_FROM}-{train_to}. Scored: **{len(y):,} matches** in {meta.tourney_id.nunique()} tournaments, "
              f"tournament starts {meta.event_date.min().date()} to {meta.event_date.max().date()}. Player A won {100 * y.mean():.1f}% (A/B order is random)."]
        if label == "2026":
            L.append("**The approved 2026 data ends with tournaments starting 2026-05-25 (Roland Garros). It is not a full season:** no grass season, no North American hard-court swing.")
        L += ["", f"### {label}.1 Model comparison\n", "Lower is better for log loss, Brier, ECE. Calibration: intercept 0 and slope 1 are ideal; slope < 1 = overconfident.\n",
              ev.md(metrics_table(y, P, grp), {"log_loss": f4, "brier": f4, "accuracy": f3, "ece": f4, "cal. intercept": f3, "cal. slope": f3}), "",
              f"### {label}.2 Paired differences in log loss (95% interval, resampled by tournament; negative = first model better)\n",
              ev.md(diff_table(y, P, grp), {c: sg for c in ("diff", "low", "high")}), "",
              f"### {label}.3 Subgroups (exploratory, log loss)\n"]
        best = np.fmin(meta.rank_a, meta.rank_b)
        band = pd.cut(best, [0, 10, 30, 100, np.inf], labels=BANDS).astype(object).fillna(BANDS[3])
        quarter = meta.event_date.dt.year.astype(str) + "-Q" + meta.event_date.dt.quarter.astype(str)
        show = ["A Elo", "B+ LR", "C LR", "D LR", "D XGB", "D-norank LR"]
        for title, lab in (("Surface", meta.surface.fillna("Unknown")), ("Ranking band (best ranking of the two players)", band), ("Quarter (by tournament start)", quarter)):
            L += [f"{title}:\n", ev.md(dg.by_group(y, {n: P[n] for n in show}, lab), {n: f4 for n in show}), ""]
        fig = FIG / f"phase6_calibration_{label}.png"
        plot_reliability(y, P, fig, f"Calibration, {label} ({len(y):,} matches, 10 probability bins)")
        L += [f"### {label}.4 Calibration\n", f"![Calibration {label}](figures/{fig.name})\n", "Reliability table for the primary model (D LR):\n",
              ev.md(ev.reliability(y, P["D LR"]), {"lo": lambda v: f"{v:.1f}", "hi": lambda v: f"{v:.1f}", "mean_pred": f3, "actual": f3}),
              "\nReliability table for Elo:\n",
              ev.md(ev.reliability(y, P["A Elo"]), {"lo": lambda v: f"{v:.1f}", "hi": lambda v: f"{v:.1f}", "mean_pred": f3, "actual": f3}), ""]
        preds.append(P.assign(period=label, y=y, tourney_id=meta.tourney_id))
    out = pd.concat(preds)
    out.to_csv(PRED, index_label="match_id")
    L += ["## Reproducibility\n", f"Predictions for every scored match: `{PRED}` (not committed: derived from CC BY-NC-SA data), sha256 `{sha(PRED)}`.\n"]
    REPORT.write_text("\n".join(L))
    print("\n".join(L))


if __name__ == "__main__":
    main()
