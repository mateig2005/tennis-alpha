"""Forward evaluation of the frozen models on matches after the snapshot (PLAN_FORWARD.md). Prepared now, run later.

Run from the project root:
  python -m src.v2.forward_eval --dry-run              # proof on the dev file: cutoff 2023-12-31, evaluates 2024
  python -m src.v2.forward_eval --data new_rows.csv    # real run: approved new rows in the V2 table format (see src/v2/sources.py)
Parameters are fitted once on tour-level matches up to the cutoff; player state (ratings, form, profiles) keeps updating
chronologically through the evaluation period by the usual release rule. Evaluation outcomes never reach a fit.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from src import evaluate as ev
from src import models
from src.canonical import DEV_PATH, LOCKED_PATH, MAX_DAYS, SCORED_LEVELS, load_with_locked
from src.dataset import ELO_K_GEN, ELO_K_SURF
from src.elo import expected
from src.features import build_features, design, elo_ratings
from src.pairs import assign_ab
from src.v2 import features as F
from src.v2.data import from_sackmann_v2, load_dev_v2
from src.v2.sources import validate_source

EXPECTED = {"models/frozen_phase5.json": "16fe577b12eb96d3", "models/frozen_v2.json": "42d6a5a803741412",
            "models/v2_config.json": "e785ce642bce11f3"}  # committed with PLAN_FORWARD.md
CUTOFF = pd.Timestamp("2026-05-25")       # snapshot data end: last tournament start used to fit parameters
RETRO_END = pd.Timestamp("2026-10-19")   # period A (retrospective hold-out) ends; B (prospective) starts after
REPORT, DRY_REPORT = Path("reports/forward_eval.md"), Path("reports/forward_eval_dryrun.md")
MIN_CONCLUSIVE = 2000


def check_frozen(expected=EXPECTED):
    bad = {p: h for p, h in expected.items() if hashlib.sha256(Path(p).read_bytes()).hexdigest()[:16] != h}
    if bad:
        raise SystemExit(f"REFUSED: frozen files changed since the protocol was committed: {sorted(bad)}")


def guard(report, approved=False):
    if Path(report).exists() and not approved:
        raise SystemExit(f"REFUSED: {report} exists. The forward evaluation of a period runs once; a rerun needs the owner's approval (--rerun-approved).")


def evaluate(table, cutoff, start=None, end=None):
    """table: V2 table covering history and evaluation rows. Returns per-row predictions for the evaluation rows."""
    fz1, fz2 = json.loads(Path("models/frozen_phase5.json").read_text()), json.loads(Path("models/frozen_v2.json").read_text())
    ab = assign_ab(table).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    meta = ab.set_index("match_id")
    rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
    fa1, fb1 = build_features(ab, rt, MAX_DAYS)
    fa, fb, ctx = F.build(ab, rt, MAX_DAYS)
    X = F.design(design(ab, rt, fa1, fb1), fa, fb, ctx).loc[meta.index]
    tour = meta.level.isin(SCORED_LEVELS)
    train = tour & (meta.event_date.dt.year >= 2015) & (meta.event_date <= cutoff)
    test = tour & (meta.event_date > (start or cutoff)) & ((meta.event_date <= end) if end is not None else True)
    y = meta.y
    P = pd.DataFrame(index=meta.index[test])
    for key in ("D LR", "C LR"):
        spec = fz1["models"][key]
        P[key[0]] = models.predict(models.fit(X.loc[train, spec["features"]], y[train], spec["C"]), X.loc[test, spec["features"]])
    cols = fz2["model"]["features"]
    P["V2"] = models.predict(models.fit(X.loc[train, cols], y[train], fz2["model"]["C"]), X.loc[test, cols])
    r = rt.loc[P.index]
    P["Elo"] = expected(r.elo_a.to_numpy(), r.elo_b.to_numpy())
    f_a, f_b = fa.loc[P.index], fb.loc[P.index]
    P["v2_available"] = ((f_a.srv != 0) | (f_a.ret != 0)) & ((f_b.srv != 0) | (f_b.ret != 0))  # a rating stays exactly 0 only without statistics
    P["y"], P["tourney"], P["event_date"] = y[P.index], meta.tourney_id[P.index], meta.event_date[P.index]
    P["surface"], P["rank_a"], P["rank_b"] = meta.surface[P.index], meta.rank_a[P.index], meta.rank_b[P.index]
    return P


def period_a_split(new, end=RETRO_END):
    """Amendment 2: a tournament is in period A only if start + MAX_DAYS[level] <= end, i.e. it must have finished by the
    cutoff (the frozen visibility rule's maximum lengths; no match dates exist). Every other tournament already downloaded
    is quarantined: not period A (it may run past the cutoff) and not period B (its results were downloaded before any
    prediction was logged). Uses tournament ids, levels and start dates only, never outcomes."""
    t = new.groupby("tourney_id").agg(level=("level", "first"), start=("event_date", "min"), rows=("match_id", "size")).reset_index()
    t["assumed_end"] = t.start + pd.to_timedelta(t.level.map(MAX_DAYS), unit="D")
    t["status"] = np.where(t.assumed_end <= end, "period A", "quarantined")
    keep = new.tourney_id.isin(t.tourney_id[t.status == "period A"])
    return new[keep], new[~keep], t.sort_values("start")


def metrics(rows, models_=("V2", "D", "C", "Elo")):
    out = []
    for m in models_:
        a, b = ev.calibration_fit(rows.y, rows[m])
        out.append(dict(model=m, n=len(rows), log_loss=ev.log_loss(rows.y, rows[m]), brier=ev.brier(rows.y, rows[m]),
                        accuracy=ev.accuracy(rows.y, rows[m]), cal_intercept=a, cal_slope=b))
    return pd.DataFrame(out)


def summarise(P):
    """Metrics on the rows where V2 is available (same rows for every model) and the primary V2-minus-D difference."""
    rows = P[P.v2_available]
    summ = metrics(rows)
    d = ev.paired_bootstrap(ev.per_match_loss(rows.y, rows.V2), ev.per_match_loss(rows.y, rows.D), rows.tourney) if len(rows) else (np.nan,) * 3
    return summ, d


def verdict(n, d):
    if n < MIN_CONCLUSIVE:
        return f"inconclusive (fewer than {MIN_CONCLUSIVE:,} scored matches; see PLAN_FORWARD.md section 5)"
    return "V2 better than Model D" if d[2] < 0 else "V2 not shown to be better than Model D"


def write(path, parts, title):
    """parts: [(label, predictions)]. One table and one primary comparison per part."""
    L = [f"# Forward evaluation: {title} (auto-generated by src/v2/forward_eval.py)\n",
         "Frozen files checked against PLAN_FORWARD.md. Parameters fitted on tour-level matches up to the cutoff only; player state updated "
         "chronologically. Every model is scored on the rows where V2 is available (both players have serve/return statistics).\n"]
    for label, P in parts:
        summ, d = summarise(P)
        n = int(P.v2_available.sum())
        L += [f"## {label}\n", f"{len(P):,} evaluation matches, {n:,} scored.\n",
              ev.md(summ, {c: lambda v: f"{v:.4f}" for c in ("log_loss", "brier", "accuracy", "cal_slope")}),
              f"\n**Primary: V2 minus Model D log loss = {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]** (95%, resampling whole tournaments). "
              f"Verdict: **{verdict(n, d)}**.\n"]
    Path(path).write_text("\n".join(L))


MIN_SUBGROUP = 300  # Amendment 2: subgroups are reported only with at least this many scored matches


def full_report(path, P, quarantine, provenance):
    """The one-time period A report: same-row comparison, Model D on all eligible rows, calibration, subgroups."""
    from src.v2.audit_tools import calibration_table
    f4 = lambda v: f"{v:.4f}"
    rows = P[P.v2_available]
    summ, d = summarise(P)
    dc = ev.paired_bootstrap(ev.per_match_loss(rows.y, rows.C), ev.per_match_loss(rows.y, rows.D), rows.tourney)
    de = ev.paired_bootstrap(ev.per_match_loss(rows.y, rows.Elo), ev.per_match_loss(rows.y, rows.D), rows.tourney)
    n, n_t = len(rows), rows.tourney.nunique()
    L = ["# Forward evaluation, period A: one-time run (auto-generated by src/v2/forward_eval.py)\n",
         "Retrospective hold-out with documented structural exposure (PLAN_FORWARD.md Amendments 1-2). Frozen parameters fitted on tour-level "
         "matches up to 2026-05-25; player state updated chronologically by the release rule. **Not a prospective test.**\n",
         "## Provenance\n", "\n".join(f"- {k}: `{v}`" for k, v in provenance.items()), "",
         "## Eligibility\n",
         f"- Tournaments in period A (start + maximum length <= 2026-10-19): {int((quarantine.status == 'period A').sum())}; quarantined: {int((quarantine.status == 'quarantined').sum())} "
         f"({int(quarantine.rows[quarantine.status == 'quarantined'].sum())} rows, excluded from history and scoring).",
         f"- Tour-level (A, M, G, F) matches in period A: **{len(P):,}**; V2 available (both players have serve/return statistics): **{n:,}** "
         f"({n / max(len(P), 1):.1%}); tournaments: {n_t}.\n",
         "Quarantined tournaments (ids, levels and dates only):\n",
         ev.md(quarantine[quarantine.status == "quarantined"].assign(start=lambda q: q.start.dt.date, assumed_end=lambda q: q.assumed_end.dt.date)), "",
         "## 1. Same rows for every model (V2 available)\n", ev.md(summ, {c: f4 for c in ("log_loss", "brier", "accuracy", "cal_intercept", "cal_slope")}), "",
         f"**Primary: V2 minus Model D log loss = {d[0]:+.4f} [{d[1]:+.4f}, {d[2]:+.4f}]** (95%, 2,000 draws resampling whole tournaments). "
         f"Verdict under the pre-registered rule: **{verdict(n, d)}**.\n",
         f"Secondary (decide nothing): Model C minus Model D = {dc[0]:+.4f} [{dc[1]:+.4f}, {dc[2]:+.4f}]; Elo minus Model D = {de[0]:+.4f} [{de[1]:+.4f}, {de[2]:+.4f}].\n",
         "## 2. Model D, Model C and Elo on all eligible matches (coverage check)\n",
         ev.md(metrics(P, ("D", "C", "Elo")), {c: f4 for c in ("log_loss", "brier", "accuracy", "cal_intercept", "cal_slope")}), "",
         "## 3. Calibration by favourite band (same rows)\n"]
    for m in ("V2", "D"):
        t = calibration_table(rows.y, rows[m], rows.tourney)
        L += [f"**{m}**\n", ev.md(t, {"predicted": lambda v: f"{v:.1%}", "observed": lambda v: f"{v:.1%}", "gap": lambda v: f"{v:+.4f}", "low": lambda v: f"{v:+.4f}", "high": lambda v: f"{v:+.4f}"}), ""]
    ra, rb = rows.rank_a.fillna(9999), rows.rank_b.fillna(9999)
    groups = {f"surface: {s_}": (rows.surface == s_).to_numpy() for s_ in ("Hard", "Clay", "Grass")}
    groups["top-20 v top-20"] = ((ra <= 20) & (rb <= 20)).to_numpy()
    groups["top-10 v top-10"] = ((ra <= 10) & (rb <= 10)).to_numpy()
    sub = []
    for g, mask in groups.items():
        k = int(mask.sum())
        if k < MIN_SUBGROUP:
            sub.append(dict(group=g, n=k, V2=np.nan, D=np.nan, diff="not reported (n < 300)"))
            continue
        r = rows[mask]
        dd = ev.paired_bootstrap(ev.per_match_loss(r.y, r.V2), ev.per_match_loss(r.y, r.D), r.tourney)
        sub.append(dict(group=g, n=k, V2=ev.log_loss(r.y, r.V2), D=ev.log_loss(r.y, r.D), diff=f"{dd[0]:+.4f} [{dd[1]:+.4f}, {dd[2]:+.4f}]"))
    L += ["## 4. Subgroups (log loss; V2 minus D with interval; only n >= 300)\n", ev.md(pd.DataFrame(sub), {"V2": f4, "D": f4}), "",
          "Aggregate results only: no individual match outcome is listed. Period B (tournaments after 2026-10-19) has no rows here; it needs "
          "predictions logged before results exist.\n"]
    Path(path).write_text("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", help="approved new rows (CSV in the V2 table format)")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rerun-approved", action="store_true")
    args = ap.parse_args()
    check_frozen()
    if args.dry_run:
        P = evaluate(load_dev_v2(), pd.Timestamp("2023-12-31"))
        write(DRY_REPORT, [("2024 (development data)", P)], "DRY RUN on 2024 development data. Not a test: 2024 is known data; the verdict line only exercises the report")
        print(f"Wrote {DRY_REPORT}"); return
    if not args.data:
        sys.exit("Give --data (approved new rows) or --dry-run.")
    guard(REPORT, args.rerun_approved)
    new = pd.read_csv(args.data, parse_dates=["event_date"])
    if new.event_date.min() <= CUTOFF:
        sys.exit("REFUSED: new rows must start after the snapshot data end (2026-05-25).")
    eligible, quarantined, tourneys = period_a_split(new)
    print("Source check (counts only):", validate_source(eligible))
    history = from_sackmann_v2(pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(LOCKED_PATH)], ignore_index=True), load_with_locked())
    P = evaluate(pd.concat([history, eligible[history.columns]], ignore_index=True), CUTOFF)
    import subprocess
    git = lambda *a: subprocess.run(["git", *a], capture_output=True, text=True).stdout.strip()
    provenance = {"code commit": git("rev-parse", "--short", "HEAD"), "uncommitted changes": bool(git("status", "--porcelain", "src", "models")),
                  **{f"{k} sha256": v for k, v in EXPECTED.items()},
                  "input file": args.data, "input sha256": hashlib.sha256(Path(args.data).read_bytes()).hexdigest()[:16],
                  "TML 2026.csv sha256": json.loads(Path("data/raw/tml/manifest.json").read_text())["files"]["2026.csv"]["sha256"][:16],
                  "run (UTC)": pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M")}
    full_report(REPORT, P, tourneys, provenance)
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
