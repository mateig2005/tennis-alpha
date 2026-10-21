"""TennisMyLife (TML) adapter: download, split protected outcomes, clean, link players, validate (Phase 14).

Source: https://stats.tennismylife.org/tennis-match-database (ATP singles, Sackmann-style columns, ATP player ids).
Permission: written permission reported by the owner on 2026-10-20 ("access to everything"); the text is held by the owner.
Raw files stay in data/raw/tml/ (git-ignored). Rows from tournaments starting after CUTOFF are written straight to
data/protected/ and are never printed or summarised here: they are the forward-evaluation outcomes (PLAN_FORWARD.md).
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src.identity import full_key
from src.prepare_data import NOT_PLAYED, ROUND_RANK
from src.v2.data import STATS, from_sackmann_v2
from src.canonical import DEV_PATH, LOCKED_PATH, from_sackmann

BASE = "https://stats.tennismylife.org/data/"
RAW, PROTECTED = Path("data/raw/tml"), Path("data/protected")
CUTOFF = pd.Timestamp("2026-05-25")  # last tournament start in the frozen snapshot
LEVELS = {"250": "A", "500": "A", "A": "A", "M": "M", "G": "G", "D": "D", "O": "O", "F": "F"}  # TML codes -> ours
STAT_COLS = [f"{p}_{s}" for p in "wl" for s in STATS]
# Thresholds fixed in reports/data_source_review_2026.md section 4, before any overlap comparison was run.
THRESHOLDS = {"linked_share": 0.98, "winner_agreement": 0.995, "stats_equal_share": 0.97, "stats_coverage_2026": 0.95}


def fetch(names, out=RAW):
    """Download files with curl (python.org Python on macOS lacks certificates); record sha256, Last-Modified and fetch time."""
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {"source": BASE, "files": {}}
    for name in names:
        tmp = out / f".{name}.part"
        # timeouts + retries: a run right after the Mac wakes can start before the network is ready (seen 2026-10-21)
        head = subprocess.run(["curl", "-fsSL", "--connect-timeout", "20", "--max-time", "180", "--retry", "4", "--retry-delay", "30",
                               "--retry-all-errors", "-D", "-", "-o", str(tmp), BASE + name], check=True, capture_output=True, text=True).stdout
        modified = next((l.split(":", 1)[1].strip() for l in head.splitlines() if l.lower().startswith("last-modified:")), None)
        data = tmp.read_bytes()
        tmp.replace(out / name)  # replace only after a complete download
        manifest["files"][name] = {"sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data), "last_modified": modified,
                                   "fetched_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return manifest


def read_split(path, cutoff=CUTOFF):
    """Read one TML match file and split it at the cutoff. Returns (known rows, protected rows); callers must not display the second."""
    df = pd.read_csv(path, dtype={"tourney_date": str, "winner_id": str, "loser_id": str, "tourney_level": str, "tourney_id": str})  # '250'/'500' must stay text
    df["tourney_date"] = pd.to_datetime(df.tourney_date, format="%Y%m%d")
    # TML dates some rows of one tournament on different days (Roland Garros 2026: 24 May, 25 May and later), so the
    # split uses the tournament's start (its earliest date), like the frozen visibility rule; never the row's own date.
    start = df.groupby(norm_tid(df.tourney_id)).tourney_date.transform("min")
    df["tourney_date"] = start
    late = start > cutoff
    return df[~late].copy(), df[late].copy()


def clean(df):
    """Same cleaning as src/prepare_data.py (completed matches only, retirements flagged, duplicates dropped), plus TML level codes."""
    out = df[~df.score.str.lower().str.contains(NOT_PLAYED, regex=True, na=True) & df["round"].notna()].copy()  # no round = no place in time
    out["retired"] = out.score.str.contains("RET")
    out = out[~out.duplicated(["tourney_id", "round", "winner_id", "loser_id", "score"])].copy()
    for c in ("winner_ht", "loser_ht"):
        out.loc[(out[c] < 150) | (out[c] > 230), c] = np.nan
    unknown = set(out.tourney_level) - set(LEVELS)
    assert not unknown, f"unknown TML levels {unknown}"
    out["tourney_level"] = out.tourney_level.map(LEVELS)
    assert not set(out["round"]) - set(ROUND_RANK), "unknown round"
    out["round_rank"] = out["round"].map(ROUND_RANK)
    # match_num is missing on some rows and may change when TML rewrites a file; tournament + round + both players is stable
    out["match_id"] = [f"{t}_{r}_{'-'.join(sorted((w, l)))}" for t, r, w, l in zip(out.tourney_id, out["round"], out.winner_id, out.loser_id)]
    assert out.match_id.is_unique, "duplicate TML match id"
    return out.reset_index(drop=True)


def norm_tid(tid):
    """'2024-0301' and '2024-301' are the same tournament (the sources pad codes differently)."""
    parts = tid.astype(str).str.split("-", n=1)
    return parts.str[0] + "-" + parts.str[1].str.lstrip("0")


def link(tml, sack):
    """One-to-one links between TML rows and Sackmann rows of the same tournament and round, by the two players' names
    in either order (so winners can then be compared). Returns linked pairs, unlinked Sackmann rows, unlinked TML rows."""
    key = lambda d, w, l: [tuple(sorted((full_key(a), full_key(b)))) for a, b in zip(d[w], d[l])]
    t = tml.assign(pair=key(tml, "winner_name", "loser_name"), tid=norm_tid(tml.tourney_id))
    s = sack.assign(pair=key(sack, "winner_name", "loser_name"), tid=norm_tid(sack.tourney_id))
    t_ok = ~t.duplicated(["tid", "round", "pair"], keep=False)  # repeated pair in one round (round robins): left unlinked
    s_ok = ~s.duplicated(["tid", "round", "pair"], keep=False)
    m = s[s_ok].merge(t[t_ok], on=["tid", "round", "pair"], suffixes=("_s", "_t"))
    return m, s[~s.match_id.isin(m.match_id_s)], t[~t.match_id.isin(m.match_id_t)]


def player_map(m):
    """ATP id -> Sackmann id from linked matches, using each player's own side. Ids mapped to more than one partner are conflicts."""
    same = m.winner_name_s.map(full_key) == m.winner_name_t.map(full_key)  # orientation: is TML's winner Sackmann's winner?
    pairs = pd.concat([pd.DataFrame({"atp": np.where(same, m.winner_id_t, m.loser_id_t), "sack": m.winner_id_s.astype(str)}),
                       pd.DataFrame({"atp": np.where(same, m.loser_id_t, m.winner_id_t), "sack": m.loser_id_s.astype(str)})])
    n_sack, n_atp = pairs.groupby("atp").sack.nunique(), pairs.groupby("sack").atp.nunique()
    bad_atp = set(n_sack[n_sack > 1].index) | set(pairs[pairs.sack.isin(n_atp[n_atp > 1].index)].atp)
    good = pairs[~pairs.atp.isin(bad_atp)].drop_duplicates()
    return dict(zip(good.atp, good.sack)), sorted(bad_atp)


def validate_overlap(tml, sack, scored=("A", "M", "G", "F")):
    """Agreement on already-known matches (tour level). Returns (metrics dict, pass/fail dict, linked frame)."""
    m, s_un, t_un = link(tml, sack)
    tour = sack.tourney_level.isin(scored)
    mt = m[m.tourney_level_s.isin(scored)]
    same = mt.winner_name_s.map(full_key) == mt.winner_name_t.map(full_key)
    both = mt[[f"{c}_s" for c in STAT_COLS]].notna().all(axis=1).to_numpy() & mt[[f"{c}_t" for c in STAT_COLS]].notna().all(axis=1).to_numpy()
    # compare each player's own statistics (TML's w_ is Sackmann's l_ when the winners disagree)
    def own(side_s, side_t):
        return np.all([np.abs(mt[f"{side_s}_{x}_s"].to_numpy() - np.where(same, mt[f"{side_s}_{x}_t"], mt[f"{'l' if side_s == 'w' else 'w'}_{x}_t"])) <= 1
                       for x in STATS], axis=0)
    eq = own("w", "w") & own("l", "l")
    k26 = tml[(tml.tourney_date.dt.year == 2026) & tml.tourney_level.isin(scored)]
    metrics = {"sackmann_tour_rows": int(tour.sum()), "linked_tour_rows": len(mt), "linked_share": len(mt) / max(int(tour.sum()), 1),
               "winner_agreement": float(same.mean()), "stats_equal_share": float(eq[both].mean()) if both.any() else float("nan"),
               "stats_coverage_2026": float(k26[STAT_COLS].notna().all(axis=1).mean()),
               "level_agreement": float((mt.tourney_level_s == mt.tourney_level_t).mean()),
               "surface_agreement": float((mt.surface_s == mt.surface_t).mean()),
               "score_agreement": float((mt.score_s.str.strip() == mt.score_t.str.strip()).mean()),
               "rank_agreement": float(((mt.winner_rank_s == np.where(same, mt.winner_rank_t, mt.loser_rank_t)) | mt.winner_rank_s.isna()).mean()),
               "unlinked_sackmann_tour": int((~s_un.match_id.isin(m.match_id_s) & s_un.tourney_level.isin(scored)).sum()),
               "unlinked_tml_tour": int(t_un.tourney_level.isin(scored).sum())}
    passed = {k: metrics[k] >= v for k, v in THRESHOLDS.items()}
    return metrics, passed, m


def to_v2_table(df, pmap):
    """Cleaned TML rows -> the V2 table (CANON + statistics), with our player ids where known.
    pmap: {ATP id: Sackmann id}. Unknown players get 'tml:<ATP id>' (they start from the usual priors)."""
    d = df.assign(match_id=df.match_id, tourney_id=norm_tid(df.tourney_id), split="forward",
                  tourney_date=df.tourney_date.dt.strftime("%Y-%m-%d"))
    v2 = from_sackmann_v2(d, from_sackmann(d))  # same mapping code as the history; prefixes are corrected below
    ids = lambda col: [f"sackmann:{pmap[a]}" if a in pmap else f"tml:{a}" for a in d[col]]
    v2["winner_id"], v2["loser_id"] = ids("winner_id"), ids("loser_id")
    v2["match_id"] = "tml:" + d.match_id.to_numpy()
    v2["source"] = "tml"
    v2["tourney_id"] = "tml:" + d.tourney_id.to_numpy()  # venue keeps the history's key, so pace history continues
    return v2


def name_fallback(atp_ids, df, known_names):
    """For players not linked through shared matches: unique normalised-name match against our known players only."""
    names = dict(zip(df.winner_id, df.winner_name)) | dict(zip(df.loser_id, df.loser_name))
    from src.identity import map_players
    return map_players({a: names[a] for a in atp_ids}, known_names)


def build_forward_input(pmap, known_names, out=PROTECTED / "forward_eval_input.csv"):
    """Protected rows -> V2 table for src/v2/forward_eval.py. Prints counts only; no result is shown."""
    _, prot = read_split(RAW / "2026.csv")
    c = clean(prot)
    new_ids = sorted((set(c.winner_id) | set(c.loser_id)) - set(pmap))
    extra, unmatched, ambiguous = name_fallback(new_ids, c, known_names)
    table = to_v2_table(c, {**pmap, **{a: k.removeprefix("sackmann:") for a, k in extra.items()}})
    table.to_csv(out, index=False)
    return {"rows": len(table), "players_linked_by_matches": len(set(c.winner_id) | set(c.loser_id)) - len(new_ids),
            "players_linked_by_name": len(extra), "new_players": len(unmatched), "ambiguous_players": len(ambiguous)}


def history_since(year=2024):
    """Our Sackmann rows (dev + locked files) from `year` on: already-known matches, the overlap used for validation."""
    s = pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(LOCKED_PATH)], ignore_index=True)
    s = s[s.tourney_date >= f"{year}-01-01"].copy()
    s["tourney_date"] = pd.to_datetime(s.tourney_date)
    return s


def validate(report=Path("reports/tml_validation.md")):
    """Overlap validation on 2024 to May 2026 (known matches only) against the thresholds fixed in Phase 13."""
    from src.v2.data import stats_ok
    t = clean(pd.concat([read_split(RAW / f"{y}.csv")[0] for y in (2024, 2025, 2026)], ignore_index=True))
    s = history_since(2024)
    met, ok, m = validate_overlap(t, s)
    valid = stats_ok(t, "w") & stats_ok(t, "l")
    met_v = validate_overlap(t.assign(**{c: t[c].where(valid) for c in STAT_COLS}), s)[0]
    pm, bad = player_map(m)
    rows = [(k, f"{met[k]:.4f}" if isinstance(met[k], float) else str(met[k]), f"{THRESHOLDS[k]:.3f}" if k in THRESHOLDS else "", "pass" if ok.get(k) else ("FAIL" if k in ok else ""))
            for k in met]
    L = ["# TennisMyLife validation against already-known matches (auto-generated by src/tml.py validate)\n",
         f"Files: data/raw/tml/manifest.json (sha256 and Last-Modified recorded). Overlap: tour-level matches from 2024 to tournaments starting "
         f"{CUTOFF.date()}. No row after that date is read here.\n", "| metric | value | threshold | result |", "|---|---|---|---|"]
    L += [f"| {a} | {b} | {c} | {d} |" for a, b, c, d in rows]
    L += [f"\nSame statistics comparison with TML rows that fail our consistency rule treated as missing (as the pipeline does): "
          f"**{met_v['stats_equal_share']:.4f}**. TML tour-level rows failing the rule: {int((~valid & t.tourney_level.isin(['A', 'M', 'G', 'F']) & t[STAT_COLS].notna().all(axis=1)).sum())}.\n",
          f"Players mapped from ATP ids to our ids through shared matches: {len(pm)}; conflicts: {len(bad)}.\n"]
    report.write_text("\n".join(L))
    return met, ok, pm


RULES = {"service points > 0": lambda c: c["svpt"] > 0, "service games > 0": lambda c: c["SvGms"] > 0,
         "first serves in <= service points": lambda c: c["1stIn"] <= c["svpt"], "first-serve points won <= first serves in": lambda c: c["1stWon"] <= c["1stIn"],
         "second-serve points won <= second serves": lambda c: c["2ndWon"] <= c["svpt"] - c["1stIn"], "break points saved <= faced": lambda c: c["bpSaved"] <= c["bpFaced"]}


def stats_exclusions(df):
    """Rows whose statistics are complete in the source but fail src/v2/data.stats_ok (a rule fixed in V2, before TML was
    used; it checks each player's own counts for internal consistency and never looks at who won). Counts by rule."""
    present = df[STAT_COLS].notna().all(axis=1)
    failing = {}
    bad_any = pd.Series(False, index=df.index)
    for name, rule in RULES.items():
        bad = present & ~(rule({x: df[f"w_{x}"] for x in STATS}) & rule({x: df[f"l_{x}"] for x in STATS}))
        failing[name] = int(bad.sum()); bad_any |= bad
    return {"rows": len(df), "rows_missing_statistics_in_source": int((~present).sum()), "rows_failing_consistency": int(bad_any.sum()),
            "statistic_values_excluded": int(bad_any.sum()) * len(STAT_COLS), "by_rule": failing}


def preflight(report=Path("reports/forward_eval_preflight.md")):
    """Chronology and data checks before the one-time period A run. Tournament ids, levels, dates and counts only."""
    from src.v2.forward_eval import RETRO_END, period_a_split
    new = pd.read_csv(PROTECTED / "forward_eval_input.csv", parse_dates=["event_date"])
    eligible, quarantined, t = period_a_split(new)
    tour = lambda d: d[d.level.isin(["A", "M", "G", "F"])]
    known = clean(pd.concat([read_split(RAW / f"{y}.csv")[0] for y in (2024, 2025, 2026)], ignore_index=True))
    _, prot = read_split(RAW / "2026.csv")
    pc = clean(prot)
    elig_raw = pc[norm_tid(pc.tourney_id).map(lambda x: f"tml:{x}").isin(eligible.tourney_id)]
    ek, ep = stats_exclusions(known), stats_exclusions(elig_raw)
    show = t.assign(start=t.start.dt.date, assumed_end=t.assumed_end.dt.date)
    L = ["# Period A preflight (auto-generated by src/tml.py preflight)\n",
         "No winner, score or other outcome is used or shown here: only tournament ids, levels, start dates and row counts.\n",
         "## 1. Date boundary\n",
         f"Rule (PLAN_FORWARD.md Amendment 2): period A = tournaments starting after 2026-05-25 whose start + maximum length "
         f"(G 14, M 13, F 8, O 14, D 4, A 7 days) is on or before {RETRO_END.date()}. All other downloaded tournaments are quarantined.\n",
         ev_md(show), "",
         f"- Eligible: {int((t.status == 'period A').sum())} tournaments, {len(eligible):,} rows, **{len(tour(eligible)):,} tour-level matches**.",
         f"- Quarantined: {int((t.status == 'quarantined').sum())} tournaments, {len(quarantined):,} rows (excluded from history and scoring).",
         f"- Latest assumed end among eligible tournaments: {t[t.status == 'period A'].assumed_end.max().date()}.\n",
         "**Roland Garros 2026** started 2026-05-25, the snapshot's last tournament start. It belongs to the frozen history: the Sackmann data holds "
         "it in full, and its results were released by the snapshot date (2026-06-08 = start + 14 days). TML's Roland Garros rows are assigned "
         "to that tournament start (TML dates some of them on later days), so they fall on the known side of the split and are not in period A. "
         "There is no double counting.\n",
         "**Period B** starts after 2026-10-19 and is prospective only for matches whose predictions are logged before their results exist. "
         "Nothing downloaded so far is relabelled as period B.\n",
         "## 2. Statistics excluded by the consistency rule (no values filled in)\n",
         "| data | rows | missing in source | failing consistency | statistic values excluded |", "|---|---|---|---|---|",
         f"| Overlap 2024 to May 2026 (all levels) | {ek['rows']:,} | {ek['rows_missing_statistics_in_source']:,} | {ek['rows_failing_consistency']:,} | {ek['statistic_values_excluded']:,} |",
         f"| Period A eligible rows (all levels) | {ep['rows']:,} | {ep['rows_missing_statistics_in_source']:,} | {ep['rows_failing_consistency']:,} | {ep['statistic_values_excluded']:,} |",
         "", f"Failures by rule, overlap: {ek['by_rule']}", f"\nFailures by rule, period A: {ep['by_rule']}\n",
         "A failing row keeps its result (it still counts for Elo, form and H2H); all 18 of its statistics are treated as missing. The rule "
         "(`stats_ok`, src/v2/data.py) was written in V2 stage 1 (commit 140d8b8), before TennisMyLife was considered.\n"]
    report.write_text("\n".join(L))
    return t, ek, ep


def ev_md(df):
    from src.evaluate import md
    return md(df)


def main():
    import sys
    cmd = sys.argv[1] if len(sys.argv) > 1 else "download"
    if cmd == "download":
        fetch(["ATP_Database.csv", "2024.csv", "2025.csv", "2026.csv"])
        PROTECTED.mkdir(parents=True, exist_ok=True)
        known, protected = read_split(RAW / "2026.csv")
        protected.to_csv(PROTECTED / "tml_2026_after_cutoff_raw.csv", index=False)
        print(f"Downloaded. 2026 file split at {CUTOFF.date()}: {len(known):,} known rows; the rest written to {PROTECTED}/ (not shown).")
    elif cmd == "validate":
        met, ok, _ = validate()
        print("Wrote reports/tml_validation.md:", ok)
    elif cmd == "prepare-eval":  # counts only; results stay unseen until the owner approves forward_eval
        from src.build_artifacts import player_names
        pm = validate()[2]
        print(build_forward_input(pm, player_names()))
    elif cmd == "preflight":
        t, ek, ep = preflight()
        print("Wrote reports/forward_eval_preflight.md", t.status.value_counts().to_dict())
    else:
        sys.exit("usage: python -m src.tml [download|validate|prepare-eval|preflight]")


if __name__ == "__main__":
    main()
