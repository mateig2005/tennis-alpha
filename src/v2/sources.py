"""Data readiness for a future, owner-approved source (Phase 12). Nothing here downloads or merges anything.

A new source gets an adapter that outputs the V2 table: CANON (src/canonical.py) + best_of, venue, score, has_stats and
w_/l_ statistics (src/v2/data.py). `validate_source` checks it before it may extend the history; problems are
reported or raised, never silently fixed.
"""
import numpy as np
import pandas as pd

from src.canonical import CANON, MAX_DAYS
from src.identity import map_players
from src.v2.data import STATS, stats_ok
from src.v2.score import parse

RESULTS_ONLY = CANON + ["best_of", "score"]          # enough for Elo, form, ranking features, H2H and Model D / C
FOR_V2 = [f"{p}_{s}" for p in ("w", "l") for s in STATS] + ["has_stats", "venue"]  # serve/return groups and pace
SURFACES = {"Hard", "Clay", "Grass"}


def validate_source(df):
    """Hard errors raise ValueError. Returns a report of soft issues (counts) for the owner to review."""
    missing = [c for c in RESULTS_ONLY if c not in df.columns]
    if missing:
        raise ValueError(f"missing columns: {missing}")
    errors = []
    if df.match_id.duplicated().any(): errors.append("duplicate match_id")
    if df[["winner_id", "loser_id", "event_date", "level"]].isna().any().any(): errors.append("missing key field")
    if (df.winner_id == df.loser_id).any(): errors.append("winner equals loser")
    if not df.level.isin(list(MAX_DAYS)).all(): errors.append(f"unknown level: {sorted(set(df.level) - set(MAX_DAYS))}")
    if not df.best_of.isin([3, 5]).all(): errors.append("best_of must be 3 or 5")
    if not (df.surface.isna() | df.surface.isin(SURFACES)).all(): errors.append("surface must be Hard, Clay, Grass or missing")
    if errors:
        raise ValueError("; ".join(errors))
    parsed = df.score.map(parse)
    complete = parsed.map(lambda r: r is not None and r["complete"])
    retired_score = parsed.map(lambda r: r is not None and not r["complete"])
    report = {"rows": len(df), "unparseable_scores": int(parsed.isna().sum()),
              "retired_flag_mismatch": int((retired_score & ~df.retired.astype(bool)).sum() + (complete & df.retired.astype(bool)).sum())}
    if all(c in df.columns for c in FOR_V2):
        ok = stats_ok(df, "w") & stats_ok(df, "l")
        report["rows_with_valid_statistics"] = int(ok.sum())
        report["has_stats_but_invalid"] = int((df.has_stats.astype(bool) & ~ok).sum())
    else:
        report["rows_with_valid_statistics"] = 0  # results-only source: V2 is unavailable for players with no statistics
    report["possible_duplicates"] = len(find_duplicates(df))
    return report


def find_duplicates(df, window_days=14):
    """Pairs of rows that are probably the same match: same two players, same round, start dates within the window, and
    either the same tournament or two different sources (tournament ids differ between sources). Two rows of one source
    from different tournaments are real rematches (e.g. Madrid then Rome) and are not flagged.
    Across sources, run it after link_players has mapped the new ids onto known ones; otherwise no pair can match."""
    d = df.assign(pair=[tuple(sorted(p)) for p in zip(df.winner_id, df.loser_id)], day=pd.to_datetime(df.event_date))
    out = []
    for _, g in d.groupby(["pair", "round_rank"]):
        g = g.sort_values("day")
        gaps = g.day.diff().dt.days
        for i in np.flatnonzero((gaps <= window_days).to_numpy()):
            a, b = g.iloc[i - 1], g.iloc[i]
            if a.source != b.source or a.tourney_id == b.tourney_id:
                out.append((a.match_id, b.match_id))
    return out


def link_players(new_names, known_names):
    """{new id: name} -> ({new id: known id}, unmatched, ambiguous). Wraps src/identity.py: anything unclear is listed, not guessed."""
    return map_players(new_names, known_names)
