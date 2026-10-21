"""Canonical match table: the only thing models read. A new data source needs just an adapter like from_sackmann().

Visibility key = (event_date, round_rank). Sources with real match days set event_date to that day and round_rank to 0;
Sackmann only has the tournament start, so event_date = tourney_date and round_rank orders the rounds.
"""
import pandas as pd

DEV_PATH = "data/processed/atp_matches_dev.csv"  # warmup + train + validation only (2025/2026 are in the locked file)
DEV_SPLITS = ("warmup", "train", "validation")
SCORED_LEVELS = ("A", "M", "G", "F")  # PLAN.md section 6; D (Davis Cup) and O (Olympics) are not scored
# Days after a tournament's start before its results count as visible (maximum possible length, PLAN.md M1).
MAX_DAYS = {"G": 14, "M": 13, "F": 8, "O": 14, "D": 4, "A": 7}
CANON = ["match_id", "source", "tourney_id", "event_date", "round_rank", "surface", "level",
         "winner_id", "loser_id", "winner_rank", "loser_rank", "winner_points", "loser_points", "retired", "split"]


def from_sackmann(df):
    """Map our cleaned Sackmann table to the canonical schema. Post-match columns (score, stats) are dropped on purpose."""
    return pd.DataFrame({
        "match_id": "sackmann:" + df.match_id.astype(str),
        "source": "sackmann",
        "tourney_id": "sackmann:" + df.tourney_id.astype(str),
        "event_date": pd.to_datetime(df.tourney_date),
        "round_rank": df.round_rank.astype(int),
        "surface": df.surface.replace({"Carpet": "Hard"}),  # 66 old carpet matches; NaN stays NaN (general Elo only)
        "level": df.tourney_level,
        "winner_id": "sackmann:" + df.winner_id.astype(str),  # prefix keeps ids from different sources apart
        "loser_id": "sackmann:" + df.loser_id.astype(str),
        "winner_rank": df.winner_rank,
        "loser_rank": df.loser_rank,
        "winner_points": df.winner_rank_points,
        "loser_points": df.loser_rank_points,
        "retired": df.retired.astype(bool),
        "split": df.split,
    })[CANON]


LOCKED_PATH = "data/processed/atp_matches_locked.csv"  # 2025 test + 2026 forward data


def load_with_locked():
    """Dev + locked rows. ONLY for the owner-approved independent evaluation (src/run_phase6.py)."""
    return validate(from_sackmann(pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(LOCKED_PATH)], ignore_index=True)))


def validate(df):
    assert list(df.columns) == CANON, "unexpected columns"
    assert df.match_id.is_unique, "duplicate match_id"
    assert (df.winner_id != df.loser_id).all(), "winner == loser"
    assert df[["event_date", "round_rank", "winner_id", "loser_id", "level"]].notna().all().all(), "missing key field"
    return df


def load_dev(path=DEV_PATH):
    """Dev data only. The dev file physically excludes 2025+, and this re-checks it."""
    df = validate(from_sackmann(pd.read_csv(path)))
    assert set(df.split) <= set(DEV_SPLITS), "locked split found in dev data"
    assert df.event_date.max() < pd.Timestamp("2025-01-01"), "2025+ data found in dev data"
    return df
