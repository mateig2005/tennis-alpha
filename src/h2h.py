"""Head-to-head records between two players, from the cleaned match data. Display only: NOT a model input.

Run from the project root:  python -m src.h2h        (writes artifacts/h2h.pkl; the model artifact is not touched)
The cleaned data already excludes walkovers, defaults and abandoned matches and keeps retirements, flagged, exactly
as the research does. Dates are tournament START dates: the data has no match dates, and none are invented.
"""
import pickle
from pathlib import Path

import pandas as pd

from src.canonical import DEV_PATH, LOCKED_PATH

OUT = Path("artifacts/h2h.pkl")
COLUMNS = ["tourney_name", "tourney_level", "tourney_date", "round", "round_rank", "surface", "score", "retired", "winner_id", "loser_id"]
SURFACES = ("Hard", "Clay", "Grass")
LEVELS = {"G": "Grand Slam", "M": "Masters 1000", "A": "Tour", "F": "Tour Finals", "D": "Davis Cup", "O": "Olympics"}


def build_index(df):
    """{(id_low, id_high): [match, ...]} with each pair's matches in chronological order (tournament start, then round)."""
    df = df.sort_values(["tourney_date", "round_rank"], kind="stable")
    index = {}
    for r in df[COLUMNS].itertuples(index=False):
        w, l = (str(x) if ":" in str(x) else f"sackmann:{x}" for x in (r.winner_id, r.loser_id))  # TML rows arrive already prefixed
        index.setdefault((min(w, l), max(w, l)), []).append({
            "date": pd.Timestamp(r.tourney_date).date(), "tournament": r.tourney_name, "level": LEVELS.get(r.tourney_level, r.tourney_level),
            "round": r.round, "surface": None if pd.isna(r.surface) else r.surface, "score": r.score, "winner_id": w, "retired": bool(r.retired)})
    return index


def summary(index, a, b):
    """Record of player a against player b. Swapping a and b swaps the win counts and nothing else."""
    matches = index.get((min(a, b), max(a, b)), [])
    wins = lambda ms, p: sum(m["winner_id"] == p for m in ms)
    on = lambda s: [m for m in matches if m["surface"] == s]
    recent = matches[::-1]  # most recent first
    return {"a_id": a, "b_id": b, "total": len(matches), "a_wins": wins(matches, a), "b_wins": wins(matches, b),
            "a_win_pct": wins(matches, a) / len(matches) if matches else None,
            "by_surface": {s: {"matches": len(on(s)), "a_wins": wins(on(s), a), "b_wins": wins(on(s), b)} for s in SURFACES},
            "other_surface": sum(m["surface"] not in SURFACES for m in matches),  # carpet or not recorded
            "retirements": sum(m["retired"] for m in matches), "last_meeting": recent[0] if recent else None, "last5": recent[:5], "matches": recent}


def load_matches():
    return pd.concat([pd.read_csv(p, usecols=COLUMNS) for p in (DEV_PATH, LOCKED_PATH)], ignore_index=True)


def main():
    index = build_index(load_matches())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(pickle.dumps(index))
    print(f"Wrote {OUT}: {len(index):,} player pairs, {sum(map(len, index.values())):,} matches.")


if __name__ == "__main__":
    main()
