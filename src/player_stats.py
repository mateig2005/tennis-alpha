"""Observed (raw) player statistics from the stored match histories, for display.

These are plain counts: wins and matches inside each window, as of the snapshot day. The models use the same windows
but shrink the rates toward 50% (features._features); a test checks the two agree. Nothing here is estimated.
Windows are measured from tournament start dates, like everywhere else in the project.
"""
from src.features import DAY, SRF, SURF, WIN

WINDOWS = {"d30": 30, "d90": 90, "d365": 365}


def _record(recs):
    wins = int(sum(r[WIN] for r in recs))
    return {"wins": wins, "matches": len(recs), "rate": wins / len(recs) if recs else None}


def raw_stats(state, player, snapshot_day):
    """state: per-player histories from the artifact. Returns last-5 results, win records by window and by surface (365 days)."""
    hist = state[player].hist if player in state else []
    recent = [r for r in hist if snapshot_day - r[DAY] <= 365]
    form = {"last10": _record(hist[-10:])} | {k: _record([r for r in recent if snapshot_day - r[DAY] <= w]) for k, w in WINDOWS.items()}
    return {"last5": ["W" if r[WIN] else "L" for r in hist[-5:]], "form": form,
            "surface_form": {name: _record([r for r in recent if r[SRF] == code]) for name, code in SURF.items()}}
