import numpy as np
import pandas as pd

from src.canonical import CANON


def synthetic(n=400, players=30, seed=0):
    """Random but valid canonical table: 'strong' players (low id) usually win; many ties in (date, round)."""
    rng = np.random.default_rng(seed)
    a = rng.integers(0, players, n)
    b = (a + 1 + rng.integers(0, players - 1, n)) % players
    a_wins = rng.random(n) < 1 / (1 + np.exp((a - b) / 8))
    win, lose = np.where(a_wins, a, b), np.where(a_wins, b, a)
    return pd.DataFrame({
        "match_id": [f"t:{i}" for i in range(n)], "source": "t", "tourney_id": "t:" + (rng.integers(0, 40, n)).astype(str),
        "event_date": pd.Timestamp("2015-01-05") + pd.to_timedelta(7 * rng.integers(0, 40, n), unit="D"),
        "round_rank": rng.integers(1, 7, n), "surface": rng.choice(["Hard", "Clay", "Grass", None], n),
        "level": rng.choice(["A", "M", "G", "D"], n), "winner_id": [f"p{i}" for i in win], "loser_id": [f"p{i}" for i in lose],
        "winner_rank": rng.integers(1, 200, n).astype(float), "loser_rank": rng.integers(1, 200, n).astype(float),
        "winner_points": rng.integers(100, 9000, n).astype(float), "loser_points": rng.integers(100, 9000, n).astype(float),
        "retired": False, "split": "train"})[CANON]
