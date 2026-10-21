"""Reproducible player A/B assignment.

The raw table is 'winner vs loser', which leaks the answer. Here A is chosen by a hash of (seed, match_id):
the same match always gets the same order, whatever the row order, and the choice never looks at who won.
"""
import zlib

import pandas as pd


def assign_ab(df, seed=2024):
    a_wins = df.match_id.map(lambda m: zlib.crc32(f"{seed}|{m}".encode()) % 2 == 0)
    out = df.copy()
    out["a_id"] = df.winner_id.where(a_wins, df.loser_id)
    out["b_id"] = df.loser_id.where(a_wins, df.winner_id)
    out["rank_a"] = df.winner_rank.where(a_wins, df.loser_rank)
    out["rank_b"] = df.loser_rank.where(a_wins, df.winner_rank)
    out["pts_a"] = df.winner_points.where(a_wins, df.loser_points)
    out["pts_b"] = df.loser_points.where(a_wins, df.winner_points)
    out["y"] = a_wins.astype(int)  # the label: 1 if A won. Used for scoring and for rating updates AFTER predicting.
    return out.drop(columns=["winner_id", "loser_id", "winner_rank", "loser_rank", "winner_points", "loser_points"])
