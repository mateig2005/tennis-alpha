"""Build the dashboard artifact: production models, player states and Elo ratings (PLAN.md Phase 7).

Run from the project root:  python -m src.build_artifacts
Writes artifacts/dashboard.pkl (not committed: it contains data derived from the CC BY-NC-SA source).
The frozen specifications are refitted on ALL approved data; nothing is tuned and nothing can be validated here.
"""
import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

from src import h2h, models
from src.canonical import DEV_PATH, LOCKED_PATH, MAX_DAYS, SCORED_LEVELS, load_with_locked
from src.dataset import ELO_K_GEN, ELO_K_SURF
from src.elo import expected, run_elo
from src.features import EPOCH, build_features, design, elo_ratings
from src.pairs import assign_ab

FROZEN, OUT = Path("models/frozen_phase5.json"), Path("artifacts/dashboard.pkl")
DASHBOARD_MODELS = {"D": "D LR", "C": "C LR"}  # primary and challenger; Elo is computed from ratings


def build(raw, frozen, names=None, fixed=None):
    """raw: canonical match table. Returns the artifact dict used by src/predictor.py.
    fixed: the 'models' dict of an earlier artifact. Its coefficients are copied unchanged (Phase 15 state refresh);
    only player state, ratings and histories are rebuilt from `raw`. Without it the frozen specs are refitted."""
    ab = assign_ab(raw).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    meta = ab.set_index("match_id")
    rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
    fa, fb, state, snap = build_features(ab, rt, MAX_DAYS, return_state=True)
    D = design(ab, rt, fa, fb).loc[meta.index]
    train = (meta.level.isin(SCORED_LEVELS) & (meta.event_date.dt.year >= models.TRAIN_FROM)).to_numpy()
    fitted = {k: dict(v) for k, v in fixed.items()} if fixed else {}
    for key, name in ({} if fixed else DASHBOARD_MODELS).items():
        spec = frozen["models"][name]
        sc, lr = models.fit(D.loc[train, spec["features"]], meta.y[train], spec["C"])
        fitted[key] = {"name": name, "features": spec["features"], "C": spec["C"], "scale": sc.scale_.tolist(), "coef": lr.coef_[0].tolist()}
    pre, (gen, _) = run_elo(ab, k=ELO_K_GEN, release_days=MAX_DAYS)
    sur = run_elo(ab, k=ELO_K_SURF, release_days=MAX_DAYS)[1][1]
    long = pd.concat([raw[["winner_id", "winner_rank", "winner_points", "event_date", "round_rank"]].set_axis(["id", "rank", "points", "date", "rr"], axis=1),
                      raw[["loser_id", "loser_rank", "loser_points", "event_date", "round_rank"]].set_axis(["id", "rank", "points", "date", "rr"], axis=1)])
    last = long.sort_values(["date", "rr"], kind="stable").groupby("id").agg(last_date=("date", "last"), rank=("rank", "last"), points=("points", "last"), matches=("date", "size"))
    last["name"] = [(names or {}).get(i, i) for i in last.index]
    last["elo"] = [gen.get(i, 1500.0) for i in last.index]
    return {"models": fitted, "state": state, "elo": gen, "surface_elo": sur, "players": last, "elo_history": elo_history(ab, pre, gen),
            "snapshot_date": (EPOCH + np.timedelta64(int(snap), "D")).astype("datetime64[D]").astype(object), "snapshot_day": int(snap),
            "data_end": raw.event_date.max().date(), "data_start": raw.event_date.min().date(), "n_matches": len(raw),
            "n_train": int(train.sum()), "half_life": frozen["decay_half_life_days"], "source": "Jeff Sackmann / Tennis Abstract (CC BY-NC-SA 4.0)"}


def elo_history(ab, pre, final, k=ELO_K_GEN, release_days=MAX_DAYS):
    """Each player's real rating path: {id: [(day, rating), ...]}, days counted from EPOCH; a rating holds until the next entry.

    Rebuilt from the engine's own pre-match ratings: every result moves K * (1 - expected) points from loser to winner on the
    day the result is released (tournament start + maximum length). Nothing is interpolated, and each path is asserted to end
    exactly at the engine's final rating.
    """
    m = ab.set_index("match_id").join(pre.set_index("match_id")[["elo_a", "elo_b"]])
    day = (m.event_date - pd.Timestamp(EPOCH)).dt.days.to_numpy()
    release = day + m.level.map(release_days).to_numpy()
    a_won = m.y.to_numpy() == 1
    won, lost = np.where(a_won, m.a_id, m.b_id), np.where(a_won, m.b_id, m.a_id)
    delta = k * (1.0 - expected(np.where(a_won, m.elo_a, m.elo_b), np.where(a_won, m.elo_b, m.elo_a)))
    moves = pd.concat([pd.DataFrame({"id": won, "day": release, "d": delta}), pd.DataFrame({"id": lost, "day": release, "d": -delta})])
    rating = moves.groupby(["id", "day"], sort=True).d.sum().groupby(level=0).cumsum() + 1500.0
    first = pd.concat([pd.Series(day, index=m.a_id.to_numpy()), pd.Series(day, index=m.b_id.to_numpy())]).groupby(level=0).min()
    out = {}
    for pid, s in rating.groupby(level=0):
        path = [(int(first[pid]), 1500.0)] + [(int(d), float(v)) for (_, d), v in s.items()]
        assert abs(path[-1][1] - final[pid]) < 1e-6, f"rating path does not end at the engine's rating for {pid}"
        out[pid] = path
    return out


def player_names():
    df = pd.concat([pd.read_csv(p, usecols=["winner_id", "winner_name", "loser_id", "loser_name"]) for p in (DEV_PATH, LOCKED_PATH)])
    ids = pd.concat([df.winner_id, df.loser_id]); nm = pd.concat([df.winner_name, df.loser_name])
    return dict(zip("sackmann:" + ids.astype(str), nm))  # later rows win: the most recent spelling


def main():
    art = build(load_with_locked(), json.loads(FROZEN.read_text()), player_names())
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(pickle.dumps(art))
    print(f"Wrote {OUT}: {len(art['players']):,} players, {art['n_matches']:,} matches, data {art['data_start']} to {art['data_end']}, "
          f"snapshot date {art['snapshot_date']}, models trained on {art['n_train']:,} tour-level matches.")
    h2h.main()  # head-to-head index (display only), in its own file


if __name__ == "__main__":
    main()
