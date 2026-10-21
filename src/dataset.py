"""Shared dev dataset for modelling phases: features, labels and row masks. Reads the dev file only."""
from types import SimpleNamespace

import pandas as pd

from src import models
from src.canonical import MAX_DAYS, SCORED_LEVELS, load_dev
from src.elo import expected
from src.features import build_features, design, elo_ratings
from src.pairs import assign_ab

ELO_K_GEN, ELO_K_SURF = 24, 32  # frozen in Phase 4 (chosen on dev under the strict rule)


def load_design():
    ab = assign_ab(load_dev()).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    meta = ab.set_index("match_id")
    rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
    D = design(ab, rt, *build_features(ab, rt, MAX_DAYS)).loc[meta.index]
    year = meta.event_date.dt.year
    elig = meta.level.isin(SCORED_LEVELS) & year.between(2015, 2024)
    r = rt.loc[meta.index]
    return SimpleNamespace(meta=meta, D=D, y=meta.y, year=year, elig=elig, dev=elig & (year <= 2023), val=elig & (year == 2024),
                           oof=elig & year.isin(models.FOLDS), tourney=meta.tourney_id,
                           p_elo=pd.Series(expected(r.elo_a.to_numpy(), r.elo_b.to_numpy()), index=meta.index))
