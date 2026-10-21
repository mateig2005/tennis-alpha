"""Build the V2 prediction artifact (Phase 11): the frozen V2 specification refitted on all approved data.

Run from the project root:  python -m src.v2.build_artifact   -> artifacts/v2.pkl (not committed: derived licensed data)
Production refit only: settings come from models/frozen_v2.json and nothing is selected here. Player states are
built with the same leak-free pass as research, and everything is released by the same snapshot as the V1 dashboard.
"""
import hashlib
import json
import pickle
import subprocess
from pathlib import Path

import pandas as pd

from src import models
from src.canonical import DEV_PATH, LOCKED_PATH, MAX_DAYS, SCORED_LEVELS, load_with_locked
from src.dataset import ELO_K_GEN, ELO_K_SURF
from src.features import build_features, design, elo_ratings
from src.pairs import assign_ab
from src.v2 import features as F
from src.v2 import quality as Q
from src.v2.data import from_sackmann_v2

FROZEN, OUT = Path("models/frozen_v2.json"), Path("artifacts/v2.pkl")


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


def build(raw, spec, fixed=None):
    """raw: V2 table (canonical + statistics). Returns the artifact dict read by src/v2/predictor.py.
    fixed: (coef, scale) of an earlier artifact, copied unchanged (Phase 15 state refresh); otherwise fitted with the frozen spec."""
    ab = assign_ab(raw).sort_values(["event_date", "round_rank", "match_id"], kind="stable").reset_index(drop=True)
    meta = ab.set_index("match_id")
    rt = elo_ratings(ab, ELO_K_GEN, ELO_K_SURF, MAX_DAYS)
    fa1, fb1, _, snap = build_features(ab, rt, MAX_DAYS, return_state=True)
    fa, fb, ctx, state, tour = F.build(ab, rt, MAX_DAYS, return_state=True)
    *_, qstate, vol0 = Q.build(ab, ctx, MAX_DAYS, return_state=True)
    X = F.design(design(ab, rt, fa1, fb1), fa, fb, ctx).loc[meta.index]
    cols, C = spec["model"]["features"], spec["model"]["C"]
    train = (meta.level.isin(SCORED_LEVELS) & (meta.event_date.dt.year >= 2015)).to_numpy()
    coef, scale = fixed if fixed else (lambda m: (m[1].coef_[0].tolist(), m[0].scale_.tolist()))(models.fit(X.loc[train, cols], meta.y[train], C))
    stat_matches = {pid: len(st.raw) for pid, st in state.items()}  # released matches with serve/return statistics
    for st in state.values():  # keep only what a prediction at the snapshot can read
        st.raw = st.raw[-F.WINDOW:]
        st.elo = [e for e in st.elo if e[0] >= snap - F.TREND]
    for q in qstate.values():
        q.all, q.load = q.all[-20:], [r for r in q.load if r[0] >= snap - Q.CFG["load_days"]]  # q.stat kept whole: the surface window looks further back
    return {"version": spec["version"], "spec": spec, "cols": cols, "C": C, "scale": list(scale), "coef": list(coef),
            "state": state, "tour": tour, "qstate": qstate, "vol0": vol0, "stat_matches": stat_matches,
            "snapshot_day": int(snap), "data_end": raw.event_date.max().date(), "n_train": int(train.sum())}


def main():
    spec = json.loads(FROZEN.read_text())
    raw = from_sackmann_v2(pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(LOCKED_PATH)], ignore_index=True), load_with_locked())
    art = build(raw, spec)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip() or "unknown"
    art["provenance"] = {"git_commit": commit, "frozen_v2_sha256_16": sha(FROZEN), "v2_config_sha256_16": sha("models/v2_config.json"),
                         "dev_csv_sha256_16": sha(DEV_PATH), "locked_csv_sha256_16": sha(LOCKED_PATH)}
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_bytes(pickle.dumps(art))
    print(f"Wrote {OUT}: V2 {art['version']}, {len(art['state']):,} players, trained on {art['n_train']:,} matches, data to {art['data_end']}.")


if __name__ == "__main__":
    main()
