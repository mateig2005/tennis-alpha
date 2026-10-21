"""Refresh the local app's player state with approved TennisMyLife results; every model coefficient stays frozen (Phase 15).

  python -m src.refresh            build in artifacts/staging/, validate, then activate (or keep the current artifacts)
  python -m src.refresh rollback   restore the most recent backup

History = pinned Sackmann data to 2026-05-25 (unchanged) + TennisMyLife tournaments that started after it AND must have
finished before the file was downloaded (start + MAX_DAYS < download date). Coefficients are copied from the June 2026
production artifacts (artifacts/snapshot_2026-06-08/), so only ratings, form, profiles and head-to-head move.
"""
import json
import os
import pickle
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from src import h2h, tml
from src.build_artifacts import build as build_v1, player_names
from src.canonical import DEV_PATH, LOCKED_PATH, MAX_DAYS, load_with_locked
from src.v2.build_artifact import FROZEN as V2_FROZEN, build as build_v2
from src.v2.data import from_sackmann_v2

ART, REF = Path("artifacts"), Path("artifacts/snapshot_2026-06-08")
STAGING, BACKUPS = ART / "staging", ART / "backups"
FILES = ("dashboard.pkl", "v2.pkl", "h2h.pkl", "status.json")
SOURCE = "Jeff Sackmann / Tennis Abstract (CC BY-NC-SA 4.0) to 2026-05-25; TennisMyLife (stats.tennismylife.org, used with permission) after"
KEY_PLAYERS = ("sackmann:206173", "sackmann:207989")  # Sinner, Alcaraz: must always be present


def content_day(entry):
    """Day the file's content was produced: the server's Last-Modified (a file downloaded today may be days old), else the fetch time."""
    when = entry.get("last_modified") or entry["fetched_utc"]
    return pd.Timestamp(pd.to_datetime(when, utc=True)).tz_convert(None).normalize()


def new_rows(raw_dir=tml.RAW):
    """Cleaned TML rows of tournaments that started after the snapshot cutoff, split into complete / not yet complete."""
    manifest = json.loads((raw_dir / "manifest.json").read_text())["files"]
    years = sorted(int(n[:4]) for n in manifest if n[:4].isdigit() and int(n[:4]) >= 2026)
    frames, fetched = [], []
    for y in years:
        frames.append(tml.read_split(raw_dir / f"{y}.csv")[1])
        fetched.append(content_day(manifest[f"{y}.csv"]))
    df = tml.clean(pd.concat(frames, ignore_index=True))
    ends = df.tourney_date + pd.to_timedelta(df.tourney_level.map(MAX_DAYS), unit="D")
    done = ends < min(fetched)  # strictly before the day the file's content was written: the whole tournament must be in it
    return df[done].copy(), df[~done].copy(), min(fetched)


def coverage(complete, pending, raw_dir=tml.RAW):
    """Source completeness for the weeks the new data claims to cover. Compares tour-level events with the same weeks one
    year earlier (by tournament code) and each event's match count with its draw size (walkovers make a few rows fewer)."""
    if complete.empty:
        return {"same_weeks_last_year": 0, "missing_vs_last_year": [], "short_draws": []}
    lo, hi = complete.tourney_date.min(), complete.tourney_date.max()
    prev_file = raw_dir / f"{lo.year - 1}.csv"
    code = lambda tid: tml.norm_tid(tid).str.split("-", n=1).str[1]
    tour = lambda d: d[d.tourney_level.isin(["A", "M", "G", "F"])]
    missing, n_prev = [], 0
    if prev_file.exists():
        prev = tour(tml.clean(tml.read_split(prev_file)[0]))
        prev = prev[prev.tourney_date.between(lo - pd.Timedelta(days=371), hi - pd.Timedelta(days=361))]  # same calendar weeks, which shift about a day per year
        have = set(code(tour(complete).tourney_id)) | set(code(tour(pending).tourney_id)) if len(pending) else set(code(tour(complete).tourney_id))
        names = prev.assign(code=code(prev.tourney_id)).groupby("code").tourney_name.first()
        n_prev, missing = len(names), sorted(names[~names.index.isin(have)].tolist())
    d = tour(complete).groupby("tourney_id").agg(name=("tourney_name", "first"), rows=("match_id", "size"), draw=("draw_size", "first"))
    short = sorted(d[d.rows < d.draw - 3].name.tolist())
    return {"same_weeks_last_year": int(n_prev), "missing_vs_last_year": missing, "short_draws": short}


def known_duplicates(new, history):
    """Boolean mask over `new`: rows already in `history` (same tournament, round and pair of players, in any order)."""
    key = lambda d, prefix: set(zip(tml.norm_tid(d.tourney_id.str.removeprefix(prefix)), d.round_rank, [tuple(sorted(p)) for p in zip(d.winner_id, d.loser_id)]))
    have = key(history, "sackmann:")
    keys = zip(tml.norm_tid(new.tourney_id.str.removeprefix("tml:")), new.round_rank, [tuple(sorted(p)) for p in zip(new.winner_id, new.loser_id)])
    return np.array([k in have for k in keys], dtype=bool)


def assemble():
    """(V2 table for the full history, frame for the head-to-head index, names, coverage facts)."""
    history = from_sackmann_v2(pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(LOCKED_PATH)], ignore_index=True), load_with_locked())
    complete, pending, fetched = new_rows()
    known = tml.clean(pd.concat([tml.read_split(tml.RAW / f"{y}.csv")[0] for y in (2024, 2025, 2026)], ignore_index=True))
    m, _, _ = tml.link(known, tml.history_since(2024))
    pmap, conflicts = tml.player_map(m)
    names = player_names()
    unseen = sorted((set(complete.winner_id) | set(complete.loser_id)) - set(pmap))
    extra, new_players, ambiguous = tml.name_fallback(unseen, complete, names)  # ambiguous players stay separate new ids
    pmap = {**pmap, **{a: k.removeprefix("sackmann:") for a, k in extra.items()}}
    new = tml.to_v2_table(complete, pmap)
    invalid = int((complete[tml.STAT_COLS].notna().all(axis=1).to_numpy() & ~new.has_stats.to_numpy(bool)).sum())  # consistency rule -> missing
    dup = known_duplicates(new, history)
    new = new[~dup]
    table = pd.concat([history, new[history.columns]], ignore_index=True)
    for _, r in complete.iterrows():  # names for players new to the data (latest spelling wins, as in player_names)
        for side in ("winner", "loser"):
            pid = f"sackmann:{pmap[r[f'{side}_id']]}" if r[f"{side}_id"] in pmap else f"tml:{r[f'{side}_id']}"
            names.setdefault(pid, r[f"{side}_name"])
    ids = lambda col: [f"sackmann:{pmap[a]}" if a in pmap else f"tml:{a}" for a in complete[col]]
    h2h_frame = pd.concat([h2h.load_matches(), complete.assign(winner_id=ids("winner_id"), loser_id=ids("loser_id"))[h2h.COLUMNS][~dup]],
                          ignore_index=True)
    ends = complete.tourney_date + pd.to_timedelta(complete.tourney_level.map(MAX_DAYS), unit="D")
    facts = {"tml_matches_added": int(len(new)), "duplicates_dropped": int(dup.sum()), "tournaments_added": int(complete.tourney_id.nunique()),
             "latest_tournament_start": str(table.event_date.max().date()), "latest_complete_through": str(ends.max().date()) if len(ends) else None,
             "pending_tournaments": sorted(pending.tourney_name.unique().tolist()), "players_linked": len(pmap), "id_conflicts": len(conflicts),
             "new_players": len(new_players), "ambiguous_players": len(ambiguous), "source_content_date": str(fetched.date()),
             "download_date": max(f["fetched_utc"] for f in json.loads((tml.RAW / "manifest.json").read_text())["files"].values())[:10],
             "statistics_rows_invalid_treated_missing": invalid, "coverage": coverage(complete, pending)}
    return table, h2h_frame, names, facts


def ensure_reference(ref=REF):
    """The June 2026 production artifacts hold the frozen coefficients. On a fresh clone they are rebuilt exactly as in
    June: the frozen specifications fitted on the pinned Sackmann data to 2026-05-25 (deterministic)."""
    if all((ref / f).exists() for f in ("dashboard.pkl", "v2.pkl", "h2h.pkl")):
        return
    ref.mkdir(parents=True, exist_ok=True)
    raw = load_with_locked()
    (ref / "dashboard.pkl").write_bytes(pickle.dumps(build_v1(raw, json.loads(Path("models/frozen_phase5.json").read_text()), player_names())))
    v2raw = from_sackmann_v2(pd.concat([pd.read_csv(DEV_PATH), pd.read_csv(LOCKED_PATH)], ignore_index=True), raw)
    (ref / "v2.pkl").write_bytes(pickle.dumps(build_v2(v2raw, json.loads(V2_FROZEN.read_text()))))
    (ref / "h2h.pkl").write_bytes(pickle.dumps(h2h.build_index(h2h.load_matches())))


def build(out=STAGING):
    """Build every app artifact into `out` with coefficients copied from the reference artifacts."""
    ensure_reference()
    out.mkdir(parents=True, exist_ok=True)
    table, h2h_frame, names, facts = assemble()
    ref_v1 = pickle.loads((REF / "dashboard.pkl").read_bytes())
    ref_v2 = pickle.loads((REF / "v2.pkl").read_bytes())
    v1 = build_v1(table, json.loads(Path("models/frozen_phase5.json").read_text()), names, fixed=ref_v1["models"])
    v1["n_train"] = ref_v1["n_train"]  # the coefficients come from that fit
    v2 = build_v2(table, json.loads(V2_FROZEN.read_text()), fixed=(ref_v2["coef"], ref_v2["scale"]))
    v2["n_train"], v2["provenance"] = ref_v2["n_train"], ref_v2.get("provenance", {}) | {"state_refreshed_utc": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (out / "dashboard.pkl").write_bytes(pickle.dumps(v1))
    (out / "v2.pkl").write_bytes(pickle.dumps(v2))
    (out / "h2h.pkl").write_bytes(pickle.dumps(h2h.build_index(h2h_frame)))
    status = {"source": SOURCE, "attribution": "Match data: Jeff Sackmann / Tennis Abstract; TennisMyLife (stats.tennismylife.org).",
              "built_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"), "snapshot_date": str(v1["snapshot_date"]),
              "data_end": str(v1["data_end"]), "model_coefficients": "frozen: June 2026 production fit (data to 2026-05-25), unchanged",
              "model_versions": {"V1": "frozen_phase5", "V2": v2["version"]},
              "tml_files": {n: f["sha256"][:16] for n, f in json.loads((tml.RAW / "manifest.json").read_text())["files"].items()}, **facts}
    (out / "status.json").write_text(json.dumps(status, indent=2))
    return status


def validate(d=STAGING, ref=REF, key_players=KEY_PLAYERS):
    """Problems found in the artifacts in `d` (empty list = safe to activate)."""
    from fastapi.testclient import TestClient
    from src.api import create_app
    from src.predictor import Predictor
    from src.v2.predictor import V2Predictor
    problems = []
    P = Predictor.load(d / "dashboard.pkl")
    V = V2Predictor.load(P, d / "v2.pkl")
    ref1 = pickle.loads((ref / "dashboard.pkl").read_bytes())
    ref2 = pickle.loads((ref / "v2.pkl").read_bytes())
    for k in ("D", "C"):
        if P.art["models"][k] != ref1["models"][k]:
            problems.append(f"Model {k} coefficients differ from the frozen reference")
    if V.art["coef"] != ref2["coef"] or V.art["scale"] != ref2["scale"] or V.art["cols"] != ref2["cols"]:
        problems.append("V2 coefficients differ from the frozen reference")
    missing = set(ref1["players"].index) - set(P.art["players"].index)
    if missing or any(p not in P.art["players"].index for p in key_players):
        problems.append(f"{len(missing)} players from the previous artifact are missing")
    if P.art["snapshot_date"] < ref1["snapshot_date"] or P.art["data_end"] < ref1["data_end"]:
        problems.append("snapshot is older than the reference")
    top = P.players(active_only=True).index[:6]
    pairs = [(top[i], top[j]) for i in range(len(top)) for j in range(i + 1, len(top))][:8]
    for a, b in pairs:
        for s in ("Hard", "Clay", "Grass"):
            r1, r2 = P.predict(a, b, s), P.predict(b, a, s)
            if any(abs(r1["p"][k] + r2["p"][k] - 1) > 1e-9 for k in ("D", "C", "Elo")):
                problems.append(f"V1 asymmetry {a} v {b} on {s}")
            v1_, v2_ = V.predict(a, b, s), V.predict(b, a, s)
            if v1_["status"] == "ok" and abs(v1_["p"] + v2_["p"] - 1) > 1e-9:
                problems.append(f"V2 asymmetry {a} v {b} on {s}")
    index = pickle.loads((d / "h2h.pkl").read_bytes())
    with TestClient(create_app(P, h2h=index, v2=V)) as c:
        a, b = key_players
        api = c.get("/predict", params={"a": a, "b": b, "surface": "Clay"}).json()
        if api["p"] != P.predict(a, b, "Clay")["p"] or abs(api["v2"]["p"] - V.predict(a, b, "Clay")["p"]) > 1e-12:
            problems.append("API and standalone predictors disagree")
        ref_h2h = pickle.loads((ref / "h2h.pkl").read_bytes())
        if c.get("/h2h", params={"a": a, "b": b}).json()["total"] < len(ref_h2h.get(tuple(sorted(key_players)), [])):
            problems.append("head-to-head lost meetings")
    return problems


def activate(d=STAGING):
    """Swap staged files in. The current files are copied to a timestamped backup first; any error restores them."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")  # microseconds: two activations in one second must not collide
    backup = BACKUPS / stamp
    backup.mkdir(parents=True)
    for f in FILES:
        if (ART / f).exists():
            shutil.copy2(ART / f, backup / f)
    try:
        for f in FILES:
            os.replace(d / f, ART / f)  # atomic per file on the same disk
    except Exception:
        restore(backup)
        raise
    return backup


def restore(backup):
    for f in FILES:
        if (backup / f).exists():
            shutil.copy2(backup / f, ART / f)


def rollback():
    latest = sorted(p for p in BACKUPS.iterdir() if p.is_dir())[-1]
    restore(latest)
    return latest


def run():
    """Build, validate, activate. Returns (activated, status or problems). Never leaves half-replaced artifacts."""
    if STAGING.exists():
        shutil.rmtree(STAGING)
    status = build()
    problems = validate()
    if problems:
        return False, problems
    activate()
    return True, status


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "rollback":
        print(f"Restored artifacts from {rollback()}")
        return
    ok, info = run()
    if not ok:
        sys.exit("NOT activated; current artifacts kept. Problems:\n- " + "\n- ".join(info))
    print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
