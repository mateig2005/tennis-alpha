"""Prospective prediction log (Phase 15): record pre-match predictions now, score them once results exist.

  python -m src.predlog log --a "Jannik Sinner" --b "Carlos Alcaraz" --surface Hard --best-of 3 \
      --tournament "Paris Masters" --tournament-start 2026-10-26 [--match-start 2026-10-29T14:00Z]
  python -m src.predlog list          # what was logged, with its timing class
  python -m src.predlog verify        # check that no record was changed or removed

The log is a private append-only JSON-lines file (data/predictions/, git-ignored). Each record carries the hash of the
previous one, so editing or deleting an old record breaks the chain and `verify` fails. Nothing is ever overwritten.

Timing classes, from strongest to weakest:
  validated  created before 00:00 UTC on the tournament's first day: certainly before the match and its result.
  declared   created before a match start time typed in by the user (not independently verified).
  uncertain  anything else; kept, but excluded from the primary prospective evaluation.
"""
import argparse
import hashlib
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

LOG = Path("data/predictions/log.jsonl")
GENESIS = "0" * 64


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16] if Path(path).exists() else None


def _records(log=LOG):
    return [json.loads(l) for l in Path(log).read_text().splitlines() if l.strip()] if Path(log).exists() else []


def _digest(rec):
    body = {k: v for k, v in rec.items() if k != "hash"}
    return hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()


def timing_class(created_utc, tournament_start, match_start=None):
    created = pd.Timestamp(created_utc)
    if created < pd.Timestamp(tournament_start).tz_localize("UTC"):
        return "validated"
    if match_start is not None and created < pd.Timestamp(match_start):
        return "declared"
    return "uncertain"


def append(entry, log=LOG, now=None):
    """Append one prediction. entry: dict of match fields and model outputs. Returns the stored record."""
    log = Path(log)
    log.parent.mkdir(parents=True, exist_ok=True)
    prev = _records(log)
    created = (now or datetime.now(timezone.utc)).isoformat(timespec="seconds")
    rec = {"id": str(uuid.uuid4()), "created_utc": created, **entry, "prev_hash": prev[-1]["hash"] if prev else GENESIS}
    rec["timing"] = timing_class(created, rec["tournament_start"], rec.get("match_start_utc"))
    rec["hash"] = _digest(rec)
    with log.open("a") as f:  # append only: earlier lines are never rewritten
        f.write(json.dumps(rec, sort_keys=True) + "\n")
    return rec


def verify(log=LOG):
    """True if every record is intact and in order; raises ValueError naming the first broken record otherwise."""
    prev = GENESIS
    for i, r in enumerate(_records(log)):
        if r["prev_hash"] != prev or _digest(r) != r["hash"]:
            raise ValueError(f"prediction log broken at record {i} ({r.get('id')}): changed, removed or reordered")
        prev = r["hash"]
    return True


def make_entry(P, V, a_id, b_id, surface, best_of, tournament, tournament_start, match_start=None, status=None):
    """Model outputs for one future match from the loaded app predictors (V1 Predictor P, V2Predictor V)."""
    r = P.predict(a_id, b_id, surface)
    v2 = V.predict(a_id, b_id, surface, best_of, v1_result=r) if V is not None else {"status": "unavailable", "p": None, "reason": "V2 artifact missing", "notes": []}
    from src.v2.forward_eval import EXPECTED
    return {"a_id": a_id, "b_id": b_id, "a_name": r["A"]["name"], "b_name": r["B"]["name"], "surface": surface, "best_of": best_of,
            "tournament": tournament, "tournament_start": str(pd.Timestamp(tournament_start).date()),
            "match_start_utc": None if match_start is None else pd.Timestamp(match_start).isoformat(),
            "p": {"D": r["p"]["D"], "C": r["p"]["C"], "Elo": r["p"]["Elo"], "V2": v2["p"]}, "v2_status": v2["status"],
            "warnings": r["notes"] + v2.get("notes", []) + ([v2["reason"]] if v2.get("reason") else []), "reliable_v1": r["reliable"],
            "model_versions": EXPECTED, "artifacts": {n: _sha(f"artifacts/{n}") for n in ("dashboard.pkl", "v2.pkl")},
            "data": {"snapshot_date": str(P.art["snapshot_date"]), "data_end": str(P.art["data_end"]), **(status or {})}}


def join_results(records, results, window_days=14):
    """Attach outcomes to logged predictions. results: canonical rows (winner_id, loser_id, event_date = tournament start).
    A result matches when the two players are the same and its tournament starts within window_days of the logged start.
    Returns (scored rows with y = 1 if player A won, unmatched record ids, ambiguous record ids). Nothing is guessed."""
    res = results.assign(pair=[tuple(sorted(p)) for p in zip(results.winner_id, results.loser_id)])
    scored, unmatched, ambiguous = [], [], []
    for r in records:
        start = pd.Timestamp(r["tournament_start"])
        hit = res[(res.pair == tuple(sorted((r["a_id"], r["b_id"])))) & ((res.event_date - start).abs() <= pd.Timedelta(days=window_days))]
        if len(hit) == 0:
            unmatched.append(r["id"])
        elif len(hit) > 1:
            ambiguous.append(r["id"])
        else:
            scored.append({"id": r["id"], "timing": r["timing"], "y": int(hit.iloc[0].winner_id == r["a_id"]), **{f"p_{k}": v for k, v in r["p"].items()}})
    return pd.DataFrame(scored), unmatched, ambiguous


def already_played(index, a, b, tournament_start, window_days=14):
    """True if the local data already holds a meeting of a and b within window_days of tournament_start (head-to-head
    index from the app artifacts). Such a match can never be logged as a pre-match prediction."""
    start = pd.Timestamp(tournament_start).date()
    return any(abs((m["date"] - start).days) <= window_days for m in index.get((min(a, b), max(a, b)), []))


def _resolve(P, s):
    if s in P.art["players"].index:
        return s
    hits = P.art["players"][P.art["players"].name.str.lower() == s.lower()]
    if len(hits) != 1:
        raise SystemExit(f"Player '{s}' matched {len(hits)} players; use the player id instead.")
    return hits.index[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["log", "list", "verify"])
    ap.add_argument("--a"); ap.add_argument("--b"); ap.add_argument("--surface", choices=["Hard", "Clay", "Grass"])
    ap.add_argument("--best-of", type=int, choices=[3, 5], default=3)
    ap.add_argument("--tournament"); ap.add_argument("--tournament-start", help="YYYY-MM-DD (first day of the tournament)")
    ap.add_argument("--match-start", help="optional scheduled start, ISO time with zone (e.g. 2026-10-29T14:00Z)")
    args = ap.parse_args()
    if args.cmd == "verify":
        verify(); print(f"OK: {len(_records())} records intact"); return
    if args.cmd == "list":
        for r in _records():
            print(f"{r['created_utc']}  {r['timing']:9s}  {r['a_name']} v {r['b_name']} ({r['surface']}, bo{r['best_of']}, {r['tournament']} {r['tournament_start']})"
                  f"  D {r['p']['D']:.3f}  V2 {r['p']['V2'] if r['p']['V2'] is None else round(r['p']['V2'], 3)}")
        return
    if not all([args.a, args.b, args.surface, args.tournament, args.tournament_start]):
        raise SystemExit("log needs --a --b --surface --tournament --tournament-start")
    from src.predictor import Predictor
    from src.v2.predictor import ARTIFACT as V2A, V2Predictor
    P = Predictor.load()
    V = V2Predictor.load(P, V2A) if V2A.exists() else None
    status = json.loads(Path("artifacts/status.json").read_text()) if Path("artifacts/status.json").exists() else None
    a, b = _resolve(P, args.a), _resolve(P, args.b)
    import pickle
    index = pickle.loads(Path("artifacts/h2h.pkl").read_bytes()) if Path("artifacts/h2h.pkl").exists() else {}
    if already_played(index, a, b, args.tournament_start):
        raise SystemExit("REFUSED: this match is already in the local data, so it cannot be logged as a pre-match prediction.")
    rec = append(make_entry(P, V, a, b, args.surface, args.best_of, args.tournament,
                            args.tournament_start, args.match_start, status))
    print(f"Logged {rec['id']} ({rec['timing']}): {rec['a_name']} {rec['p']['D']:.1%} (Model D) v {rec['b_name']}")


if __name__ == "__main__":
    main()
