"""TENNIS ALPHA API: a thin FastAPI layer over src/predictor.py (the same Predictor the Streamlit dashboard uses).

Run from the project root (localhost only):   python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
Interactive documentation:                    http://127.0.0.1:8000/docs
The trained artifact is loaded once at startup. Nothing is trained, tuned or recalibrated here, and every number is a
hypothetical-match estimate as of the historical snapshot date, never a live forecast.
"""
import datetime as dt
import json
import pickle
from contextlib import asynccontextmanager
from datetime import date, timedelta
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from fastapi import Depends, FastAPI, HTTPException, Query, Request
from pydantic import BaseModel, Field

from src import h2h as h2h_records
from src.features import EPOCH
from src.player_stats import raw_stats
from src.predictor import STALE_DAYS, SURFACES, UNRELIABLE_DAYS, Predictor
from src.v2.predictor import V2Predictor

ROOT = Path(__file__).resolve().parent.parent
ARTIFACT = ROOT / "artifacts" / "dashboard.pkl"
H2H_ARTIFACT = ROOT / "artifacts" / "h2h.pkl"
V2_ARTIFACT = ROOT / "artifacts" / "v2.pkl"
STATUS_FILE, SYNC_STATE = ROOT / "artifacts" / "status.json", ROOT / "data" / "sync_state.json"
STALE_DATA_DAYS, STALE_SYNC_DAYS = 14, 3
V2_MISSING = "The V2 artifact is missing. Build it with: python -m src.v2.build_artifact"
Surface = Literal["Hard", "Clay", "Grass"]
GROUP_LABELS = {"elo": "Elo ratings", "rank": "Ranking", "last": "Last 5 / 10", "win": "Win-rate windows", "surf": "Surface form",
                "rest": "Rest / workload", "dec": "Decayed form", "adj": "Opponent-adjusted", "adjdec": "Opp-adj. decayed"}
DISCLAIMER = ("Historical snapshot, not a live forecast. Probabilities describe a hypothetical match on the snapshot date, "
              "using only results in the data by then. No real-time ATP data. Research tool, not betting advice.")
_EPOCH = pd.Timestamp(EPOCH).date()


# ---- response schemas ------------------------------------------------------------------------------------------------
class WinRecord(BaseModel):
    wins: int
    matches: int
    rate: float | None = Field(description="wins / matches; null when the player has no match in the window")


class PlayerSummary(BaseModel):
    id: str = Field(description="Stable player identifier; use this, not the name")
    name: str
    rank: int | None = Field(description="Last ATP ranking in the data (as of the player's last tournament); null if none")
    elo: float
    matches: int = Field(description="Matches in the data")
    last_event: date | None = Field(description="Start date of the player's last tournament in the data")
    days_since_last_event: int | None = Field(description="Days from last_event to the snapshot date")
    active: bool = Field(description=f"Played within {UNRELIABLE_DAYS} days of the snapshot")
    stale: bool = Field(description=f"Last played more than {STALE_DAYS} days before the snapshot")


class EloPoint(BaseModel):
    date: date
    elo: float


class PlayerDetail(PlayerSummary):
    points: int | None
    surface: Surface = Field(description="Surface the model inputs below were computed for")
    surface_elo: dict[Surface, float | None] = Field(description="null = no rated match on that surface (the model then uses 1500)")
    last5: list[Literal["W", "L"]] = Field(description="Most recent last")
    form: dict[str, WinRecord] = Field(description="Observed win records: last10, d30, d90, d365 (windows end at the snapshot)")
    surface_form: dict[Surface, WinRecord] = Field(description="Observed win records over the last 365 days, by surface")
    adj_form: float = Field(description="Model input adj10: shrunk mean of (result - Elo expectation) over the last 10 matches")
    inputs: dict[str, float] = Field(description="Model D per-player inputs for this surface (rates shrunk toward 50%)")
    elo_history: list[EloPoint] = Field(description="Real rating path; each rating holds until the next date")


class Contribution(BaseModel):
    group: str
    label: str
    value: float = Field(description="Log-odds pushed toward player A: trained coefficient x standardised A-minus-B difference")


class V2Quality(BaseModel):
    points_vs_expected20: float = Field(description="Shrunk mean of (points won share - pre-match expectation), last 20 matches. Descriptive (V2.1 research), not a V2 input")
    luck10: float = Field(description="Shrunk mean of (result - win chance implied by the points actually won), last 10. Descriptive, not a V2 input")


class V2Player(BaseModel):
    stat_matches: int = Field(description="Released matches with serve/return statistics")
    serve: float = Field(description="Opponent-adjusted serve rating: serve points won above the tour average (0.01 = 1 point in 100)")
    ret: float = Field(description="Opponent-adjusted return rating, same units")
    serve_surface: float
    ret_surface: float
    serve_recent: float = Field(description="Serve rating with a 60-day half-life")
    ret_recent: float
    spw20: float = Field(description="Serve points won, last 20 matches with statistics (shrunk toward the tour average)")
    rpw20: float
    hold20: float
    break20: float
    ace20: float
    elo_trend: float = Field(description="General Elo change over the last 90 days, divided by 400")
    quality: V2Quality | None


class V2Matchup(BaseModel):
    pa: float = Field(description="Expected share of serve points player A wins against this opponent on this surface")
    pb: float
    tour_serve_avg: float
    p_points_model: float = Field(description="Raw point-to-match probability for A (uncalibrated; one V2 input)")


class V2H2H(BaseModel):
    residual: float = Field(description="Shrunk (result - Elo expectation) over earlier meetings, A's side; the V2 input")
    meetings: int


class V2Prediction(BaseModel):
    status: Literal["ok", "unavailable"]
    reason: str | None
    version: str | None = None
    best_of: int | None = None
    validation: str | None = None
    data_end: date | None = None
    p: float | None = Field(None, description="V2 probability that player A wins; null when unavailable")
    logit: float | None = None
    contributions: list[Contribution] = []
    notes: list[str] = []
    players: dict[Literal["a", "b"], V2Player] | None = None
    matchup: V2Matchup | None = None
    h2h: V2H2H | None = None


class Prediction(BaseModel):
    surface: Surface
    snapshot_date: date
    a: PlayerDetail
    b: PlayerDetail
    p: dict[Literal["D", "C", "Elo"], float] = Field(description="Probability that player A wins; player B's is 1 minus this")
    logit: float = Field(description="Model D log-odds for player A; equals the sum of contributions")
    contributions: list[Contribution]
    notes: list[str]
    reliable: bool = Field(description="False when an input is missing or too old for an honest estimate")
    v2: V2Prediction = Field(description="Tennis Alpha V2 (experimental), separate from the V1 numbers above")


class ModelInfo(BaseModel):
    key: str
    name: str
    role: Literal["primary", "challenger", "benchmark", "experimental"]
    kind: str
    n_features: int


class DataStatus(BaseModel):
    source: str
    attribution: str
    latest_tournament_start: date | None = Field(None, description="Start of the most recent tournament in the player data")
    latest_complete_through: date | None = Field(None, description="Date by which every included tournament must have finished")
    pending: list[str] = Field([], description="Downloaded tournaments not yet included because they may still be running")
    known_gaps: list[str] = Field([], description="Tour events held in the same weeks a year earlier but absent from the source, or with short draws")
    model_coefficients: str = Field(description="Coefficients are frozen; this is NOT the data date")
    last_success_utc: str | None = None
    last_attempt_utc: str | None = None
    last_error: str | None = None
    data_age_days: int | None = Field(None, description="Days from latest_complete_through to today")
    warnings: list[str] = []


def data_status(today=None) -> dict | None:
    """Freshness of the local data: artifacts/status.json (written by src/refresh.py) and data/sync_state.json (src/sync.py)."""
    if not STATUS_FILE.exists():
        return None
    st = json.loads(STATUS_FILE.read_text())
    sync = json.loads(SYNC_STATE.read_text()) if SYNC_STATE.exists() else {}
    today = today or dt.datetime.now(dt.timezone.utc).date()
    through = date.fromisoformat(st["latest_complete_through"]) if st.get("latest_complete_through") else None
    age = (today - through).days if through else None
    warnings = []
    if age is not None and age > STALE_DATA_DAYS:
        warnings.append(f"The newest results are {age} days old; recent matches are missing from player form and ratings.")
    last = sync.get("last_success_utc")
    if last is None:
        warnings.append("No successful data update recorded yet.")
    elif (today - dt.datetime.fromisoformat(last).date()).days > STALE_SYNC_DAYS:
        warnings.append(f"The last successful update was on {last[:10]}. Run ./update.sh.")
    if sync.get("last_result") == "failed" and sync.get("last_error"):
        warnings.append("The most recent update failed; the previous data is still in use.")
    return dict(source=st["source"], attribution=st["attribution"], latest_tournament_start=st.get("latest_tournament_start"),
                latest_complete_through=through, pending=st.get("pending_tournaments", []), model_coefficients=st["model_coefficients"],
                known_gaps=st.get("coverage", {}).get("missing_vs_last_year", []) + [f"{n} (short draw)" for n in st.get("coverage", {}).get("short_draws", [])],
                last_success_utc=last, last_attempt_utc=sync.get("last_attempt_utc"), last_error=sync.get("last_error"), data_age_days=age, warnings=warnings)


class Meta(BaseModel):
    snapshot_date: date
    data_start: date
    data_end: date = Field(description="Start date of the last tournament in the data")
    n_matches: int
    n_players: int
    n_active_players: int
    n_training_matches: int
    source: str
    live: bool = Field(description="Always false: this service has no real-time data")
    disclaimer: str
    models: list[ModelInfo]
    data_status: DataStatus | None = Field(None, description="Local data freshness; null when no refresh has run yet")


class Health(BaseModel):
    status: Literal["ok", "unavailable"]
    artifact_loaded: bool
    v2_loaded: bool
    detail: str


class H2HMatch(BaseModel):
    date: dt.date = Field(description="Tournament START date; the data has no match dates")  # dt.date: a field named `date` hides the type
    tournament: str
    level: str
    round: str
    surface: str | None = Field(description="null when the data records no surface")
    score: str = Field(description="As recorded, from the winner's side")
    winner_id: str
    retired: bool = Field(description="The match ended by retirement; it counts as a win, as in the research data")


class SurfaceRecord(BaseModel):
    matches: int
    a_wins: int
    b_wins: int


class HeadToHead(BaseModel):
    a_id: str
    b_id: str
    total: int
    a_wins: int
    b_wins: int
    a_win_pct: float | None = Field(description="null when the players have never met in the data")
    by_surface: dict[Surface, SurfaceRecord]
    other_surface: int = Field(description="Meetings on carpet or with no recorded surface")
    retirements: int
    last_meeting: H2HMatch | None
    last5: list[H2HMatch] = Field(description="Most recent first")
    matches: list[H2HMatch] = Field(description="Every recorded meeting, most recent first")
    data_start: date = Field(description="Start of the first tournament in the data; earlier meetings are missing")
    data_end: date = Field(description="Start of the last tournament in the data; later meetings are missing")
    note: str


class PlayerList(BaseModel):
    total: int = Field(description="Players matching the query before the limit")
    players: list[PlayerSummary]


# ---- helpers --------------------------------------------------------------------------------------------------------
def _summary(P, pid, row):
    gap = (P.art["snapshot_date"] - row["last_date"].date()).days
    return dict(id=pid, name=row["name"], rank=None if pd.isna(row["rank"]) else int(row["rank"]), elo=float(row["elo"]), matches=int(row["matches"]),
                last_event=row["last_date"].date(), days_since_last_event=gap, active=gap <= UNRELIABLE_DAYS, stale=gap > STALE_DAYS)


def _detail(P, profile, surface):
    art, pid = P.art, profile["id"]
    stats = raw_stats(art["state"], pid, art["snapshot_day"])
    return _summary(P, pid, art["players"].loc[pid]) | stats | dict(
        points=None if np.isnan(profile["points"]) else int(profile["points"]), surface=surface,
        surface_elo={s: art["surface_elo"].get((s, pid)) for s in SURFACES}, adj_form=float(profile["features"]["adj10"]),
        inputs={k: float(v) for k, v in profile["features"].items()},
        elo_history=[dict(date=_EPOCH + timedelta(days=d), elo=e) for d, e in art.get("elo_history", {}).get(pid, [])])


H2H_NOTE = ("Recorded meetings in the historical dataset only; meetings before the data starts and after the data cutoff are missing. Walkovers, defaults and abandoned "
            "matches are excluded; retirements count and are flagged. Head-to-head is not an input to any model and is not a forecast.")


def create_app(predictor: Predictor | None = None, artifact: Path = ARTIFACT, h2h: dict | None = None, h2h_artifact: Path = H2H_ARTIFACT,
               v2: V2Predictor | None = None, v2_artifact: Path = V2_ARTIFACT) -> FastAPI:
    """predictor=None loads the artifacts at startup; tests pass their own Predictor (and head-to-head index, V2 predictor)."""

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.predictor = predictor or (Predictor.load(artifact) if Path(artifact).exists() else None)  # loaded once
        app.state.h2h = h2h if h2h is not None else (pickle.loads(Path(h2h_artifact).read_bytes()) if Path(h2h_artifact).exists() else None)
        P = app.state.predictor
        app.state.v2, app.state.v2_reason = v2, V2_MISSING
        if v2 is None and P is not None and Path(v2_artifact).exists():
            try:
                app.state.v2 = V2Predictor.load(P, v2_artifact)
            except ValueError as e:  # built from a different snapshot than the V1 artifact: refuse rather than mix data
                app.state.v2_reason = f"V2 is unavailable: {e}."
        if P is not None:  # player list sorted once: active players first, then strongest first
            rows = [_summary(P, pid, row) for pid, row in P.art["players"].iterrows()]
            app.state.players = sorted(rows, key=lambda r: (not r["active"], -r["elo"]))
        yield

    app = FastAPI(title="Tennis Alpha API", version="1.0", lifespan=lifespan,
                  description="Hypothetical ATP match win probabilities from a frozen research model, as of a historical snapshot. " + DISCLAIMER)

    def model(request: Request) -> Predictor:
        if request.app.state.predictor is None:
            raise HTTPException(503, "The model artifact is missing. Build it with: python -m src.build_artifacts")
        return request.app.state.predictor

    def known(P: Predictor, pid: str) -> str:
        if pid not in P.art["players"].index:
            raise HTTPException(404, f"Unknown player id: {pid}")
        return pid

    @app.get("/health", response_model=Health, summary="Is the service up and the model loaded?")
    def health(request: Request):
        ok = request.app.state.predictor is not None
        return dict(status="ok" if ok else "unavailable", artifact_loaded=ok, v2_loaded=request.app.state.v2 is not None,
                    detail="ready" if ok else "Model artifact missing. Build it with: python -m src.build_artifacts")

    @app.get("/meta", response_model=Meta, summary="Data snapshot and model availability")
    def meta(request: Request, P: Predictor = Depends(model)):
        a = P.art
        models = [ModelInfo(key=k, name=f"Model {k}", role="primary" if k == "D" else "challenger", kind="Logistic Regression (frozen specification)",
                            n_features=len(m["features"])) for k, m in a["models"].items()]
        models.append(ModelInfo(key="Elo", name="Elo", role="benchmark", kind="Elo rating", n_features=1))
        if request.app.state.v2 is not None:
            v = request.app.state.v2.art
            models.append(ModelInfo(key="V2", name=f"Tennis Alpha V2 ({v['version']})", role="experimental", kind="Logistic Regression (frozen V2 specification)", n_features=len(v["cols"])))
        return dict(snapshot_date=a["snapshot_date"], data_start=a["data_start"], data_end=a["data_end"], n_matches=a["n_matches"],
                    n_players=len(a["players"]), n_active_players=sum(p["active"] for p in request.app.state.players),
                    n_training_matches=a["n_train"], source=a["source"], live=False, disclaimer=DISCLAIMER, models=models,
                    data_status=data_status())

    @app.get("/surfaces", response_model=list[Surface], summary="Court surfaces the models support")
    def surfaces():
        return list(SURFACES)

    @app.get("/players", response_model=PlayerList, summary="Players in the snapshot, optionally filtered by name")
    def players(request: Request, q: str = Query("", description="Case-insensitive part of the name"), active_only: bool = False,
                limit: int = Query(2000, ge=1, le=5000), P: Predictor = Depends(model)):
        words = q.lower().split()
        hits = [p for p in request.app.state.players if all(w in p["name"].lower() for w in words) and (p["active"] or not active_only)]
        return dict(total=len(hits), players=hits[:limit])

    @app.get("/players/{player_id}", response_model=PlayerDetail, summary="One player's observed statistics, model inputs and rating history")
    def player(player_id: str, surface: Surface = "Hard", P: Predictor = Depends(model)):
        return _detail(P, P._profile(known(P, player_id), surface), surface)

    @app.get("/predict", response_model=Prediction, summary="Model D, Model C, Elo and experimental V2 probabilities for a hypothetical match")
    def predict(request: Request, a: str = Query(description="Player A id"), b: str = Query(description="Player B id"), surface: Surface = "Hard",
                best_of: int = Query(3, description="Match format, 3 or 5; used by V2 only (V1 models do not use it)"), P: Predictor = Depends(model)):
        if a == b:
            raise HTTPException(400, "Choose two different players.")
        if best_of not in (3, 5):
            raise HTTPException(422, "best_of must be 3 or 5.")
        r = P.predict(known(P, a), known(P, b), surface)
        V = request.app.state.v2
        v2 = V.predict(a, b, surface, best_of, v1_result=r) if V is not None else dict(status="unavailable", reason=request.app.state.v2_reason)
        contributions = [dict(group=g, label=GROUP_LABELS.get(g, g), value=v) for g, v in r["contributions"]["D"].items()]
        return dict(surface=surface, snapshot_date=P.art["snapshot_date"], a=_detail(P, r["A"], surface), b=_detail(P, r["B"], surface),
                    p=r["p"], logit=sum(c["value"] for c in contributions), contributions=contributions, notes=r["notes"], reliable=r["reliable"], v2=v2)

    @app.get("/h2h", response_model=HeadToHead, summary="Recorded head-to-head between two players (history only, not a model input)")
    def head_to_head(request: Request, a: str = Query(description="Player A id"), b: str = Query(description="Player B id"), P: Predictor = Depends(model)):
        if a == b:
            raise HTTPException(400, "Choose two different players.")
        if request.app.state.h2h is None:
            raise HTTPException(503, "The head-to-head index is missing. Build it with: python -m src.h2h")
        return h2h_records.summary(request.app.state.h2h, known(P, a), known(P, b)) | dict(data_start=P.art["data_start"], data_end=P.art["data_end"], note=H2H_NOTE)

    return app


app = create_app()
