// The single seam between the interface and the Python models: every number on the prediction and analytics pages
// comes through here from the FastAPI service (src/api.py), which wraps the same Predictor as the Streamlit dashboard.
import type { Contribution, HeadToHead, Matchup, Meta, ModelKey, Player, PlayerSummary, Surface, V2Player, V2Prediction, WinRecord } from "./types";

const BASE = "/api";
export const UNREACHABLE = "The prediction service is not reachable. Start it from the project folder with ./dev.sh";

export class ApiError extends Error {
  constructor(message: string, readonly status: number) {
    super(message);
    this.name = "ApiError";
  }
}

async function get<T>(path: string): Promise<T> {
  let res: Response;
  try {
    res = await fetch(`${BASE}${path}`, { headers: { accept: "application/json" } });
  } catch {
    throw new ApiError(UNREACHABLE, 0);
  }
  if (!res.ok) {
    let detail: unknown;
    try { detail = (await res.json())?.detail; } catch { /* not JSON: the proxy answered, not the API */ }
    if (typeof detail === "string") throw new ApiError(detail, res.status);
    throw new ApiError(res.status >= 500 ? UNREACHABLE : `The prediction service returned an error (${res.status}).`, res.status);
  }
  return (await res.json()) as T;
}

// ---- wire format (snake_case, as documented at http://127.0.0.1:8000/docs) ----
interface RawSummary {
  id: string; name: string; rank: number | null; elo: number; matches: number; last_event: string | null;
  days_since_last_event: number | null; active: boolean; stale: boolean;
}
interface RawPlayer extends RawSummary {
  points: number | null; surface_elo: Record<Surface, number | null>; last5: ("W" | "L")[];
  form: Record<"last10" | "d30" | "d90" | "d365", WinRecord>; surface_form: Record<Surface, WinRecord>; adj_form: number;
  elo_history: { date: string; elo: number }[];
}
interface RawV2Player {
  stat_matches: number; serve: number; ret: number; serve_surface: number; ret_surface: number; serve_recent: number; ret_recent: number;
  spw20: number; rpw20: number; hold20: number; break20: number; ace20: number; elo_trend: number;
  quality: { points_vs_expected20: number; luck10: number } | null;
}
interface RawV2 {
  status: "ok" | "unavailable"; reason: string | null; version?: string | null; best_of?: number | null; validation?: string | null; data_end?: string | null;
  p?: number | null; logit?: number | null; contributions?: Contribution[]; notes?: string[]; players?: { a: RawV2Player; b: RawV2Player } | null;
  matchup?: { pa: number; pb: number; tour_serve_avg: number; p_points_model: number } | null; h2h?: { residual: number; meetings: number } | null;
}
interface RawPrediction {
  surface: Surface; snapshot_date: string; a: RawPlayer; b: RawPlayer; p: Record<ModelKey, number>; logit: number;
  contributions: Contribution[]; notes: string[]; reliable: boolean; v2: RawV2;
}

const toV2Player = (r: RawV2Player): V2Player => ({
  statMatches: r.stat_matches, serve: r.serve, ret: r.ret, serveSurface: r.serve_surface, retSurface: r.ret_surface, serveRecent: r.serve_recent,
  retRecent: r.ret_recent, spw20: r.spw20, rpw20: r.rpw20, hold20: r.hold20, break20: r.break20, ace20: r.ace20, eloTrend: r.elo_trend,
  quality: r.quality ? { pointsVsExpected20: r.quality.points_vs_expected20, luck10: r.quality.luck10 } : null,
});

const toV2 = (r: RawV2): V2Prediction => ({
  status: r.status, reason: r.reason, version: r.version ?? null, bestOf: r.best_of ?? null, validation: r.validation ?? null, dataEnd: r.data_end ?? null,
  p: r.p ?? null, logit: r.logit ?? null, contributions: r.contributions ?? [], notes: r.notes ?? [],
  players: r.players ? { a: toV2Player(r.players.a), b: toV2Player(r.players.b) } : null,
  matchup: r.matchup ? { pa: r.matchup.pa, pb: r.matchup.pb, tourServeAvg: r.matchup.tour_serve_avg, pPointsModel: r.matchup.p_points_model } : null,
  h2h: r.h2h ?? null,
});
interface RawDataStatus {
  source: string; attribution: string; latest_tournament_start: string | null; latest_complete_through: string | null; pending: string[]; known_gaps?: string[];
  model_coefficients: string; last_success_utc: string | null; last_error: string | null; data_age_days: number | null; warnings: string[];
}
interface RawMeta {
  snapshot_date: string; data_start: string; data_end: string; n_matches: number; n_players: number; n_active_players: number;
  n_training_matches: number; source: string; disclaimer: string; data_status?: RawDataStatus | null;
}

const toSummary = (r: RawSummary): PlayerSummary => ({
  id: r.id, name: r.name, rank: r.rank, elo: r.elo, matches: r.matches, lastEvent: r.last_event,
  daysSinceLastEvent: r.days_since_last_event, active: r.active, stale: r.stale,
});

const toPlayer = (r: RawPlayer): Player => ({
  ...toSummary(r), points: r.points, surfaceElo: r.surface_elo, last5: r.last5, form: r.form, surfaceForm: r.surface_form,
  adjForm: r.adj_form, eloHistory: r.elo_history.map((h) => ({ t: Date.parse(h.date), elo: h.elo })),
});

// The player list and the snapshot metadata do not change while the service runs, so each is fetched once.
// A failed request is not cached: the next call tries again.
let players: Promise<PlayerSummary[]> | null = null;
let meta: Promise<Meta> | null = null;

export function fetchPlayers(): Promise<PlayerSummary[]> {
  players ??= get<{ players: RawSummary[] }>("/players").then((r) => r.players.map(toSummary)).catch((e) => { players = null; throw e; });
  return players;
}

export function fetchMeta(): Promise<Meta> {
  meta ??= get<RawMeta>("/meta").then((r) => ({
    snapshotDate: r.snapshot_date, dataStart: r.data_start, dataEnd: r.data_end, nMatches: r.n_matches, nPlayers: r.n_players,
    nActivePlayers: r.n_active_players, nTrainingMatches: r.n_training_matches, source: r.source, disclaimer: r.disclaimer,
    dataStatus: r.data_status ? {
      source: r.data_status.source, attribution: r.data_status.attribution, latestTournamentStart: r.data_status.latest_tournament_start,
      latestCompleteThrough: r.data_status.latest_complete_through, pending: r.data_status.pending, knownGaps: r.data_status.known_gaps ?? [], modelCoefficients: r.data_status.model_coefficients,
      lastSuccessUtc: r.data_status.last_success_utc, lastError: r.data_status.last_error, dataAgeDays: r.data_status.data_age_days, warnings: r.data_status.warnings,
    } : null,
  })).catch((e) => { meta = null; throw e; });
  return meta;
}

/** Real Model D, Model C, Elo and experimental V2 probabilities for a hypothetical match, with both players' statistics. */
export async function fetchMatchup(aId: string, bId: string, surface: Surface, bestOf: 3 | 5 = 3): Promise<Matchup> {
  const query = new URLSearchParams({ a: aId, b: bId, surface, best_of: String(bestOf) });
  const r = await get<RawPrediction>(`/predict?${query}`);
  return {
    a: toPlayer(r.a), b: toPlayer(r.b), surface: r.surface, snapshotDate: r.snapshot_date,
    prediction: { p: r.p, contributions: r.contributions, logit: r.logit, notes: r.notes, reliable: r.reliable, v2: toV2(r.v2) },
  };
}

interface RawH2HMatch { date: string; tournament: string; level: string; round: string; surface: string | null; score: string; winner_id: string; retired: boolean }
interface RawH2H {
  a_id: string; b_id: string; total: number; a_wins: number; b_wins: number; a_win_pct: number | null;
  by_surface: Record<Surface, { matches: number; a_wins: number; b_wins: number }>; other_surface: number; retirements: number;
  matches: RawH2HMatch[]; data_start: string; data_end: string;
}

/** Recorded meetings between two players. Separate from the prediction: head-to-head is not a model input. */
export async function fetchH2H(aId: string, bId: string): Promise<HeadToHead> {
  const r = await get<RawH2H>(`/h2h?${new URLSearchParams({ a: aId, b: bId })}`);
  const surface = (s: Surface) => ({ matches: r.by_surface[s].matches, aWins: r.by_surface[s].a_wins, bWins: r.by_surface[s].b_wins });
  return {
    aId: r.a_id, bId: r.b_id, total: r.total, aWins: r.a_wins, bWins: r.b_wins, aWinPct: r.a_win_pct,
    bySurface: { Hard: surface("Hard"), Clay: surface("Clay"), Grass: surface("Grass") }, otherSurface: r.other_surface, retirements: r.retirements,
    matches: r.matches.map((m) => ({ date: m.date, tournament: m.tournament, level: m.level, round: m.round, surface: m.surface, score: m.score, winnerId: m.winner_id, retired: m.retired })),
    dataStart: r.data_start, dataEnd: r.data_end,
  };
}

/** Test helper: forget cached responses. */
export function resetCache() {
  players = null;
  meta = null;
}
