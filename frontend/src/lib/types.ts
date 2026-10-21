// Data contract with the Python API (src/api.py). Field names are the API's, converted to camelCase in src/lib/api.ts.
export type Surface = "Hard" | "Clay" | "Grass";
export const SURFACES: Surface[] = ["Hard", "Clay", "Grass"];

/** Observed results in a window. rate is null when there are no matches: nothing is filled in. */
export interface WinRecord { wins: number; matches: number; rate: number | null }

export interface PlayerSummary {
  id: string; // stable identifier; names are for display only
  name: string;
  rank: number | null; // last ranking in the data, not today's
  elo: number;
  matches: number;
  lastEvent: string | null; // ISO date: start of the player's last tournament in the data
  daysSinceLastEvent: number | null; // counted to the snapshot date
  active: boolean;
  stale: boolean;
  // Only set when a properly licensed photograph exists; otherwise the interface shows a monogram.
  portrait?: { src: string; credit: string };
}

export interface Player extends PlayerSummary {
  points: number | null;
  surfaceElo: Record<Surface, number | null>; // null = no rated match on that surface
  last5: ("W" | "L")[]; // most recent last
  form: Record<"last10" | "d30" | "d90" | "d365", WinRecord>;
  surfaceForm: Record<Surface, WinRecord>; // last 365 days
  adjForm: number; // Model D input: shrunk mean of (result - Elo expectation), last 10 matches
  eloHistory: { t: number; elo: number }[]; // real rating path; a rating holds until the next point
}

export type ModelKey = "D" | "C" | "Elo";

export interface Contribution {
  group: string;
  label: string;
  value: number; // log-odds pushed toward player A (negative = toward player B)
}

/** Model shown in the hero: the validated baseline or the experimental V2. */
export type HeroModel = "D" | "V2";

/** Tennis Alpha V2 inputs for one player. Ratings are serve/return points won above the tour average (0.01 = 1 point in 100). */
export interface V2Player {
  statMatches: number; serve: number; ret: number; serveSurface: number; retSurface: number; serveRecent: number; retRecent: number;
  spw20: number; rpw20: number; hold20: number; break20: number; ace20: number; eloTrend: number;
  quality: { pointsVsExpected20: number; luck10: number } | null; // descriptive (V2.1 research), not a V2 input
}

/** Experimental V2 output, separate from the V1 numbers. p is null when V2 is unavailable (reason says why). */
export interface V2Prediction {
  status: "ok" | "unavailable"; reason: string | null; version: string | null; bestOf: number | null; validation: string | null;
  dataEnd: string | null; p: number | null; logit: number | null; contributions: Contribution[]; notes: string[];
  players: { a: V2Player; b: V2Player } | null;
  matchup: { pa: number; pb: number; tourServeAvg: number; pPointsModel: number } | null;
  h2h: { residual: number; meetings: number } | null;
}

export interface Prediction {
  p: Record<ModelKey, number>; // probability that player A wins
  contributions: Contribution[]; // Model D; they sum to logit
  logit: number;
  notes: string[];
  reliable: boolean;
  v2: V2Prediction;
}

export interface Matchup { a: Player; b: Player; surface: Surface; snapshotDate: string; prediction: Prediction }

/** Local data freshness (written by the update pipeline). The data date is NOT the model training cutoff. */
export interface DataStatus {
  source: string;
  attribution: string;
  latestTournamentStart: string | null;
  latestCompleteThrough: string | null; // every included tournament must have finished by this date
  pending: string[];
  knownGaps: string[]; // tour events the source lacks for the covered weeks (compared with a year earlier)
  modelCoefficients: string;
  lastSuccessUtc: string | null;
  lastError: string | null;
  dataAgeDays: number | null;
  warnings: string[];
}

export interface Meta {
  snapshotDate: string;
  dataStart: string;
  dataEnd: string;
  nMatches: number;
  nPlayers: number;
  nActivePlayers: number;
  nTrainingMatches: number;
  source: string;
  disclaimer: string;
  dataStatus: DataStatus | null;
}

/** One recorded meeting. `date` is the tournament START date: the data has no match dates. */
export interface H2HMatch {
  date: string; tournament: string; level: string; round: string; surface: string | null; score: string; winnerId: string; retired: boolean;
}

/** Recorded head-to-head between player A and player B. History only: not a model input and not a forecast. */
export interface HeadToHead {
  aId: string;
  bId: string;
  total: number;
  aWins: number;
  bWins: number;
  aWinPct: number | null; // null when the players have never met in the data
  bySurface: Record<Surface, { matches: number; aWins: number; bWins: number }>;
  otherSurface: number; // carpet, or surface not recorded
  retirements: number;
  matches: H2HMatch[]; // most recent first
  dataStart: string;
  dataEnd: string;
}
