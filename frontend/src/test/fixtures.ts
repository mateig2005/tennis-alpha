// Test fixtures in the API's wire format (snake_case), using real ids and the real Sinner v Alcaraz hard-court output.
export const SINNER = "sackmann:206173";
export const ALCARAZ = "sackmann:207989";

const record = (wins: number, matches: number) => ({ wins, matches, rate: matches ? wins / matches : null });

export const rawPlayer = (id: string, name: string, rank: number, elo: number, days = 14) => ({
  id, name, rank, elo, matches: 447, last_event: "2026-05-25", days_since_last_event: days, active: days <= 365, stale: days > 180,
  points: 14750, surface: "Hard", surface_elo: { Hard: elo + 4, Clay: elo - 196, Grass: null },
  last5: ["W", "W", "W", "W", "L"], form: { last10: record(9, 10), d30: record(1, 2), d90: record(24, 25), d365: record(77, 84) },
  surface_form: { Hard: record(51, 56), Clay: record(18, 19), Grass: record(0, 0) }, adj_form: -0.026, inputs: { wr10: 0.8077 },
  elo_history: [{ date: "2019-04-22", elo: 1500 }, { date: "2026-06-08", elo }],
});

const v2Player = (serve: number, ret: number, stat_matches: number) => ({
  stat_matches, serve, ret, serve_surface: serve, ret_surface: ret, serve_recent: serve, ret_recent: ret, spw20: 0.713, rpw20: 0.428,
  hold20: 0.925, break20: 0.334, ace20: 0.1, elo_trend: 0.068, quality: { points_vs_expected20: 0.022, luck10: 0.021 },
});

/** Experimental V2 block as the API sends it (real V2 output for Sinner v Alcaraz on hard, best of 3). */
export const rawV2 = (overrides: Record<string, unknown> = {}) => ({
  status: "ok", reason: null, version: "v2.0", best_of: 3, data_end: "2026-05-25",
  validation: "Experimental. Better than Model D on development folds 2023-2024 (held out from V2 choices but not independent); no independent prospective test yet.",
  p: 0.7321, logit: 1.005, notes: [],
  contributions: [{ group: "momentum", label: "Recent serve/return & Elo trend", value: 0.39 }, { group: "matchup", label: "Serve-vs-return matchup", value: 0.3 }],
  players: { a: v2Player(0.101, 0.08, 442), b: v2Player(0.063, 0.081, 368) },
  matchup: { pa: 0.672, pb: 0.633, tour_serve_avg: 0.641, p_points_model: 0.634 }, h2h: { residual: -0.098, meetings: 17 }, ...overrides,
});

export const rawV2Unavailable = () => ({ status: "unavailable", reason: "No match with serve/return statistics in the data for Roger Federer: V2 needs them and does not invent them." });

export const rawPrediction = (overrides: Record<string, unknown> = {}) => ({
  surface: "Hard", snapshot_date: "2026-06-08",
  a: rawPlayer(SINNER, "Jannik Sinner", 1, 2189.3), b: rawPlayer(ALCARAZ, "Carlos Alcaraz", 2, 2119.0, 56),
  p: { D: 0.6878366326951671, C: 0.6765118282730975, Elo: 0.5997956664305136 }, logit: 0.79,
  contributions: [{ group: "elo", label: "Elo ratings", value: 0.52 }, { group: "rank", label: "Ranking", value: 0.27 }],
  notes: [], reliable: true, v2: rawV2(), ...overrides,
});

export const rawSummaries = [
  { id: SINNER, name: "Jannik Sinner", rank: 1, elo: 2189.3, matches: 447, last_event: "2026-05-25", days_since_last_event: 14, active: true, stale: false },
  { id: ALCARAZ, name: "Carlos Alcaraz", rank: 2, elo: 2119.0, matches: 370, last_event: "2026-04-13", days_since_last_event: 56, active: true, stale: false },
  { id: "sackmann:103819", name: "Roger Federer", rank: 8, elo: 1978.0, matches: 687, last_event: "2021-06-28", days_since_last_event: 1806, active: false, stale: true },
];

export const rawMeta = {
  snapshot_date: "2026-06-08", data_start: "2010-01-03", data_end: "2026-05-25", n_matches: 46648, n_players: 1788, n_active_players: 445,
  n_training_matches: 29271, source: "Jeff Sackmann / Tennis Abstract (CC BY-NC-SA 4.0)", disclaimer: "Historical snapshot, not a live forecast.",
  data_status: {
    source: "Sackmann to 2026-05-25; TennisMyLife after", attribution: "Match data: Jeff Sackmann / Tennis Abstract; TennisMyLife.",
    latest_tournament_start: "2026-09-30", latest_complete_through: "2026-10-07", pending: ["Shanghai"], model_coefficients: "frozen",
    last_success_utc: "2026-10-20T15:39:09+00:00", last_error: null, data_age_days: 13, warnings: [] as string[],
  },
};

export const json = (body: unknown, status = 200) =>
  Promise.resolve(new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } }));

const meeting = (date: string, tournament: string, round: string, surface: string | null, winner: string, score = "6-4 6-4", retired = false, level = "Tour") =>
  ({ date, tournament, level, round, surface, score, winner_id: winner, retired });

/** Seven recorded meetings, most recent first: Sinner 3, Alcaraz 4; one retirement; one with no surface recorded. */
export const rawH2H = (overrides: Record<string, unknown> = {}) => {
  const matches = [
    meeting("2026-04-05", "Monte Carlo Masters", "F", "Clay", SINNER, "7-6(5) 6-3", false, "Masters 1000"),
    meeting("2025-11-09", "Tour Finals", "F", "Hard", SINNER, "7-6(4) 7-5", false, "Tour Finals"),
    meeting("2025-08-25", "Us Open", "F", "Hard", ALCARAZ, "6-2 3-6 6-1 6-4", false, "Grand Slam"),
    meeting("2025-08-07", "Cincinnati Masters", "F", "Hard", ALCARAZ, "5-0 RET", true, "Masters 1000"),
    meeting("2025-06-30", "Wimbledon", "F", "Grass", SINNER, "4-6 6-4 6-4 6-4", false, "Grand Slam"),
    meeting("2025-05-25", "Roland Garros", "F", "Clay", ALCARAZ, "4-6 6-7(4) 6-4 7-6(3) 7-6(2)", false, "Grand Slam"),
    meeting("2021-11-01", "Davis Cup", "RR", null, ALCARAZ, "7-6(1) 7-5", false, "Davis Cup"),
  ];
  return {
    a_id: SINNER, b_id: ALCARAZ, total: 7, a_wins: 3, b_wins: 4, a_win_pct: 3 / 7,
    by_surface: { Hard: { matches: 3, a_wins: 1, b_wins: 2 }, Clay: { matches: 2, a_wins: 1, b_wins: 1 }, Grass: { matches: 1, a_wins: 1, b_wins: 0 } },
    other_surface: 1, retirements: 1, last_meeting: matches[0], last5: matches.slice(0, 5), matches,
    data_start: "2010-01-03", data_end: "2026-05-25", note: "Recorded meetings only.", ...overrides,
  };
};

export const rawNoMeetings = () => rawH2H({
  total: 0, a_wins: 0, b_wins: 0, a_win_pct: null, other_surface: 0, retirements: 0, last_meeting: null, last5: [], matches: [],
  by_surface: { Hard: { matches: 0, a_wins: 0, b_wins: 0 }, Clay: { matches: 0, a_wins: 0, b_wins: 0 }, Grass: { matches: 0, a_wins: 0, b_wins: 0 } },
});
