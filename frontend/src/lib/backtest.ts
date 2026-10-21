// REAL RESULTS, copied from ../reports/phase6_backtest.md (independent evaluation, run once, models frozen beforehand).
export interface BacktestRow { model: string; role: string; logLoss: number; brier: number; accuracy: number; slope: number }

export const BACKTEST: { period: string; note: string; matches: number; rows: BacktestRow[] }[] = [
  { period: "2025", note: "Independent test, full season", matches: 2588, rows: [
    { model: "Model D", role: "Primary", logLoss: 0.6207, brier: 0.2166, accuracy: 0.651, slope: 0.918 },
    { model: "Model C", role: "Challenger", logLoss: 0.6203, brier: 0.2163, accuracy: 0.648, slope: 0.94 },
    { model: "Elo", role: "Benchmark", logLoss: 0.6291, brier: 0.2201, accuracy: 0.636, slope: 0.801 } ] },
  { period: "2026", note: "Forward test, January to Roland Garros only", matches: 1310, rows: [
    { model: "Model D", role: "Primary", logLoss: 0.6143, brier: 0.2135, accuracy: 0.662, slope: 0.947 },
    { model: "Model C", role: "Challenger", logLoss: 0.6128, brier: 0.2128, accuracy: 0.658, slope: 0.988 },
    { model: "Elo", role: "Benchmark", logLoss: 0.6317, brier: 0.2202, accuracy: 0.653, slope: 0.775 } ] },
];

export const DIFFS = [
  { period: "2025", comparison: "Model D minus Elo", tier: "Primary", diff: -0.0083, low: -0.0153, high: -0.002 },
  { period: "2025", comparison: "Model D minus Model C", tier: "Exploratory", diff: 0.0005, low: -0.0013, high: 0.0024 },
  { period: "2026", comparison: "Model D minus Elo", tier: "Primary", diff: -0.0174, low: -0.0289, high: -0.0046 },
  { period: "2026", comparison: "Model D minus Model C", tier: "Exploratory", diff: 0.0015, low: -0.0011, high: 0.0042 },
];

// 2025 reliability, exact values from the report: mean predicted probability (x) vs share of matches actually won (y), 10 bins.
export const CALIBRATION_2025 = {
  modelD: [[0.073, 0.1], [0.157, 0.147], [0.252, 0.277], [0.352, 0.388], [0.45, 0.429], [0.55, 0.541], [0.648, 0.648], [0.748, 0.711], [0.839, 0.808], [0.928, 0.976]],
  elo: [[0.065, 0.119], [0.155, 0.18], [0.253, 0.261], [0.352, 0.415], [0.452, 0.474], [0.55, 0.524], [0.646, 0.608], [0.745, 0.699], [0.844, 0.767], [0.942, 0.955]],
};
