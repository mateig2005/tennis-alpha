/** Model-implied fair decimal odds (no bookmaker margin): 1 / probability. Not market prices. */
export function fairOdds(p: number | null | undefined): number | null {
  return p != null && p > 0 && p < 1 ? 1 / p : null;
}

export const fmtOdds = (p: number | null | undefined) => {
  const o = fairOdds(p);
  return o === null ? "—" : o.toFixed(2);
};
