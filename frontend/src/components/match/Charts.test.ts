import { describe, expect, it } from "vitest";
import { eloRows, ratingAt } from "./Charts";

const day = (iso: string) => Date.parse(iso);
const A = [{ t: day("2025-01-01"), elo: 1500 }, { t: day("2025-03-01"), elo: 1540 }, { t: day("2025-06-01"), elo: 1520 }];
const B = [{ t: day("2025-04-01"), elo: 1500 }, { t: day("2025-05-01"), elo: 1480 }];

describe("Elo chart data", () => {
  it("holds a rating until the next recorded step and never interpolates", () => {
    expect(ratingAt(A, day("2025-02-15"))).toBe(1500); // between steps: still the earlier rating, not a blend
    expect(ratingAt(A, day("2025-03-01"))).toBe(1540); // the step applies from its own day
    expect(ratingAt(A, day("2026-01-01"))).toBe(1520);
  });

  it("gives no rating before a player's first match", () => {
    expect(ratingAt(B, day("2025-03-31"))).toBeUndefined();
    expect(ratingAt([], day("2025-03-31"))).toBeUndefined();
  });

  it("builds one row per day on which either rating changed, inside the chosen period", () => {
    const rows = eloRows(A, B, day("2025-02-01"), day("2025-07-01"));
    expect(rows.map((r) => r.t)).toEqual(["2025-02-01", "2025-03-01", "2025-04-01", "2025-05-01", "2025-06-01", "2025-07-01"].map(day));
    expect(rows.map((r) => r.a)).toEqual([1500, 1540, 1540, 1540, 1520, 1520]);
    expect(rows.map((r) => r.b)).toEqual([undefined, undefined, 1500, 1480, 1480, 1480]);
    expect(new Set(rows.flatMap((r) => [r.a, r.b]))).toEqual(new Set([undefined, 1500, 1540, 1520, 1480])); // only recorded values appear
  });
});
