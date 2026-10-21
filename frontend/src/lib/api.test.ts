import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { ALCARAZ, SINNER, json, rawMeta, rawPrediction, rawSummaries } from "../test/fixtures";
import { ApiError, UNREACHABLE, fetchMatchup, fetchMeta, fetchPlayers, resetCache } from "./api";

const fetchMock = vi.fn();

beforeEach(() => { resetCache(); fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });
afterEach(() => vi.unstubAllGlobals());

describe("fetchMatchup", () => {
  it("asks the Python API for the selected players by id and surface", async () => {
    fetchMock.mockReturnValue(json(rawPrediction()));
    await fetchMatchup(SINNER, ALCARAZ, "Clay");
    expect(fetchMock).toHaveBeenCalledOnce();
    const url = new URL(fetchMock.mock.calls[0][0], "http://localhost");
    expect(url.pathname).toBe("/api/predict");
    expect(Object.fromEntries(url.searchParams)).toEqual({ a: SINNER, b: ALCARAZ, surface: "Clay", best_of: "3" });
  });

  it("passes the model's probabilities through unchanged", async () => {
    fetchMock.mockReturnValue(json(rawPrediction()));
    const m = await fetchMatchup(SINNER, ALCARAZ, "Hard");
    expect(m.prediction.p).toEqual({ D: 0.6878366326951671, C: 0.6765118282730975, Elo: 0.5997956664305136 });
    expect(m.prediction.logit).toBe(0.79);
    expect(m.prediction.contributions).toHaveLength(2);
    expect(m.snapshotDate).toBe("2026-06-08");
  });

  it("maps player statistics without inventing missing values", async () => {
    fetchMock.mockReturnValue(json(rawPrediction()));
    const { a, b } = await fetchMatchup(SINNER, ALCARAZ, "Hard");
    expect(a).toMatchObject({ id: SINNER, name: "Jannik Sinner", rank: 1, daysSinceLastEvent: 14, active: true, adjForm: -0.026 });
    expect(a.surfaceElo.Grass).toBeNull(); // unrated stays unrated
    expect(a.surfaceForm.Grass).toEqual({ wins: 0, matches: 0, rate: null }); // no matches stays "no matches"
    expect(a.form.d90).toEqual({ wins: 24, matches: 25, rate: 0.96 });
    expect(a.eloHistory).toEqual([{ t: Date.parse("2019-04-22"), elo: 1500 }, { t: Date.parse("2026-06-08"), elo: 2189.3 }]);
    expect(b.daysSinceLastEvent).toBe(56);
  });

  it("surfaces the API's own error message", async () => {
    fetchMock.mockReturnValue(json({ detail: "Unknown player id: nobody" }, 404));
    await expect(fetchMatchup("nobody", ALCARAZ, "Hard")).rejects.toMatchObject({ name: "ApiError", status: 404, message: "Unknown player id: nobody" });
  });

  it("explains how to start the service when it is down", async () => {
    fetchMock.mockReturnValue(Promise.resolve(new Response("Internal Server Error", { status: 500 }))); // what the Next proxy returns
    await expect(fetchMatchup(SINNER, ALCARAZ, "Hard")).rejects.toMatchObject({ message: UNREACHABLE, status: 500 });
    fetchMock.mockReturnValue(Promise.reject(new TypeError("Failed to fetch")));
    const error = await fetchMatchup(SINNER, ALCARAZ, "Hard").catch((e) => e);
    expect(error).toBeInstanceOf(ApiError);
    expect(error).toMatchObject({ message: UNREACHABLE, status: 0 });
  });
});

describe("fetchPlayers and fetchMeta", () => {
  it("loads the player list once and keeps ids as the identifier", async () => {
    fetchMock.mockReturnValue(json({ total: 3, players: rawSummaries }));
    const [first, second] = await Promise.all([fetchPlayers(), fetchPlayers()]);
    expect(fetchMock).toHaveBeenCalledOnce();
    expect(second).toBe(first);
    expect(first.map((p) => p.id)).toEqual([SINNER, ALCARAZ, "sackmann:103819"]);
    expect(first[2]).toMatchObject({ name: "Roger Federer", active: false, stale: true, daysSinceLastEvent: 1806 });
  });

  it("does not cache a failure", async () => {
    fetchMock.mockReturnValueOnce(Promise.reject(new TypeError("down"))).mockReturnValue(json({ total: 3, players: rawSummaries }));
    await expect(fetchPlayers()).rejects.toBeInstanceOf(ApiError);
    await expect(fetchPlayers()).resolves.toHaveLength(3);
  });

  it("reads the snapshot dates", async () => {
    fetchMock.mockReturnValue(json(rawMeta));
    expect(await fetchMeta()).toMatchObject({ snapshotDate: "2026-06-08", dataEnd: "2026-05-25", nPlayers: 1788, nActivePlayers: 445 });
  });
});
