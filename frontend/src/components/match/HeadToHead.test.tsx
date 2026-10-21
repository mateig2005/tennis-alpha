import { fireEvent, render, screen, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/lib/api";
import { ALCARAZ, SINNER, json, rawH2H, rawNoMeetings } from "@/test/fixtures";
import type { PlayerSummary } from "@/lib/types";
import { HeadToHead } from "./HeadToHead";

const player = (id: string, name: string): PlayerSummary => ({ id, name, rank: 1, elo: 2100, matches: 400, lastEvent: "2026-05-25", daysSinceLastEvent: 14, active: true, stale: false });
const sinner = player(SINNER, "Jannik Sinner");
const alcaraz = player(ALCARAZ, "Carlos Alcaraz");
const fetchMock = vi.fn();

beforeEach(() => { api.resetCache(); fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });

describe("fetchH2H", () => {
  it("asks for the pair by id and maps the record", async () => {
    fetchMock.mockReturnValue(json(rawH2H()));
    const h = await api.fetchH2H(SINNER, ALCARAZ);
    const url = new URL(fetchMock.mock.calls[0][0], "http://localhost");
    expect(url.pathname).toBe("/api/h2h");
    expect(Object.fromEntries(url.searchParams)).toEqual({ a: SINNER, b: ALCARAZ });
    expect(h).toMatchObject({ total: 7, aWins: 3, bWins: 4, otherSurface: 1, retirements: 1, dataStart: "2010-01-03", dataEnd: "2026-05-25" });
    expect(h.bySurface.Hard).toEqual({ matches: 3, aWins: 1, bWins: 2 });
    expect(h.matches[0]).toEqual({ date: "2026-04-05", tournament: "Monte Carlo Masters", level: "Masters 1000", round: "F", surface: "Clay", score: "7-6(5) 6-3", winnerId: SINNER, retired: false });
    expect(h.matches.at(-1)!.surface).toBeNull(); // an unrecorded surface stays unrecorded
  });
});

describe("Head-to-head section", () => {
  it("shows the central score, who leads and the last meeting", async () => {
    fetchMock.mockReturnValue(json(rawH2H()));
    render(<HeadToHead a={sinner} b={alcaraz} surface="Hard" />);
    const score = await screen.findByLabelText("Jannik Sinner 3, Carlos Alcaraz 4");
    expect(score.textContent).toBe("3–4");
    const summary = screen.getByText(/recorded meetings/).textContent!;
    expect(summary).toContain("7 recorded meetings");
    expect(summary).toContain("Carlos Alcaraz leads, winning 57%");
    expect(summary).toContain("last met in 2026 (Monte Carlo Masters), won by Jannik Sinner");
  });

  it("shows the record on each surface and marks the selected one", async () => {
    fetchMock.mockReturnValue(json(rawH2H()));
    render(<HeadToHead a={sinner} b={alcaraz} surface="Grass" />);
    await screen.findByLabelText("Jannik Sinner 3, Carlos Alcaraz 4");
    expect(screen.getByRole("img", { name: "On Hard: Jannik Sinner 1, Carlos Alcaraz 2" })).toBeTruthy();
    expect(screen.getByRole("img", { name: "On Clay: Jannik Sinner 1, Carlos Alcaraz 1" })).toBeTruthy();
    const grass = screen.getByRole("img", { name: "On Grass: Jannik Sinner 1, Carlos Alcaraz 0" }).closest("div.rounded-xl")!;
    expect(grass.textContent).toContain("selected");
    expect(grass.textContent).toContain("1 meeting");
    expect(screen.getByText("1 more on carpet or with no surface recorded.")).toBeTruthy();
  });

  it("lists the last five meetings oldest to newest and flags the retirement", async () => {
    fetchMock.mockReturnValue(json(rawH2H()));
    render(<HeadToHead a={sinner} b={alcaraz} surface="Hard" />);
    await screen.findByLabelText("Jannik Sinner 3, Carlos Alcaraz 4");
    const items = within(screen.getByRole("list", { name: "" })).getAllByRole("listitem");
    expect(items).toHaveLength(5);
    expect(items.map((li) => li.textContent!.slice(0, 4))).toEqual(["2025", "2025", "2025", "2025", "2026"]);
    expect(items[0].textContent).toContain("Wimbledon");
    expect(items[4].textContent).toContain("Monte Carlo Masters");
    expect(items[1].textContent).toContain("5-0"); // Cincinnati: the recorded score, with the retirement shown once as a badge
    expect(within(items[1]).getByTitle(/ended by retirement/).textContent).toBe("RET");
    expect(items[1].textContent!.match(/RET/g)).toHaveLength(1);
    expect(items[0].textContent).not.toContain("RET");
  });

  it("expands to every recorded meeting, most recent first", async () => {
    fetchMock.mockReturnValue(json(rawH2H()));
    render(<HeadToHead a={sinner} b={alcaraz} surface="Hard" />);
    const toggle = await screen.findByRole("button", { name: /Show all 7 meetings/ });
    expect(toggle.getAttribute("aria-expanded")).toBe("false");
    expect(screen.queryByRole("list", { name: /All recorded meetings/ })).toBeNull();
    fireEvent.click(toggle);
    expect(toggle.getAttribute("aria-expanded")).toBe("true");
    const rows = within(screen.getByRole("list", { name: "All recorded meetings, most recent first" })).getAllByRole("listitem");
    expect(rows).toHaveLength(7);
    expect(rows[0].textContent).toContain("2026"); expect(rows[0].textContent).toContain("Final");
    expect(rows[6].textContent).toContain("Davis Cup"); expect(rows[6].textContent).toContain("Round robin");
    expect(rows[6].textContent).toContain("n/a"); // no surface recorded: said so, not guessed
  });

  it("mirrors the record when the players are swapped", async () => {
    fetchMock.mockReturnValue(json(rawH2H({ a_id: ALCARAZ, b_id: SINNER, a_wins: 4, b_wins: 3, a_win_pct: 4 / 7 })));
    render(<HeadToHead a={alcaraz} b={sinner} surface="Hard" />);
    expect((await screen.findByLabelText("Carlos Alcaraz 4, Jannik Sinner 3")).textContent).toBe("4–3");
    expect(screen.getByText(/recorded meetings/).textContent).toContain("Carlos Alcaraz leads, winning 57%");
  });

  it("says so plainly when the players have never met", async () => {
    fetchMock.mockReturnValue(json(rawNoMeetings()));
    render(<HeadToHead a={sinner} b={alcaraz} surface="Hard" />);
    expect(await screen.findByText("No recorded meetings")).toBeTruthy();
    expect(screen.getByText("Jannik Sinner and Carlos Alcaraz have not played each other in the data.")).toBeTruthy();
    expect(screen.queryByRole("button", { name: /Show all/ })).toBeNull();
    expect(screen.queryByText(/%/)).toBeNull(); // no percentage is invented
  });

  it("states the data period and that the record is not a model input or a forecast", async () => {
    fetchMock.mockReturnValue(json(rawH2H()));
    render(<HeadToHead a={sinner} b={alcaraz} surface="Hard" />);
    const note = (await screen.findByText(/Recorded results from the historical dataset/)).textContent!;
    for (const part of ["3 Jan 2010", "25 May 2026", "before or after that period", "no match dates", "Walkovers, defaults and abandoned matches are excluded",
      "not an input to Model D, Model C or Elo", "not a forecast"]) expect(note).toContain(part);
  });

  it("shows a loading state, then an error if the service fails", async () => {
    fetchMock.mockReturnValue(json({ detail: "The head-to-head index is missing. Build it with: python -m src.h2h" }, 503));
    const { container } = render(<HeadToHead a={sinner} b={alcaraz} surface="Hard" />);
    expect(container.querySelector('[aria-busy="true"]')).toBeTruthy();
    expect((await screen.findByRole("alert")).textContent).toContain("head-to-head index is missing");
  });
});
