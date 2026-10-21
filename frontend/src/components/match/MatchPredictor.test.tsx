import { fireEvent, render, screen, waitFor, within } from "@testing-library/react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import * as api from "@/lib/api";
import { ALCARAZ, SINNER, json, rawH2H, rawMeta, rawPrediction, rawSummaries, rawV2Unavailable } from "@/test/fixtures";
import { MatchPredictor } from "./MatchPredictor";

// Recharts needs a real layout engine; the charts are replaced by markers so these tests cover data flow and page states.
vi.mock("./Charts", () => ({
  ContributionChart: () => <div data-testid="contribution-chart" />, EloChart: () => <div data-testid="elo-chart" />,
  FormChart: () => <div data-testid="form-chart" />, SurfaceChart: () => <div data-testid="surface-chart" />,
}));

const fetchMock = vi.fn();
/** Routes requests like the real service; `predict` decides what /predict answers. */
function serve(predict: (url: URL) => Promise<Response>, { players = true } = {}) {
  fetchMock.mockImplementation((input: string) => {
    const url = new URL(input, "http://localhost");
    if (url.pathname === "/api/players") return players ? json({ total: rawSummaries.length, players: rawSummaries }) : Promise.reject(new TypeError("down"));
    if (url.pathname === "/api/meta") return players ? json(rawMeta) : Promise.reject(new TypeError("down"));
    if (url.pathname === "/api/h2h") return json(rawH2H()); // a separate request: head-to-head never feeds the prediction
    return predict(url);
  });
}
const percent = (text: string) => (_: string, el: Element | null) => el?.tagName === "P" && el.textContent === text;
const hero = () => screen.getByRole("region", { name: "Estimated probability of winning" });

beforeEach(() => { api.resetCache(); fetchMock.mockReset(); vi.stubGlobal("fetch", fetchMock); });

describe("Match Prediction page", () => {
  it("shows a loading state, then the real model output", async () => {
    let release!: (r: Response) => void;
    serve(() => new Promise<Response>((r) => { release = r; }));
    render(<MatchPredictor />);

    await within(hero()).findByText("Jannik Sinner"); // names arrive with the player list, before the prediction
    expect(hero().getAttribute("aria-busy")).toBe("true");
    expect(screen.queryByText(percent("68.8%"))).toBeNull();

    release(await json(rawPrediction()));
    await within(hero()).findByText(percent("68.8%"));
    expect(within(hero()).getByText(percent("31.2%"))).toBeTruthy(); // the two sides sum to 100%
    expect(hero().getAttribute("aria-busy")).toBe("false");
    expect(within(hero()).getByText("Favourite")).toBeTruthy();
    expect(within(hero()).getByText("Underdog")).toBeTruthy();
    const details = screen.getByRole("region", { name: "How to read the prediction" });
    for (const value of ["67.7%", "32.3%", "60.0%", "40.0%"]) expect(within(details).getByText(value)).toBeTruthy(); // Model C and Elo
    expect(screen.getAllByTestId("contribution-chart")).toHaveLength(2); // Model D and V2
    expect(await screen.findByText("Previous meetings")).toBeTruthy(); // the head-to-head section is on the page
    expect(await screen.findByLabelText("Jannik Sinner 3, Carlos Alcaraz 4")).toBeTruthy();
    expect(screen.getAllByText(/sum of bars/).map((e) => e.textContent).join(" ")).toContain("+0.790"); // Model D; V2 has its own line
  });

  it("requests the stored selection by player id", async () => {
    window.localStorage.setItem("tennis-alpha:selection:v2", JSON.stringify({ aId: ALCARAZ, bId: SINNER, surface: "Grass" }));
    serve(() => json(rawPrediction({ surface: "Grass" })));
    render(<MatchPredictor />);
    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u).includes("/api/predict"))).toBe(true));
    const url = new URL(fetchMock.mock.calls.map(([u]) => String(u)).find((u) => u.includes("/api/predict"))!, "http://localhost");
    expect(Object.fromEntries(url.searchParams)).toEqual({ a: ALCARAZ, b: SINNER, surface: "Grass", best_of: "3" });
  });

  it("always says the data is a historical snapshot, not a live forecast", async () => {
    serve(() => json(rawPrediction()));
    render(<MatchPredictor />);
    expect(screen.getByText("Historical snapshot, not a live forecast.")).toBeTruthy(); // before any data has loaded
    await screen.findAllByText(/tournaments starting 25 May 2026/);
    expect(screen.getByText(/hypothetical match on 8 Jun 2026/)).toBeTruthy();
    expect(await screen.findByText("Model output · snapshot 8 Jun 2026")).toBeTruthy();
    expect(screen.queryByText(/illustrative|fictional/i)).toBeNull();
  });

  it("shows how fresh the local data is, separately from the frozen model", async () => {
    serve(() => json(rawPrediction()));
    render(<MatchPredictor />);
    const line = await screen.findByLabelText("Data status");
    expect(line.textContent).toContain("Results included through 7 Oct 2026 (13 days old)");
    expect(line.textContent).not.toContain("missing from the source");
    expect(line.textContent).toContain("last update 20 Oct 2026");
    expect(line.textContent).toContain("TennisMyLife");
    expect(line.textContent).toContain("model coefficients frozen");
    expect(line.textContent).toContain("Shanghai");
  });

  it("warns when the local data is stale", async () => {
    fetchMock.mockImplementation((input: string) => {
      const url = new URL(input, "http://localhost");
      if (url.pathname === "/api/meta") return json({ ...rawMeta, data_status: { ...rawMeta.data_status, data_age_days: 45, warnings: ["The newest results are 45 days old; recent matches are missing from player form and ratings."] } });
      if (url.pathname === "/api/players") return json({ total: rawSummaries.length, players: rawSummaries });
      if (url.pathname === "/api/h2h") return json(rawH2H());
      return json(rawPrediction());
    });
    render(<MatchPredictor />);
    expect((await screen.findByText(/45 days old; recent matches are missing/)).getAttribute("role")).toBe("status");
  });

  it("names tournaments missing from the source instead of implying full coverage", async () => {
    fetchMock.mockImplementation((input: string) => {
      const url = new URL(input, "http://localhost");
      if (url.pathname === "/api/meta") return json({ ...rawMeta, data_status: { ...rawMeta.data_status, known_gaps: ["Hamburg", "Gstaad (short draw)"] } });
      if (url.pathname === "/api/players") return json({ total: rawSummaries.length, players: rawSummaries });
      if (url.pathname === "/api/h2h") return json(rawH2H());
      return json(rawPrediction());
    });
    render(<MatchPredictor />);
    expect((await screen.findByLabelText("Data status")).textContent).toContain("missing from the source: Hamburg, Gstaad (short draw)");
  });

  it("marks an unreliable prediction and shows the reasons", async () => {
    const note = "Roger Federer: last match in the data started 2021-06-28, 1806 days before the snapshot.";
    serve(() => json(rawPrediction({ reliable: false, notes: [note] })));
    render(<MatchPredictor />);
    expect(await screen.findByText(/Not a reliable estimate/)).toBeTruthy();
    expect(screen.getByText("This prediction is not reliable")).toBeTruthy();
    expect(screen.getByText(note)).toBeTruthy();
    expect(screen.getByText("Not reliable")).toBeTruthy();
  });

  it("shows the API's message when a prediction fails", async () => {
    serve(() => json({ detail: "The model artifact is missing. Build it with: python -m src.build_artifacts" }, 503));
    render(<MatchPredictor />);
    expect((await screen.findByRole("alert")).textContent).toContain("The model artifact is missing");
    expect(screen.queryByText(percent("68.8%"))).toBeNull(); // no number is shown without a real prediction
  });

  it("explains how to start the service when it is unreachable", async () => {
    serve(() => Promise.reject(new TypeError("down")), { players: false });
    render(<MatchPredictor />);
    expect((await screen.findByRole("alert")).textContent).toContain(api.UNREACHABLE);
    expect((screen.getAllByPlaceholderText("Players unavailable")[0] as HTMLInputElement).disabled).toBe(true);
    expect(hero().getAttribute("aria-busy")).toBe("false"); // failed is not the same as loading
  });

  it("falls back to the default matchup when a stored player no longer exists", async () => {
    window.localStorage.setItem("tennis-alpha:selection:v2", JSON.stringify({ aId: "p1", bId: "p2", surface: "Hard" })); // ids from the old prototype
    serve((url) => (url.searchParams.get("a") === "p1" ? json({ detail: "Unknown player id: p1" }, 404) : json(rawPrediction())));
    render(<MatchPredictor />);
    await within(hero()).findByText(percent("68.8%"));
    expect(JSON.parse(window.localStorage.getItem("tennis-alpha:selection:v2")!)).toEqual({ aId: SINNER, bId: ALCARAZ, surface: "Hard", bestOf: 3 });
  });

  it("compares V2 with the other models, with fair odds, and switches the headline model", async () => {
    serve(() => json(rawPrediction()));
    render(<MatchPredictor />);
    await within(hero()).findByText(percent("68.8%"));
    expect(within(hero()).getByText("Model D · validated baseline")).toBeTruthy();
    expect(within(hero()).getByText("1.45")).toBeTruthy(); // fair odds = 1 / 0.688
    const details = screen.getByRole("region", { name: "How to read the prediction" });
    for (const value of ["73.2%", "26.8%"]) expect(within(details).getByText(value)).toBeTruthy(); // V2's own number
    expect(within(details).getByText("Experimental")).toBeTruthy();
    expect(within(details).getByLabelText("Tennis Alpha V2 fair odds: Jannik Sinner 1.37, Carlos Alcaraz 3.73")).toBeTruthy();

    fireEvent.click(within(hero()).getByRole("radio", { name: "V2 · experimental" }));
    await within(hero()).findByText(percent("73.2%"));
    expect(within(hero()).getByText("Tennis Alpha V2 · experimental")).toBeTruthy();
    expect(within(details).getByText("Validated baseline")).toBeTruthy(); // Model D moves into the comparison row
    expect(within(details).getByText("68.8%")).toBeTruthy();

    const panel = screen.getByRole("region", { name: "Advanced matchup intelligence" });
    expect(within(panel).getAllByText("+10.1").length).toBeGreaterThan(0); // Sinner's serve rating, points per 100
    expect(within(panel).getByText(/no independent prospective test yet/)).toBeTruthy();
    expect(within(panel).getByText(/descriptive, not a V2 input/)).toBeTruthy();
  });

  it("sends the match format for V2", async () => {
    serve(() => json(rawPrediction()));
    render(<MatchPredictor />);
    const panel = await screen.findByRole("region", { name: "Advanced matchup intelligence" });
    fireEvent.click(within(panel).getByRole("radio", { name: "Best of 5" }));
    await waitFor(() => expect(fetchMock.mock.calls.some(([u]) => String(u).includes("best_of=5"))).toBe(true));
  });

  it("says when V2 has no estimate instead of inventing one", async () => {
    serve(() => json(rawPrediction({ v2: rawV2Unavailable() })));
    render(<MatchPredictor />);
    await within(hero()).findByText(percent("68.8%"));
    const toggle = within(hero()).getByRole("radio", { name: "V2 · experimental" }) as HTMLButtonElement;
    expect(toggle.disabled).toBe(true);
    const panel = screen.getByRole("region", { name: "Advanced matchup intelligence" });
    expect(within(panel).getByText(/does not invent them/)).toBeTruthy();
    const details = screen.getByRole("region", { name: "How to read the prediction" });
    expect(within(details).getByText(/does not invent them/)).toBeTruthy();
    expect(within(details).queryByText("73.2%")).toBeNull();
  });
});
