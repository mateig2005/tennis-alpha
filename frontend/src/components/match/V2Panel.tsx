"use client";

import { FlaskConical, TriangleAlert } from "lucide-react";
import { fmtDate } from "@/lib/hooks";
import { setSelection } from "@/lib/selection";
import type { Player, Surface, V2Player, V2Prediction } from "@/lib/types";
import { Segmented } from "../ui";
import { ContributionChart } from "./Charts";

const pts = (v: number) => `${v >= 0 ? "+" : ""}${(v * 100).toFixed(1)}`; // rating -> points per 100 above the tour average
const pct = (v: number) => `${(v * 100).toFixed(1)}%`;

type Row = { label: string; hint: string; get: (p: V2Player) => number; fmt: (v: number) => string };

function rows(surface: Surface): Row[] {
  const s = surface.toLowerCase();
  return [
    { label: "Serve strength", hint: "Serve points won above the tour average, adjusted for each opponent's return (points per 100)", get: (p) => p.serve, fmt: pts },
    { label: "Return strength", hint: "Return points won above the tour average, adjusted for each opponent's serve", get: (p) => p.ret, fmt: pts },
    { label: `Serve on ${s}`, hint: `Serve rating from ${s} matches, pulled toward the overall rating when ${s} matches are few`, get: (p) => p.serveSurface, fmt: pts },
    { label: `Return on ${s}`, hint: `Return rating from ${s} matches, same rule`, get: (p) => p.retSurface, fmt: pts },
    { label: "Serve, recent", hint: "Serve rating with a 60-day half-life", get: (p) => p.serveRecent, fmt: pts },
    { label: "Return, recent", hint: "Return rating with a 60-day half-life", get: (p) => p.retRecent, fmt: pts },
    { label: "Serve points won", hint: "Last 20 matches with statistics, pulled toward the tour average", get: (p) => p.spw20, fmt: pct },
    { label: "Return points won", hint: "Last 20 matches with statistics", get: (p) => p.rpw20, fmt: pct },
    { label: "Service games held", hint: "Last 20 matches with statistics", get: (p) => p.hold20, fmt: pct },
    { label: "Return games won", hint: "Last 20 matches with statistics", get: (p) => p.break20, fmt: pct },
    { label: "Aces per serve point", hint: "Last 20 matches with statistics", get: (p) => p.ace20, fmt: pct },
    { label: "Elo change, 90 days", hint: "General Elo now minus 90 days ago", get: (p) => p.eloTrend * 400, fmt: (v) => `${v >= 0 ? "+" : ""}${v.toFixed(0)}` },
  ];
}

function Compare({ label, hint, a, b }: { label: string; hint: string; a: string; b: string }) {
  return (
    <div className="grid grid-cols-[1fr_auto_1fr] items-baseline gap-3 border-b border-line/60 py-2 last:border-0">
      <span className="num text-[13.5px] text-fg">{a}</span>
      <span className="text-center text-[12px] text-muted" title={hint}>{label}</span>
      <span className="num text-right text-[13.5px] text-fg">{b}</span>
    </div>
  );
}

/** Tennis Alpha V2: the serve/return inputs behind the experimental probability. Every value comes from the API. */
export function V2Panel({ v2, a, b, surface, snapshot, bestOf }: { v2: V2Prediction; a: Player; b: Player; surface: Surface; snapshot: string; bestOf: 3 | 5 }) {
  const ok = v2.status === "ok" && v2.players && v2.matchup;
  return (
    <section className="card border-dashed border-sage/45" aria-labelledby="v2-title">
      <div className="p-5 sm:p-7">
        <div className="flex flex-wrap items-start justify-between gap-3">
          <div>
            <p className="eyebrow flex items-center gap-1.5 text-sage"><FlaskConical size={12} aria-hidden="true" />Tennis Alpha V2 · experimental</p>
            <h2 id="v2-title" className="mt-1 text-[16px] font-medium tracking-tight">Advanced matchup intelligence</h2>
          </div>
          <Segmented label="Match format (V2 only)" value={String(bestOf) as "3" | "5"} onChange={(v) => setSelection({ bestOf: v === "5" ? 5 : 3 })}
            options={[{ value: "3", label: "Best of 3" }, { value: "5", label: "Best of 5" }]} />
        </div>
        <p className="mt-3 text-[12.5px] leading-relaxed text-muted">
          {v2.version ? <>Version <span className="num text-fg">{v2.version}</span> · </> : null}snapshot {fmtDate(snapshot)}
          {v2.dataEnd ? <> · data to tournaments starting {fmtDate(v2.dataEnd)}</> : null}. {v2.validation ?? "Experimental; no independent prospective test yet."}
        </p>

        {!ok ? (
          <p className="mt-4 flex items-start gap-2.5 rounded-lg border border-coral/40 bg-coral/5 px-3 py-2 text-[13px]" role="status">
            <TriangleAlert size={15} className="mt-0.5 shrink-0 text-coral" aria-hidden="true" />
            <span><span className="text-fg">V2 has no estimate for this matchup.</span> {v2.reason}</span>
          </p>
        ) : (
          <>
            <div className="mt-5 grid gap-3 sm:grid-cols-3">
              <div className="rounded-xl border border-line p-4">
                <p className="eyebrow">Expected serve points won</p>
                <p className="num mt-2 text-[15px]"><span className="text-fg">{pct(v2.matchup!.pa)}</span> <span className="text-muted">vs</span> <span className="text-fg">{pct(v2.matchup!.pb)}</span></p>
                <p className="mt-1 text-[12px] leading-relaxed text-muted">{a.name} and {b.name} on serve against each other. Tour average on {surface.toLowerCase()}: {pct(v2.matchup!.tourServeAvg)}.</p>
              </div>
              <div className="rounded-xl border border-line p-4">
                <p className="eyebrow">Point-to-match model</p>
                <p className="num mt-2 text-[15px] text-fg">{pct(v2.matchup!.pPointsModel)} <span className="text-[12px] text-muted">for {a.name}</span></p>
                <p className="mt-1 text-[12px] leading-relaxed text-muted">Those serve chances played out over best of {v2.bestOf}. Uncalibrated and overconfident alone; V2 uses it as one input.</p>
              </div>
              <div className="rounded-xl border border-line p-4">
                <p className="eyebrow">Head-to-head input</p>
                <p className="num mt-2 text-[15px] text-fg">{v2.h2h!.residual >= 0 ? "+" : ""}{v2.h2h!.residual.toFixed(3)} <span className="text-[12px] text-muted">over {v2.h2h!.meetings} meetings</span></p>
                <p className="mt-1 text-[12px] leading-relaxed text-muted">{a.name}&apos;s results against {b.name} minus Elo&apos;s expectation, shrunk by 5 pseudo-meetings so a few matches cannot dominate.</p>
              </div>
            </div>

            <div className="mt-5 grid gap-6 lg:grid-cols-12">
              <div className="lg:col-span-5">
                <div className="grid grid-cols-[1fr_auto_1fr] gap-3 pb-1 text-[12px] font-medium">
                  <span className="truncate">{a.name}</span><span /><span className="truncate text-right">{b.name}</span>
                </div>
                {rows(surface).map((r) => <Compare key={r.label} label={r.label} hint={r.hint} a={r.fmt(r.get(v2.players!.a))} b={r.fmt(r.get(v2.players!.b))} />)}
                <Compare label="Matches with statistics" hint="Released matches with serve/return statistics" a={String(v2.players!.a.statMatches)} b={String(v2.players!.b.statMatches)} />
                {v2.players!.a.quality && v2.players!.b.quality && (
                  <>
                    <p className="eyebrow mt-4">Match quality · descriptive, not a V2 input</p>
                    <Compare label="Points vs expectation, last 20" hint="Share of points won minus what was expected before each match (points per 100)"
                      a={pts(v2.players!.a.quality.pointsVsExpected20)} b={pts(v2.players!.b.quality.pointsVsExpected20)} />
                    <Compare label="Luck, last 10" hint="Results minus the win chance implied by the points actually won; positive = won more than the points deserved"
                      a={pts(v2.players!.a.quality.luck10)} b={pts(v2.players!.b.quality.luck10)} />
                  </>
                )}
                <p className="mt-3 text-[11.5px] leading-relaxed text-dim">
                  Ratings are points per 100 above the tour average (+5.0 = five more points won in every 100). Hover a label for its definition.
                </p>
              </div>
              <div className="lg:col-span-7">
                <p className="eyebrow">What moves V2</p>
                <ContributionChart contributions={v2.contributions} a={a} b={b} aFav={(v2.p ?? 0.5) >= 0.5} />
                <p className="num mt-3 rounded-lg border border-line bg-ink/50 px-3 py-2 text-[12.5px] leading-relaxed text-muted">
                  sum of bars = <span className="text-fg">{(v2.logit ?? 0) >= 0 ? "+" : ""}{(v2.logit ?? 0).toFixed(3)}</span> log-odds
                  {"  →  "}<span className="text-fg">{((v2.p ?? 0) * 100).toFixed(1)}%</span> for {a.name}
                </p>
                <p className="mt-3 text-[11.5px] leading-relaxed text-dim">
                  V2 keeps Model D&apos;s inputs and adds serve/return ratings, the serve-vs-return matchup and recent serve/return trends. Correlated groups
                  share credit, so read the total. No court-speed value is shown: V2&apos;s pace term is a statistical proxy, and with no venue chosen it uses the surface average.
                </p>
              </div>
            </div>

            {v2.notes.length > 0 && (
              <ul className="mt-4 space-y-2 text-[13px] leading-relaxed text-muted">
                {v2.notes.map((n) => <li key={n} className="flex gap-2.5"><TriangleAlert size={15} className="mt-0.5 shrink-0 text-sage" aria-hidden="true" />{n}</li>)}
              </ul>
            )}
          </>
        )}
      </div>
    </section>
  );
}
