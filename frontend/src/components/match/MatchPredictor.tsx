"use client";

import { CircleSlash, TriangleAlert } from "lucide-react";
import { MotionConfig, motion } from "motion/react";
import { useState } from "react";
import { fmtDate, useMatchup, useMeta, usePlayers } from "@/lib/hooks";
import type { DataStatus, HeroModel } from "@/lib/types";
import { Card, CardHeader, Skeleton, SnapshotBadge } from "../ui";
import { ContributionChart, EloChart, FormChart, SurfaceChart } from "./Charts";
import { HeadToHead } from "./HeadToHead";
import { MatchHero, PredictionDetails } from "./ProbabilityHero";
import { V2Panel } from "./V2Panel";
import { StatCompare } from "./StatCompare";

const rise = { initial: { opacity: 0, y: 10 }, animate: { opacity: 1, y: 0 }, transition: { duration: 0.4, ease: [0.16, 1, 0.3, 1] as const } };

export function ServiceError({ message }: { message: string }) {
  return (
    <Card className="flex items-start gap-3 border-coral/40">
      <CircleSlash size={20} className="mt-0.5 shrink-0 text-coral" aria-hidden="true" />
      <div role="alert">
        <p className="text-[15px] font-medium">The prediction could not be produced</p>
        <p className="mt-1 text-[13.5px] text-muted">{message}</p>
      </div>
    </Card>
  );
}

/** Says, on every page that shows model output, that this is a historical snapshot and not a live forecast. */
export function SnapshotNotice() {
  const meta = useMeta();
  return (
    <p className="mb-4 flex items-start gap-2.5 rounded-xl border border-dashed border-sage/30 bg-panel/60 px-3.5 py-2 text-[12.5px] leading-relaxed text-muted">
      <TriangleAlert size={15} className="mt-0.5 shrink-0 text-sage" aria-hidden="true" />
      <span>
        <span className="text-fg">Historical snapshot, not a live forecast.</span>{" "}
        {meta ? (
          <>The data ends with tournaments starting {fmtDate(meta.dataEnd)}. <span className="hidden sm:inline">Probabilities describe a hypothetical match on {fmtDate(meta.snapshotDate)}; nothing after that date is known to the models.</span></>
        ) : "Probabilities describe a hypothetical match on the snapshot date; there is no real-time ATP data."}
        {meta?.dataStatus && <DataStatusLine status={meta.dataStatus} />}
      </span>
    </p>
  );
}

/** One discreet line: how fresh the local data is, where it comes from, and any warning. Coefficients stay frozen. */
function DataStatusLine({ status }: { status: DataStatus }) {
  const stale = status.warnings.length > 0;
  return (
    <span className="mt-1 block text-[11.5px] text-dim" aria-label="Data status">
      Results included through <span className="text-muted">{status.latestCompleteThrough ? fmtDate(status.latestCompleteThrough) : "unknown"}</span>
      {status.dataAgeDays !== null && <> ({status.dataAgeDays} days old)</>}
      {status.lastSuccessUtc && <> · last update {fmtDate(status.lastSuccessUtc)}</>}
      {" · "}source: TennisMyLife and Jeff Sackmann / Tennis Abstract · model coefficients frozen (June 2026 fit)
      {status.pending.length > 0 && <> · not yet included (may still be running): {status.pending.join(", ")}</>}
      {status.knownGaps.length > 0 && <> · <span className="text-coral">missing from the source: {status.knownGaps.join(", ")}</span></>}
      {stale && status.warnings.map((w) => <span key={w} className="mt-0.5 block text-coral" role="status">{w}</span>)}
    </span>
  );
}

export function MatchPredictor() {
  const { aId, bId, surface, bestOf, data, error } = useMatchup();
  const [chosen, setChosen] = useState<HeroModel>("D");
  const { players, error: listError } = usePlayers();
  const a = data?.a ?? players?.find((p) => p.id === aId);
  const b = data?.b ?? players?.find((p) => p.id === bId);
  const prediction = data?.prediction;
  const aFav = (prediction?.p.D ?? 0.5) >= 0.5;
  const failure = error ?? (data ? undefined : listError);
  const snapshot = data ? fmtDate(data.snapshotDate) : null;
  const v2Available = prediction ? prediction.v2.status === "ok" : true; // unknown while loading: keep the choice
  const model: HeroModel = chosen === "V2" && v2Available ? "V2" : "D"; // V2 without an estimate falls back to Model D

  return (
    <MotionConfig reducedMotion="user">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-x-4 gap-y-2">
        <h1 className="text-2xl font-medium tracking-tight sm:text-[28px]">
          <span className="eyebrow mr-3 align-middle text-sage">Match prediction</span>Head to head
        </h1>
        <span className="rounded-full border border-line px-2.5 py-0.5 text-[11px] text-muted">Hypothetical match{snapshot ? ` · snapshot ${snapshot}` : ""}</span>
      </div>
      <SnapshotNotice />

      <MatchHero aId={aId} bId={bId} a={a} b={b} detail={data} surface={surface} players={players} prediction={prediction}
        badge={snapshot ? `Model output · snapshot ${snapshot}` : "Model output"} failed={!!failure}
        model={model} onModel={setChosen} v2Available={v2Available} />

      <div className="mt-5 space-y-5" aria-live="polite">
        {failure ? (
          <ServiceError message={failure} />
        ) : !data || !prediction ? (
          <>
            <Card><Skeleton className="h-4 w-64" /><div className="mt-5 grid gap-3 md:grid-cols-3"><Skeleton className="h-24" /><Skeleton className="h-24" /><Skeleton className="h-24" /></div></Card>
            <div className="grid gap-5 lg:grid-cols-2"><Card><Skeleton className="h-72 w-full" /></Card><Card><Skeleton className="h-72 w-full" /></Card></div>
          </>
        ) : (
          <motion.div key={`${aId}|${bId}|${surface}`} {...rise} className="space-y-5">
            <PredictionDetails a={data.a} b={data.b} prediction={prediction} model={model} />

            {prediction.notes.length > 0 && (
              <Card className={prediction.reliable ? "" : "border-coral/40"}>
                <CardHeader eyebrow="Read before using this number" title={prediction.reliable ? "Notes on the inputs" : "This prediction is not reliable"} />
                <ul className="space-y-2 text-[13.5px] leading-relaxed text-muted">
                  {prediction.notes.map((n) => (
                    <li key={n} className="flex gap-2.5"><TriangleAlert size={15} className={`mt-0.5 shrink-0 ${prediction.reliable ? "text-sage" : "text-coral"}`} aria-hidden="true" />{n}</li>
                  ))}
                </ul>
              </Card>
            )}

            <V2Panel v2={prediction.v2} a={data.a} b={data.b} surface={surface} snapshot={data.snapshotDate} bestOf={bestOf} />

            <HeadToHead a={data.a} b={data.b} surface={surface} />

            <div className="grid gap-5 lg:grid-cols-12">
              <Card className="lg:col-span-5">
                <CardHeader eyebrow="Player analytics" title="Side by side" right={<SnapshotBadge />} />
                <StatCompare a={data.a} b={data.b} surface={surface} />
              </Card>
              <Card className="lg:col-span-7">
                <CardHeader eyebrow="Model D" title="What moves the prediction" right={<SnapshotBadge label="Trained coefficients" />} />
                <ContributionChart contributions={prediction.contributions} a={data.a} b={data.b} aFav={aFav} />
                <p className="num mt-3 rounded-lg border border-line bg-ink/50 px-3 py-2 text-[12.5px] leading-relaxed text-muted">
                  sum of bars = <span className="text-fg">{prediction.logit >= 0 ? "+" : ""}{prediction.logit.toFixed(3)}</span> log-odds
                  {"  →  "}1 / (1 + e<sup>−sum</sup>) = <span className="text-fg">{(prediction.p.D * 100).toFixed(1)}%</span> for {data.a.name}
                </p>
                <p className="mt-3 text-[11.5px] leading-relaxed text-dim">
                  Each bar is Model D&apos;s trained coefficients for one feature group times the standardised difference between the players, so the bars add up exactly
                  to the prediction. Bars to the right favour {data.a.name}, bars to the left favour {data.b.name}. Groups carry overlapping information
                  (form and Elo, for example), so opposite bars often cancel: read the total rather than single bars.
                </p>
              </Card>
            </div>

            <Card>
              <CardHeader eyebrow="Rating history" title="Elo progression" right={<SnapshotBadge />} />
              <EloChart a={data.a} b={data.b} aFav={aFav} snapshot={data.snapshotDate} />
            </Card>

            <div className="grid gap-5 lg:grid-cols-2">
              <Card>
                <CardHeader eyebrow="Recent performance" title="Form by window" right={<SnapshotBadge />} />
                <FormChart a={data.a} b={data.b} aFav={aFav} />
              </Card>
              <Card>
                <CardHeader eyebrow="Surface breakdown" title="Performance by surface" right={<SnapshotBadge />} />
                <SurfaceChart a={data.a} b={data.b} aFav={aFav} surface={surface} />
              </Card>
            </div>
            <p className="text-[11.5px] text-dim">In the bar charts, solid bars belong to Model D&apos;s favourite and hatched bars to the underdog. A missing bar means no matches in that window.</p>

          </motion.div>
        )}
      </div>
    </MotionConfig>
  );
}
