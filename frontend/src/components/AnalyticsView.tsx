"use client";

import { ArrowLeftRight, Swords } from "lucide-react";
import Link from "next/link";
import { useMatchup, usePlayers } from "@/lib/hooks";
import { setSelection } from "@/lib/selection";
import { EloChart, FormChart, SurfaceChart } from "./match/Charts";
import { HeadToHead } from "./match/HeadToHead";
import { ServiceError, SnapshotNotice } from "./match/MatchPredictor";
import { IdentitySkeleton, PlayerIdentity } from "./match/PlayerPanel";
import { StatCompare } from "./match/StatCompare";
import { PlayerSelect } from "./PlayerSelect";
import { SurfaceSelector } from "./SurfaceSelector";
import { Card, CardHeader, Skeleton, SnapshotBadge } from "./ui";

/** Uses the same matchup as the Match Prediction page: change it on either page and both follow. */
export function AnalyticsView() {
  const { aId, bId, surface, data, error } = useMatchup();
  const { players, error: listError } = usePlayers();
  const a = data?.a ?? players?.find((p) => p.id === aId);
  const b = data?.b ?? players?.find((p) => p.id === bId);
  const lead = (a?.elo ?? 0) >= (b?.elo ?? 0); // on this page "solid" simply means the higher-rated player
  const failure = error ?? (data ? undefined : listError);
  return (
    <div className="space-y-5">
      <SnapshotNotice />
      <Card>
        <div className="grid items-end gap-3 md:grid-cols-[1fr_auto_1fr_auto] md:gap-5">
          <PlayerSelect label="Player A" players={players} value={aId} selectedName={a?.name} unavailable={!!failure} takenId={bId} onChange={(id) => setSelection({ aId: id })} />
          <button type="button" onClick={() => setSelection({ aId: bId, bId: aId })} aria-label="Swap player A and player B" title="Swap players"
            className="mx-auto grid h-11 w-11 place-items-center rounded-full border border-line text-muted transition-colors hover:border-dim hover:text-fg">
            <ArrowLeftRight size={16} aria-hidden="true" />
          </button>
          <PlayerSelect label="Player B" players={players} value={bId} selectedName={b?.name} unavailable={!!failure} takenId={aId} onChange={(id) => setSelection({ bId: id })} />
          <div className="md:w-60"><SurfaceSelector value={surface} onChange={(s) => setSelection({ surface: s })} /></div>
        </div>
        <div className="mt-6 grid grid-cols-2 gap-4 sm:gap-8">
          {a ? <PlayerIdentity player={a} detail={data?.a} surface={surface} tone={lead ? "mint" : "coral"} /> : <IdentitySkeleton />}
          {b ? <PlayerIdentity player={b} detail={data?.b} surface={surface} tone={lead ? "coral" : "mint"} align="right" /> : <IdentitySkeleton align="right" />}
        </div>
        <div className="mt-5 flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
          <p className="text-[12px] text-dim">The same matchup is used on the Match Prediction page. Green marks the higher-rated player here, not a prediction.</p>
          <Link href="/" className="inline-flex items-center gap-2 rounded-lg border border-mint/40 px-3 py-1.5 text-[12.5px] text-mint transition-colors hover:bg-mint/10">
            <Swords size={14} aria-hidden="true" />See the prediction
          </Link>
        </div>
      </Card>
      {failure ? (
        <ServiceError message={failure} />
      ) : !data ? (
        <><Card><Skeleton className="h-72 w-full" /></Card><div className="grid gap-5 lg:grid-cols-2"><Card><Skeleton className="h-64 w-full" /></Card><Card><Skeleton className="h-64 w-full" /></Card></div></>
      ) : (
        <>
          <HeadToHead a={data.a} b={data.b} surface={surface} />
          <Card>
            <CardHeader eyebrow="Elo trends" title="Rating progression" right={<SnapshotBadge />} />
            <EloChart a={data.a} b={data.b} aFav={lead} snapshot={data.snapshotDate} />
          </Card>
          <div className="grid gap-5 lg:grid-cols-2">
            <Card>
              <CardHeader eyebrow="Recent-form analytics" title="Form by window" right={<SnapshotBadge />} />
              <FormChart a={data.a} b={data.b} aFav={lead} />
            </Card>
            <Card>
              <CardHeader eyebrow="Surface performance" title="Performance by surface" right={<SnapshotBadge />} />
              <SurfaceChart a={data.a} b={data.b} aFav={lead} surface={surface} />
            </Card>
          </div>
          <p className="text-[11.5px] text-dim">Solid bars: the higher-rated player. Hatched bars: the lower-rated player. A missing bar means no matches in that window.</p>
          <Card>
            <CardHeader eyebrow="Player comparison" title="Side by side" right={<SnapshotBadge />} />
            <StatCompare a={data.a} b={data.b} surface={surface} />
          </Card>
        </>
      )}
    </div>
  );
}
