"use client";

import { ArrowDownRight, ArrowLeftRight, ArrowUpRight, FlaskConical, Gauge, Info, Scale, ShieldAlert, ShieldCheck } from "lucide-react";
import { animate, motion, useReducedMotion } from "motion/react";
import { useEffect, useRef, useState } from "react";
import { fmtOdds } from "@/lib/odds";
import { setSelection } from "@/lib/selection";
import type { HeroModel, Player, PlayerSummary, Prediction, Surface } from "@/lib/types";
import { PlayerSelect } from "../PlayerSelect";
import { SurfaceSelector } from "../SurfaceSelector";
import { Segmented, Skeleton, SnapshotBadge } from "../ui";
import { IdentitySkeleton, PlayerIdentity, type Tone } from "./PlayerPanel";

const EASE = [0.16, 1, 0.3, 1] as const;

function AnimatedPct({ value }: { value: number }) {
  const reduce = useReducedMotion();
  const from = useRef(value);
  const [shown, setShown] = useState(value);
  useEffect(() => {
    const controls = animate(from.current, value, { duration: reduce ? 0 : 0.7, ease: EASE, onUpdate: setShown });
    from.current = value;
    return () => controls.stop();
  }, [value, reduce]);
  return <>{(shown * 100).toFixed(1)}</>;
}

function Probability({ p, favourite, align, muted }: { p: number | null; favourite: boolean; align: "left" | "right"; muted: boolean }) {
  const right = align === "right";
  if (p === null) {
    return <div className={`flex flex-col gap-2 ${right ? "items-end" : ""}`}><Skeleton className="h-5 w-24" /><Skeleton className="h-[clamp(2.6rem,8vw,5.25rem)] w-36 sm:w-52" /></div>;
  }
  const Icon = favourite ? ArrowUpRight : ArrowDownRight;
  return (
    <div className={`shrink-0 ${right ? "text-right" : ""}`}>
      <p className={`inline-flex items-center gap-1 rounded-full border px-2 py-0.5 text-[10.5px] font-medium tracking-wider uppercase ${favourite ? "border-mint/40 text-mint" : "border-coral/40 text-coral"}`}>
        <Icon size={12} aria-hidden="true" />
        {favourite ? "Favourite" : "Underdog"}
      </p>
      <p className={`mt-1 text-[clamp(2.6rem,8vw,5.25rem)] leading-none font-medium tracking-[-0.045em] tabular-nums ${favourite ? "text-mint" : "text-coral"} ${muted ? "opacity-55" : favourite ? "glow-mint" : "glow-coral"}`}>
        <AnimatedPct value={p} />
        <span className="ml-1 text-[0.36em] tracking-normal opacity-70">%</span>
      </p>
      <p className="num mt-1.5 text-[11.5px] text-muted" title="Model-implied fair decimal odds (1 / probability). Not a bookmaker price.">
        fair odds <span className="text-fg">{fmtOdds(p)}</span>
      </p>
    </div>
  );
}

interface HeroProps {
  aId: string;
  bId: string;
  a?: PlayerSummary; // from the player list or the prediction response, whichever arrived first
  b?: PlayerSummary;
  detail?: { a: Player; b: Player }; // from the prediction response
  surface: Surface;
  players?: PlayerSummary[];
  prediction?: Prediction;
  badge: string;
  failed?: boolean; // the service gave no answer: stop the loading animation, show no numbers
  model: HeroModel; // which model's probability is the headline
  onModel: (m: HeroModel) => void;
  v2Available: boolean;
}

export const MODEL_LABEL: Record<HeroModel, string> = { D: "Model D · validated baseline", V2: "Tennis Alpha V2 · experimental" };

/** The headline probability for player A from the chosen model; null while loading (or if V2 has no estimate). */
export const heroP = (prediction: Prediction | undefined, model: HeroModel) => (!prediction ? null : model === "V2" ? prediction.v2.p : prediction.p.D);

/** Matchup controls, player identities and the primary Model D probability in one panel, so the answer is visible on arrival. */
export function MatchHero({ aId, bId, a, b, detail, surface, players, prediction, badge, failed = false, model, onModel, v2Available }: HeroProps) {
  const pA = heroP(prediction, model);
  const aFav = (pA ?? 0.5) >= 0.5;
  const tone = (isA: boolean): Tone => (pA === null ? "neutral" : isA === aFav ? "mint" : "coral");
  const muted = !!prediction && !prediction.reliable;
  const label = pA === null || !a || !b ? "Calculating" : `${a.name} ${(pA * 100).toFixed(1)} percent, ${b.name} ${((1 - pA) * 100).toFixed(1)} percent`;
  return (
    <section className={`card ${failed ? "[&_.animate-pulse]:animate-none" : ""}`} aria-labelledby="hero-title" aria-busy={pA === null && !failed}>
      {/* One column whose order changes: on phones the answer comes before the controls; from md up the controls come first. */}
      <div className="flex flex-col gap-5 p-5 sm:gap-6 sm:p-7">
        <div className="order-1 flex flex-wrap items-center justify-between gap-x-4 gap-y-3">
          <div>
            <p className="eyebrow flex items-center gap-1.5 text-sage">{model === "V2" && <FlaskConical size={12} aria-hidden="true" />}{MODEL_LABEL[model]}</p>
            <h2 id="hero-title" className="mt-1 text-[16px] font-medium tracking-tight">Estimated probability of winning</h2>
          </div>
          <div className="flex flex-wrap items-center gap-3">
            <Segmented label="Headline model" value={model} onChange={onModel} disabled={v2Available ? [] : ["V2"]}
              options={[{ value: "D", label: "Model D" }, { value: "V2", label: "V2 · experimental" }]} />
            <SnapshotBadge label={badge} />
            <SurfaceSelector value={surface} onChange={(s) => setSelection({ surface: s })} hideLabel />
          </div>
        </div>

        <div className="order-4 grid items-end gap-3 md:order-2 md:grid-cols-[1fr_auto_1fr] md:gap-5">
          <PlayerSelect label="Player A" players={players} value={aId} selectedName={a?.name} unavailable={failed} takenId={bId} onChange={(id) => setSelection({ aId: id })} />
          <div className="flex items-center justify-center gap-2 md:pb-0.5">
            <span className="grid h-11 w-11 place-items-center rounded-full border border-mint/40 bg-ink text-[11.5px] font-semibold tracking-[0.2em] text-mint shadow-[0_0_28px_-6px_rgb(57_255_156/0.55)]">VS</span>
            <button type="button" onClick={() => setSelection({ aId: bId, bId: aId })} aria-label="Swap player A and player B" title="Swap players"
              className="grid h-11 w-11 place-items-center rounded-full border border-line text-muted transition-colors hover:border-dim hover:text-fg">
              <ArrowLeftRight size={16} aria-hidden="true" />
            </button>
          </div>
          <PlayerSelect label="Player B" players={players} value={bId} selectedName={b?.name} unavailable={failed} takenId={aId} onChange={(id) => setSelection({ bId: id })} />
        </div>

        {/* Identity and probability share a row on wide screens; on smaller ones the number sits under the name. */}
        <div className="order-2 grid grid-cols-2 gap-4 sm:gap-8 md:order-3">
          <div className="flex min-w-0 flex-col gap-4 xl:flex-row xl:items-end xl:justify-between">
            {a ? <PlayerIdentity player={a} detail={detail?.a} surface={surface} tone={tone(true)} /> : <IdentitySkeleton />}
            <Probability p={pA} favourite={aFav} align="left" muted={muted} />
          </div>
          <div className="flex min-w-0 flex-col items-end gap-4 xl:flex-row-reverse xl:items-end xl:justify-between">
            {b ? <PlayerIdentity player={b} detail={detail?.b} surface={surface} tone={tone(false)} align="right" /> : <IdentitySkeleton align="right" />}
            <Probability p={pA === null ? null : 1 - pA} favourite={!aFav} align="right" muted={muted} />
          </div>
        </div>

        <div className="relative order-3 md:order-4">
          <div className="flex h-3 overflow-hidden rounded-full bg-ink" role="img" aria-label={label}>
            {pA === null ? <div className="w-full animate-pulse bg-raised" /> : (
              <>
                <motion.div initial={false} animate={{ width: `${pA * 100}%` }} transition={{ duration: 0.8, ease: EASE }}
                  className={aFav ? "bg-mint shadow-[0_0_24px_rgb(57_255_156/0.6)]" : "bg-coral"} />
                <div className="w-1 bg-ink" />
                <div className={`flex-1 ${aFav ? "bg-coral" : "bg-mint shadow-[0_0_24px_rgb(57_255_156/0.6)]"}`} />
              </>
            )}
          </div>
          <div className="absolute -top-1.5 left-1/2 h-6 w-px -translate-x-1/2 bg-fg/70" aria-hidden="true" />
          <p className="num mt-1.5 text-center text-[10.5px] tracking-widest text-dim">50 / 50</p>
        </div>

        {muted && (
          <p className="order-5 flex items-center gap-2 rounded-lg border border-coral/40 bg-coral/5 px-3 py-2 text-[13px] text-fg" role="status">
            <ShieldAlert size={15} className="shrink-0 text-coral" aria-hidden="true" />
            Not a reliable estimate: an input is missing or too old. The numbers are dimmed and shown for transparency only.
          </p>
        )}
      </div>
    </section>
  );
}

function edge(p: number) {
  const d = Math.abs(p - 0.5);
  if (d < 0.05) return { label: "Toss-up", text: "The model sees almost no difference between the players." };
  if (d < 0.15) return { label: "Lean", text: "A modest edge. Upsets at this level are routine." };
  if (d < 0.3) return { label: "Clear edge", text: "A clear favourite, though far from a sure thing." };
  return { label: "Strong edge", text: "A heavy favourite on these inputs." };
}

function MiniModel({ name, role, pA, a, b, experimental = false, unavailable }: { name: string; role: string; pA: number | null; a: Player; b: Player; experimental?: boolean; unavailable?: string | null }) {
  const aFav = (pA ?? 0.5) >= 0.5;
  return (
    <div className={`rounded-xl border p-4 ${experimental ? "border-dashed border-sage/45 bg-sage/[0.04]" : "border-line bg-raised/60"}`}>
      <div className="flex items-baseline justify-between gap-2">
        <p className="flex items-center gap-1.5 text-[13.5px] font-medium">{experimental && <FlaskConical size={13} className="text-sage" aria-hidden="true" />}{name}</p>
        <p className={`eyebrow ${experimental ? "text-sage" : ""}`}>{role}</p>
      </div>
      {pA === null ? (
        <p className="mt-3 text-[12.5px] leading-relaxed text-muted">{unavailable ?? "No estimate."}</p>
      ) : (
        <>
          <div className="mt-3 flex items-baseline justify-between gap-3">
            <span className={`num text-xl ${aFav ? "text-mint" : "text-coral"}`}>{(pA * 100).toFixed(1)}%</span>
            <span className={`num text-xl ${aFav ? "text-coral" : "text-mint"}`}>{((1 - pA) * 100).toFixed(1)}%</span>
          </div>
          <div className="mt-2 flex h-1.5 overflow-hidden rounded-full bg-ink" role="img" aria-label={`${name}: ${a.name} ${(pA * 100).toFixed(1)} percent, ${b.name} ${((1 - pA) * 100).toFixed(1)} percent`}>
            <motion.div className={aFav ? "bg-mint" : "bg-coral"} initial={false} animate={{ width: `${pA * 100}%` }} transition={{ duration: 0.6, ease: EASE }} />
            <div className="w-0.5 bg-ink" />
            <div className={`flex-1 ${aFav ? "bg-coral/70" : "bg-mint/70"}`} />
          </div>
          <div className="num mt-2 flex justify-between text-[11px] text-muted" aria-label={`${name} fair odds: ${a.name} ${fmtOdds(pA)}, ${b.name} ${fmtOdds(1 - pA)}`}>
            <span>fair odds {fmtOdds(pA)}</span><span>{fmtOdds(1 - pA)}</span>
          </div>
        </>
      )}
    </div>
  );
}

/** Probability is not certainty: three separate readings of how far to trust the headline number, then the other two models. */
export function PredictionDetails({ a, b, prediction, model }: { a: Player; b: Player; prediction: Prediction; model: HeroModel }) {
  const pA = heroP(prediction, model) ?? prediction.p.D;
  const pFav = Math.max(pA, 1 - pA);
  const e = edge(pA);
  const spread = Math.max(...Object.values(prediction.p)) - Math.min(...Object.values(prediction.p)); // V1 models only, as before
  const agreement = spread < 0.04 ? "High" : spread < 0.1 ? "Medium" : "Low";
  const per100 = Math.round((1 - pFav) * 100);
  const upsets = per100 < 1 ? "fewer than 1 in 100" : `about ${per100} in 100`;
  return (
    <section className="card" aria-label="How to read the prediction">
      <div className="p-5 sm:p-7">
        <p className="flex items-start gap-2 text-[13px] leading-relaxed text-muted">
          <Info size={15} className="mt-0.5 shrink-0 text-sage" aria-hidden="true" />
          <span><span className="text-fg">A probability is not a certainty.</span> At {(pFav * 100).toFixed(1)}%, the underdog still wins {upsets} matches like this one.</span>
        </p>
        <dl className="mt-4 grid gap-3 md:grid-cols-3">
          <div className="rounded-xl border border-line p-4">
            <dt className="eyebrow flex items-center gap-1.5"><Gauge size={13} aria-hidden="true" />Size of the edge</dt>
            <dd className="mt-2 text-[15px] font-medium">{e.label}</dd>
            <dd className="mt-1 text-[12.5px] leading-relaxed text-muted">{e.text}</dd>
          </div>
          <div className="rounded-xl border border-line p-4">
            <dt className="eyebrow flex items-center gap-1.5"><Scale size={13} aria-hidden="true" />Model agreement</dt>
            <dd className="mt-2 text-[15px] font-medium">{agreement} <span className="num text-[12.5px] font-normal text-muted">· spread {(spread * 100).toFixed(1)} pts</span></dd>
            <dd className="mt-1 text-[12.5px] leading-relaxed text-muted">How far Model D, Model C and Elo are from each other.</dd>
          </div>
          <div className="rounded-xl border border-line p-4">
            <dt className="eyebrow flex items-center gap-1.5">
              {prediction.reliable ? <ShieldCheck size={13} aria-hidden="true" /> : <ShieldAlert size={13} className="text-coral" aria-hidden="true" />}Input quality
            </dt>
            <dd className={`mt-2 text-[15px] font-medium ${prediction.reliable ? "" : "text-coral"}`}>{prediction.reliable ? "Complete" : "Not reliable"}</dd>
            <dd className="mt-1 text-[12.5px] leading-relaxed text-muted">{prediction.reliable ? "Both players have recent history." : "An input is missing or too old. See the notes below."}</dd>
          </div>
        </dl>
      </div>
      <div className="grid gap-3 border-t border-line p-5 sm:grid-cols-3 sm:px-7">
        {model === "V2"
          ? <MiniModel name="Model D" role="Validated baseline" pA={prediction.p.D} a={a} b={b} />
          : <MiniModel name="Tennis Alpha V2" role="Experimental" pA={prediction.v2.p} a={a} b={b} experimental unavailable={prediction.v2.reason} />}
        <MiniModel name="Model C" role="Challenger" pA={prediction.p.C} a={a} b={b} />
        <MiniModel name="Elo" role="Benchmark" pA={prediction.p.Elo} a={a} b={b} />
        <p className="text-[11.5px] text-dim sm:col-span-3">
          Left value: {a.name}. Right value: {b.name}. Green marks each model&apos;s favourite, red its underdog. Fair odds are model-implied
          (1 ÷ probability, no margin), not bookmaker prices.
        </p>
      </div>
    </section>
  );
}
