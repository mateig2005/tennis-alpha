"use client";

import { ChevronDown, CircleSlash, History } from "lucide-react";
import { motion } from "motion/react";
import { useId, useState } from "react";
import { fmtDate, useH2H } from "@/lib/hooks";
import { SURFACES, type H2HMatch, type HeadToHead as H2H, type PlayerSummary, type Surface } from "@/lib/types";
import { Card, CardHeader, Skeleton, SnapshotBadge } from "../ui";

const ROUNDS: Record<string, string> = { F: "Final", SF: "Semi-final", QF: "Quarter-final", RR: "Round robin", BR: "Bronze match" };
const round = (r: string) => ROUNDS[r] ?? r;
const year = (iso: string) => iso.slice(0, 4);
type Side = "a" | "b" | "level";
const INK: Record<Side, string> = { a: "text-mint", b: "text-coral", level: "text-fg" };

/** Green marks the player ahead in the overall head-to-head, red the player behind; a tied record uses neither. */
function tones(aWins: number, bWins: number): { a: Side; b: Side } {
  if (aWins === bWins) return { a: "level", b: "level" };
  return aWins > bWins ? { a: "a", b: "b" } : { a: "b", b: "a" };
}

function Split({ aWins, bWins, tone, label }: { aWins: number; bWins: number; tone: { a: Side; b: Side }; label: string }) {
  const fill = (t: Side) => (t === "a" ? "bg-mint" : t === "b" ? "bg-coral" : "bg-muted");
  if (aWins + bWins === 0) return <div className="h-1.5 rounded-full bg-raised" role="img" aria-label={label} />;
  return (
    <div className="flex h-1.5 overflow-hidden rounded-full bg-ink" role="img" aria-label={label}>
      <motion.div className={fill(tone.a)} initial={false} animate={{ width: `${(aWins / (aWins + bWins)) * 100}%` }} transition={{ duration: 0.6, ease: [0.16, 1, 0.3, 1] }} />
      <div className="w-0.5 bg-ink" />
      <div className={`flex-1 ${fill(tone.b)} ${tone.b === "level" ? "" : "opacity-80"}`} />
    </div>
  );
}

function Winner({ m, a, b, tone }: { m: H2HMatch; a: PlayerSummary; b: PlayerSummary; tone: { a: Side; b: Side } }) {
  const aWon = m.winnerId === a.id;
  return <span className={`font-medium ${INK[aWon ? tone.a : tone.b]}`}>{aWon ? a.name : b.name}</span>;
}

/** The recorded score without its trailing "RET": the retirement is shown once, as a badge. */
const scoreText = (m: H2HMatch) => (m.retired ? m.score.replace(/\s*RET$/i, "") : m.score);

const Ret = () => (
  <span className="ml-1.5 rounded border border-muted/40 px-1 text-[10px] tracking-wide text-muted" title="The match ended by retirement. It counts as a win, as in the research data.">RET</span>
);

function Body({ h, a, b, surface }: { h: H2H; a: PlayerSummary; b: PlayerSummary; surface: Surface }) {
  const [open, setOpen] = useState(false);
  const listId = useId();
  const tone = tones(h.aWins, h.bWins);
  const leader = h.aWins === h.bWins ? null : h.aWins > h.bWins ? a : b;
  const recent = h.matches.slice(0, 5).reverse(); // oldest of the last five on the left, newest on the right
  const last = h.matches[0];
  return (
    <>
      {/* Central score */}
      {/* On phones the names share a row above the score, so neither is cut off; from sm up the score sits between them. */}
      <div className="grid grid-cols-2 items-center gap-x-3 gap-y-2 sm:grid-cols-[1fr_auto_1fr] sm:gap-6">
        <p className="truncate text-[15px] font-medium sm:text-xl">{a.name}</p>
        <p className="order-3 col-span-2 flex items-baseline justify-center gap-3 text-[clamp(2.75rem,9vw,4.5rem)] sm:order-none sm:col-span-1 sm:gap-4 leading-none font-medium tracking-[-0.04em] tabular-nums" aria-label={`${a.name} ${h.aWins}, ${b.name} ${h.bWins}`}>
          <span className={`${INK[tone.a]} ${tone.a === "a" ? "glow-mint" : ""}`}>{h.aWins}</span>
          <span className="text-[0.45em] text-dim" aria-hidden="true">–</span>
          <span className={`${INK[tone.b]} ${tone.b === "a" ? "glow-mint" : ""}`}>{h.bWins}</span>
        </p>
        <p className="truncate text-right text-[15px] font-medium sm:text-xl">{b.name}</p>
      </div>
      <div className="mt-4">
        <Split aWins={h.aWins} bWins={h.bWins} tone={tone} label={`${a.name} won ${h.aWins} of ${h.total} meetings, ${b.name} won ${h.bWins}`} />
        <p className="mt-2 text-center text-[12.5px] text-muted">
          <span className="num text-fg">{h.total}</span> recorded {h.total === 1 ? "meeting" : "meetings"} ·{" "}
          {leader ? <><span className="text-fg">{leader.name}</span> leads, winning <span className="num text-fg">{Math.round((Math.max(h.aWins, h.bWins) / h.total) * 100)}%</span></> : "level"}
          {last && <> · last met in <span className="num text-fg">{year(last.date)}</span> ({last.tournament}), won by <Winner m={last} a={a} b={b} tone={tone} /></>}
        </p>
      </div>

      {/* By surface */}
      <div className="mt-6 grid gap-3 sm:grid-cols-3">
        {SURFACES.map((s) => {
          const r = h.bySurface[s];
          const t = tone; // each player keeps the colour of the overall record, so a name never changes colour inside this section
          return (
            <div key={s} className={`rounded-xl border p-4 ${s === surface ? "border-mint/40 bg-mint/[0.04]" : "border-line"}`}>
              <p className="eyebrow flex items-center justify-between">{s}{s === surface && <span className="text-mint normal-case tracking-normal">selected</span>}</p>
              {r.matches ? (
                <>
                  <p className="num mt-2 text-2xl"><span className={INK[t.a]}>{r.aWins}</span><span className="mx-1.5 text-dim">–</span><span className={INK[t.b]}>{r.bWins}</span></p>
                  <div className="mt-2"><Split aWins={r.aWins} bWins={r.bWins} tone={t} label={`On ${s}: ${a.name} ${r.aWins}, ${b.name} ${r.bWins}`} /></div>
                  <p className="mt-2 text-[11.5px] text-dim">{r.matches} {r.matches === 1 ? "meeting" : "meetings"}</p>
                </>
              ) : <p className="mt-2 text-[13px] text-muted">No meetings</p>}
            </div>
          );
        })}
      </div>
      {h.otherSurface > 0 && <p className="mt-2 text-[11.5px] text-dim">{h.otherSurface} more on carpet or with no surface recorded.</p>}

      {/* Recent meetings */}
      <p className="eyebrow mt-7 mb-3">Last {recent.length === 1 ? "meeting" : `${recent.length} meetings`} · oldest to newest</p>
      <ol className="grid gap-2 sm:grid-cols-5">
        {recent.map((m, i) => (
          <li key={`${m.date}-${m.round}-${i}`} className="relative rounded-xl border border-line bg-raised/50 p-3 text-[12.5px]">
            <p className="flex items-center justify-between gap-2"><span className="num text-fg">{year(m.date)}</span><span className="truncate text-dim">{m.surface ?? "Surface n/a"}</span></p>
            <p className="mt-1 truncate text-muted" title={m.tournament}>{m.tournament}</p>
            <p className="text-dim">{round(m.round)}</p>
            <p className="mt-2 truncate"><Winner m={m} a={a} b={b} tone={tone} /></p>
            <p className="num mt-0.5 text-[11.5px] text-muted">{scoreText(m)}{m.retired && <Ret />}</p>
          </li>
        ))}
      </ol>

      {/* Full history */}
      <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-controls={listId}
        className="mt-5 inline-flex items-center gap-2 rounded-lg border border-line px-3 py-2 text-[13px] text-muted transition-colors hover:border-dim hover:text-fg">
        <History size={14} aria-hidden="true" />
        {open ? "Hide" : "Show"} all {h.total} {h.total === 1 ? "meeting" : "meetings"}
        <ChevronDown size={14} className={`transition-transform ${open ? "rotate-180" : ""}`} aria-hidden="true" />
      </button>
      {open && (
        <ul id={listId} className="mt-3 divide-y divide-line border-y border-line" aria-label="All recorded meetings, most recent first">
          {h.matches.map((m, i) => (
            <li key={`${m.date}-${m.round}-${i}`} className="grid grid-cols-[3.25rem_1fr] gap-x-3 gap-y-0.5 py-2.5 text-[13px] md:grid-cols-[3.25rem_minmax(0,1.5fr)_7rem_4.5rem_minmax(0,1fr)_minmax(0,1.2fr)] md:items-center">
              <span className="num row-span-4 text-fg md:row-span-1">{year(m.date)}</span>
              <span className="truncate text-fg" title={`${m.tournament} · ${m.level} · began ${fmtDate(m.date)}`}>{m.tournament} <span className="text-dim">· {m.level}</span></span>
              <span className="text-muted">{round(m.round)}</span>
              <span className="hidden text-muted md:block">{m.surface ?? "n/a"}</span>
              <span className="truncate"><span className="text-dim md:hidden">{m.surface ?? "Surface n/a"} · </span><Winner m={m} a={a} b={b} tone={tone} /></span>
              <span className="num col-start-2 text-[12px] text-muted md:col-start-auto md:text-right">{scoreText(m)}{m.retired && <Ret />}</span>
            </li>
          ))}
        </ul>
      )}
    </>
  );
}

/** Recorded meetings between the two selected players. History only: separate from, and never fed into, the models. */
export function HeadToHead({ a, b, surface }: { a: PlayerSummary; b: PlayerSummary; surface: Surface }) {
  const { data, error } = useH2H(a.id, b.id);
  return (
    <Card>
      <CardHeader eyebrow="Head to head" title="Previous meetings" right={<SnapshotBadge label="Recorded results" />} />
      {error ? (
        <p role="alert" className="flex items-start gap-2.5 text-[13.5px] text-muted"><CircleSlash size={16} className="mt-0.5 shrink-0 text-coral" aria-hidden="true" />{error}</p>
      ) : !data ? (
        <div aria-busy="true"><Skeleton className="mx-auto h-16 w-56" /><div className="mt-6 grid gap-3 sm:grid-cols-3"><Skeleton className="h-24" /><Skeleton className="h-24" /><Skeleton className="h-24" /></div></div>
      ) : data.total === 0 ? (
        <div className="grid place-items-center rounded-xl border border-dashed border-line px-4 py-10 text-center">
          <p className="text-[15px] font-medium">No recorded meetings</p>
          <p className="mt-1 max-w-md text-[13px] text-muted">{a.name} and {b.name} have not played each other in the data.</p>
        </div>
      ) : (
        <Body h={data} a={a} b={b} surface={surface} />
      )}
      {data && (
        <p className="mt-5 text-[11.5px] leading-relaxed text-dim">
          Recorded results from the historical dataset, covering tournaments from {fmtDate(data.dataStart)} to {fmtDate(data.dataEnd)}: meetings before or after that period
          are missing. Years are tournament start dates; the data has no match dates. Walkovers, defaults and abandoned matches are excluded; retirements count as wins and
          are marked RET. Scores are shown from the winner&apos;s side. Head-to-head is not an input to Model D, Model C or Elo, and a past record is not a forecast.
        </p>
      )}
    </Card>
  );
}
