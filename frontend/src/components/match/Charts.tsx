"use client";

import { useState } from "react";
import { Bar, BarChart, CartesianGrid, Cell, LabelList, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { SURFACES, type Contribution, type Player, type Surface } from "@/lib/types";
import { Segmented } from "../ui";

export const MINT = "#39FF9C";
export const CORAL = "#FF4D5A";
const INK = "#C9D3CE";
const AXIS = { stroke: "#5C6B64", fontSize: 11, tickLine: false, axisLine: false } as const;
const GRID = "#1E2A25";
const ANIM = { animationDuration: 450, animationEasing: "ease-out" } as const; // quick, so charts settle before the eye gets there
const TIP = "rounded-lg border border-line bg-raised px-3 py-2 text-[12px] shadow-[0_12px_32px_-8px_rgb(0_0_0/0.9)]";

interface TipItem { name?: string; value?: number | string; color?: string; dataKey?: string | number }
/** Shared dark tooltip. Recharts injects active/payload/label. */
export function ChartTooltip({ active, payload, label, format = (v: number) => String(v), footer, labelFormat, missing = "no data" }: {
  active?: boolean; payload?: TipItem[]; label?: string | number; format?: (v: number) => string; footer?: (items: TipItem[]) => string | null;
  labelFormat?: (label: string | number) => string; missing?: string;
}) {
  if (!active || !payload?.length) return null;
  const foot = footer?.(payload);
  return (
    <div className={TIP}>
      {label !== undefined && label !== "" && <p className="mb-1 text-muted">{labelFormat ? labelFormat(label) : label}</p>}
      {payload.map((i, n) => (
        <p key={`${n}-${i.name}`} className="flex items-center justify-between gap-5">
          <span className="flex items-center gap-1.5 text-muted">
            <span className="h-2 w-2 rounded-full" style={{ background: i.color?.startsWith("url") ? CORAL : i.color }} />
            {i.name}
          </span>
          <span className="num text-fg">{typeof i.value === "number" ? format(i.value) : missing}</span>
        </p>
      ))}
      {foot && <p className="num mt-1.5 border-t border-line pt-1.5 text-muted">{foot}</p>}
    </div>
  );
}

/** Hatched fill for the second player, so bars differ by texture as well as colour. */
function Hatch({ id, color }: { id: string; color: string }) {
  return (
    <defs>
      <pattern id={id} width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
        <rect width="6" height="6" fill={color} opacity="0.22" />
        <line x1="0" y1="0" x2="0" y2="6" stroke={color} strokeWidth="3" />
      </pattern>
    </defs>
  );
}

// Legend labels stay in neutral ink; the swatch beside them carries the series identity.
// itemSorter null keeps the series order (player A, then player B) instead of sorting names alphabetically.
export const legend = { iconSize: 9, itemSorter: null, wrapperStyle: { fontSize: 12, paddingTop: 8 }, formatter: (v: string) => <span style={{ color: INK }}>{v}</span> };

interface Pair { a: Player; b: Player; aFav: boolean }
type Range = "6" | "12" | "24" | "all";
type Path = Player["eloHistory"];
const DAY_MS = 86_400_000;
const monthYear = new Intl.DateTimeFormat("en-GB", { month: "short", year: "2-digit", timeZone: "UTC" });
const fullDate = new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" });

/** A player's rating on a given day: the last recorded step on or before it (ratings hold between steps; nothing is interpolated). */
export function ratingAt(path: Path, t: number): number | undefined {
  let v: number | undefined;
  for (const p of path) { if (p.t > t) break; v = p.elo; }
  return v;
}

/** One row per day on which either rating changed, inside [from, to]. */
export function eloRows(a: Path, b: Path, from: number, to: number) {
  const days = new Set<number>([from, to]);
  for (const p of [...a, ...b]) if (p.t > from && p.t < to) days.add(p.t);
  return [...days].sort((x, y) => x - y).map((t) => ({ t, a: ratingAt(a, t), b: ratingAt(b, t) }));
}

export function EloChart({ a, b, aFav, snapshot }: Pair & { snapshot: string }) {
  const [range, setRange] = useState<Range>("24");
  const [narrow, setNarrow] = useState(false);
  const to = Date.parse(snapshot);
  const first = Math.min(...[a, b].filter((p) => p.eloHistory.length).map((p) => p.eloHistory[0].t));
  if (!Number.isFinite(first)) return <Empty text="Neither player has a rating history in the data." />;
  const from = range === "all" ? first : Math.max(first, to - Number(range) * 30.4375 * DAY_MS);
  const data = eloRows(a.eloHistory, b.eloHistory, from, to);
  const all = data.flatMap((d) => [d.a, d.b]).filter((v): v is number => v !== undefined);
  const lo = Math.floor((Math.min(...all) - 20) / 50) * 50;
  const hi = Math.ceil((Math.max(...all) + 20) / 50) * 50;
  const step = hi - lo > 600 ? 200 : hi - lo > 300 ? 100 : 50;
  const yTicks = Array.from({ length: Math.floor((hi - lo) / step) + 1 }, (_, i) => lo + i * step);
  const n = narrow ? 4 : 7;
  const xTicks = Array.from({ length: n }, (_, i) => Math.round(from + ((to - from) * i) / (n - 1)));
  return (
    <div>
      <div className="mb-3 flex flex-wrap items-center justify-between gap-3">
        <p className="text-[11.5px] text-dim">Solid line: {aFav ? a.name : b.name}. Dashed line: {aFav ? b.name : a.name}.</p>
        <Segmented label="Period" value={range} onChange={setRange}
          options={[{ value: "6", label: "6M" }, { value: "12", label: "12M" }, { value: "24", label: "24M" }, { value: "all", label: "All" }]} />
      </div>
      <ResponsiveContainer width="100%" height={270} onResize={(w) => setNarrow(w < 520)}>
        <LineChart data={data} margin={{ top: 8, right: 16, bottom: 0, left: -12 }}>
          <CartesianGrid stroke={GRID} vertical={false} />
          <XAxis dataKey="t" type="number" scale="time" domain={[from, to]} ticks={xTicks} {...AXIS} tickMargin={6} tickFormatter={(t: number) => monthYear.format(t)} />
          <YAxis {...AXIS} domain={[lo, hi]} ticks={yTicks} />
          <Tooltip cursor={{ stroke: "#5C6B64", strokeDasharray: "3 3" }} content={<ChartTooltip missing="not yet rated" format={(v) => String(Math.round(v))}
            labelFormat={(t) => fullDate.format(Number(t))} footer={(items) => {
              const [x, y] = [items[0]?.value, items[1]?.value];
              if (typeof x !== "number" || typeof y !== "number") return null;
              return Math.round(x) === Math.round(y) ? "Level" : `Gap ${Math.round(Math.abs(x - y))} · ${x > y ? a.name : b.name} ahead`;
            }} />} />
          <Legend {...legend} />
          <Line name={a.name} dataKey="a" type="stepAfter" stroke={aFav ? MINT : CORAL} strokeWidth={2} strokeDasharray={aFav ? undefined : "6 4"} dot={false} activeDot={{ r: 5 }} {...ANIM} />
          <Line name={b.name} dataKey="b" type="stepAfter" stroke={aFav ? CORAL : MINT} strokeWidth={2} strokeDasharray={aFav ? "6 4" : undefined} dot={false} activeDot={{ r: 5 }} {...ANIM} />
        </LineChart>
      </ResponsiveContainer>
      <p className="mt-2 text-[11.5px] leading-relaxed text-dim">
        Real ratings from the model&apos;s Elo engine. Each step is the rating after a tournament&apos;s results were counted. Because the data has no match dates,
        a tournament is counted only after its maximum possible length, so steps appear up to two weeks after the event began.
      </p>
    </div>
  );
}

function PairBars({ data, a, b, aFav, id, unit, max, highlight, missing }: Pair & { data: { name: string; a: number | null; b: number | null }[]; id: string; unit: string; max: number; highlight?: string; missing: string }) {
  const [labels, setLabels] = useState(true); // value labels only where they fit without touching
  const fill = (isA: boolean) => (isA === aFav ? MINT : `url(#${id})`);
  const step = max / 4;
  return (
    <ResponsiveContainer width="100%" height={260} onResize={(w) => setLabels(w >= 440)}>
      <BarChart data={data} margin={{ top: 18, right: 8, bottom: 0, left: -18 }} barGap={6} barCategoryGap="26%">
        <Hatch id={id} color={CORAL} />
        <CartesianGrid stroke={GRID} vertical={false} />
        <XAxis dataKey="name" {...AXIS} interval={0} tickMargin={6}
          tick={(t) => {
            const name = String(t.payload.value);
            return (
              <text x={Number(t.x)} y={Number(t.y) + 10} textAnchor="middle" fontSize={11} fill={name === highlight ? MINT : "#8B9A93"} fontWeight={name === highlight ? 600 : 400}>
                {name}
              </text>
            );
          }} />
        <YAxis {...AXIS} domain={[0, max]} ticks={[0, step, step * 2, step * 3, max]} tickFormatter={(v: number) => `${v}${unit}`} />
        <Tooltip content={<ChartTooltip missing={missing} format={(v) => `${v}${unit}`} footer={(items) => {
          const [x, y] = [items[0]?.value, items[1]?.value];
          return typeof x === "number" && typeof y === "number" && x !== y ? `${x > y ? a.name : b.name} +${Math.abs(x - y)}${unit === "%" ? " pts" : ""}` : null;
        }} />} cursor={{ fill: "rgb(255 255 255 / 0.03)" }} />
        <Legend {...legend} />
        <Bar name={a.name} dataKey="a" fill={fill(true)} stroke={aFav ? MINT : CORAL} strokeWidth={aFav ? 0 : 1} radius={[4, 4, 0, 0]} maxBarSize={36} {...ANIM}>
          {labels && <LabelList dataKey="a" position="top" fontSize={10.5} fill={INK} formatter={(v) => (typeof v === "number" ? `${v}${unit}` : "")} />}
        </Bar>
        <Bar name={b.name} dataKey="b" fill={fill(false)} stroke={aFav ? CORAL : MINT} strokeWidth={aFav ? 1 : 0} radius={[4, 4, 0, 0]} maxBarSize={36} {...ANIM}>
          {labels && <LabelList dataKey="b" position="top" fontSize={10.5} fill={INK} formatter={(v) => (typeof v === "number" ? `${v}${unit}` : "")} />}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

const percent = (r: { rate: number | null }) => (r.rate === null ? null : Math.round(r.rate * 100));
const topOf = (data: { a: number | null; b: number | null }[], unit: number) =>
  Math.max(unit, Math.ceil(Math.max(0, ...data.flatMap((d) => [d.a ?? 0, d.b ?? 0])) / unit) * unit);

export function FormChart(p: Pair) {
  const [metric, setMetric] = useState<"rate" | "matches">("rate");
  const windows = [["30 days", "d30"], ["90 days", "d90"], ["365 days", "d365"]] as const;
  const data = metric === "rate"
    ? [{ name: "Last 10", a: percent(p.a.form.last10), b: percent(p.b.form.last10) }, ...windows.map(([name, k]) => ({ name, a: percent(p.a.form[k]), b: percent(p.b.form[k]) }))]
    : windows.map(([name, k]) => ({ name, a: p.a.form[k].matches, b: p.b.form[k].matches }));
  return (
    <div>
      <div className="mb-1 flex justify-end">
        <Segmented label="Measure" value={metric} onChange={setMetric} options={[{ value: "rate", label: "Win rate" }, { value: "matches", label: "Matches played" }]} />
      </div>
      <PairBars {...p} data={data} id="hatch-form" unit={metric === "rate" ? "%" : ""} max={metric === "rate" ? 100 : topOf(data, 20)} missing="no matches" />
    </div>
  );
}

export function SurfaceChart({ surface, ...p }: Pair & { surface: Surface }) {
  const [metric, setMetric] = useState<"rate" | "elo">("rate");
  const rating = (v: number | null) => (v === null ? null : Math.round(v));
  const data = SURFACES.map((s) => metric === "rate"
    ? { name: s, a: percent(p.a.surfaceForm[s]), b: percent(p.b.surfaceForm[s]) }
    : { name: s, a: rating(p.a.surfaceElo[s]), b: rating(p.b.surfaceElo[s]) });
  return (
    <div>
      <div className="mb-1 flex flex-wrap items-center justify-between gap-3">
        <p className="text-[11.5px] text-dim">{metric === "rate" ? "Last 365 days. " : ""}Selected surface: <span className="text-mint">{surface}</span></p>
        <Segmented label="Measure" value={metric} onChange={setMetric} options={[{ value: "rate", label: "Win rate" }, { value: "elo", label: "Surface Elo" }]} />
      </div>
      <PairBars {...p} data={data} id="hatch-surface" unit={metric === "rate" ? "%" : ""} max={metric === "rate" ? 100 : topOf(data, 400)} highlight={surface}
        missing={metric === "rate" ? "no matches" : "unrated"} />
    </div>
  );
}

export function ContributionChart({ contributions, a, b, aFav }: Pair & { contributions: Contribution[] }) {
  const [narrow, setNarrow] = useState(false);
  const data = [...contributions].sort((x, y) => y.value - x.value);
  const max = Math.ceil(Math.max(0.1, ...data.map((d) => Math.abs(d.value))) * 1.25 * 5) / 5; // round up to 0.2: clean ticks, room for labels
  const color = (v: number) => ((v >= 0) === aFav ? MINT : CORAL);
  return (
    <ResponsiveContainer width="100%" height={Math.max(330, data.length * 30)} onResize={(w) => setNarrow(w < 460)}>
      <BarChart data={data} layout="vertical" margin={{ top: 4, right: 12, bottom: 0, left: 4 }}>
        <CartesianGrid stroke={GRID} horizontal={false} />
        <XAxis type="number" {...AXIS} domain={[-max, max]} ticks={[-max, -max / 2, 0, max / 2, max]} tickFormatter={(v: number) => (v > 0 ? "+" : "") + v.toFixed(1)} />
        <YAxis type="category" dataKey="label" {...AXIS} stroke="#8B9A93" width={narrow ? 104 : 136} tick={{ fontSize: narrow ? 10.5 : 11.5, width: narrow ? 104 : 136 }} interval={0} />
        <Tooltip cursor={{ fill: "rgb(255 255 255 / 0.03)" }} content={({ active, payload }) => {
          const d = payload?.[0]?.payload as Contribution | undefined;
          if (!active || !d) return null;
          return (
            <div className={TIP}>
              <p className="text-fg">{d.label}</p>
              <p className="num mt-0.5 text-muted">{d.value >= 0 ? "+" : ""}{d.value.toFixed(3)} log-odds</p>
              <p className="mt-0.5 text-muted">Pushes toward <span className="text-fg">{d.value >= 0 ? a.name : b.name}</span></p>
            </div>
          );
        }} />
        <ReferenceLine x={0} stroke="#8B9A93" />
        <Bar dataKey="value" radius={[0, 4, 4, 0]} maxBarSize={18} {...ANIM}>
          {data.map((d) => <Cell key={d.group} fill={color(d.value)} fillOpacity={0.9} />)}
          <LabelList dataKey="value" position="right" fontSize={10.5} fill={INK} formatter={(v) => (typeof v === "number" ? `${v >= 0 ? "+" : ""}${v.toFixed(2)}` : "")} />
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  );
}

export function Empty({ text }: { text: string }) {
  return <div className="grid h-40 place-items-center rounded-xl border border-dashed border-line px-4 text-center text-[13px] text-muted">{text}</div>;
}
