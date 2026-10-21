import { ChevronLeft, ChevronRight, Minus } from "lucide-react";
import type { ReactNode } from "react";
import type { Player, Surface, WinRecord } from "@/lib/types";

const pct = (v: number) => `${Math.round(v * 100)}%`;
const record = (r: WinRecord) => (r.rate === null ? "no matches" : `${pct(r.rate)} (${r.matches})`);

function Last5({ results, right }: { results: Player["last5"]; right?: boolean }) {
  if (!results.length) return <span className="text-dim">no matches</span>;
  return (
    <span className={`inline-flex gap-1 ${right ? "justify-end" : ""}`} aria-label={`Last ${results.length}: ${results.join(" ")}`}>
      {results.map((r, i) => (
        <span key={i} aria-hidden="true"
          className={`num grid h-6 w-6 place-items-center rounded-md border text-[11px] ${r === "W" ? "border-mint/50 bg-mint/10 text-mint" : "border-coral/40 text-coral"}`}>
          {r}
        </span>
      ))}
    </span>
  );
}

interface Row { label: string; a: ReactNode; b: ReactNode; lead: -1 | 0 | 1 } // lead: 1 = A ahead, -1 = B ahead

/** Every value here is observed data or a real model input from the API. Missing values stay missing. */
export function StatCompare({ a, b, surface }: { a: Player; b: Player; surface: Surface }) {
  // A missing value on either side means no comparison is drawn.
  const cmp = (x: number | null, y: number | null, higherIsBetter = true): Row["lead"] => (x === null || y === null || x === y ? 0 : (x > y) === higherIsBetter ? 1 : -1);
  const wins = (p: Player) => (p.last5.length ? p.last5.filter((r) => r === "W").length : null);
  const days = (p: Player) => (p.daysSinceLastEvent === null ? "never" : `${p.daysSinceLastEvent.toLocaleString("en-US")}d`);
  const rating = (v: number | null) => (v === null ? "unrated" : Math.round(v));
  const form = (label: string, k: "last10" | "d30" | "d90" | "d365"): Row => ({ label, a: record(a.form[k]), b: record(b.form[k]), lead: cmp(a.form[k].rate, b.form[k].rate) });
  const rows: Row[] = [
    { label: "Overall Elo", a: Math.round(a.elo), b: Math.round(b.elo), lead: cmp(a.elo, b.elo) },
    { label: `${surface} Elo`, a: rating(a.surfaceElo[surface]), b: rating(b.surfaceElo[surface]), lead: cmp(a.surfaceElo[surface], b.surfaceElo[surface]) },
    { label: "Ranking (last in data)", a: a.rank ? `#${a.rank}` : "none", b: b.rank ? `#${b.rank}` : "none", lead: cmp(a.rank, b.rank, false) },
    { label: "Last 5 matches", a: <Last5 results={a.last5} />, b: <Last5 results={b.last5} right />, lead: cmp(wins(a), wins(b)) },
    form("Last 10 win rate", "last10"), form("Form, 30 days", "d30"), form("Form, 90 days", "d90"), form("Form, 365 days", "d365"),
    { label: `Win rate on ${surface.toLowerCase()}, 365 days`, a: record(a.surfaceForm[surface]), b: record(b.surfaceForm[surface]), lead: cmp(a.surfaceForm[surface].rate, b.surfaceForm[surface].rate) },
    { label: "Opponent-adjusted form", a: `${a.adjForm >= 0 ? "+" : ""}${a.adjForm.toFixed(3)}`, b: `${b.adjForm >= 0 ? "+" : ""}${b.adjForm.toFixed(3)}`, lead: cmp(a.adjForm, b.adjForm) },
    { label: "Last event before snapshot", a: days(a), b: days(b), lead: cmp(a.daysSinceLastEvent, b.daysSinceLastEvent, false) },
  ];
  return (
    <div>
      <div className="mb-2 grid grid-cols-2 items-center gap-3 text-[12.5px] text-muted sm:grid-cols-[1fr_auto_1fr]">
        <span className="truncate">{a.name}</span>
        <span className="eyebrow hidden w-48 text-center sm:block">Metric</span>
        <span className="truncate text-right">{b.name}</span>
      </div>
      <ul className="divide-y divide-line border-y border-line">
        {rows.map((r) => (
          <li key={r.label} className="grid grid-cols-2 items-center gap-x-3 gap-y-1.5 py-2.5 sm:grid-cols-[1fr_auto_1fr]">
            {/* On phones the label sits on its own line above the two values; from sm up it sits between them. */}
            <span className={`num order-2 flex items-center gap-1.5 text-[14px] sm:order-1 ${r.lead === 1 ? "font-medium text-fg" : "text-muted"}`}>{r.a}</span>
            <span className="order-1 col-span-2 flex items-center justify-center gap-1.5 text-center text-[12px] text-muted sm:order-2 sm:col-span-1 sm:w-48">
              {r.lead === 1 ? <ChevronLeft size={13} className="shrink-0 text-mint" aria-label="Player A leads" /> : <span className="w-[13px] shrink-0" />}
              <span>{r.label}</span>
              {r.lead === -1 ? <ChevronRight size={13} className="shrink-0 text-mint" aria-label="Player B leads" />
                : r.lead === 0 ? <Minus size={13} className="shrink-0 text-dim" aria-label="Level or not comparable" /> : <span className="w-[13px] shrink-0" />}
            </span>
            <span className={`num order-3 flex items-center justify-end gap-1.5 text-right text-[14px] ${r.lead === -1 ? "font-medium text-fg" : "text-muted"}`}>{r.b}</span>
          </li>
        ))}
      </ul>
      <p className="mt-3 text-[11.5px] leading-relaxed text-dim">
        The arrow points to the player ahead on each metric. Win rates are observed results, with the number of matches in brackets; windows end at the snapshot date and are
        measured from tournament start dates. Opponent-adjusted form is a Model D input: the average of (result minus the Elo-expected result) over the last 10 matches,
        pulled toward zero when matches are few. Positive means the player beat expectations.
      </p>
    </div>
  );
}
