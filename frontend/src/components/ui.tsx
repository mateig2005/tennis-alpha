import { Database } from "lucide-react";
import type { ReactNode } from "react";

export function Card({ children, className = "" }: { children: ReactNode; className?: string }) {
  return <section className={`card p-5 sm:p-6 ${className}`}>{children}</section>;
}

export function CardHeader({ eyebrow, title, right }: { eyebrow?: string; title: string; right?: ReactNode }) {
  return (
    <header className="mb-5 flex flex-wrap items-start justify-between gap-3">
      <div>
        {eyebrow && <p className="eyebrow mb-1.5">{eyebrow}</p>}
        <h2 className="text-[17px] font-medium tracking-tight text-fg">{title}</h2>
      </div>
      {right}
    </header>
  );
}

/** Marks a card as real model output or real data from the historical snapshot. It is never live data. */
export function SnapshotBadge({ label = "Historical data" }: { label?: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-sage/40 px-2 py-0.5 text-[10.5px] font-medium tracking-wider text-sage uppercase"
      title="Computed from the historical snapshot. Not live data.">
      <Database size={11} aria-hidden="true" />
      {label}
    </span>
  );
}

export function RealData({ source }: { source: string }) {
  return (
    <span className="inline-flex items-center gap-1.5 rounded-full border border-mint/30 bg-mint/5 px-2 py-0.5 text-[10.5px] font-medium tracking-wider text-mint uppercase">
      Real results<span className="text-sage normal-case tracking-normal">· {source}</span>
    </span>
  );
}

export function Skeleton({ className = "" }: { className?: string }) {
  return <div className={`animate-pulse rounded-lg bg-raised ${className}`} />;
}

export function PageIntro({ eyebrow, title, children }: { eyebrow: string; title: string; children: ReactNode }) {
  return (
    <div className="mb-8 max-w-3xl">
      <p className="eyebrow mb-2 text-sage">{eyebrow}</p>
      <h1 className="text-3xl font-medium tracking-tight sm:text-4xl">{title}</h1>
      <p className="mt-3 text-[15px] leading-relaxed text-muted">{children}</p>
    </div>
  );
}

/** Small option switch for charts (a radio group: arrow keys move between options). */
export function Segmented<T extends string>({ label, options, value, onChange, disabled = [] }: { label: string; options: { value: T; label: string }[]; value: T; onChange: (v: T) => void; disabled?: T[] }) {
  return (
    <div role="radiogroup" aria-label={label} className="inline-flex rounded-lg border border-line bg-raised p-0.5">
      {options.map((o, i) => (
        <button key={o.value} type="button" role="radio" aria-checked={o.value === value} tabIndex={o.value === value ? 0 : -1}
          disabled={disabled.includes(o.value)} onClick={() => onChange(o.value)}
          onKeyDown={(e) => {
            const step = e.key === "ArrowRight" ? 1 : e.key === "ArrowLeft" ? -1 : 0;
            if (!step) return;
            e.preventDefault();
            const next = options[(i + step + options.length) % options.length].value;
            if (!disabled.includes(next)) onChange(next);
            const group = e.currentTarget.parentElement;
            requestAnimationFrame(() => group?.querySelector<HTMLButtonElement>('[aria-checked="true"]')?.focus());
          }}
          className={`num rounded-md px-2.5 py-1 text-[11.5px] transition-colors ${o.value === value ? "bg-mint/12 text-mint" : "text-muted hover:text-fg"} disabled:cursor-not-allowed disabled:opacity-40 disabled:hover:text-muted`}>
          {o.label}
        </button>
      ))}
    </div>
  );
}
