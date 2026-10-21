"use client";

import { Check, ChevronsUpDown, Search } from "lucide-react";
import { useId, useMemo, useRef, useState } from "react";
import type { PlayerSummary } from "@/lib/types";

const MAX_SHOWN = 60; // the snapshot has about 1,800 players: show the best matches and ask for more letters
const fold = (t: string) => t.normalize("NFD").replace(/[\u0300-\u036f]/g, "").toLowerCase();

interface Props {
  label: string;
  players: PlayerSummary[] | undefined; // undefined while the list is loading
  value: string | null;
  selectedName?: string; // shown before the list has arrived
  unavailable?: boolean; // the player list could not be loaded
  takenId: string | null; // the other side's player: shown but not selectable
  onChange: (id: string) => void;
}

/** Searchable, keyboard-operable player picker (WAI-ARIA combobox pattern). */
export function PlayerSelect({ label, players, value, selectedName, unavailable = false, takenId, onChange }: Props) {
  const id = useId();
  const input = useRef<HTMLInputElement>(null);
  const [open, setOpen] = useState(false);
  const [query, setQuery] = useState("");
  const [active, setActive] = useState(0);
  const selected = players?.find((p) => p.id === value) ?? null;
  const matches = useMemo(() => {
    const words = fold(query).split(/\s+/).filter(Boolean); // every typed word must appear in the name, in any order
    return (players ?? []).filter((p) => { const name = fold(p.name); return words.every((w) => name.includes(w)); });
  }, [players, query]);
  const results = matches.slice(0, MAX_SHOWN);

  const choose = (p: PlayerSummary | undefined) => {
    if (!p || p.id === takenId) return;
    onChange(p.id);
    setOpen(false);
    setQuery("");
    input.current?.blur();
  };
  const move = (step: number) => {
    if (!results.length) return;
    let i = active;
    for (let n = 0; n < results.length; n++) {
      i = (i + step + results.length) % results.length;
      if (results[i].id !== takenId) break;
    }
    setActive(i);
  };

  return (
    <div className="relative">
      <label htmlFor={id} className="eyebrow mb-2 block">{label}</label>
      <div className={`flex items-center gap-2 rounded-xl border bg-raised px-3 transition-colors ${open ? "border-mint/60 shadow-[0_0_0_3px_rgb(57_255_156/0.12)]" : "border-line hover:border-dim"}`}>
        <Search size={15} className="shrink-0 text-dim" aria-hidden="true" />
        <input ref={input} id={id} role="combobox" aria-expanded={open} aria-controls={`${id}-list`} aria-autocomplete="list" autoComplete="off"
          aria-activedescendant={open && results[active] ? `${id}-${results[active].id}` : undefined}
          placeholder={players ? "Search player" : unavailable ? "Players unavailable" : "Loading players…"} disabled={!players} value={open ? query : selected?.name ?? selectedName ?? ""}
          onChange={(e) => { setQuery(e.target.value); setActive(0); setOpen(true); }}
          onFocus={() => { setOpen(true); setQuery(""); setActive(Math.max(0, results.findIndex((p) => p.id === value))); }}
          onBlur={() => setOpen(false)}
          onKeyDown={(e) => {
            if (e.key === "ArrowDown") { e.preventDefault(); if (open) move(1); else setOpen(true); }
            else if (e.key === "ArrowUp") { e.preventDefault(); move(-1); }
            else if (e.key === "Enter" && open) { e.preventDefault(); choose(results[active]); }
            else if (e.key === "Escape") { setOpen(false); input.current?.blur(); }
          }}
          className="h-11 w-full min-w-0 bg-transparent text-[14.5px] text-fg outline-none placeholder:text-dim focus-visible:outline-none" />
        <ChevronsUpDown size={15} className="shrink-0 text-dim" aria-hidden="true" />
      </div>
      {open && (
        <ul id={`${id}-list`} role="listbox" aria-label={label}
          className="absolute z-30 mt-2 max-h-72 w-full overflow-auto rounded-xl border border-line bg-raised p-1 shadow-[0_24px_48px_-12px_rgb(0_0_0/0.9)]">
          {results.length === 0 && <li role="presentation" className="px-3 py-6 text-center text-[13px] text-muted">No player in the snapshot matches “{query}”.</li>}
          {results.map((p, i) => {
            const taken = p.id === takenId;
            return (
              <li key={p.id} id={`${id}-${p.id}`} role="option" aria-selected={p.id === value} aria-disabled={taken}
                onMouseDown={(e) => { e.preventDefault(); choose(p); }} onMouseEnter={() => !taken && setActive(i)}
                className={`flex items-center justify-between gap-3 rounded-lg px-3 py-2.5 text-[14px] ${taken ? "cursor-not-allowed opacity-40" : "cursor-pointer"} ${i === active && !taken ? "bg-panel" : ""}`}>
                <span className="flex min-w-0 items-center gap-2">
                  <span className="w-4 shrink-0">{p.id === value && <Check size={14} className="text-mint" aria-hidden="true" />}</span>
                  <span className="truncate">{p.name}</span>
                </span>
                <span className="num shrink-0 text-[12px] text-muted">
                  {taken ? "selected opposite" : `${p.active ? "" : "inactive · "}${p.rank ? `#${p.rank}` : "unranked"} · ${Math.round(p.elo)}`}
                </span>
              </li>
            );
          })}
          {matches.length > results.length && (
            <li role="presentation" className="px-3 py-2.5 text-center text-[11.5px] text-dim">Showing {results.length} of {matches.length.toLocaleString("en-US")}. Type more letters to narrow.</li>
          )}
        </ul>
      )}
    </div>
  );
}
