"use client";

// One shared matchup (player A, player B, surface) for every page. It lives in localStorage, so it survives
// navigation and reloads, and useSyncExternalStore keeps all pages in step without hydration mismatches.
// Players are stored by their stable ids from the API, never by name.
import { useSyncExternalStore } from "react";
import { SURFACES, type Surface } from "./types";

export interface Selection { aId: string; bId: string; surface: Surface; bestOf: 3 | 5 } // bestOf: match format, used by V2 only

const KEY = "tennis-alpha:selection:v2"; // v2: real player ids (v1 held the prototype's fictional ids)
export const DEFAULT_SELECTION: Selection = { aId: "sackmann:206173", bId: "sackmann:207989", surface: "Hard", bestOf: 3 }; // Sinner, Alcaraz
const listeners = new Set<() => void>();
let cache: { raw: string | null; value: Selection } = { raw: null, value: DEFAULT_SELECTION };

function valid(v: unknown): v is Selection {
  const s = v as Selection;
  const id = (x: unknown) => typeof x === "string" && x.length > 0;
  return !!s && id(s.aId) && id(s.bId) && s.aId !== s.bId && SURFACES.includes(s.surface) && (s.bestOf === 3 || s.bestOf === 5);
}

function read(): Selection {
  let raw: string | null;
  try { raw = window.localStorage.getItem(KEY); } catch { return cache.value; } // storage blocked: keep the in-memory value
  if (raw === cache.raw) return cache.value;
  let value = DEFAULT_SELECTION;
  try {
    const parsed: unknown = raw ? { bestOf: 3, ...JSON.parse(raw) } : null; // entries saved before the format existed keep their players
    if (valid(parsed)) value = parsed;
  } catch { /* corrupt entry: fall back to the default */ }
  cache = { raw, value };
  return value;
}

function subscribe(listener: () => void) {
  listeners.add(listener);
  window.addEventListener("storage", listener); // other tabs
  return () => { listeners.delete(listener); window.removeEventListener("storage", listener); };
}

/** Change part of the matchup. Choosing the same player on both sides is ignored. */
export function setSelection(patch: Partial<Selection>) {
  const next = { ...read(), ...patch };
  if (!valid(next)) return;
  const raw = JSON.stringify(next);
  try { window.localStorage.setItem(KEY, raw); } catch { /* storage blocked: the in-memory value still updates */ }
  cache = { raw, value: next };
  listeners.forEach((l) => l());
}

export function useSelection(): Selection {
  return useSyncExternalStore(subscribe, read, () => DEFAULT_SELECTION);
}
