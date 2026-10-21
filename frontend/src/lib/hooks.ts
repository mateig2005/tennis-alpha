"use client";

import { useEffect, useState } from "react";
import { ApiError, fetchH2H, fetchMatchup, fetchMeta, fetchPlayers } from "./api";
import { DEFAULT_SELECTION, setSelection, useSelection } from "./selection";
import type { HeadToHead, Matchup, Meta, PlayerSummary } from "./types";

/** The searchable player list (fetched once per session). */
export function usePlayers() {
  const [state, set] = useState<{ players?: PlayerSummary[]; error?: string }>({});
  useEffect(() => {
    let live = true;
    fetchPlayers().then((players) => live && set({ players })).catch((e: Error) => live && set({ error: e.message }));
    return () => { live = false; };
  }, []);
  return state;
}

/** Snapshot dates and dataset size, for the "not live" notices. */
export function useMeta(): Meta | undefined {
  const [meta, set] = useState<Meta>();
  useEffect(() => {
    let live = true;
    fetchMeta().then((m) => live && set(m)).catch(() => { /* the pages show the service error; the notice just stays generic */ });
    return () => { live = false; };
  }, []);
  return meta;
}

/** The current matchup with real model output. `data` is undefined while the request for the current selection is in flight. */
export function useMatchup() {
  const selection = useSelection();
  const { aId, bId, surface, bestOf } = selection;
  const key = `${aId}|${bId}|${surface}|${bestOf}`;
  const [result, setResult] = useState<{ key: string; data?: Matchup; error?: string } | null>(null);
  useEffect(() => {
    let live = true;
    fetchMatchup(aId, bId, surface, bestOf)
      .then((data) => live && setResult({ key, data }))
      .catch((e: ApiError) => {
        if (!live) return;
        const isDefault = aId === DEFAULT_SELECTION.aId && bId === DEFAULT_SELECTION.bId;
        if (e.status === 404 && !isDefault) setSelection(DEFAULT_SELECTION); // a stored player no longer exists: fall back, do not guess
        else setResult({ key, error: e.message });
      });
    return () => { live = false; };
  }, [key, aId, bId, surface, bestOf]);
  const current = result?.key === key ? result : null; // a result for an older selection counts as "still loading"
  return { ...selection, data: current?.data, error: current?.error, loading: !current };
}

/** Recorded meetings for the current pair. `data` is undefined while the request for this pair is in flight. */
export function useH2H(aId: string, bId: string) {
  const key = `${aId}|${bId}`;
  const [result, setResult] = useState<{ key: string; data?: HeadToHead; error?: string } | null>(null);
  useEffect(() => {
    let live = true;
    fetchH2H(aId, bId).then((data) => live && setResult({ key, data })).catch((e: Error) => live && setResult({ key, error: e.message }));
    return () => { live = false; };
  }, [key, aId, bId]);
  const current = result?.key === key ? result : null;
  return { data: current?.data, error: current?.error };
}

export const fmtDate = (iso: string) =>
  new Intl.DateTimeFormat("en-GB", { day: "numeric", month: "short", year: "numeric", timeZone: "UTC" }).format(new Date(iso));
