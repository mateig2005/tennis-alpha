import { act, renderHook } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import { ALCARAZ, SINNER } from "../test/fixtures";
import { DEFAULT_SELECTION, setSelection, useSelection } from "./selection";

const KEY = "tennis-alpha:selection:v2";

describe("shared matchup selection", () => {
  it("starts on real players, identified by id", () => {
    expect(DEFAULT_SELECTION).toEqual({ aId: SINNER, bId: ALCARAZ, surface: "Hard", bestOf: 3 });
    expect(renderHook(() => useSelection()).result.current).toEqual(DEFAULT_SELECTION);
  });

  it("updates every subscriber and survives a reload", () => {
    const one = renderHook(() => useSelection());
    const two = renderHook(() => useSelection()); // e.g. Match Prediction and Analytics
    act(() => setSelection({ bId: "sackmann:103819", surface: "Grass" }));
    const expected = { aId: SINNER, bId: "sackmann:103819", surface: "Grass", bestOf: 3 };
    expect(one.result.current).toEqual(expected);
    expect(two.result.current).toEqual(expected);
    expect(JSON.parse(window.localStorage.getItem(KEY)!)).toEqual(expected);
    expect(renderHook(() => useSelection()).result.current).toEqual(expected); // a fresh mount reads the stored value
  });

  it("never allows the same player on both sides", () => {
    const { result } = renderHook(() => useSelection());
    act(() => setSelection({ bId: SINNER }));
    expect(result.current).toEqual(DEFAULT_SELECTION);
  });

  it("swaps players in one step", () => {
    const { result } = renderHook(() => useSelection());
    act(() => setSelection({ aId: ALCARAZ, bId: SINNER }));
    expect(result.current).toMatchObject({ aId: ALCARAZ, bId: SINNER });
  });

  it("keeps players saved before the match format existed", () => {
    window.localStorage.setItem(KEY, JSON.stringify({ aId: ALCARAZ, bId: SINNER, surface: "Clay" }));
    expect(renderHook(() => useSelection()).result.current).toEqual({ aId: ALCARAZ, bId: SINNER, surface: "Clay", bestOf: 3 });
    window.localStorage.setItem(KEY, JSON.stringify({ aId: ALCARAZ, bId: SINNER, surface: "Clay", bestOf: 4 }));
    expect(renderHook(() => useSelection()).result.current).toEqual(DEFAULT_SELECTION);
  });

  it("ignores a corrupt or invalid stored value", () => {
    window.localStorage.setItem(KEY, JSON.stringify({ aId: SINNER, bId: SINNER, surface: "Carpet" }));
    expect(renderHook(() => useSelection()).result.current).toEqual(DEFAULT_SELECTION);
    window.localStorage.setItem(KEY, "{not json");
    expect(renderHook(() => useSelection()).result.current).toEqual(DEFAULT_SELECTION);
  });
});
