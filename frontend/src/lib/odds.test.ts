import { describe, expect, it } from "vitest";
import { fairOdds, fmtOdds } from "./odds";

describe("fair odds", () => {
  it("is one over the probability", () => {
    expect(fairOdds(0.5)).toBe(2);
    expect(fmtOdds(0.6069)).toBe("1.65");
    expect(fmtOdds(1 - 0.6069)).toBe("2.54");
  });
  it("refuses probabilities that have no finite fair price", () => {
    for (const p of [0, 1, -0.1, null, undefined]) expect(fairOdds(p)).toBeNull();
    expect(fmtOdds(null)).toBe("—");
  });
});
