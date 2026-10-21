import { render, screen } from "@testing-library/react";
import { describe, expect, it } from "vitest";
import Methodology from "./page";

describe("Methodology page", () => {
  it("credits TennisMyLife with a visible link and keeps the Sackmann attribution", () => {
    render(<Methodology />);
    const link = screen.getByRole("link", { name: "Data: stats.tennismylife.org" });
    expect(link.getAttribute("href")).toBe("https://stats.tennismylife.org");
    expect(screen.getByText(/Jeff Sackmann \/ Tennis Abstract \(CC BY-NC-SA 4\.0\)/)).toBeTruthy();
    expect(screen.getByText(/Non-commercial use only/)).toBeTruthy();
  });
});
