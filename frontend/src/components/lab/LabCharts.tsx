"use client";

import { Bar, BarChart, CartesianGrid, Legend, Line, LineChart, ReferenceLine, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import { BACKTEST, CALIBRATION_2025 } from "@/lib/backtest";
import { ChartTooltip, MINT, legend } from "../match/Charts";

const AXIS = { stroke: "#5C6B64", fontSize: 11, tickLine: false, axisLine: false } as const;

export function ImprovementChart() {
  const gain = (period: number, model: string) => {
    const rows = BACKTEST[period].rows;
    return +(rows.find((r) => r.model === "Elo")!.logLoss - rows.find((r) => r.model === model)!.logLoss).toFixed(4);
  };
  const data = ["Model D", "Model C"].map((m) => ({ name: m, y2025: gain(0, m), y2026: gain(1, m) }));
  return (
    <ResponsiveContainer width="100%" height={260}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 0, left: -4 }} barGap={6} barCategoryGap="32%">
        <defs>
          <pattern id="hatch-lab" width="6" height="6" patternUnits="userSpaceOnUse" patternTransform="rotate(45)">
            <rect width="6" height="6" fill={MINT} opacity="0.18" /><line x1="0" y1="0" x2="0" y2="6" stroke={MINT} strokeWidth="2.5" />
          </pattern>
        </defs>
        <CartesianGrid stroke="#1E2A25" vertical={false} />
        <XAxis dataKey="name" {...AXIS} />
        <YAxis {...AXIS} tickFormatter={(v: number) => v.toFixed(3)} />
        <Tooltip content={<ChartTooltip format={(v) => v.toFixed(4)} />} cursor={{ fill: "rgb(255 255 255 / 0.03)" }} />
        <Legend {...legend} />
        <Bar name="2025 (full season)" dataKey="y2025" fill={MINT} radius={[4, 4, 0, 0]} maxBarSize={44} />
        <Bar name="2026 (Jan to Roland Garros)" dataKey="y2026" fill="url(#hatch-lab)" stroke={MINT} strokeWidth={1} radius={[4, 4, 0, 0]} maxBarSize={44} />
      </BarChart>
    </ResponsiveContainer>
  );
}

export function CalibrationChart() {
  // Each line gets its own data key; sharing one key ("y") gave React two children with the same key in the tooltip and legend.
  const modelD = CALIBRATION_2025.modelD.map(([x, v]) => ({ x, modelD: v }));
  const elo = CALIBRATION_2025.elo.map(([x, v]) => ({ x, elo: v }));
  const ticks = [0, 0.25, 0.5, 0.75, 1];
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart margin={{ top: 8, right: 12, bottom: 18, left: -8 }}>
        <CartesianGrid stroke="#1E2A25" />
        <XAxis type="number" dataKey="x" domain={[0, 1]} ticks={ticks} {...AXIS} tickFormatter={(v: number) => `${v * 100}%`}
          label={{ value: "Predicted probability", position: "insideBottom", offset: -10, fill: "#8B9A93", fontSize: 11 }} />
        <YAxis type="number" domain={[0, 1]} ticks={ticks} {...AXIS} tickFormatter={(v: number) => `${v * 100}%`} />
        <Tooltip content={<ChartTooltip format={(v) => `${(v * 100).toFixed(1)}%`} />} labelFormatter={() => ""} />
        <Legend {...legend} verticalAlign="top" />
        <ReferenceLine segment={[{ x: 0, y: 0 }, { x: 1, y: 1 }]} stroke="#5C6B64" strokeDasharray="4 4" />
        <Line name="Model D" data={modelD} dataKey="modelD" stroke={MINT} strokeWidth={2} dot={{ r: 3.5, fill: MINT }} isAnimationActive={false} />
        <Line name="Elo" data={elo} dataKey="elo" stroke="#C9D3CE" strokeWidth={2} strokeDasharray="6 4" dot={{ r: 3, fill: "#080D0B", stroke: "#C9D3CE" }} isAnimationActive={false} />
      </LineChart>
    </ResponsiveContainer>
  );
}
