import { CalibrationChart, ImprovementChart } from "@/components/lab/LabCharts";
import { Card, CardHeader, PageIntro, RealData } from "@/components/ui";
import { BACKTEST, DIFFS } from "@/lib/backtest";

export const metadata = { title: "Model Lab · Tennis Alpha" };
const SOURCE = "reports/phase6_backtest.md";

export default function ModelLab() {
  return (
    <>
      <PageIntro eyebrow="Model Lab" title="Model D vs Model C vs Elo">
        These are the results of the project&apos;s independent evaluation: the models were frozen first, then scored once on matches they had never seen.
        The models behind the Match Prediction page are the same specifications, refitted on all the data.
      </PageIntro>
      <div className="space-y-5">
        <div className="grid gap-5 lg:grid-cols-2">
          {BACKTEST.map((b) => (
            <Card key={b.period}>
              <CardHeader eyebrow={`${b.note} · ${b.matches.toLocaleString("en-US")} matches`} title={`${b.period} results`} right={<RealData source={SOURCE} />} />
              <div className="overflow-x-auto">
                <table className="w-full min-w-[420px] text-left text-[13.5px]">
                  <thead className="eyebrow">
                    <tr className="border-b border-line">
                      <th className="py-2 pr-3 font-medium">Model</th><th className="px-3 py-2 text-right font-medium">Log loss</th>
                      <th className="px-3 py-2 text-right font-medium">Brier</th><th className="px-3 py-2 text-right font-medium">Accuracy</th>
                      <th className="py-2 pl-3 text-right font-medium">Cal. slope</th>
                    </tr>
                  </thead>
                  <tbody className="divide-y divide-line">
                    {b.rows.map((r) => (
                      <tr key={r.model}>
                        <th scope="row" className="py-3 pr-3 font-medium">{r.model}<span className="ml-2 text-[11.5px] font-normal text-muted">{r.role}</span></th>
                        <td className="num px-3 text-right">{r.logLoss.toFixed(4)}</td>
                        <td className="num px-3 text-right">{r.brier.toFixed(4)}</td>
                        <td className="num px-3 text-right">{(r.accuracy * 100).toFixed(1)}%</td>
                        <td className="num pl-3 text-right">{r.slope.toFixed(2)}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </Card>
          ))}
        </div>
        <p className="text-[12px] leading-relaxed text-dim">
          Lower log loss and Brier score are better; always guessing 50% scores 0.6931 and 0.2500. A calibration slope of 1.00 is ideal; below 1 means overconfident.
          The 2026 period covers tournaments starting up to 25 May 2026 only, so it is not a full season.
        </p>

        <div className="grid gap-5 lg:grid-cols-2">
          <Card>
            <CardHeader eyebrow="Log loss" title="Improvement over Elo" right={<RealData source={SOURCE} />} />
            <ImprovementChart />
            <p className="mt-2 text-[11.5px] text-dim">Elo log loss minus the model&apos;s log loss. Higher is better.</p>
          </Card>
          <Card>
            <CardHeader eyebrow="Calibration · 2025" title="Do stated probabilities come true?" right={<RealData source={SOURCE} />} />
            <CalibrationChart />
            <p className="mt-2 text-[11.5px] text-dim">Points on the dashed diagonal mean the stated probability matched how often the player actually won.</p>
          </Card>
        </div>

        <Card>
          <CardHeader eyebrow="Paired differences in log loss" title="Is the difference real?" right={<RealData source={SOURCE} />} />
          <div className="overflow-x-auto">
            <table className="w-full min-w-[560px] text-left text-[13.5px]">
              <thead className="eyebrow">
                <tr className="border-b border-line">
                  <th className="py-2 pr-3 font-medium">Period</th><th className="px-3 py-2 font-medium">Comparison</th><th className="px-3 py-2 font-medium">Tier</th>
                  <th className="px-3 py-2 text-right font-medium">Difference</th><th className="px-3 py-2 text-right font-medium">95% interval</th>
                  <th className="py-2 pl-3 font-medium">Reading</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-line">
                {DIFFS.map((d) => (
                  <tr key={d.period + d.comparison}>
                    <td className="num py-3 pr-3">{d.period}</td><td className="px-3">{d.comparison}</td><td className="px-3 text-muted">{d.tier}</td>
                    <td className="num px-3 text-right">{d.diff > 0 ? "+" : ""}{d.diff.toFixed(4)}</td>
                    <td className="num px-3 text-right text-muted">{d.low.toFixed(4)} to {d.high > 0 ? "+" : ""}{d.high.toFixed(4)}</td>
                    <td className="pl-3">{d.high < 0 ? "First model clearly better" : "No clear difference"}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          <p className="mt-3 text-[12px] leading-relaxed text-dim">Negative means the first model is better. Intervals come from resampling whole tournaments. Model D beats Elo in both periods; Model C matches Model D.</p>
        </Card>

        <Card>
          <CardHeader eyebrow="Research limitations" title="What these results do not show" />
          <ul className="grid gap-x-8 gap-y-3 text-[13.5px] leading-relaxed text-muted md:grid-cols-2">
            <li><span className="text-fg">A small edge.</span> About 1.5 points of accuracy over Elo in 2025.</li>
            <li><span className="text-fg">No market benchmark.</span> The models were never compared with bookmaker odds.</li>
            <li><span className="text-fg">2026 is partial.</span> Hard and clay only, 29 tournaments, wide intervals.</li>
            <li><span className="text-fg">Used test sets.</span> 2025 and early 2026 can no longer validate changes to the models.</li>
            <li><span className="text-fg">Slight overconfidence.</span> Model D&apos;s calibration slope is 0.92 to 0.95; no recalibration was applied.</li>
            <li><span className="text-fg">XGBoost was tested</span> on the same features and was no better than Logistic Regression.</li>
          </ul>
        </Card>
      </div>
    </>
  );
}
