import { Clock, Database, GitBranch, Layers, Lock, Scale } from "lucide-react";
import { Card, PageIntro } from "@/components/ui";

export const metadata = { title: "Methodology · Tennis Alpha" };

const SECTIONS: { icon: typeof Scale; title: string; text: string; link?: { href: string; label: string } }[] = [
  { icon: Scale, title: "How the model works",
    text: "Each player has an Elo rating that rises after wins and falls after losses. Model D is a Logistic Regression that combines the rating gap with ranking, recent form and opponent-adjusted form, and turns the total into a win probability. Model C is the same without the opponent-adjusted inputs. Elo alone is the benchmark." },
  { icon: Layers, title: "Feature engineering",
    text: "Overall and surface Elo; ATP ranking and points; win rate over the last 5 and 10 matches and the last 30, 90 and 365 days; surface form; rest and workload; exponentially decayed form with a 60-day half-life; opponent-adjusted form, the average of result minus Elo expectation. Every input is a difference between the two players, so swapping them flips the probability exactly." },
  { icon: Database, title: "Data sources and licensing",
    text: "Results and serve statistics to May 2026: Jeff Sackmann / Tennis Abstract (CC BY-NC-SA 4.0). From June 2026: TennisMyLife, used with the publisher's permission for this non-commercial project, including daily automatic updates; upstream source the ATP. Non-commercial use only; the raw data is not redistributed and cannot be rebuilt from this app or its code. No betting odds are used anywhere.",
    link: { href: "https://stats.tennismylife.org", label: "Data: stats.tennismylife.org" } },
  { icon: GitBranch, title: "Chronological backtesting",
    text: "2010 to 2014 warms up the ratings, 2015 to 2023 trains the models, 2024 validates them. 2025 and early 2026 were locked away, then scored once after every setting had been frozen and committed. Differences are reported with intervals that resample whole tournaments." },
  { icon: Lock, title: "Data-leakage safeguards",
    text: "The source has tournament start dates, not match dates. A result is therefore used only after its tournament's maximum possible length, even for later rounds of the same event. Automated tests flip every future result and confirm that earlier features and predictions do not change." },
  { icon: Clock, title: "Limitations and data freshness",
    text: "Player data is refreshed daily on this Mac from TennisMyLife, and a tournament is added only once it must have finished, so results arrive a few days late. Model coefficients stay frozen. There is no real-time data and no live forecast. Model D does not model injuries, best-of-five or head-to-head; V2 is experimental. The measured edges are small." },
];

export default function Methodology() {
  return (
    <>
      <PageIntro eyebrow="Methodology" title="How it works, and where it stops">
        The research behind TENNIS ALPHA was built to be checked: a written plan before each step, a single independent test, and limits stated plainly.
      </PageIntro>
      <div className="grid gap-5 md:grid-cols-2">
        {SECTIONS.map(({ icon: Icon, title, text, link }) => (
          <Card key={title}>
            <div className="mb-4 grid h-10 w-10 place-items-center rounded-xl border border-line bg-raised text-mint"><Icon size={18} aria-hidden="true" /></div>
            <h2 className="text-[17px] font-medium tracking-tight">{title}</h2>
            <p className="mt-2 text-[13.5px] leading-relaxed text-muted">{text}</p>
            {link && <a href={link.href} target="_blank" rel="noopener noreferrer" className="mt-3 inline-block text-[13px] text-mint underline-offset-4 hover:underline">{link.label}</a>}
          </Card>
        ))}
      </div>
    </>
  );
}
