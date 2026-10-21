import { AnalyticsView } from "@/components/AnalyticsView";
import { PageIntro } from "@/components/ui";

export const metadata = { title: "Analytics · Tennis Alpha" };

export default function Analytics() {
  return (
    <>
      <PageIntro eyebrow="Analytics" title="Player analytics">
        Compare two players on real ratings, results and surface records from the historical snapshot. Nothing on this page is estimated or filled in: where a player has no matches, the chart shows a gap.
      </PageIntro>
      <AnalyticsView />
    </>
  );
}
