import { Clock } from "lucide-react";
import Image from "next/image";
import type { Player, PlayerSummary, Surface } from "@/lib/types";
import { Skeleton } from "../ui";

const initials = (name: string) => name.split(/[\s-]+/).map((w) => w[0]).slice(0, 2).join("").toUpperCase();
export type Tone = "mint" | "coral" | "neutral";

/** Activity is measured to the snapshot date from the start of the player's last tournament in the data. */
export function activity(p: PlayerSummary): { short: string; tone: "ok" | "warn" | "bad" } {
  const d = p.daysSinceLastEvent;
  if (d === null) return { short: "No match history", tone: "bad" };
  if (d > 365) return { short: `Inactive · ${d.toLocaleString("en-US")}d`, tone: "bad" };
  if (d > 180) return { short: `Stale · ${d}d`, tone: "warn" };
  return { short: `Active · ${d}d`, tone: "ok" };
}
const ACT = { ok: "text-sage border-sage/30", warn: "text-fg border-muted/40", bad: "text-coral border-coral/40" };
const RING: Record<Tone, string> = {
  mint: "border-mint/50 shadow-[0_0_36px_-10px_rgb(57_255_156/0.6)]",
  coral: "border-coral/45 shadow-[0_0_36px_-12px_rgb(255_77_90/0.5)]",
  neutral: "border-line",
};
const INK: Record<Tone, string> = { mint: "text-mint", coral: "text-coral", neutral: "text-sage" };

/** A licensed photograph when one exists; otherwise a monogram. No portrait is ever invented. */
export function Portrait({ player, tone, small = false }: { player: PlayerSummary; tone: Tone; small?: boolean }) {
  const box = small ? "h-12 w-12 rounded-xl text-[15px]" : "h-16 w-16 rounded-2xl text-xl sm:h-24 sm:w-24 sm:rounded-3xl sm:text-[30px]";
  if (player.portrait) {
    return (
      <figure className={`relative shrink-0 overflow-hidden border ${box} ${RING[tone]}`} title={`Photo: ${player.portrait.credit}`}>
        <Image src={player.portrait.src} alt={`Portrait of ${player.name}`} fill sizes="96px" className="object-cover" />
        <figcaption className="sr-only">Photo: {player.portrait.credit}</figcaption>
      </figure>
    );
  }
  return (
    <div aria-hidden="true" title="No licensed portrait available: showing a monogram"
      className={`relative grid shrink-0 place-items-center overflow-hidden border bg-raised font-medium tracking-[0.06em] ${box} ${RING[tone]} ${INK[tone]}`}>
      {/* soft light from the top-left, and a faint baseline like a court line */}
      <span className="absolute inset-0 bg-[radial-gradient(120%_90%_at_20%_0%,rgb(255_255_255/0.09),transparent_60%)]" />
      <span className="absolute inset-x-3 bottom-[22%] h-px bg-current opacity-20" />
      <span className="relative">{initials(player.name)}</span>
    </div>
  );
}

/** player: from the list (available at once). detail: from the prediction response (adds the surface rating). */
export function PlayerIdentity({ player, detail, surface, tone, align = "left" }: { player: PlayerSummary; detail?: Player; surface: Surface; tone: Tone; align?: "left" | "right" }) {
  const act = activity(player);
  const right = align === "right";
  const surfaceElo = detail?.surfaceElo[surface];
  return (
    // max-w-full: on the right the column aligns its content to the end, so without it a long name grows past its half and over the other player
    <div className={`flex min-w-0 max-w-full flex-col gap-3 sm:flex-row sm:items-center sm:gap-5 ${right ? "items-end sm:flex-row-reverse" : ""}`}>
      <Portrait player={player} tone={tone} />
      <div className={`min-w-0 max-w-full ${right ? "text-right" : ""}`}>
        <p className="line-clamp-3 text-[22px] leading-tight font-medium tracking-tight break-words sm:text-[32px] xl:text-[38px]" title={player.name}>{player.name}</p>
        <div className={`mt-1.5 flex flex-wrap gap-x-4 gap-y-0.5 text-[12.5px] text-muted ${right ? "justify-end" : ""}`}>
          <span title="Last ranking in the data, not today's">Rank <span className="num text-fg">{player.rank ? `#${player.rank}` : "none"}</span></span>
          <span>Elo <span className="num text-fg">{Math.round(player.elo)}</span></span>
          <span className="inline-flex items-center gap-1">{surface} Elo{" "}
            {detail ? <span className="num text-fg">{surfaceElo == null ? "unrated" : Math.round(surfaceElo)}</span> : <Skeleton className="inline-block h-3 w-8" />}
          </span>
        </div>
        <span title="Days from the start of the player's last tournament in the data to the snapshot date"
          className={`mt-2.5 inline-flex items-center gap-1.5 rounded-full border px-2.5 py-0.5 text-[11.5px] whitespace-nowrap ${ACT[act.tone]}`}>
          <Clock size={12} aria-hidden="true" />
          {act.short}{player.daysSinceLastEvent !== null && <span className="hidden sm:inline">{" "}before snapshot</span>}
        </span>
      </div>
    </div>
  );
}

export function IdentitySkeleton({ align = "left" }: { align?: "left" | "right" }) {
  return (
    <div className={`flex items-center gap-5 ${align === "right" ? "flex-row-reverse" : ""}`} aria-hidden="true">
      <Skeleton className="h-16 w-16 rounded-2xl sm:h-24 sm:w-24 sm:rounded-3xl" />
      <div className={`flex flex-col gap-2 ${align === "right" ? "items-end" : ""}`}><Skeleton className="h-8 w-40 sm:w-56" /><Skeleton className="h-3 w-32" /></div>
    </div>
  );
}
