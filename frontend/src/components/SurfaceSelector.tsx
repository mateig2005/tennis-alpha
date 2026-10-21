"use client";

import { motion } from "motion/react";
import { useId } from "react";
import { SURFACES, type Surface } from "@/lib/types";

/** Radio group: Tab reaches it, arrow keys change the surface. */
export function SurfaceSelector({ value, onChange, hideLabel = false }: { value: Surface; onChange: (s: Surface) => void; hideLabel?: boolean }) {
  const id = useId(); // unique per instance: a previous page can stay mounted in the background, and radio names must not collide
  return (
    <fieldset>
      <legend className={hideLabel ? "sr-only" : "eyebrow mb-2 w-full text-center"}>Surface</legend>
      <div className="flex rounded-xl border border-line bg-raised p-1">
        {SURFACES.map((s) => (
          <label key={s} className="relative flex-1 cursor-pointer">
            <input type="radio" name={`surface-${id}`} value={s} checked={value === s} onChange={() => onChange(s)} className="peer sr-only" />
            {value === s && (
              <motion.span layoutId={`surface-pill-${id}`} transition={{ type: "spring", stiffness: 420, damping: 34 }}
                className="absolute inset-0 rounded-lg border border-mint/50 bg-mint/10 shadow-[0_0_18px_-4px_rgb(57_255_156/0.5)]" />
            )}
            <span className={`relative block rounded-lg px-3.5 py-2 text-center text-[13px] font-medium transition-colors peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-mint ${value === s ? "text-mint" : "text-muted hover:text-fg"}`}>
              {s}
            </span>
          </label>
        ))}
      </div>
    </fieldset>
  );
}
