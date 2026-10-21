"use client";

import { BookOpen, FlaskConical, LineChart, Menu, Swords, X } from "lucide-react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import { useState } from "react";
import { fmtDate, useMeta } from "@/lib/hooks";
import { Wordmark } from "./Logo";

const LINKS = [
  { href: "/", label: "Match Prediction", icon: Swords },
  { href: "/analytics", label: "Analytics", icon: LineChart },
  { href: "/model-lab", label: "Model Lab", icon: FlaskConical },
  { href: "/methodology", label: "Methodology", icon: BookOpen },
];

export function Nav() {
  const path = usePathname();
  const [open, setOpen] = useState(false);
  const meta = useMeta();
  const snapshot = meta ? `Snapshot ${fmtDate(meta.snapshotDate)}` : "Historical snapshot";
  return (
    <header className="sticky top-0 z-40 border-b border-line bg-ink/85 backdrop-blur-md">
      <div className="mx-auto flex h-16 max-w-7xl items-center justify-between gap-4 px-4 sm:px-6">
        <Link href="/" aria-label="Tennis Alpha home" className="shrink-0">
          <Wordmark />
        </Link>
        <nav aria-label="Main" className="hidden items-center gap-1 lg:flex">
          {LINKS.map(({ href, label }) => {
            const active = path === href;
            return (
              <Link key={href} href={href} aria-current={active ? "page" : undefined}
                className={`relative rounded-lg px-3.5 py-2 text-[13.5px] transition-colors ${active ? "text-fg" : "text-muted hover:text-fg"}`}>
                {label}
                {active && <span className="absolute inset-x-3.5 -bottom-[13px] h-px bg-mint shadow-[0_0_12px_#39ff9c]" />}
              </Link>
            );
          })}
        </nav>
        <div className="flex items-center gap-2">
          <span className="hidden items-center gap-2 rounded-full border border-line bg-panel px-3 py-1.5 text-[11.5px] text-muted sm:flex">
            <span className="h-1.5 w-1.5 rounded-full bg-sage" aria-hidden="true" />
            {snapshot} <span className="text-dim">· not live</span>
          </span>
          <button type="button" onClick={() => setOpen((o) => !o)} aria-expanded={open} aria-controls="mobile-nav" aria-label={open ? "Close menu" : "Open menu"}
            className="grid h-10 w-10 place-items-center rounded-lg border border-line text-muted hover:text-fg lg:hidden">
            {open ? <X size={18} /> : <Menu size={18} />}
          </button>
        </div>
      </div>
      {open && (
        <nav id="mobile-nav" aria-label="Main" className="border-t border-line px-4 py-3 lg:hidden">
          {LINKS.map(({ href, label, icon: Icon }) => (
            <Link key={href} href={href} onClick={() => setOpen(false)} aria-current={path === href ? "page" : undefined}
              className={`flex items-center gap-3 rounded-lg px-3 py-3 text-[15px] ${path === href ? "bg-raised text-fg" : "text-muted"}`}>
              <Icon size={17} className={path === href ? "text-mint" : ""} aria-hidden="true" />
              {label}
            </Link>
          ))}
          <p className="px-3 pt-3 text-[11.5px] text-dim">{snapshot} · not live data</p>
        </nav>
      )}
    </header>
  );
}
