// The mark: a tennis ball whose seam is a logistic curve (how a rating gap becomes a probability), with a dot at 50%.
export function LogoMark({ size = 28 }: { size?: number }) {
  return (
    <svg width={size} height={size} viewBox="0 0 32 32" fill="none" aria-hidden="true">
      <circle cx="16" cy="16" r="13.25" stroke="#39FF9C" strokeWidth="1.5" opacity="0.9" />
      <path d="M3.5 22.5C10 22.5 12.5 19 16 16s6-6.5 12.5-6.5" stroke="#39FF9C" strokeWidth="1.75" strokeLinecap="round" />
      <circle cx="16" cy="16" r="2.4" fill="#080D0B" stroke="#F3F7F5" strokeWidth="1.5" />
    </svg>
  );
}

export function Wordmark({ compact = false }: { compact?: boolean }) {
  return (
    <span className="flex items-center gap-2.5">
      <LogoMark />
      <span className="flex flex-col leading-none">
        <span className="text-[15px] font-semibold tracking-[0.22em] text-fg">
          TENNIS<span className="ml-[0.35em] text-mint">ALPHA</span>
        </span>
        {!compact && <span className="mt-1 text-[9.5px] tracking-[0.2em] text-dim uppercase">Precision in every point</span>}
      </span>
    </span>
  );
}
