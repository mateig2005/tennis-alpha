import type { Metadata } from "next";
import { Geist, Geist_Mono } from "next/font/google";
import { Suspense } from "react";
import { Nav } from "@/components/Nav";
import "./globals.css";

const geistSans = Geist({ variable: "--font-geist-sans", subsets: ["latin"] });
const geistMono = Geist_Mono({ variable: "--font-geist-mono", subsets: ["latin"] });

export const metadata: Metadata = {
  title: "Tennis Alpha · Precision in every point",
  description: "ATP match win probabilities from a frozen research model, as of a historical data snapshot. Not live data.",
};

export default function RootLayout({ children }: LayoutProps<"/">) {
  return (
    <html lang="en" className={`${geistSans.variable} ${geistMono.variable} h-full antialiased`}>
      <body className="flex min-h-full flex-col font-sans">
        <a href="#main" className="sr-only focus:not-sr-only focus:absolute focus:left-4 focus:top-4 focus:z-50 focus:rounded-lg focus:bg-mint focus:px-3 focus:py-2 focus:text-ink">
          Skip to content
        </a>
        <Suspense fallback={<div className="h-16 border-b border-line" />}>
          <Nav />
        </Suspense>
        <main id="main" className="mx-auto w-full max-w-7xl flex-1 px-4 py-8 sm:px-6 sm:py-10">
          {children}
        </main>
        <footer className="border-t border-line">
          <div className="mx-auto flex max-w-7xl flex-col gap-2 px-4 py-6 text-[12px] leading-relaxed text-dim sm:px-6 md:flex-row md:items-center md:justify-between">
            <p>TENNIS ALPHA is a research prototype. It shows no live ATP data and is not betting advice.</p>
            <p>Match data in the research: Jeff Sackmann / Tennis Abstract, CC BY-NC-SA 4.0.</p>
          </div>
        </footer>
      </body>
    </html>
  );
}
