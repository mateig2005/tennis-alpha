# TENNIS ALPHA frontend: architecture and design plan

**Status: connected to the real models.** Match Prediction and Analytics show real output from the Python API; Model Lab shows the real backtest results copied from `../reports/phase6_backtest.md`. Nothing is simulated, and nothing is live: everything is as of the historical snapshot.

## Architecture
```
browser  ->  Next.js (localhost:3100)  --/api/* proxy-->  FastAPI (127.0.0.1:8000, ../src/api.py)  ->  ../src/predictor.py
                                                                                                   (same Predictor as Streamlit)
frontend/                 Next.js 16 (App Router) + TypeScript + Tailwind 4 + Motion + Recharts + Lucide
  src/app/                routes: / (Match Prediction), /analytics, /model-lab, /methodology
  src/components/         brand, navigation, controls, match page sections, charts
  src/lib/types.ts        data contract with the API
  src/lib/api.ts          the ONLY place that calls the API (and converts its snake_case to camelCase)
  src/lib/hooks.ts        usePlayers, useMeta, useMatchup: loading, error and stale-selection handling
  src/lib/selection.ts    one shared matchup (player ids + surface) for all pages, kept in localStorage
  src/lib/backtest.ts     real 2025/2026 results from the committed report
  src/**/*.test.ts(x)     Vitest + Testing Library
```
The API loads the trained artifact once at startup and trains nothing. Players are identified by stable ids (`sackmann:206173`), never by name. The Python research code is not modified by anything in this folder.

## Design system
- **Colour:** background `#080D0B`; panels `#0D1512` / `#121C18` with 1px `#1E2A25` borders; text `#F3F7F5`, muted `#8B9A93`; accent green `#39FF9C`; red `#FF4D5A`.
- **Meaning of colour:** green = favourite / positive, red = underdog / negative. Never colour alone: every use also has a label, an icon, a line style (solid vs dashed) or a hatch pattern.
- **Type:** Geist Sans for interface text; Geist Mono with tabular figures for every number (terminal feel, digits do not jump while animating).
- **Surfaces:** flat dark panels, hairline borders, a faint top highlight; glow only on the primary probability and the active controls.
- **Motion:** 200-500 ms eases on probability, bar and card entry; disabled when the system asks for reduced motion.
- **Logo:** a tennis-ball outline whose seam is a logistic curve (the shape that turns a rating gap into a probability), with a dot at the 50% point.

## Integrity rules carried into the UI
- A persistent snapshot notice on every page with model output: historical snapshot, not a live forecast, no real-time ATP data.
- Real players only. Every statistic is observed data or a real model input; a missing value is shown as missing ("no matches", "unrated"), never filled in.
- Players have an optional `portrait` field for a licensed photograph with a credit; none exists, so every player shows a monogram.
- Probability and certainty are shown separately; an unreliable prediction (inactive player, no history) is dimmed and explained.
- Feature contributions are the trained Model D coefficients times the standardised differences, and they sum exactly to the displayed log-odds.
- The Elo chart plots recorded rating steps; ratings are held between steps, never interpolated.
- Rankings are each player's last ranking in the data, labelled as such, not today's.
- Head-to-head shows recorded meetings only, states the period the data covers (2010 to the cutoff), keeps one colour per player inside the section, and is labelled as history: not a model input and not a forecast.
- No betting features, no odds, no Valuebetennis data.
