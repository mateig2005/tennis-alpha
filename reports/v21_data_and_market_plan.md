# V2.1 data availability, shot-level data plan and market benchmark plan

This report covers the approved dataset only: Sackmann ATP matches 2010-2024 in the dev file. The V2 matrix ([v2_feature_availability.md](v2_feature_availability.md)) still applies; this adds what V2.1 needed.

## 1. Availability matrix
| Feature | Status | Notes |
|---|---|---|
| Games won %, sets won %, straight sets vs deciding set, tiebreaks | **Derived reliably** from the score | The parser reads 42,244 of 42,372 scores. The 128 it rejects (Next Gen first-to-4 sets, truncated scores) are treated as missing. |
| Total points won | **Derived exactly** | Serve points won + return points won. Both come from the serve totals. |
| Serve and return points won, holds, breaks, break points created and faced | Available | Already in V2. |
| Performance vs pre-match expectation (points, games) | **Derived** (V2.1 `quality`) | The expectation comes from the V2 pre-match serve/return ratings: opponent- and surface-adjusted. |
| Winning ugly / losing well ("luck") | **Derived** | Result minus the win chance implied by the points actually won (V2 point-to-match model). |
| Statistical momentum, consistency, surface quality | **Derived** (V2.1 `qmomentum`) | Rolling, shrunk with 3 pseudo-matches. |
| Workload | **Approximate** | Sets played. Minutes are missing for 10% of matches and include implausible outliers (up to 1,146), so they are not used. |
| Retirements | Handled | They give a workload record only, never a performance record. Walkovers are already removed. |
| Unforced errors, winners, forced errors | **Unavailable** without point-level data | Not inferred. Double faults are not unforced errors. |
| Net points, rally length, forehand/backhand, shot direction | **Needs point-by-point (charted) data** | See section 2. |
| Indoor/outdoor, altitude | **Unavailable** without a venue table | Needs approval. |
| Exact rest, in-tournament fatigue, weather | **Needs real match timestamps** | The source has tournament start dates only. |
| Bookmaker odds | **Unavailable** (no approved source) | Valuebetennis is excluded until its licence is resolved. |

## 2. Match Charting Project: integration plan, not integrated
- **What it is:** volunteer, shot-by-shot charting of professional matches by Jeff Sackmann and contributors. It includes serve direction, rally length, net points, winners and errors by shot type.
- **Licence:** CC BY-NC-SA 4.0, the same licence as our match data. The public repository states this (checked 2026-10-19). It is compatible with this non-commercial project if attribution is given and derived data is shared alike.
- **Size:** "over 5,000" charted matches per the repository, across many seasons and both tours. That is a small fraction of our roughly 42,000 ATP matches.
- **Representativeness: the main problem.** Volunteers chart what interests them: top players, Slams and finals. A typical first-round match has no chart. A feature that exists only for famous matches would mostly encode *which matches were charted*.
- **Subjectivity:** whether an error is "forced" or "unforced" is the charter's judgement. Definitions can differ between charters and years.

**Plan if approved:**
1. Download the files into their own folder, pinned to a commit, and record checksums.
2. Map player identities to our ids. The V1 identity checker already exists.
3. Measure coverage by player, level and year *before* building any feature.
4. Use charted statistics only as **player-level shrunk profiles** (for example, a career-to-date unforced-error rate pulled strongly toward the tour average), never per match.
5. Add a coverage flag and evaluate separately on charted and uncharted matches.
6. Report charter-dependence as a limitation.
7. Run the same leakage rule as everything else: a chart counts only after its tournament's release day.

## 3. Market benchmark: framework built, no prices used
Built and tested in `src/v2/odds.py`:
- fair odds from a probability;
- the overround (bookmaker margin);
- no-vig probabilities, by two methods: proportional, and the power method (which takes more margin from the longshot);
- `benchmark()`, which compares model and market log loss, Brier score and accuracy, the average disagreement, and losses by disagreement size.

`benchmark()` **refuses** any odds not timestamped strictly before the match start, and any duplicate match.

**Rules for a future odds source:**
1. The licence must allow research use, and the data must have provenance (bookmaker, timestamp, opening or closing price).
2. Every row must match a match by tournament, date window, round and both player ids. Unmatched rows are reported, not guessed.
3. Odds are a **benchmark only**. They are never a model input and never a tuning target. A model copying the market would show no independent value.
4. Questions to answer:
   - Is our log loss above or below the no-vig market's?
   - When model and market disagree by more than 5 or 10 points, who was right more often?
   - Do favourite/underdog and top-10 patterns differ?
5. The benchmark needs a period the models have not been tuned on. That is the reserved prospective period, not 2025-2026.

**Sinner-Alcaraz:** only fair odds can be shown (Model D 60.7% → 1.65 / 2.54). No market price is available, and none was invented.
