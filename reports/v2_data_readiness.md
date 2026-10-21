# Data readiness: adding ATP results and statistics through October 2026

Status: **prepared, nothing merged.** A new source needs the owner's approval of its licence and provenance first. The tools are in `src/v2/sources.py`, `src/identity.py` and `src/v2/odds.py`; the tests are in `tests/test_phase12.py`.

## 1. Columns V2 needs
| Group | Columns (one row per match, winner/loser oriented) | Used by |
|---|---|---|
| **Results** (required) | match_id, source, tourney_id, event_date, round_rank, surface, level, winner_id, loser_id, retired, best_of, score | Elo, all form windows, rest/workload, H2H, best-of-5 terms |
| Ranking (recommended) | winner_rank, loser_rank, winner_points, loser_points | Model D/C/V2 ranking group. Missing values are allowed and flagged, never invented. |
| **Serve/return statistics** (needed for V2's main signal) | w_/l_ ace, df, svpt, 1stIn, 1stWon, 2ndWon, SvGms, bpSaved, bpFaced; has_stats; venue | V2 serve/return ratings, rolling rates, matchup term, pace proxy, recent serve/return |

**From results alone:** Elo (general and surface), every Model D and C feature, H2H, best-of-5 × Elo, and the Elo trend.

**Only with statistics:** opponent-adjusted serve/return ratings, rolling serve/return rates, break-point rates, the serve-vs-return matchup term, the pace proxy, and recent (60-day) serve/return ratings. These carry V2's gain over Model D.

## 2. How missing statistics affect predictions
- A match without statistics still updates Elo, form and H2H. It adds nothing to serve/return profiles.
- A player's serve/return rating fades toward the tour average with time (240-day half-life) when no new statistics arrive. A results-only update would make V2's serve/return picture gradually stale while Model D stays current. In that case the app should show a staleness note for V2 (this is not implemented yet; it should be added together with the data).
- **V2 is unavailable** for a player with no released match carrying statistics. The API already returns `unavailable` with the reason.
- The forward evaluation scores every model on the rows where V2 is available, so the comparison stays fair.

## 3. Linking player ids
1. Every source keeps its own id prefix (for example `newsource:123`). Ids are never re-used across sources.
2. `link_players` (wrapping `src/identity.py`) matches names after normalising accents and case. It returns the mapping, the **unmatched** ids and the **ambiguous** ids (two players with one name, or two new ids pointing at one known player).
3. Unmatched or ambiguous players are resolved by hand in a small committed mapping file (ids only, no data), never guessed.
4. New players with no match in our history start at the usual priors (Elo 1500, neutral form, no V2 profile).

## 4. Duplicates and inconsistent scores
- `validate_source` **raises** on structural errors: missing columns, duplicate match ids, winner = loser, unknown level, best_of not 3 or 5, unknown surface.
- It **counts** soft issues for review: unparseable scores, retirement flag vs score mismatches, rows flagged as having statistics that fail consistency checks, and possible duplicates.
- `find_duplicates` flags the same two players in the same round within 14 days, from the same tournament or from two different sources. Repeat meetings in different tournaments of one source are real rematches and are not flagged. On our own dev file the check gives 0 duplicates; an earlier version of the rule flagged 201 Madrid-then-Rome-type rematches, which is why the rule now looks at the tournament. Across sources, run it after linking ids.
- `src/identity.py:validate_extension` refuses a new source whose dates overlap the existing data. Overlaps must be reconciled first, preferably by taking a whole period from one source rather than mixing two.
- Our own data, through the same check: 128 unparseable scores (Next Gen first-to-4 sets and truncated scores), 1 retirement-flag mismatch (an abandoned match), 0 duplicates, 39,887 rows with valid statistics.

## 5. What exact match dates would change
The canonical table already supports real match days: set `event_date` to the match day and `round_rank` to 0.

**What they would improve:**
- exact rest days and in-tournament fatigue;
- results visible the next day instead of after the tournament's maximum length (today a Slam result waits up to 14 days);
- form that includes earlier rounds of the same tournament;
- odds matching that can prove a price was taken before the match started.

**But the visibility rule is part of the frozen setup.** For the forward evaluation, a new source must use **tournament start dates** like the history, so the frozen models run exactly as specified. A model that uses exact dates is a new version (V3) with its own development, freeze and evaluation.

## 6. Steps when a source is approved
1. Write an adapter: source file → V2 table (section 1). Keep the raw files out of git, pinned and checksummed, like `src/download_data.py`.
2. Run `validate_source` and `link_players`. Review every soft issue and every unmatched player.
3. Freeze nothing new. Run `python -m src.v2.forward_eval --data <adapter output>` once for the evaluation periods (PLAN_FORWARD.md).
4. Only after that, rebuild the app artifacts with the extended history so the snapshot moves forward. The evaluation must come first, so the app never shows models fitted on evaluation outcomes.
5. Market odds, if a licensed source is approved: `odds.match_odds` links prices to matches (unmatched and ambiguous prices are listed); `odds.benchmark` refuses any price not timestamped before the match start, which needs real match times.
