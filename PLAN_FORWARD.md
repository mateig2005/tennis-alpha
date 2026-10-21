# Forward evaluation protocol (frozen 2026-10-19, before any match after 2026-05-25 has been loaded)

This is the only test that can tell whether V2 is better than Model D on data nobody has tuned on. 2025 and January-May 2026 are already known and are not used as test evidence.

## 1. What is fixed (no change allowed during the evaluation)
| Item | File | sha256 (first 16) |
|---|---|---|
| Model D, Model C, Elo settings | `models/frozen_phase5.json` | `16fe577b12eb96d3` |
| V2 settings (52 features, C = 0.03) | `models/frozen_v2.json` | `42d6a5a803741412` |
| V2 feature settings | `models/v2_config.json` | `e785ce642bce11f3` |
| Visibility rule | `MAX_DAYS` in `src/canonical.py` | unchanged since Phase 4 |

**Model parameters** (coefficients and scales) are fitted once, on tour-level matches from 2015 to the snapshot data end (2026-05-25), exactly as in the app's artifacts. They are never refitted on evaluation matches.

## 2. What may change: player state
Elo ratings, form, serve/return profiles and H2H **update chronologically** with each evaluation result after its release day (the same rule as always). This is how the models would run live. It is not training: no coefficient moves.

## 3. Periods
- **A. Retrospective hold-out:** matches from tournaments starting 2026-05-26 to 2026-10-19. They happened before this protocol was written, but no model, setting or person in this project has looked at their outcomes for modelling. Reported as "held-out, collected after freezing". This is not truly prospective. **Corrected in Amendment 1: structural metadata of this period was seen; no outcomes were.**
- **B. Prospective:** tournaments starting after 2026-10-19. This is the real test.
- A and B are reported separately and pooled.

## 4. Data
- Only a source the owner has approved for licence and provenance, loaded through an adapter that passes `src/v2/sources.py` validation.
- Player ids are linked with `src/identity.py`. Unmatched or ambiguous players are listed, never guessed.
- Matches without serve/return statistics still count for Model D and Elo. V2 is scored on the rows where it is available, and every comparison uses exactly the same rows.

## 5. Metrics and decision rule
- Scored matches: tour level (A, M, G, F), excluding walkovers. Retirements count, as in all earlier phases.
- **Primary:** log loss of V2 minus Model D, with a paired bootstrap that resamples whole tournaments (2,000 draws). **V2 is called better only if the 95% interval is entirely below 0.**
- **Secondary (reported, decide nothing):** Brier score, accuracy, calibration slope and intercept, and calibration by favourite band, surface, best-of-3 vs best-of-5, and top-10/top-20 matchups. Model C and Elo are also reported.
- **Sample size:** the development difference was about 0.006, and its interval half-width was about 0.0033 on 5,559 matches.
  - Below roughly 2,000 matches, a real 0.006 gain usually would not reach the rule. Any result there is reported as **inconclusive**, not as a loss.
  - A clear verdict needs about 5,000 matches (roughly 18 months of tour-level play).
  - Period A alone (June to October 2026) will hold roughly 1,000-1,500.
- Market comparison: only if a licensed pre-match odds source is approved (see `reports/v21_data_and_market_plan.md`). Odds are never a model input.

## 6. Running it
- `python -m src.v2.forward_eval --data <adapter output>` checks the frozen hashes (the values above are hard-coded in the script), builds player state over all data chronologically, fits on rows up to 2026-05-25 only, predicts only evaluation rows, and writes `reports/forward_eval.md`.
- **It refuses to run a second time on the same period** unless the owner approves (`--rerun-approved`), like Phase 6.
- `--dry-run` reproduces the 2024 development fold from the dev file. That proves the pipeline without touching new outcomes.

## 7. Forbidden during the evaluation
- Any tuning, feature change or recalibration based on evaluation results.
- Any change to the scoring rows after seeing outcomes.
- Presenting a period-A result as a prospective result.

If a bug is found: document it, assess whether it invalidates the run, and get owner approval before rerunning.

## Amendment 1 (2026-10-20): prior exposure to period A, corrected
Section 3 said no one had looked at period A outcomes. That is not quite right. On 2026-10-13 the Valuebetennis 2026 file was downloaded for the source audit. It contains June-October 2026 results and odds. Structural metadata of that period was displayed: per-tournament match counts, first/last match times and monthly counts (the private Valuebetennis audit (not published)). No June-October 2026 winner, score or odds was displayed or analysed, as checked against the full session record (`reports/data_source_review_2026.md`, section 6). No model setting was influenced: all were frozen without outcome information from the period. Period A is therefore reported as a **retrospective hold-out with documented structural exposure**. Public knowledge of results by people following tennis is outside the project's control, which is why period B remains the decisive test. The rules, frozen files and decision rule are unchanged.

## Amendment 2 (2026-10-20): period A boundary, subgroups, source acceptance. Written and committed before the one-time run
1. **Boundary.** The data has tournament start dates only. A tournament counts as period A only if its start plus the frozen maximum length (`MAX_DAYS`: G 14, M 13, F 8, O 14, D 4, A 7 days) is on or before 2026-10-19, so all its matches must have been played by the cutoff. Any other tournament already downloaded is **quarantined**: excluded from history and scoring, not period A (it may run past the cutoff) and not period B (its results were downloaded before predictions were logged). The cutoff is not moved to enlarge the sample. Preflight: `reports/forward_eval_preflight.md` (55 tournaments eligible, 1 quarantined, 972 tour-level matches).
2. **Roland Garros 2026** (start 2026-05-25) belongs to the frozen history, as in the June 2026 snapshot. TennisMyLife rows are assigned to their tournament's start, so they never enter period A.
3. **Period B** is prospective only for matches whose predictions are logged before their results exist. Nothing already downloaded is relabelled as period B.
4. **Subgroups** (surface, top-20 v top-20, top-10 v top-10) are reported only with at least 300 scored matches; otherwise their size is shown without metrics. Secondary comparisons decide nothing.
5. **Source accepted by the owner** with documented limitations:
   - Raw statistics agreement on known matches was 96.3%, below the pre-set 97%. It stays recorded as a failed threshold.
   - Adjusted agreement is 98.1%, after statistics failing the pre-existing consistency rule (`stats_ok`, written in V2 stage 1) are treated as missing.
   - Exclusions: 126 rows (2,268 values) in the 2024 to May 2026 overlap, 119 of them because service games were 0; none in period A.

## Period A result (one-time run 2026-10-20, code 27f0b5e; `reports/forward_eval.md`)
- **Sample:** 972 eligible tour-level matches; V2 available for 929 (95.6%).
- **On the same 929 matches:** V2 0.6271 vs Model D 0.6333 log loss. V2 minus D = **-0.0062 [-0.0165, +0.0043]**: the same size as the development estimate (-0.0062), but the interval includes 0.
- **Verdict:** inconclusive under the pre-registered rule (fewer than 2,000 matches). No model, setting or conclusion was changed. The guard now refuses a rerun of period A. Period B awaits logged pre-match predictions.

## Period B: prospective logging (Phase 15)
- **Retrospective vs prospective.** Period A was a retrospective hold-out: its matches had been played before the models were frozen. Period B evidence comes only from `src/predlog.py`: predictions written to a hash-chained, append-only log *before* the match.
- **Timing classes.** Only records classed **validated** (created before 00:00 UTC on the tournament's first day) count as prospective in the primary analysis. **declared** records (before a typed-in match start) are reported separately; **uncertain** records are excluded.
- **Nothing relabelled.** No downloaded October 2026 match is relabelled prospective.
- **Scoring.** After results arrive, `predlog.join_results` matches each record to exactly one completed match: the same two player ids, and a tournament start within 14 days of the logged one. Unmatched and ambiguous records are listed, not guessed. The probabilities used are the ones stored at logging time, never recomputed.
- **Analysis rules.** The primary metric, the paired tournament bootstrap and the decision rule are those of section 5. The predictions come from the app's refreshed player state with frozen coefficients.
