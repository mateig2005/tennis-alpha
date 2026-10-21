# TENNIS ALPHA V2: Advanced Matchup Intelligence (research protocol)

Written and committed **before** any V2 experiment was run. Settings are versioned in `models/v2_config.json` (`v2.0-dev`). V1 (Elo, Model C, Model D, the frozen specification, the Phase 6 results, the API and the frontend) is not changed by V2.

## 1. Status of the data periods
| Period | V1 role | V2 role |
|---|---|---|
| 2010-2014 | warm-up | warm-up (ratings, serve/return profiles, pace history) |
| 2015-2019 | training | training only |
| 2020-2022 | V1 selection folds | **V2 selection folds**: every V2 choice (C, which groups) is made here |
| 2023-2024 | V1 fold 2023, V1 validation 2024 | **V2 reporting folds**: scored with choices frozen on 2020-2022 |
| 2025, Jan-May 2026 | V1 independent test (used once) | **not read by V2 experiments.** They are already-examined data, not fresh evidence for V2 |
| future data | none | **prospective test**: the first period after V2 is frozen, from a new legally usable source |

The 2023-2024 reporting folds are held out from V2 *selection*, but they are not independent: V1's Model D was selected using 2020-2023 and confirmed on 2024, and the same people designed V2 knowing the V1 results. They are honest development evidence, nothing more. Only the prospective period can validate V2.

## 2. Rules
- Same leakage rules as V1: features use only results released by `event_date + MAX_DAYS[level]`; A/B by hash; all model inputs are A-minus-B differences (or antisymmetric interactions), so P(A) = 1 - P(B) exactly.
- Match statistics (`w_*`, `l_*`) are post-match. They enter a player's history only when that match is released, exactly like results.
- Tournament pace is estimated from **previous editions only**: a tournament's own matches are released after it ends, so they can only inform its next edition.
- No fabricated values: missing statistics stay missing and simply add nothing; players without statistics get the prior (tour average). Weather, altitude, indoor/outdoor, ball type and match dates are not in the data and are not invented.
- Priors (shrinkage) and half-lives are fixed in `models/v2_config.json`, not tuned. Only the L2 strength C is selected, on the selection folds.
- The Phase 10 display H2H index (`artifacts/h2h.pkl`) stays display-only. V2 builds its own leak-free pre-match H2H inside its chronological pass, for research only.

## 3. Feature groups (V2 additions on top of the Model D features)
All ratings are measured in "points won above the tour average".
- **sr_adj** (opponent-adjusted serve and return): each match gives a serve sample = serve points won % minus the surface's tour average plus the opponent's pre-match return rating (and the mirror for return). Samples are weighted by points, decayed with a 240-day half-life and shrunk toward 0 with 300 pseudo-points. Surface-specific versions are shrunk toward the player's general rating (a two-level hierarchy).
- **sr_raw** (raw rolling rates, last 20 matches with statistics, shrunk toward the tour average): aces and double faults per service point, first-serve in %, first- and second-serve points won %, service games held %, first- and second-serve return points won %, return games won %.
- **pressure** (hypothesis, not "mental strength"): break points saved % and converted %, shrunk.
- **matchup**: a service-return model. Each player's chance of winning a point on serve = surface tour average + own serve rating - opponent's return rating. A standard point-game-set-match calculation (best of 3 or 5, tiebreak sets) turns the two point chances into a match probability; its log-odds is the feature. Plus one cross term (A serve x B return minus B serve x A return).
- **pace**: tournament pace index = average over previous editions of (actual minus expected ace rate) and (actual minus expected serve points won), shrunk toward the surface average with 40 pseudo-matches. It is a **statistical proxy**, not a measured court speed. Features: pace x (serve rating difference), pace x (return rating difference), pace x (ace rate difference).
- **h2h**: pre-match head-to-head residual = sum over earlier released meetings of (result minus the Elo-expected result), divided by (meetings + 5). Same on the same surface. A 2-0 record against an expected 80% favourite moves the value by only 0.4 / 7 = 0.06.
- **context**: best-of-five x Elo difference (longer matches should favour the stronger player more).
- **momentum**: 60-day half-life serve and return ratings, and the change in general Elo over the last 90 days.

Not in this stage (listed in the availability report): tiebreak and deciding-set records (needs score parsing; noisy), fatigue within a tournament (no match dates), physical attributes, clustering into discrete styles.

## 4. Models compared (identical eligible matches: tour-level A/M/G/F, as V1)
A. **Model D**, frozen specification refitted per fold (benchmark). Also Elo.
B. **V2 LR**: Model D features + V2 groups, L2, C chosen on selection folds. Ablations: D + each group; full V2 minus each group.
C. **V2 XGBoost** on the full V2 features (frozen Phase 5 tree settings, mirrored rows, early stopping inside training years).
D. **Matchup model alone**: the service-return probability as is, and recalibrated by a one-feature LR.
E. **Ensemble**: plain average of Model D and V2 LR (no fitted weights). Only claimed if it helps.

## 5. Evaluation
- Primary metric: log loss. Also Brier, accuracy, calibration intercept and slope, reliability diagram.
- Subgroups (descriptive only, choose nothing): surface, pace tercile, close matches (Model D between 40% and 60%), strong favourites (above 75%), both players top-10, season.
- Uncertainty: paired bootstrap resampling whole tournaments (2,000 draws).
- **Decision rule, fixed now:** a V2 variant counts as an improvement over Model D only if (1) its mean log loss on the selection folds is lower than D's and (2) on the pooled reporting folds its log loss difference vs D has a 95% interval entirely below 0. About 20 comparisons are made, so a single interval barely excluding 0 is weak evidence.
- Not tuning toward anecdotes: the Sinner vs Alcaraz case is a diagnostic only (`reports/v2_sinner_alcaraz_audit.md`) and influences no V2 choice.

## 6. After this phase
Nothing replaces Model D in the app until the owner approves. V2 is frozen only after review; then it waits for a prospective period from a new, properly licensed source.

## Amendments (after the first run; none was chosen to improve a result)
1. **Two implementation bugs, found by the pace sanity check in the first run's report, fixed before interpreting anything.**
   (a) In the first week of 2010 no tour average existed yet, so expected rates fell back to 0.5. The ace-rate residuals of those matches (about -0.4) permanently distorted the pace of venues that hold an event that week (Chennai, Doha, Brisbane). Residual-based updates (opponent-adjusted ratings, pace) now start only once a tour average exists; raw counts are kept. A regression test covers it.
   (b) The source writes the same tournament code with and without leading zeros ('891' and '0891'), which split one event's edition history in two. Codes are now compared without leading zeros. Some events still change code (for example Brisbane 2016-18 is 'M020'); that remains a known limitation.
   First-run headline, for the record: V2 LR (all groups) minus Model D on the reporting folds = -0.0061 [-0.0093, -0.0030]. The first-run report is replaced by the rerun; the numbers above are the only part kept.
2. **Reference row added:** "D (C re-selected)", Model D's features with C chosen on the selection folds like every V2 variant. Without it, a "D + group" row mixes the group's effect with a change of C. It is a comparison row, not a candidate.

## Stage 1 outcome (dev folds only; reports/v2_dev_experiments.md)
- **V2 LR (all groups) beats Model D** on the reporting folds 2023-2024: log loss 0.6097 vs 0.6160, difference -0.0062 [-0.0095, -0.0030]. It is also better on every selection fold (2020-2022). Calibration is similar (slope 0.94).
- **The gain comes from serve and return information.** Added on its own to D, each of opponent-adjusted serve/return (-0.0040), the matchup model (-0.0046) and recent serve/return momentum (-0.0039) helps. Raw rolling rates help less (-0.0024). They overlap: leaving any one group out of the full model changes almost nothing.
- **No measurable gain** from H2H, the pace interactions, break-point "pressure" rates, or best-of-five × Elo. They stay in the code as tested hypotheses, not as findings.
- **XGBoost was not better than LR** (-0.0036 vs D, interval touching 0). The raw matchup model alone is overconfident (slope 0.69) and worse than D. As an input to LR it is the strongest single addition.
- Best LR variant on the selection folds: "V2 LR - pressure", practically equal to the full model.
- Not frozen. Next decision for the owner: freeze a V2 candidate (suggested: the full V2 LR or the simpler "V2 core"), then wait for a prospective period.

# V2.1 protocol: match quality, statistical momentum, calibration, odds (written before any V2.1 experiment)
Settings: `models/v21_config.json` (`v2.1-dev`). V1 and V2 code, settings and reports are unchanged; V2.1 adds `src/v2/score.py`, `quality.py`, `odds.py` and `run_v21.py`. Same data periods as V2 (dev file only; selection folds 2020-2022, reporting folds 2023-2024, not independent; 2025/2026 not read except by the diagnostic audit).

**Q1. Per-match performance, each compared with a pre-match expectation** (the expectation uses the V2 pre-match serve/return ratings of both players on that surface, so it is opponent- and surface-adjusted):
- *Total points won residual* = points won share - expected share. Total points won is exact here: serve points won + return points won, both in the data.
- *Serve residual* and *return residual*: serve points won % minus expected, return points won % minus expected.
- *Scoreline dominance residual* = games won share (parsed from the score) minus the share implied by the expected hold probabilities.
- *Luck (winning ugly / losing well)* = result (1/0) minus the match-win probability implied by the points the players **actually** won in that match (V2 point-to-match model). Positive = won more than the points deserved.
- *Workload*: sets played. Minutes are not used (10% missing, implausible outliers).
Retirements, abandoned and unparseable scores give no performance record (counted only for workload when sets are parseable); matches without statistics give only the scoreline record.

**Q2. Rolling indicators (shrunk toward 0 with 3 pseudo-matches, not tuned).** Group `quality`: points residual over the last 3, 5, 10 and 20 matches; dominance over the last 5 and 20; luck over the last 10. Group `qmomentum`: trend (last 5 minus last 20) of points, serve and return residuals; volatility of the points residual (last 10, shrunk toward the tour-wide spread); same-surface points residual (last 10); sets played in the last 30 days.

**Q3. Models.** A frozen Model D; B full V2 (as in stage 1); C V2 + quality; D V2 + qmomentum; E V2 + both; leave-one-group-out from E; F XGBoost on E (frozen depth-2 settings). C chosen on selection folds as in V2. Decision rule as in V2: better than full V2 only if lower selection-fold log loss and a reporting-fold interval vs V2 entirely below 0.

**Q4. Calibration.** Diagnostics on the reporting folds by subgroup: strong favourites, underdogs, both top-10, a player inactive 45+ days, general and surface Elo pointing to different players. Recalibration (Platt slope-only, which keeps A/B symmetry, and isotonic fitted on mirrored data) is fitted on selection-fold predictions only and scored on the reporting folds. Adopted only if it improves reporting-fold log loss with an interval below 0.

**Q5. Does statistical form lead Elo?** Descriptive test on dev data: regress each player's Elo change over the next 90 days on pre-match points residual (last 20), luck (last 10) and the past 90-day Elo change.

**Q6. Market benchmark.** No legally usable odds are available, so only the tools are built: fair odds, overround, no-vig probabilities (proportional and power methods) and a benchmark function that refuses odds not timestamped before the match. Odds are never a model input. Valuebetennis stays excluded.

**Q7.** The Sinner-Alcaraz case is a diagnostic only and influences no choice.

## V2.1 amendments (after the first run)
3. **Post hoc diagnostic rows:** the first run showed overconfidence when a player had been inactive 45+ days. Two rows were added that split this group by whether the inactive player was Model D's favourite or underdog. They are descriptive only and choose nothing.

## V2.1 outcome (dev folds only; reports/v21_dev_experiments.md)
- **V2.1 does not improve on full V2.** V2 + quality + momentum minus V2 on the reporting folds = +0.0001 [-0.0010, +0.0011]. No single new feature family helps, and XGBoost on all features is worse (+0.0024).
- **Why:** the information overlaps. Added to Model D *without* the V2 groups, the quality features do help (D + both: 0.6131 vs D 0.6160). V2's opponent-adjusted serve/return ratings already capture "how well a player plays beneath the results".
- **"Luck" (winning ugly) is real but already priced in.** Players who won more than their points deserved tended to lose Elo over the next 90 days (strongly negative coefficient). Points-based form slightly anticipated Elo gains. This does not improve match prediction beyond V2, because V2's ratings are built from points, not results.
- **Calibration:** Platt rescaling changes almost nothing (-0.0002, interval includes 0) and isotonic regression makes things worse. Neither is adopted.
  - Descriptive pockets of overconfidence (about 2-4 points): matches with an inactive player (especially when the underdog was the inactive one), and matches where general and surface Elo disagree. Hypotheses for a pre-registered test, not corrections.
- **Decision:** V2 (stage 1) stays the leading candidate. V2.1 code stays as tested research tooling (performance diagnostics, odds and calibration utilities).

# Phase 11: V2 integration (no research decisions)
- **Frozen:** `models/frozen_v2.json` records the full V2 LR (52 features, C = 0.03, scale-only standardisation, no intercept) as selected on the selection folds. It also records the code commits, the config and V1 spec hashes, and the expected development results. Rerunning `python -m src.v2.run_v2` reproduces `reports/v2_dev_experiments.md` byte for byte.
- **Three separate fits, never mixed up:**
  1. *Experimental*: 2015-2024 development folds (the research numbers).
  2. *Production*: the same frozen settings refitted on all approved data to May 2026 (`artifacts/v2.pkl`, like V1's dashboard artifact). Refitting is not tuning.
  3. *Prospective*: reserved for the first period after this freeze, from a new licensed source.
- **App:** `/predict` returns a separate `v2` block (status, version, validation statement, probability, contributions, serve/return profile). The V1 fields are byte-identical with or without V2. V2 is `unavailable` for players with no released match carrying statistics (86 of 445 active players) and warns below 10 such matches. The match format (best of 3 or 5) is a user choice used by V2 only.
- **Sinner-Alcaraz:** V2 66.1% vs V2.1 61.4% (both fitted 2015-2024). Stronger regularisation alone moves V2 only to 65.8%; the rest comes from V2.1's added features. They are highly correlated with existing inputs (sets in 30 days vs matches in 30 days r = 0.98), so the same evidence is split differently. The production V2 says 68.5% (clay, best of 3). Details: `reports/v2_sinner_alcaraz_audit.md`, sections 6-7.
- **Bugs found and fixed during integration:**
  1. The `best_of` query parameter rejected every value, because an int-only Literal type does not accept query strings. Found by the new tests.
  2. V2 config paths were relative to the working directory, so the API crashed when started from another folder. Found in browser verification.
  3. V2's 17-group contribution chart had overlapping labels; the chart height now scales with its rows.
  None affects any model output.

# Phase 12: V2 forecast audit and data readiness (no model change, no retraining)
- **Audit** (`python -m src.v2.audit12` → `reports/v2_forecast_audit.md`): uses the out-of-sample development predictions for 2020-2024 with the frozen settings, and reproduces the stage 1 numbers exactly.
  - V2 is not systematically overconfident (favourites: 67.0% predicted, 66.1% won; Model D 66.4% vs 65.2%).
  - V2's stronger favourites, including above 90% and in best-of-5, win at about the predicted rate. Model D is underconfident in best-of-5.
  - Top-10 v top-10 (265 matches) shows no measurable miscalibration, with wide intervals.
  - The one consistent weakness for both models is a player inactive 45+ days (about 2.5 points overconfident, interval excluding 0).
- **Markov check:**
  - An independent solver agrees with the production formulas to about 1e-16, and a point-by-point simulation agrees within its error.
  - A Slam 10-point final-set tiebreak would move results by at most about 0.2 points.
  - The raw matchup probability is overconfident (slope 0.68 in best of 3), although its point-level inputs are not over-spread (slope 1.13). The cause is the assumption of a fixed serve chance per match. V2 down-weights the term, and V2's calibration shows the correction works.
- **Uncertainty:** refitting on resampled tournaments moves a typical prediction by about 1.8 points (standard deviation), from parameters alone. V2 and V2.1 differ by more than 5 points in 0.2% of matches. These parameter-only intervals are not shown in the app.
- **Bugs found in new audit code:**
  1. The independent solver recursed forever at deuce; fixed with the closed form.
  2. The duplicate check flagged 201 real rematches; the rule now considers the tournament.
  3. Two forward-evaluation tests were wrongly built (an outcome flip that mixed up player columns, and a comparison that ignored the A/B swap); the tests were corrected, the code was right.
  None affects a model or a prediction.
- **Prepared:** `PLAN_FORWARD.md` (frozen forward protocol), `src/v2/forward_eval.py` (refuses changed frozen files and second runs; its dry run reproduces the 2024 development fold exactly), `src/v2/sources.py` and `odds.match_odds` (data readiness), and `reports/v2_data_readiness.md`.

