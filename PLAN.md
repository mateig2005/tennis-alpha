# Evaluation plan

Committed **before** any model is run, so it acts as a timestamped pre-registration. Changes after this point must be made in new commits with a stated reason.

## 1. Target
For each ATP match, the pre-match probability that player A beats player B. Output is a probability, judged on calibration as well as accuracy.

## 2. Data and splits
Sackmann ATP data (see DATA.md), split by tournament-start year:

| Split | Years | Phase 3 use |
|---|---|---|
| warmup | 2010-2014 | ratings warm up; not scored |
| train ("dev") | 2015-2023 | score and choose Elo settings |
| validation | 2024 | confirm settings, compare model families |
| **test** | 2025 | **locked**: not loaded by Phase 3 code |
| **forward_oos** | 2026 | **locked**: not loaded by Phase 3 code |

Valuebetennis is **not** part of this plan. It is not merged (provenance and licence pending, and its Jan-May 2026 coverage is incomplete; see the private Valuebetennis audit (not published)).

## 3. Time and visibility (no match dates)
- `tourney_date` is the tournament start. Order is `(event_date, round_rank)`; matches with the same key form one **group** and are treated as simultaneous.
- For every group: (1) predict all its matches with ratings as they were **before** the group; (2) only then update ratings. Updates inside a group use the pre-group ratings, so row order inside a group cannot matter.
- A match therefore never sees its own result, same-round results, or any later round / later tournament start.
- **Residual assumption:** a player never plays in two tournaments whose true dates interleave in the opposite order to their start dates. Phase 3 includes a diagnostic counting players with overlapping tournament windows.
- Round robins share one rank, so a player's round-robin matches do not see each other.

## 4. Player A/B order
Winner/loser columns reveal the label. A is chosen by `crc32(seed|match_id)`, a fixed seed, independent of row order and of the result. Target `y = 1` if A won. Probabilities must be symmetric: `P(A) = 1 - P(B)`.

## 5. Models (Phase 3)
1. Always 50%.
2. Rank baseline: `P(A) = sigmoid(b * (ln rank_B - ln rank_A))`, one parameter, fitted on dev only; unranked player -> rank 500 (a documented assumption).
3. Standard Elo (init 1500, constant K). 4. Surface Elo (one rating per player and surface, Carpet counted as Hard). 5. Blend of the two ratings with weight `w`.
Settings grid: `K in {16, 24, 32, 48}`, surface K equal to K, `w in {0.25, 0.5, 0.75}`; D and O matches update ratings or not. **Chosen by dev (2015-2023) log loss**, then frozen. Elo has no fitted parameters, so dev predictions are already out-of-sample and 2024 stays an unbiased check.

## 6. Scored matches
Tour-level only: levels A, M, G, F. Davis Cup (D) and Olympics (O) may update ratings but are not scored. Retirements are kept.

## 7. Metrics and uncertainty
Primary **log loss**; also Brier score, accuracy, expected calibration error (10 equal-width bins) and a reliability table. Differences between models use a **paired bootstrap resampled by tournament** (2,000 draws, 95% interval), because matches inside a tournament are correlated. A difference whose interval contains 0 is not reported as an improvement.

## 8. Locks
- Phase 3 code only loads warmup, train and validation. Tests check this.
- 2025 and 2026 are each evaluated **once**, after models and hyperparameters are frozen. If anything is changed after looking at 2025, 2025 is declared burned and reported as such.
- Later phases: Logistic Regression and XGBoost are fitted on <= 2023 and tuned on 2024.

## 9. Adding newer sources later
Models only read the **canonical match table** (`src/canonical.py`): ids, event date, round rank, surface, level, winner/loser, optional ranks. A new source needs only an adapter producing that table. Conditions before any new source is used: written clarification of provenance and licence, one-to-one reconciliation against Sackmann, completeness validation, a separate split name, and day-level dates only.

## 10. Out of scope for Phase 3
Form features, inactivity and recency Elo, Logistic Regression, XGBoost, dashboard, any use of betting odds as inputs.

## Amendments (made after the first commit of this plan; none changes a selected setting)
1. **Physical lock (clarifies section 8):** `src/prepare_data.py` now writes `atp_matches_dev.csv` (warmup, train, validation) and `atp_matches_locked.csv` (2025, 2026). Phase 3 code reads only the dev file, so locked rows never enter memory. Before this, the plan said the locked splits are "not loaded"; the file split makes that literally true.
2. **Strict-visibility sensitivity (section 3b of the Phase 3 report):** added *after* the first dev/validation numbers were seen, to test the overlapping-tournament assumption. It is reported, not used to choose settings.
3. **Selection is per family:** the best settings are picked within standard Elo (w=0), surface Elo (w=1) and blend (w in {0.25, 0.5, 0.75}) by dev log loss.

## Phase 4 protocol (written and committed BEFORE any Phase 4 model is evaluated)
**Integrity.** 2025 and 2026 stay locked (only the dev file is read). Features and hyperparameters are chosen on chronological **development folds**; 2024 is a **validation** set used only to confirm, never to choose. It is not an independent test: the only independent tests are 2025 and 2026, still untouched.

**M1. Primary visibility = conservative tournament-completion.** A result becomes visible only after `event_date + MAX_DAYS[level]` (Slam 14, Masters 13, Finals 8, Olympics 14, tour events 7, Davis Cup 4), including to later rounds of its own tournament. This applies to ratings *and* every feature. The Phase 3 group rule (earlier rounds visible) is re-run as a sensitivity and its impact reported. Phase 3 chose Elo settings under the group rule, so **Elo K is re-selected under the strict rule on dev (2015-2023) log loss** before anything else.

**M2. Eligible matches.** Tour-level (A/M/G/F) matches in 2015-2024 whose features exist (everyone, debuts included). Every model is scored on exactly these rows (asserted). History (form, Elo updates) uses all levels, 2010 onwards.

**M3. Folds.** Expanding window: validate on 2020, 2021, 2022, 2023, each trained on 2015 up to the year before. Selection = lowest mean out-of-fold log loss. Final models are fitted on 2015-2023 and scored once on 2024.

**M4. Models.** A standard Elo (probability as is) · B LR on Elo diffs (general + surface) · **B+ (extra step): B + ranking/points** · C = B+ + recent form · D = C + opponent-adjusted form. Plus feature-group ablations: groups added one at a time, and each group left out of D.

**M5. Features (all as A minus B differences, so P(A) = 1 - P(B) exactly).** Elo; log rank, log rank points, explicit missing flags (missing rank -> 500, missing points -> 0); win rate over last 5/10 matches and 30/90/365 days; same-surface win rate (last 10, last 365 days); rest (days since the last *released* tournament, a lower bound) and workload (matches in 30/90/365 days, has-history flag); exponentially decayed win rate; opponent-adjusted performance = mean of (result - pre-match Elo expectation) over last 10 matches and 90/365 days, plus mean opponent Elo over the last 10 and a decayed version. Windows are measured from tournament **start** dates, so a match looks at least as old as it really is (its true date is on or after the start); this only makes recent form slightly stale, it cannot leak. *(Clarified after the first commit; the earlier sentence was garbled.)* Rates are shrunk toward 50% (residuals toward 0) with a fixed prior of 3 pseudo-matches, **not tuned**. Debut players get the neutral prior.

**M6. Hyperparameters.** Half-life of decayed features in {30, 60, 120, 240} days; L2 strength C in {0.01, 0.1, 1, 10}. Both chosen by M3 only.

**M7. Preprocessing.** Scale-only standardisation (no centring, which would break A/B symmetry) fitted on training rows only; no intercept (A/B order is random, so none is needed).

**M8. Calibration.** LR is fitted with log loss, so no extra recalibration step is applied. Calibration is *measured* (reliability table, ECE, slope/intercept of a logistic fit of outcome on logit(p)) on predictions from models that never saw the evaluated rows. B is the "recalibrated Elo" comparator, trained on dev only. Slope/intercept are diagnostics, not corrections.

**M9. Uncertainty.** Paired bootstrap resampled by tournament (2,000 draws) vs A, and for each added group vs the previous model. Many comparisons are made; an interval just touching 0 is not a finding.

**M10. Package added:** scikit-learn (and scipy) for L2 Logistic Regression.

**Phase 4 amendments (after the first run, before interpretation).** (a) Report section 3b (cumulative block effects: ranking, all form, opponent-adjusted) was added after seeing the first run; it compares already-selected models and changes no choice. (b) The C grid {0.01..10} puts several selections on its edge (0.01 for B and most form steps, 10 for the +adj step). The grid was *not* changed after seeing this; it is reported as a limitation.

## Phase 5 protocol (written and committed BEFORE any Phase 5 model is evaluated)
**Integrity.** Only the dev file is read. **2024 is not used for any choice in Phase 5** (no tuning, no early stopping, no feature decisions); it is reported once as validation. All selection, ablation, importance and diagnostics use the dev folds (validate on 2020, 2021, 2022, 2023; train on 2015 up to the year before).

**N1. Primary specification (owner's decision, not a data-driven one).** Model D (Elo, ranking, recent form, surface form, rest/workload, decayed form, opponent-adjusted form) is the primary feature-rich candidate. Phase 4 did not demonstrate that opponent-adjusted features improve on C; they are kept by design choice. If a simpler model scores better, the report says so.

**N2. XGBoost on the Model D features, with exact player symmetry.** Trees are not automatically symmetric, so (a) every training row is also added mirrored (features negated, label flipped) and (b) the prediction is `(f(x) + 1 - f(-x)) / 2`, which makes `P(A) = 1 - P(B)` exact. Inputs stay the Phase 4 differences (no new features).

**N3. Tuning, dev folds only, conservative.** Fixed: learning rate 0.05, subsample 0.8, column sample 0.8, at most 800 trees, seed 0. Grid: `max_depth in {1, 2, 3}`, `min_child_weight in {5, 25}`, `reg_lambda in {1, 10}` (12 settings). **Early stopping** (patience 40) is done inside each fold's training period: fit on years up to Y-2, monitor Y-1, take the best number of trees, refit on years up to Y-1 with that number, predict Y. The fold's own year never influences the model. Selection: among settings within 0.0003 mean log loss of the best, take the simplest (smaller depth, then larger min_child_weight, then larger lambda).

**N4. Logistic Regression re-selected with a wider grid.** Phase 4 selections sat on the edge of the C grid. Grid is now `C in {0.001, 0.003, 0.01, 0.03, 0.1, 1, 10, 100}` (half-life grid unchanged), chosen on dev folds as before. This was prompted by the dev-fold selections, not by 2024. Phase 4's report keeps its original grid.

**N5. Models compared on identical eligible matches.** A standard Elo · B+ LR · C LR · **D LR** · **D XGBoost** · rank-independent D (LR and XGBoost). Elo K (24 general, 32 surface), the strict visibility rule and the feature definitions are frozen from Phase 4.

**N6. Rank-independent candidate.** Model D without the ranking group (no rank, no points), for periods or sources without ranking data. LR re-selects (half-life, C) on dev folds; XGBoost reuses the settings chosen for D (trees re-determined by early stopping).

**N7. Diagnostics (dev folds only).** Multicollinearity: correlation matrix and variance inflation factors. Nonlinearity: additive trees (depth 1) vs linear LR. Interactions: depth 2-3 vs depth 1. Importance: XGBoost gain, and **group permutation importance** (shuffle one feature group in the held-out fold, 3 repeats). Ablation: leave-one-group-out for LR D and XGBoost D with hyperparameters held fixed, with tournament-bootstrap intervals.

**N8. Subgroups.** Log loss by surface, season and strength band (best ranking of the two players: 1-10, 11-30, 31-100, 100+/unranked), on dev folds and on 2024. Descriptive only; subgroup results choose nothing.

**N9. Choice of the primary algorithm for D (decided on dev folds, stated in advance).** If XGBoost D beats LR D on dev-fold log loss with a 95% interval excluding 0, XGBoost D is primary; otherwise LR D is primary (simpler, exactly symmetric by construction) and XGBoost D is reported as secondary.

**N10. Freeze.** At the end of Phase 5 the feature lists, Elo settings, half-life, C, XGBoost settings and tree counts are written to `models/frozen_phase5.json` and committed. Nothing in it may change before the independent evaluation.

**N11. Package added:** xgboost. On macOS it needs the OpenMP runtime (`brew install libomp`); without Homebrew the code falls back to the copy bundled with scikit-learn.

**Phase 5 clarifications (no choice changed).** (a) XGBoost D uses the decay half-life selected for LR D on dev folds (60 days), so both algorithms share one Model D feature set. (b) The ablation holds hyperparameters fixed (N7); a group can look useful there yet add nothing once regularisation is re-tuned (compare `adj` in the ablation with D vs C). (c) On this Mac xgboost runs on the libomp bundled with scikit-learn (N11).

## Phase 5 outcome and freeze
- **Primary: LR D** (rule N9: XGBoost D minus LR D on dev folds = +0.0008 [-0.0006, +0.0022], not better). XGBoost D is secondary. Rank-independent primary: **LR D-norank**.
- Frozen in `models/frozen_phase5.json`: Elo K 24/32, strict visibility, half-life 60, LR C (B+ 0.1, C 0.003, D 1.0, D-norank 100), XGBoost depth 2, min_child_weight 25, lambda 10, 231 trees (163 for the rank-independent version).
- Honest status: on dev folds and on 2024, Model D is **not** demonstrably better than B+ (Elo + ranking) or C. D stays primary by the owner's design choice (N1).

## Phase 6 protocol for the independent evaluation (PROPOSED; needs owner approval before any locked data is read)
1. **One run, no iteration.** A dedicated script reads `atp_matches_locked.csv` for the first time, writes predictions and a report, and the output is committed together with the frozen file's hash. No setting in `models/frozen_phase5.json` may change.
2. **Training.** Each frozen model is refitted once on 2015-2024 with its frozen hyperparameters (XGBoost tree counts fixed). Elo ratings and features keep updating through 2025 under the strict visibility rule; model coefficients do not.
3. **Hypotheses, in this order.** H1 (primary): LR D has lower log loss than Elo on 2025 tour-level matches. H2: LR D vs LR B+ (does the comprehensive model beat the simple one?). H3: XGBoost D vs LR D. H4: LR D-norank vs Elo. Only H1 is confirmatory; the rest are reported with intervals and no claims beyond them.
4. **2026 (Jan-May, Sackmann).** Evaluated after 2025, once, with the same frozen settings; models refitted on 2015-2025. Reported separately as forward out-of-sample.
5. **Metrics.** Log loss, Brier, accuracy, ECE, calibration slope, reliability table, tournament-bootstrap intervals; subgroup tables are descriptive.
6. **If anything is changed after the 2025 result is seen, 2025 is declared burned.**

## Phase 6 protocol: APPROVED by the owner (final; committed with the evaluation script before any locked data is read)
The proposal above is adopted with these additions. Frozen spec: `models/frozen_phase5.json` at commit `f50cab2` (sha256 `16fe577b12eb96d3...`).
1. **Models scored:** A Elo, B+ LR, C LR, **D LR (primary)**, D XGBoost, D-norank LR, D-norank XGBoost. No feature selection, no tuning, no change to any frozen value.
2. **2025:** refit once on tour-level matches 2015-2024 (2010-2014 remain warm-up history); predict 2025. **2026:** refit once on 2015-2025; predict the available 2026 matches (Sackmann snapshot, tournaments starting up to 2026-05-25, i.e. through Roland Garros; **not a full season**). Elo and form features update through the test period under the tournament-completion rule; coefficients and trees do not.
3. **Comparison tiers.** *Primary (confirmatory):* D LR vs Elo on 2025 log loss. *Secondary (pre-specified):* D LR vs B+ LR; D XGBoost vs D LR; D-norank LR vs Elo. *Exploratory:* every other pair, all subgroup tables (surface, ranking band, quarter) and all 2026 comparisons other than the same four, which keep their tier but are labelled forward out-of-sample with a small sample.
4. **Metrics:** log loss (with interval), Brier, accuracy, ECE, calibration intercept and slope, reliability table and plot; paired tournament-bootstrap 95% intervals (2,000 draws) for differences.
5. **Safeguards in the script:** (a) refuses to run unless `src/`, `models/`, `tests/` and `PLAN.md` match the commit; (b) verifies the frozen file against the code; (c) first reproduces Phase 5's 2024 numbers from the dev file with the frozen settings (a dry run that proves the script before locked data is opened); (d) checks that adding 2025/2026 rows changes no earlier feature; (e) refuses to run a second time once the report exists.
6. **Bugs:** if one is found after the run, it is documented, its effect on validity assessed, and the owner's approval obtained before any rerun.
7. **Packages added:** matplotlib (calibration plots only).

## Phase 6 outcome (recorded after the single run)
Run once at code commit `0550761` with the frozen spec from `f50cab2`; no setting changed, no bug found, no rerun. Primary hypothesis confirmed: LR D beats Elo on 2025 (log loss -0.0083, 95% interval -0.0153 to -0.0020; 2,588 matches). Details: `reports/phase6_backtest.md` (auto-generated) and `reports/phase6_assessment.md` (hand-written). **2025 and January-May 2026 are now used test sets:** they cannot validate any future change. The next untouched period must be reserved for that.

## Phase 7: dashboard (no research decisions; nothing here is evaluated on test data)
- **Production fit.** LR D (primary) and LR C (challenger) are refitted on all approved tour-level matches from 2015 to the end of the data, with the frozen settings (C values, half-life 60, Elo K 24/32, tournament-completion rule). No hyperparameter, feature or calibration change. This refit **cannot be validated**: no held-out data remains. The dashboard therefore shows the Phase 6 backtest of the same specification, clearly labelled.
- **No recalibration** on 2025/2026, no betting odds, no Valuebetennis data.
- **Historical-demo mode.** Predictions are for a hypothetical match on the *snapshot date* (the first day on which every result in the data counts as released under the visibility rule), never "today". Rankings are each player's last known ranking in the data, shown with its date; a player without one is treated as unranked exactly as in training.
- **Same feature code.** The dashboard computes features with the research pipeline's own functions; a test checks that they equal the pipeline's output for the same hypothetical match.
- **Packages added:** streamlit.

## Phase 9: web application backend (no research decisions)
- **What changed:** a FastAPI layer (`src/api.py`) over the existing `Predictor`, raw display statistics (`src/player_stats.py`), and a real Elo rating path stored in the dashboard artifact (`build_artifacts.elo_history`, rebuilt from the engine's own pre-match ratings and asserted to end at the engine's final rating for every player).
- **Models unchanged:** the artifact was rebuilt to add the rating path. Model D and Model C coefficients and scales, every general Elo rating and every real-surface Elo rating are identical to the previous artifact, and the Sinner v Alcaraz probabilities on all three surfaces are bit-for-bit the same. No frozen value, feature definition or research module was modified.
- **Discrepancy found (documented, not fixed):** `src/elo.py` is documented to use general Elo only when a match has no surface. With the real data the missing value is not converted, so those matches are tracked under a separate pseudo-surface. **Scope:** 52 matches, all Davis Cup (level D), 2015-2023. **Impact: none.** They are never trained on or scored (tour-level only), their surface feature is never used, and real-surface ratings, general Elo, form features, every backtest result and every dashboard number are unaffected. The unit test for this case passes because it uses `None`, not the `NaN` the CSV produces. Fixing it would touch the frozen pipeline, so it is left as is pending the owner's approval; a fix would change no reported number.

## Phase 10: head-to-head display (no research decisions)
Recorded meetings between two players are served from a separate index (`artifacts/h2h.pkl`, built by `src/h2h.py` from the same cleaned match files). It is display-only: no model, feature, frozen value or backtest result reads it, and the model artifact was not rebuilt (its file hash is unchanged). The index inherits the research data's rules: walkovers, defaults and abandoned matches excluded, retirements kept and flagged. It covers only the dataset's period (2010 to the cutoff), so earlier and later meetings are missing; the interface says so.
