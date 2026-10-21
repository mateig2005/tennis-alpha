# Phase 6 assessment (written by hand after the single run; numbers come from `phase6_backtest.md`)

The independent evaluation ran once, at code commit `0550761`, with the specification frozen in commit `f50cab2`. No setting was changed, no bug was found, and no rerun was made. All intervals are 95%, resampling whole tournaments.

## 1. What was tested and what happened

| Tier | Question | 2025 (2,588 matches) | 2026, Jan to Roland Garros (1,310 matches) |
|---|---|---|---|
| **Primary** | Does LR D beat Elo? | **Yes:** -0.0083 (-0.0153, -0.0020) | Yes: -0.0174 (-0.0289, -0.0046) |
| Secondary | Does LR D beat the simpler B+ (Elo + ranking)? | Yes, narrowly: -0.0031 (-0.0061, -0.0001) | Not shown: -0.0029 (-0.0071, +0.0020) |
| Secondary | Does XGBoost D beat LR D? | No: +0.0019 (-0.0007, +0.0043) | No: +0.0006 (-0.0026, +0.0034) |
| Secondary | Does rank-independent LR D beat Elo? | Yes: -0.0059 (-0.0116, -0.0007) | Yes: -0.0145 (-0.0249, -0.0019) |

Values are differences in log loss; negative means the first model is better.

| Model | 2025 log loss | 2025 accuracy | 2026 log loss | 2026 accuracy |
|---|---|---|---|---|
| A: Elo | 0.6291 | 63.6% | 0.6317 | 65.3% |
| B+: LR, Elo + ranking | 0.6238 | 64.4% | 0.6172 | 66.0% |
| C: LR, + recent form | 0.6203 | 64.8% | 0.6128 | 65.8% |
| **D: LR, all features (primary)** | 0.6207 | 65.1% | 0.6143 | 66.2% |
| D: XGBoost | 0.6226 | 64.4% | 0.6149 | 66.2% |
| D without ranking: LR | 0.6232 | 64.1% | 0.6173 | 66.0% |

## 2. Does comprehensive Model D deliver value beyond simpler alternatives?

- **Against Elo: yes.** The primary hypothesis is confirmed on unseen 2025 data and holds again on 2026. The gain is real but modest: about 1.3% lower log loss and 1.5 points of accuracy in 2025.
- **Against B+ (Elo + ranking): probably, by a small margin.** The 2025 interval excludes zero only just, and the 2026 interval does not. The direction is the same in 2024, 2025 and 2026 (about -0.001 to -0.003).
- **Against C (D without the opponent-adjusted features): no.** C scored slightly *better* than D in 2024, 2025 and 2026 (2025: D minus C = +0.0005, interval -0.0013 to +0.0024; 2026: +0.0015, -0.0011 to +0.0042). These are exploratory comparisons and none is clear, but there is no evidence that the opponent-adjusted features help, on any dataset we have.
- **Where the value comes from (exploratory):** recent form. C beats B+ in both years with intervals excluding zero (2025: -0.0035; 2026: -0.0045). This is a stronger result than on the development data, where form was not demonstrated.
- **XGBoost adds nothing** over Logistic Regression on these features, in any period.
- **Ranking data is worth about 0.002 to 0.003 log loss.** The rank-independent model still beats Elo clearly.

Model D remains the primary research model by the owner's decision. Stated plainly: D is a sound model that beats Elo, and the simpler C matches it.

## 3. Calibration
Elo is overconfident (slope 0.80 in 2025, 0.78 in 2026). LR D is much closer but still slightly overconfident (0.92 and 0.95): its 80-90% predictions came true 81% of the time in 2025. Intercepts are near zero. Reliability plots: `figures/phase6_calibration_2025.png`, `figures/phase6_calibration_2026.png`.

## 4. Limits on these conclusions
- **2026 is a partial season** (tournament starts up to 2026-05-25): hard and clay only, 29 tournaments, wide intervals.
- **Many comparisons were made.** Only the primary one is confirmatory. A secondary or exploratory interval that just excludes zero (such as D vs B+ in 2025) is weak evidence.
- **Subgroup tables are descriptive.** Example: all models are near 0.50 log loss when a top-10 player is involved and near 0.67 for players ranked 31-100. No subgroup difference between models was tested.
- **No market benchmark.** Betting odds were not used, so we do not know how the model compares with bookmaker probabilities, which are usually the hardest benchmark.
- **Match dates are unknown** in this source; the conservative rule delays information by up to two weeks, which costs a little accuracy.
- **Both test sets are now used.** Any future change to the model cannot be validated on 2025 or on January-May 2026.

## 5. Recommendation for forecasting and the dashboard
1. **Use LR D as the production model** (owner's primary), refitted on all data with the frozen settings. Show **C as a challenger** next to it: it is simpler and has matched D in every period.
2. **Do not ship XGBoost.** It is more complex and no better.
3. **Show probabilities, not picks,** with the calibration caveat. A recalibration step may help, but it must be specified in advance and judged on new data, not on 2025/2026.
4. **Data freshness is now the binding constraint.** The approved data ends in May 2026. A current forecast needs a licensed, verified source. The rank-independent LR D is the fallback wherever rankings are missing, at a known cost of about 0.003 log loss.
5. **Reserve the next untouched period as the next test.** If a source for June-October 2026 is approved, score the frozen models on it once before changing anything.
6. **Add the odds benchmark** (as a comparison only, never as an input) before describing the model as useful for anything financial.
7. The dashboard should state that it is a research tool, not betting advice, and show the sample size behind every figure.
