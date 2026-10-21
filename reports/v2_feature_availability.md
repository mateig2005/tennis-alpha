# V2 feature availability and reliability

This report is based on the approved dataset only: Sackmann ATP matches 2010-2024 in the dev file, 42,372 matches. The locked 2025-2026 file was not used for this audit.

**What one row contains:**
- tournament name, an id whose code identifies the event, surface, draw size, level and start date;
- round, best-of format, score and minutes;
- each player's id, name, hand, height, country, age, seed, entry type, ranking and ranking points;
- per-player serve totals: aces, double faults, service points, first serves in, first- and second-serve points won, service games, break points faced and saved.

**Statistics coverage:**
- 99.5% of the 25,373 tour-level matches from 2015 to 2024 have complete, consistent statistics for both players.
- Davis Cup has statistics in only 36% of matches.

**What is not in the data:** match dates, court speed, indoor or outdoor, altitude, weather, ball brand, and injuries or withdrawals.

## 1. Available and reliably computable now (built in V2 stage 1 unless marked)
| Feature | How it is computed | Notes |
|---|---|---|
| Serve and return points won %, opponent-adjusted | point-weighted, decayed, shrunk; uses the opponent's pre-match rating | V2 `sr_adj`. The strongest new signal. |
| Surface-specific serve and return | the same, per surface, shrunk toward the general rating | V2 `srv_s`, `ret_s` |
| Aces and double faults per service point | rolling over the last 20 matches with statistics, shrunk | Aces per service *game* is also possible; not added because it is redundant. |
| First-serve in %, and first- and second-serve points won % | rolling over the last 20, shrunk | V2 `sr_raw` |
| Service games held % | games held = service games − (break points faced − saved) | Exact. 0 inconsistent rows in dev. |
| First- and second-serve return points won % | from the opponent's serve totals | Exact identities. |
| Return games won %, break points converted % | breaks = opponent's break points faced − saved | V2 `sr_raw`, `pressure` |
| Break points saved % | shrunk with 15 pseudo break points | V2 `pressure`. Treated as a hypothesis, not "mental strength". |
| Recent serve and return trend | 60-day half-life ratings | V2 `momentum` |
| Elo change over 90 days | engine pre-match ratings | V2 `momentum` |
| Best of 3 vs best of 5 | `best_of` column | V2 `context` (best-of-5 × Elo) and inside the matchup model |
| Tournament level and round | columns | V1 uses them for visibility. Level and round interactions are not yet tested. |
| Pre-match head-to-head | meetings released before the match, shrunk | V2 `h2h`. Separate from the display H2H. |
| Strength of schedule | mean opponent Elo | Already in V1 (`opp10`). |
| Matches and recent activity counts | counts over 30, 90 and 365 days | Already in V1 (`ln30`, `ln90`, `ln365`). |
| Handedness, height, age | player columns | Not used. Possible style dimensions (left-hander matchups) for a later stage. |

## 2. Partially computable, with explicit assumptions or uncertainty
| Feature | Problem | What V2 does |
|---|---|---|
| **Tournament pace index** | No measured court speed. The proxy is ace-rate and serve-points-won residuals against the players' expected values, taken from **previous editions** only. Some events change code between years (e.g. Brisbane 2016-18) or change venue under the same code (Chennai to Pune). The returner's effect on aces is ignored. | Built, labelled as a statistical proxy, shrunk toward the surface average. Ranking check passes: grass events, Atlanta and Marseille fastest; Monte Carlo and Barcelona slowest. |
| Fast, medium and slow hard courts | Only through the pace proxy above | Thirds of the proxy, for evaluation only. |
| Rest and recovery | Only tournament **start** dates, so days since the last match are a lower bound | V1's `rest` is a conservative lower bound. Nothing more precise is invented. |
| Workload within a tournament (previous rounds, sets, minutes) | The round order is known but match days are not; minutes are missing for 10% of matches | Not built. Possible as "sets or minutes played earlier in this tournament", since earlier rounds' scores are known by the time a later round starts. But V1's visibility rule hides same-tournament results, so this needs a separate, carefully argued rule. |
| Tiebreak frequency and results | Parse "7-6" sets from the score (38% of matches have one). Very noisy per player. | Not built yet. Possible next step, heavily shrunk. |
| Deciding-set performance, long-match performance | Parse the score; small samples | Not built (hypothesis-level). |
| Returning after an absence | The size of the gap is known (V1 `rest`); **the reason is not** | Present only as a gap length. The Sinner-Alcaraz audit shows this matters. |
| Style dimensions and clusters | Can be derived from serve and return rates | Continuous dimensions are already there (serve rating, return rating, ace rate, first-serve dependence). Discrete clusters are not built: no evidence yet that they add anything beyond continuous ratings. |
| Performance against strong servers or high-ace opponents | Possible from history; samples per player are small | Not built. The cross term `dom_cross` and the matchup model capture part of it. |

## 3. Unavailable until properly licensed data is added
| Feature | Needed source |
|---|---|
| **Actual match dates** | A source with match-level dates (it would also allow exact rest, fatigue and within-tournament visibility) |
| Measured court speed (for example the ITF Court Pace Rating) | Official or published surface ratings |
| Indoor or outdoor | A curated venue table: small, but it is external data and needs approval and provenance |
| Altitude | The same venue table (venue elevations) |
| Weather: temperature, humidity, wind | Historical weather by venue and day. Needs match dates first. |
| Ball type | Tournament ball documentation, not in any current source |
| Injuries, withdrawals, medical time-outs | Not in the data. Walkovers are removed by cleaning and their reasons are unknown. |
| Point-by-point and shot-level data (rally length, serve direction) | Licensed point-by-point data |
| Current rankings after May 2026 | A verified, legally usable rankings source |
| Player identity across sources | A mapping table. V1 already has the canonical-id design (`sackmann:` prefix) and an identity checker. |

## 4. Reliability notes
- **Statistics are post-match.** They enter a player's profile only on the release day (tournament start plus maximum length), the same rule as results. Tests check this.
- **Retirements:** their statistics are kept (rates per point are still valid), but they are short matches, so they carry little weight. Walkovers have no statistics.
- **Missing statistics** are treated as missing: the match still counts for Elo, form and H2H but adds nothing to serve and return profiles. 0.5% of tour-level matches and 64% of Davis Cup matches have none.
- **Shrinkage:**
  - Every rate is pulled toward the tour average with fixed pseudo-counts: 240 points (about 3 matches), 36 games and 15 break points.
  - Opponent-adjusted ratings use 300 points.
  - H2H uses 5 meetings and pace uses 40 matches.
  - None of these was tuned.
- **Data-quality checks run:**
  - Each side's figures must be internally consistent: first serves in ≤ service points, first-serve points won ≤ first serves in, second-serve points won ≤ second serves, break points saved ≤ faced, service points > 0. Rows that fail are treated as missing.
  - Rows where both players have zero aces make up 0.7% of matches. They are spread across events, not concentrated in particular tournaments, so they are kept.

## 5. Roadmap for richer data (each step needs the owner's approval and a licence check)
1. **Match dates.** This is the biggest win. It enables exact rest, fatigue and within-tournament form, and weather.
2. **Venue table** (indoor or outdoor, altitude, city), mapped to the event code. The data adapter already carries a `venue` key.
3. **Measured court pace.** If available, compare it with the V2 proxy and keep both, labelled.
4. **Fresh results after May 2026** from a licensed source, through a new adapter producing the V2 columns (`src/v2/data.py` documents the schema). This also supplies the **prospective test period**.
5. **Withdrawal and injury records**, if a source documents them, to separate rest from injury after an absence.
