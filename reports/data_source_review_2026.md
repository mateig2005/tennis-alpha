# Free ATP data sources for June-October 2026: review and recommendation (Phase 13)

Checked on 2026-10-20 by reading the sources' public pages, repository READMEs and licence files. **No dataset was downloaded.** Nothing here is legal advice; the permissions marked "to confirm" need a written answer from the publisher.

## 1. Comparison
| | **TennisMyLife (TML)** stats.tennismylife.org | **Open Tennis Data** github.com/ryantjx/tennis-match-data | **tennisdata.app** | **livetennisapi-data** |
|---|---|---|---|---|
| Reachable | Yes | Yes | **No**: connection reset from this network (twice, browser fetch and curl) | **No**: the repository and the GitHub organisation return 404. Live Tennis API is a commercial API with a free tier, not a data repository |
| ATP singles Jun-Oct 2026 | Yes. `2026.csv`; the page shows files updated 20/10/2026, "updated daily" | Partly: a "preview" release; an example build is dated as of 2026-07-24; Oct coverage unconfirmed | Unknown | Not applicable |
| **Serve/return statistics** | **Yes**: same columns as Sackmann (`w_ace`, `w_df`, `w_svpt`, `w_1stIn`, `w_1stWon`, `w_2ndWon`, `w_SvGms`, `w_bpSaved`, `w_bpFaced`, and the loser's), plus `indoor` | **No** (results only, by design) | Unknown | Free tier: limited; in-play statistics are paid |
| Real match dates | No (`tourney_date` = tournament week, as in Sackmann) | **Yes** (`date` = match date; never filled from tournament dates) | Unknown | API only |
| Surface, tournament | Yes (also `indoor`) | Tournament table (surface, indoor, level) | Unknown | API only |
| Player ids | **ATP ids** (plus `ATP_Database.csv`) | Own ids; scheme not documented | Unknown | API ids |
| Bookmaker odds | No | No | Unknown | No |
| Download | Free CSVs, "Download All" zip, and an API file list. No account | GitHub release (parquet) with SHA256 checksums | Unknown | API key |
| **Stated permission** | **Conflicting** (section 2) | MIT for code only. The data is a "source-attributed research dataset", "not represented as commercially reusable", and inherits CC BY-NC-SA (Sackmann) and CC BY-SA (Wikipedia) | Unknown | Commercial API terms |
| Provenance | ATP website (ids, results, tournaments) plus Sackmann's repository plus "newspapers, tennis blogs and other statistics websites" | tennis-data.co.uk, Wikipedia, Sackmann, community corrections (CC0); it refuses WTA and Tennis TV sources without permission | Unknown | Vendor |

**Other sources considered:**
- **Jeff Sackmann `tennis_atp`** (our current source): still unavailable (404). Our pinned mirror ends in May 2026.
- **tennis-data.co.uk** (results plus odds): blocked on this network by company policy (category "gambling"). I did not try to get around that. Its terms would need checking from an unrestricted network.
- **ATP website and Tennis TV:** their terms prohibit automated collection. Not used.
- **Zenodo "ATP Tennis dataset":** CC BY 4.0, but it covers 2020-2023 only. Not useful here.
- **Commercial scrapers** (Apify and others): not free in the sense we need. Excluded.

## 2. Free to view vs legally reusable: the TennisMyLife conflict
- The **download page** says "License: MIT License" and "Free to use".
- The project's **GitHub README** (TML-Database, now kept for historical reference) says the data is based on Jeff Sackmann's CC BY-NC-SA repository and on the official ATP website. It calls the data "non-commercial unless explicitly permitted" and "intended for educational, analytical, and research purposes only". It warns that "redistribution, commercial use, or selling of the raw database without permission from TennisMyLife and/or the ATP" may violate copyright or terms of use.
- The site has **no terms page and no contact address** other than an X account (@TennisMyLife68).

**Reading:** the MIT label cannot override the share-alike terms of data derived from Sackmann, or the ATP's rights in its statistics. The README's narrower terms are the safe interpretation.

**Our use would fit them:**
- non-commercial research;
- a portfolio that shows results, not the raw data;
- no redistribution of files, which stay git-ignored as with Sackmann;
- attribution, and share-alike for derived data.

**A written confirmation is still needed before use**, especially for machine-learning training and for showing derived statistics in a public app.

## 3. Recommendation
1. **Primary source for results plus serve statistics: TennisMyLife**, *after* a written confirmation (draft A). It is the only free source found with 2026 serve/return statistics in our exact schema. That matters because V2's gain comes from those statistics.
2. **Independent cross-check for results and real match dates: Open Tennis Data** (non-commercial, attribution, share-alike), after confirming that machine-learning training on it is fine (draft B). Use it for completeness and date checks, and as an exact-date source for a future model, not as model input to the frozen models.
3. **Odds: none legally usable today.** Valuebetennis stays excluded (licence and provenance unanswered). tennis-data.co.uk needs checking from another network and a terms review. The benchmark code is ready and waits.
4. **Do not use tennisdata.app** until it can be reached and its terms read.

## 4. Quality checks before any June-October row is opened
TML overlaps our data from 2010 to May 2026, so its quality can be measured on **already-known** matches first. No exposure of new outcomes is needed:
- **Player ids:** map ATP ids to our Sackmann ids with `src/v2/sources.link_players`, using names; review unmatched and ambiguous players by hand. ATP ids are stable, which helps later.
- **Agreement on 2025 to May 2026:** compare winners, scores, levels and surfaces with the Sackmann rows, and **serve statistics** within ±1 per count. Measure the share of tour-level matches with complete statistics.
- **Thresholds, set before looking:**
  - at least 98% of matches linked one-to-one;
  - at least 99.5% winner agreement;
  - at least 97% of linked tour-level matches with all nine statistics equal to Sackmann's;
  - at least 95% of 2026 tour-level matches with statistics.
  If any threshold fails, stop and report.
- **Then** run `validate_source` on the June-October file. It reports counts only (unparseable scores, duplicates, missing statistics), never individual results.

## 5. Staged ingestion and forward evaluation
| Stage | What happens | Outcomes seen? |
|---|---|---|
| 0. Permissions | Send drafts A/B; wait for answers; record them in `DATA.md` | No |
| 1. Pinned download | Into `data/raw/external/tml/`, git-ignored, with checksums and the date recorded. A loader splits **up to 2026-05-25** from **after 2026-05-25** at read time; the second part goes to a separate file nobody opens | No |
| 2. Overlap validation | Section 4 on the up-to-May-2026 part only | Only already-known matches |
| 3. Adapter plus checks | TML → V2 table (`tourney_date` → `event_date`; tournament start dates kept, as the frozen rule requires). `validate_source` counts on the new part | Counts only |
| 4. Forward evaluation, once | `python -m src.v2.forward_eval --data <adapter output>`. Checks frozen hashes, fits up to 2026-05-25, reports **aggregate** metrics only, periods A and B separately. Expected result for period A: "inconclusive" (about 1,000-1,500 matches) | Aggregates only |
| 5. App refresh | Only after stage 4: rebuild the dashboard, H2H and V2 artifacts with the extended history and a new snapshot date | Yes, after the test |
| 6. Prospective period B | Weekly appends after 2026-10-19, evaluated together at review points (protocol unchanged) | Aggregates only |

## 6. Earlier exposure to June-October 2026 (correction to PLAN_FORWARD.md)
The protocol said period A outcomes had never been looked at. **That needs a correction.** The audit was made by searching the full session record (the same record from 2026-10-13 to today).
- **What was downloaded:** the Valuebetennis file was downloaded on 2026-10-13 for the Valuebetennis audit. It holds ATP/WTA rows to 2026-10-13 **including winners, scores and odds**. It is git-ignored and stays on disk under `data/raw/external/valuebetennis/`.
- **What was seen:** structural metadata only (tournament names, per-tournament match counts and first/last match times). The details are in the private research archive.
- **What was not seen:** no June-October 2026 winner, score, odds or player-level result was printed or analysed. The audit script loaded the whole file but compared outcomes only on rows up to 2026-05-25, where Sackmann exists.
- **2026-10-20 (this review):** fetching the TennisMyLife homepage, which lists recent 2026 results, returned only a summary with no results in it.
- **What it could have affected:** nothing in the models. Model D/C were frozen in Phase 5; V2's settings were chosen on 2020-2022 development folds and frozen on 2026-10-19; structural match counts carry no information about who won.
- **What remains uncontrolled:** the owner and anyone else following tennis may know public results from June to October. That is inherent to a retrospective hold-out, and why period B (after 2026-10-19) is the real test.
- **Classification:** period A becomes a "retrospective hold-out with documented structural exposure (no outcomes examined)". Recommended: move the Valuebetennis 2026 file out of the project folder, or delete it (owner's decision), so it cannot be opened by accident.

## 7. Permission requests

Handled privately; no correspondence is published.

## Sources consulted
[TennisMyLife match database](https://stats.tennismylife.org/tennis-match-database) · [TennisMyLife home](https://stats.tennismylife.org/) · [TML-Database README](https://github.com/Tennismylife/TML-Database) · [Open Tennis Data](https://github.com/ryantjx/tennis-match-data) · [its DATA_LICENSE](https://github.com/ryantjx/tennis-match-data/blob/main/DATA_LICENSE.md) · [SOURCES](https://github.com/ryantjx/tennis-match-data/blob/main/docs/SOURCES.md) · [SCHEMA](https://github.com/ryantjx/tennis-match-data/blob/main/docs/SCHEMA.md) · [Live Tennis API listing](https://sportsapis.dev/apis/live-tennis-api) · [Zenodo ATP dataset](https://zenodo.org/record/10108666) · unreachable: tennisdata.app, tennis-data.co.uk (network policy), github.com/livetennisapi/livetennisapi-data (404), github.com/JeffSackmann/tennis_atp (404).
