# Data provenance

## Source
- **Author:** Jeff Sackmann / [Tennis Abstract](http://www.tennisabstract.com/) ("Tennis databases, files, and algorithms").
- **Original repository:** `github.com/JeffSackmann/tennis_atp`. It returned HTTP 404 when checked on 2026-10-13, so it could not be used or compared against.
- **Copy used:** [Aneeshers/tennis-sackmann-archive](https://huggingface.co/datasets/Aneeshers/tennis-sackmann-archive) on Hugging Face (a third-party archival mirror), pinned to commit `8ac86f7c62889b5d5c4cfec4c6403edcb8c5d993` (last modified 2026-06-25). Files: `atp/atp_matches_2010.csv` ... `atp_matches_2026.csv`.
- **Column definitions:** `data/raw/matches_data_dictionary.txt` (downloaded with the data).
- **Integrity:** SHA-256 of each downloaded file is in `data/raw/manifest.json`.

## Licence
[CC BY-NC-SA 4.0](https://creativecommons.org/licenses/by-nc-sa/4.0/). Verified in the Hugging Face dataset card and in Sackmann's original README, which the mirror preserves (`UPSTREAM_README.md`).
- **Attribution:** credit Jeff Sackmann / Tennis Abstract and link to the source.
- **NonCommercial:** no commercial use.
- **ShareAlike:** derived data must carry the same licence.
- **Changes made:** completed-match filtering, de-duplication, implausible heights set to missing, added columns (`retired`, `round_rank`, `match_id`, `split`). Nothing else was altered.

Raw and processed data are **not** committed to this repository.

## Reproduce
```bash
source .venv/bin/activate
pip install -r requirements.txt
python src/download_data.py     # ~9 MB into data/raw/
python src/prepare_data.py      # writes data/processed/atp_matches_clean.csv + reports/data_report.md
```
Results of the last run (matches per year and split, latest date, missing values): [reports/data_report.md](reports/data_report.md).

## Caveats
- **Mirror vs original:** the mirror could not be compared with Sackmann's original repository (404). It is a snapshot of 2026-06-25; matches after the French Open (Roland Garros 2026, started 2026-05-25) are absent.
- **No match dates:** only `tourney_date` (tournament start, usually a Monday) exists. The latest "match date" is therefore the latest tournament start, not an exact day.
- **Davis Cup (`D`) and Olympics (`O`)** are included. Davis Cup has ~67% missing serve stats. Decide in Phase 3 whether to keep them.

## Chronological split (by tournament-start year)
| Split | Years | Use |
|---|---|---|
| warmup | 2010-2014 | build initial Elo / history only |
| train | 2015-2023 | fit models |
| validation | 2024 | tune hyperparameters |
| test | 2025 | touch once, after models and settings are frozen |
| forward_oos | 2026 | evaluate once, after the test set, with frozen settings |

## Leakage rules
1. **Visibility:** a match may use only matches with a strictly smaller `(tourney_date, round_rank)`. Same week and round counts as simultaneous, so ratings must be updated per group, after predicting it. This is conservative on purpose, because true match days are unknown.
2. **Post-match columns** (`score`, `minutes`, all `w_*`/`l_*` stats) describe the match itself. Use them only as lagged history from earlier matches.
3. **Label leak:** `winner_*` / `loser_*` reveals the result. Randomly assign players to A/B before building features.
4. **No fitting on future data:** scalers, imputers and hyperparameters are fitted on train only; the test and 2026 sets are never used to make decisions.

## Candidate external source (not merged)
Valuebetennis (CC BY 4.0, includes real match dates) was audited against Sackmann in [the private Valuebetennis audit (not published)](the private Valuebetennis audit (not published)). Status: **not merged**, pending written clarification of upstream provenance. Its Jan-May 2026 coverage is incomplete (Rome, Hamburg, Geneva and Roland Garros round 1 are largely missing) and its Jun-Oct 2026 completeness is unverified. Its odds columns are never used as model inputs. Downloaded files stay in the untracked `data/raw/external/`.

## Second source: TennisMyLife (from Phase 14)
- **Source:** [stats.tennismylife.org](https://stats.tennismylife.org/tennis-match-database) (ATP singles, Sackmann-style columns plus `indoor`, ATP player ids). Credit: TennisMyLife.
- **Permission (as reported by the owner; the correspondence is held privately and is not in this repository):**
  1. 2026-10-20: non-commercial machine-learning research, model training, evaluation, derived analytics and publication of the source code on GitHub.
  2. 2026-10-21: automated daily checks and downloads of the TennisMyLife ATP files on the owner's personal Mac (`./update.sh`, launchd at 07:30 and at login; at most one successful download per 20 hours).
  - Conditions: strictly non-commercial use; visible attribution to https://stats.tennismylife.org; credit to upstream providers (ATP, and Jeff Sackmann / Tennis Abstract for earlier data); no redistribution or substantial reconstruction of the raw datasets; contact TennisMyLife before any commercial use or material change of purpose; the data is not MIT-licensed.
- **Files:** `data/raw/tml/` (git-ignored). `manifest.json` records sha256, server Last-Modified and fetch time per file. TML rewrites past seasons (corrections); there is no changelog.
- **Adapter:** `src/tml.py`, run as `python -m src.tml download | validate | prepare-eval`.
  - Level codes: `250`/`500`/`A` → `A`.
  - Tournament ids are compared without zero padding.
  - Each row gets its tournament's earliest date. TML dates some rows of one event on different days, for example Roland Garros 2026.
  - Match ids are tournament + round + both ATP ids, because `match_num` is sometimes missing.
  - Players are linked to our ids through shared matches first (528 linked, 0 conflicts), then by unique name; anything unclear stays a new id.
- **Validation on already-known matches** (2024 to May 2026, `reports/tml_validation.md`): linked 98.2%, same winner 100%, statistics identical 96.3% (98.1% after TML's internally impossible rows are treated as missing). One threshold fails on the raw comparison.
- **Owner acceptance (2026-10-20):** accepted with the limitations above. Both statistics figures (96.3% raw, failed; 98.1% adjusted) are kept. Rows failing the consistency rule: 126 in the 2024 to May 2026 overlap (2,268 statistic values treated as missing; by rule: service games > 0: 119, second-serve points: 5, break points: 5, first serves: 3, others: 2; a row can fail more than one rule), 0 in period A. Other discrepancies kept for later investigation: 118 unlinked known tour-level rows per source, score agreement 99.0%, rank agreement 97.7%.
- **Known TML defect:** service games recorded as 0 for about 120 Australian Open matches. Our consistency rule treats those statistics as missing; nothing is filled in.
- **Protected rows:** TML rows from tournaments starting after 2026-05-25 are written to `data/protected/` (git-ignored) and are not displayed until the owner approves the forward evaluation (PLAN_FORWARD.md). The raw `2026.csv` also contains them.
- **History is not replaced:** 2010 to May 2026 stays the pinned Sackmann data, so the frozen models and every reported result remain reproducible. TML is appended only after the cutoff.
- **App refresh (Phase 15):** the local player state includes TennisMyLife tournaments that started after 2026-05-25 and must have finished before the download date (start + MAX_DAYS < download day). As of 2026-10-20: 55 tournaments and 1,072 matches added, results complete through 2026-10-07, Shanghai pending. The coverage of every build is in `artifacts/status.json` (local).
