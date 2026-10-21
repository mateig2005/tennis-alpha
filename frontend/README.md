# TENNIS ALPHA frontend

*Precision in every point.* The Next.js interface for the ATP match-prediction research in the parent folder.

Every number on the Match Prediction and Analytics pages comes from the Python models through a local API (`../src/api.py`), the same `Predictor` the Streamlit dashboard uses. Model Lab shows the independent 2025/2026 backtest from `../reports/phase6_backtest.md`. There is no live ATP data: predictions are for a hypothetical match on the snapshot date (8 June 2026), and the interface says so on every page. No betting functionality.

## Run it
From the project root (the folder above this one):
```bash
./dev.sh
```
This starts the Python API on http://127.0.0.1:8000 and the web app on http://localhost:3100, both on this computer only. Press Ctrl+C to stop both. The web app forwards `/api/*` to the Python API, so the browser only talks to one address.

Checks, run inside `frontend/`: `npm test` (40 tests), `npm run lint`, `npm run typecheck`, `npm run build`.

## Where things are
See [DESIGN.md](DESIGN.md). In short: pages in `src/app`, components in `src/components`, and `src/lib/api.ts` is the only file that talks to the Python API.
