#!/usr/bin/env bash
# TENNIS ALPHA: update the local ATP data now (TennisMyLife). Model coefficients are never changed.
#   ./update.sh            check for new results, rebuild the player data only if something changed
#   ./update.sh --force    rebuild even if nothing changed
#   ./update.sh --status   show the last attempt, success and error
# A running app picks up new data after a restart of ./dev.sh.
set -euo pipefail
cd "$(dirname "$0")"
exec .venv/bin/python -m src.sync "$@"
