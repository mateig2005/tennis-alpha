"""Local data sync (Phase 15): TennisMyLife -> validated player state. Never retrains a model.

  ./update.sh                    (same as: python -m src.sync)      check, download changes, rebuild only if needed
  python -m src.sync --force     rebuild even if no file changed
  python -m src.sync --status    print the last attempt / success / error

Steps: download the needed TML files into a temporary folder; check each parses with the expected columns; replace only
files whose sha256 changed (a changed past season = a correction); rebuild and validate the artifacts in staging; activate
them only if every check passes. Any failure keeps the previous files and artifacts and is recorded in logs/sync.log and
data/sync_state.json. TML offers whole files only (no incremental feed), so changed files are downloaded in full.
"""
import argparse
import json
import shutil
import subprocess
import tempfile
import traceback
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from src import refresh, tml

STATE, LOG = Path("data/sync_state.json"), Path("logs/sync.log")
MIN_HOURS = 20  # at most one successful download per day: the 07:30 run and the login catch-up never both download
REQUIRED_COLUMNS = {"tourney_id", "tourney_date", "tourney_level", "round", "winner_id", "loser_id", "winner_name", "loser_name", "score", *tml.STAT_COLS}


def now():
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def log(msg):
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(f"{now()}  {msg}\n")


def state():
    return json.loads(STATE.read_text()) if STATE.exists() else {}


def save_state(**kw):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(state() | kw, indent=2))


def needed_files(today=None):
    year = (today or datetime.now(timezone.utc)).year
    return ["2024.csv", "2025.csv"] + [f"{y}.csv" for y in range(2026, year + 1)]  # 2024-25: player-id linking overlap


def download_changes(names, raw=tml.RAW):
    """Download into a temp folder, check, then move only changed files into place. Returns the changed names."""
    manifest_path = raw / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    changed = []
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for name in names:
            try:
                tml.fetch([name], out=tmp)
            except subprocess.CalledProcessError:
                if name[:4].isdigit() and int(name[:4]) > 2026 and name not in manifest["files"]:
                    continue  # a new season's file may not exist yet in early January
                raise
            cols = set(pd.read_csv(tmp / name, nrows=5).columns)
            if not REQUIRED_COLUMNS <= cols:
                raise ValueError(f"{name}: unexpected format, missing {sorted(REQUIRED_COLUMNS - cols)}")
            entry = json.loads((tmp / "manifest.json").read_text())["files"][name]
            if manifest["files"].get(name, {}).get("sha256") != entry["sha256"]:
                changed.append(name)
            manifest["files"][name] = entry  # fetch time is refreshed even when the content is unchanged
        for name in changed:
            shutil.move(tmp / name, raw / name)
    manifest_path.write_text(json.dumps(manifest, indent=2))
    return changed


def run(force=False, names=None):
    """One sync. Returns 0 on success (including 'nothing to do'), 1 on failure; previous artifacts are kept on failure."""
    last = state().get("last_success_utc")
    if not force and last and datetime.now(timezone.utc) - datetime.fromisoformat(last) < pd.Timedelta(hours=MIN_HOURS):
        log(f"skipped: last successful check {last} is less than {MIN_HOURS} h ago")
        return 0
    save_state(last_attempt_utc=now())
    try:
        changed = download_changes(names or needed_files())
        status = json.loads((refresh.ART / "status.json").read_text()) if (refresh.ART / "status.json").exists() else {}
        today = pd.Timestamp.now(tz="UTC").tz_convert(None).normalize()
        pending_may_finish = status.get("pending_tournaments") and status.get("download_date") != str(today.date())
        if not (changed or force or pending_may_finish or not status):
            log("no file changed; artifacts already current")
            save_state(last_success_utc=now(), last_result="no change", last_error=None)
            return 0
        ok, info = refresh.run()
        if not ok:
            raise RuntimeError("validation failed: " + "; ".join(info))
        log(f"activated new artifacts: changed files {changed or 'none'}, data to {info['latest_complete_through']}, +{info['tml_matches_added']} TML matches")
        save_state(last_success_utc=now(), last_result=f"artifacts rebuilt (data complete through {info['latest_complete_through']})", last_error=None)
        return 0
    except Exception as e:
        log(f"FAILED, previous artifacts kept: {e!r}\n{traceback.format_exc()}")
        save_state(last_error=f"{now()}: {e}", last_result="failed")
        return 1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--status", action="store_true")
    args = ap.parse_args()
    if args.status:
        print(json.dumps(state(), indent=2) or "never run")
        return
    code = run(force=args.force)
    print("Sync OK" if code == 0 else f"Sync FAILED (see {LOG}); the previous data is still in use.", "|", state().get("last_result"))
    raise SystemExit(code)


if __name__ == "__main__":
    main()
