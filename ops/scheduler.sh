#!/usr/bin/env bash
# TENNIS ALPHA daily update with macOS launchd (a per-user agent: no admin rights, nothing leaves this Mac).
#   ops/scheduler.sh install     run ./update.sh every day at 07:30 and at each login (catch-up after the Mac was off)
#   ops/scheduler.sh uninstall   stop and remove it
#   ops/scheduler.sh status      is it installed and loaded?
# Installing is the owner's decision: nothing in the project runs this script automatically.
set -euo pipefail
cd "$(dirname "$0")/.."
LABEL="com.tennisalpha.sync"
PLIST="$HOME/Library/LaunchAgents/$LABEL.plist"
case "${1:-status}" in
  install)
    mkdir -p "$HOME/Library/LaunchAgents" logs
    sed -e "s#__PROJECT__#$(pwd)#g" ops/launchd.plist.template > "$PLIST"   # machine paths go only into ~/Library, never into git
    launchctl bootstrap "gui/$(id -u)" "$PLIST"
    sleep 8   # RunAtLoad starts one run immediately: check that macOS lets it read the project
    if launchctl print "gui/$(id -u)/$LABEL" | grep -q "last exit code = 126"; then
      launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true; rm -f "$PLIST"
      echo "NOT installed: macOS blocked the background job from reading this folder ('Operation not permitted')."
      echo "Folders such as ~/Documents, ~/Desktop and ~/Downloads are privacy-protected. Move the project to e.g. ~/Developer,"
      echo "or allow /bin/bash in System Settings > Privacy & Security > Full Disk Access, then run this again."
      exit 1
    fi
    echo "Installed $LABEL: daily at 07:30 and at login. Log: logs/launchd.log";;
  uninstall)
    launchctl bootout "gui/$(id -u)/$LABEL" 2>/dev/null || true
    rm -f "$PLIST"; echo "Removed $LABEL";;
  status)
    if launchctl print "gui/$(id -u)/$LABEL" >/dev/null 2>&1; then echo "$LABEL is installed and loaded"; else echo "$LABEL is not installed"; fi;;
  *) echo "usage: ops/scheduler.sh install|uninstall|status"; exit 1;;
esac
