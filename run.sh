#!/usr/bin/env bash
# Bash runner for baby-monitor-pi (ASCII only)

set -Eeuo pipefail

# Basic env (avoid UTF-8 locales in logs)
export HOME=/home/admin
export LANG=C
export LC_ALL=C
export PATH=/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin

# Toggle Firebase push (1 = skip push, 0 = enable push)
export FIREBASE_SKIP=1

# Mic device for cry detector (use the value you tested)
export CRY_DEVICE="pulse"

# Project paths
ROOT="/home/admin/baby-monitor-pi/baby-monitor-pi"
LOG="$ROOT/boot.log"
VENV="$ROOT/.venv/bin/activate"
PYBIN="$ROOT/.venv/bin/python"

# Firebase credentials (adjust if your file moves)
export FIREBASE_CRED_JSON=/home/admin/baby-monitor-pi/baby-monitor-pi/src/baby-sleep-tracker-a4f6c-firebase-adminsdk-fbsvc-beb6936966.json
export FIREBASE_DB_URL="https://baby-sleep-tracker-a4f6c-default-rtdb.firebaseio.com/"
export FIREBASE_BASE_PATH="sleepData"

# Go to project
cd "$ROOT" || exit 1

# Activate venv
# shellcheck source=/dev/null
source "$VENV"

# Wait for network up to ~60s
for i in {1..30}; do
  if ping -c1 -W1 8.8.8.8 >/dev/null 2>&1; then
    break
  fi
  sleep 2
done

# Optional: ensure only one instance uses the camera
pgrep -f "python -u -m src.main" | xargs -r kill || true

# Unbuffered Python output
export PYTHONUNBUFFERED=1

# Start app
exec "$PYBIN" -u -m src.main >> "$LOG" 2>&1