#!/usr/bin/env bash
# Start ClipStorm. The first run sets everything up by itself.
set -e
cd "$(dirname "$0")"
# settings for hosting (CLIPSTORM_HOST, CLIPSTORM_PASSWORD...) can live in a .env file next to this script
[ -f .env ] && set -a && . ./.env && set +a
command -v ffmpeg >/dev/null || { echo "ffmpeg is missing. Install it first:  sudo apt install ffmpeg   (Mac: brew install ffmpeg)"; exit 1; }
if [ ! -x .venv/bin/python ]; then
  echo "First run: setting up (a minute or two)..."
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
exec .venv/bin/python -m clipstorm.server
