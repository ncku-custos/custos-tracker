#!/usr/bin/env bash
# Drop zone for project demo clips (drone footage etc.). Nothing public and
# stable to auto-fetch yet — place mp4s under data/clips/ by hand; they stay
# out of git. This script only prepares the directory and lists what's there.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p clips
ls -lh clips/ 2>/dev/null || true
echo "clips: put demo mp4s in data/clips/ (gitignored)"
