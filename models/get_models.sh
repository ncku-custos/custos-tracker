#!/usr/bin/env bash
# Fetch all model weights pinned in manifest.json into models/cache/,
# verifying SHA256. Idempotent: verified files are not re-downloaded.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p cache

python3 - <<'EOF' | while read -r file url sha; do
import json
with open("manifest.json") as f:
    for m in json.load(f)["models"]:
        print(m["file"], m["url"], m["sha256"])
EOF
  dst="cache/$file"
  if [[ -f $dst ]] && echo "$sha  $dst" | sha256sum -c --quiet 2>/dev/null; then
    echo "ok        $file"
    continue
  fi
  echo "fetching  $file"
  curl -fSL --retry 3 "$url" -o "$dst"
  echo "$sha  $dst" | sha256sum -c --quiet || { echo "SHA256 MISMATCH: $file" >&2; exit 1; }
done
echo "models: all verified"
