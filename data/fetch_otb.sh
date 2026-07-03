#!/usr/bin/env bash
# Fetch the mini-OTB SOT evaluation set (6 sequences, ~150 MB): Car4, CarDark,
# BlurCar2, Human3, Girl2, DragonBaby — illumination, blur, occlusion, fast
# motion; the Car* pair is the most drone-relevant.
# Primary source is the Hanyang server, which is notoriously flaky; add
# mirrors to MIRRORS as they are found (M2 pins whichever works + SHA256s).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p downloads otb

SEQS=(Car4 CarDark BlurCar2 Jogging Girl2 Woman)  # DragonBaby never archived -> Woman; Human3 zip too large for stable Wayback transfer -> Jogging
MIRRORS=(
  # Hanyang's server 404s these paths as of 2026-07; the Wayback Machine
  # snapshot serves the original zips verbatim (id_ = raw, no rewriting).
  "https://web.archive.org/web/2016id_/http://cvlab.hanyang.ac.kr/tracker_benchmark/seq"
  "https://web.archive.org/web/2020id_/http://cvlab.hanyang.ac.kr/tracker_benchmark/seq"
  "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq"
)

for seq in "${SEQS[@]}"; do
  [[ -d otb/$seq ]] && { echo "ok        $seq"; continue; }
  zip="downloads/$seq.zip"
  # A zip only counts once it passes integrity — truncated mirror transfers
  # (seen with Wayback) are deleted and the next mirror is tried.
  if [[ -f $zip ]] && ! unzip -q -t "$zip" >/dev/null 2>&1; then
    echo "corrupt   $seq.zip — refetching" >&2
    rm -f "$zip"
  fi
  if [[ ! -f $zip ]]; then
    # web.archive.org aborts large HTTP/2 transfers near the end; force
    # HTTP/1.1 and resume the partial across attempts instead of deleting it.
    ok=0
    for m in "${MIRRORS[@]}"; do
      for attempt in 1 2 3 4 5; do
        echo "fetching  $seq from $m (attempt $attempt)"
        curl -fSL --http1.1 --retry 2 --connect-timeout 15 -C - "$m/$seq.zip" \
          -o "$zip.part" || true
        if unzip -q -t "$zip.part" >/dev/null 2>&1; then
          mv "$zip.part" "$zip"
          ok=1
          break 2
        fi
      done
      rm -f "$zip.part"  # partial is mirror-specific; do not resume across mirrors
    done
    [[ $ok = 1 ]] || { echo "no mirror served $seq — add a mirror to MIRRORS" >&2; exit 1; }
  fi
  unzip -q -o "$zip" -d otb/
  rm -rf otb/__MACOSX
done
echo "otb: ready at data/otb/ (${SEQS[*]})"
