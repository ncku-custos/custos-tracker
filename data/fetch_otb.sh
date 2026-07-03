#!/usr/bin/env bash
# Fetch the mini-OTB SOT evaluation set (6 sequences, ~150 MB): Car4, CarDark,
# BlurCar2, Human3, Girl2, DragonBaby — illumination, blur, occlusion, fast
# motion; the Car* pair is the most drone-relevant.
# Primary source is the Hanyang server, which is notoriously flaky; add
# mirrors to MIRRORS as they are found (M2 pins whichever works + SHA256s).
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p downloads otb

SEQS=(Car4 CarDark BlurCar2 Human3 Girl2 DragonBaby)
MIRRORS=(
  "http://cvlab.hanyang.ac.kr/tracker_benchmark/seq"
)

for seq in "${SEQS[@]}"; do
  [[ -d otb/$seq ]] && { echo "ok        $seq"; continue; }
  zip="downloads/$seq.zip"
  if [[ ! -f $zip ]]; then
    ok=0
    for m in "${MIRRORS[@]}"; do
      echo "fetching  $seq from $m"
      if curl -fSL --retry 2 --connect-timeout 15 "$m/$seq.zip" -o "$zip"; then ok=1; break; fi
    done
    [[ $ok = 1 ]] || { echo "no mirror served $seq — add a mirror to MIRRORS" >&2; exit 1; }
  fi
  unzip -q -t "$zip"
  unzip -q -o "$zip" -d otb/
done
echo "otb: ready at data/otb/ (${SEQS[*]})"
