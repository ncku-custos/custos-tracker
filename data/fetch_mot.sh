#!/usr/bin/env bash
# Fetch the M1 evaluation slice: MOT16-04 frames (static cam, crowded — good
# ID-switch signal). MOTChallenge has no per-sequence downloads; we take the
# full MOT16.zip (~1.9 GB) once, keep only what we need (~400 MB), and drop
# the rest. KEEP_ZIP=1 keeps the archive under data/downloads.
#
# Ground truth preference: MOT17-04 GT (more accurate re-annotation of the
# same pixels) when motchallenge.net is reachable; otherwise MOT16's own gt
# (shipped inside MOT16.zip) — flagged in the eval output filename.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p downloads mot

MOT16_MIRRORS=(
  "https://motchallenge.net/data/MOT16.zip"
  "https://bj.bcebos.com/v1/paddledet/data/mot/MOT16.zip"  # PaddleDetection docs mirror
)

if [[ ! -d mot/MOT16-04 ]]; then
  zip=downloads/MOT16.zip
  if [[ ! -f $zip ]]; then
    ok=0
    for m in "${MOT16_MIRRORS[@]}"; do
      echo "fetching MOT16.zip from $m"
      if curl -fSL --retry 2 --connect-timeout 15 -C - "$m" -o "$zip"; then ok=1; break; fi
    done
    [[ $ok = 1 ]] || { echo "no mirror served MOT16.zip" >&2; exit 1; }
  fi
  unzip -q -o "$zip" 'train/MOT16-04/*' -d downloads/mot16
  mv downloads/mot16/train/MOT16-04 mot/
  rm -rf downloads/mot16
  [[ ${KEEP_ZIP:-0} = 1 ]] || rm -f "$zip"
fi

if [[ ! -f mot/MOT16-04/gt/gt_eval.txt ]]; then
  lz=downloads/MOT17Labels.zip
  if [[ -f $lz ]] || curl -fSL --retry 2 --connect-timeout 15 \
      https://motchallenge.net/data/MOT17Labels.zip -o "$lz" 2>/dev/null; then
    unzip -q -o "$lz" 'train/MOT17-04-*/gt/gt.txt' -d downloads/mot17labels
    # All three MOT17-04-* GT files are identical (detector split only).
    cp "$(find downloads/mot17labels -name gt.txt | head -1)" mot/MOT16-04/gt/gt_eval.txt
    rm -rf downloads/mot17labels
    [[ ${KEEP_ZIP:-0} = 1 ]] || rm -f "$lz"
    echo "gt: using MOT17-04 re-annotation"
  else
    cp mot/MOT16-04/gt/gt.txt mot/MOT16-04/gt/gt_eval.txt
    echo "gt: motchallenge.net unreachable — using MOT16 native gt" >&2
  fi
fi

echo "mot: ready at data/mot/MOT16-04 (frames + gt/gt_eval.txt)"
