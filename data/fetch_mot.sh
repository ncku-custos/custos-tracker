#!/usr/bin/env bash
# Fetch the M1 evaluation slice: MOT16-04 frames (static cam, crowded — good
# ID-switch signal) paired with the more accurate MOT17 ground truth.
# MOTChallenge has no per-sequence downloads; we take the full MOT16.zip
# (~1.9 GB) once, keep only what we need (~400 MB), and drop the rest.
# KEEP_ZIP=1 to keep the archive in data/downloads for later sequences.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p downloads mot

if [[ ! -d mot/MOT16-04 ]]; then
  zip=downloads/MOT16.zip
  [[ -f $zip ]] || curl -fSL --retry 3 -C - https://motchallenge.net/data/MOT16.zip -o "$zip"
  unzip -q -o "$zip" 'train/MOT16-04/*' -d downloads/mot16
  mv downloads/mot16/train/MOT16-04 mot/
  rm -rf downloads/mot16
  [[ ${KEEP_ZIP:-0} = 1 ]] || rm -f "$zip"
fi

if [[ ! -f mot/MOT16-04/gt/gt_mot17.txt ]]; then
  lz=downloads/MOT17Labels.zip
  [[ -f $lz ]] || curl -fSL --retry 3 https://motchallenge.net/data/MOT17Labels.zip -o "$lz"
  unzip -q -o "$lz" 'train/MOT17-04-*/gt/gt.txt' -d downloads/mot17labels
  # All three MOT17-04-* GT files are identical (detector split only).
  cp "$(find downloads/mot17labels -name gt.txt | head -1)" mot/MOT16-04/gt/gt_mot17.txt
  rm -rf downloads/mot17labels
  [[ ${KEEP_ZIP:-0} = 1 ]] || rm -f "$lz"
fi

echo "mot: ready at data/mot/MOT16-04 (frames + gt/gt_mot17.txt)"
