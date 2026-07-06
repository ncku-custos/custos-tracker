#!/usr/bin/env bash
# Fetch the MOT evaluation slices: MOT16-04 (static cam, crowded — good
# ID-switch signal, the M1 scenario) and MOT16-13 (bus-mounted moving camera —
# the GMC eval scenario). MOTChallenge has no per-sequence downloads; we take
# the full MOT16.zip (~1.9 GB) once, keep only what we need, and drop the rest.
# KEEP_ZIP=1 keeps the archive under data/downloads; MOT_SEQS="MOT16-04 ..."
# overrides the sequence list.
#
# Ground truth preference: MOT17 GT (more accurate re-annotation of the same
# pixels) when motchallenge.net is reachable; otherwise MOT16's own gt
# (shipped inside MOT16.zip) — flagged in the eval output filename.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p downloads mot

read -ra SEQS <<< "${MOT_SEQS:-MOT16-04 MOT16-13}"

MOT16_MIRRORS=(
  "https://motchallenge.net/data/MOT16.zip"
  "https://bj.bcebos.com/v1/paddledet/data/mot/MOT16.zip"  # PaddleDetection docs mirror
)

zip=downloads/MOT16.zip
used_zip=0
for seq in "${SEQS[@]}"; do
  [[ -d mot/$seq ]] && continue
  if [[ ! -f $zip ]]; then
    ok=0
    for m in "${MOT16_MIRRORS[@]}"; do
      echo "fetching MOT16.zip from $m"
      if curl -fSL --retry 2 --connect-timeout 15 -C - "$m" -o "$zip"; then ok=1; break; fi
    done
    [[ $ok = 1 ]] || { echo "no mirror served MOT16.zip" >&2; exit 1; }
  fi
  used_zip=1
  # Two known internal layouts: official 'train/...' and the PaddleDetection
  # mirror's 'MOT16/images/train/...'.
  if unzip -l "$zip" | grep -q " train/$seq/"; then
    unzip -q -o "$zip" "train/$seq/*" -d downloads/mot16
    mv "downloads/mot16/train/$seq" mot/
  else
    unzip -q -o "$zip" "MOT16/images/train/$seq/*" -d downloads/mot16
    mv "downloads/mot16/MOT16/images/train/$seq" mot/
  fi
  rm -rf downloads/mot16
done
[[ ${KEEP_ZIP:-0} = 1 || $used_zip = 0 ]] || rm -f "$zip"

used_labels=0
for seq in "${SEQS[@]}"; do
  [[ -f mot/$seq/gt/gt_eval.txt ]] && continue
  n=${seq#MOT16-}
  lz=downloads/MOT17Labels.zip
  if [[ -f $lz ]] || curl -fSL --retry 2 --connect-timeout 15 \
      https://motchallenge.net/data/MOT17Labels.zip -o "$lz" 2>/dev/null; then
    used_labels=1
    unzip -q -o "$lz" "train/MOT17-$n-*/gt/gt.txt" -d downloads/mot17labels
    # All MOT17-<n>-* GT files are identical (detector split only).
    cp "$(find downloads/mot17labels -name gt.txt | head -1)" "mot/$seq/gt/gt_eval.txt"
    rm -rf downloads/mot17labels
    echo "gt($seq): using MOT17-$n re-annotation"
  else
    cp "mot/$seq/gt/gt.txt" "mot/$seq/gt/gt_eval.txt"
    echo "gt($seq): motchallenge.net unreachable — using MOT16 native gt" >&2
  fi
done
[[ ${KEEP_ZIP:-0} = 1 || $used_labels = 0 ]] || rm -f downloads/MOT17Labels.zip

echo "mot: ready at data/mot/{${SEQS[*]}} (frames + gt/gt_eval.txt)"
