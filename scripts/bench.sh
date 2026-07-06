#!/usr/bin/env bash
# Reproducible latency benchmark for the optimization work in docs/RESULTS.md.
#
# Runs each scenario REPS times, writes per-run bench JSONs under
# results/bench/, and prints a markdown table of the per-run p50 of the
# scenario's headline stage plus the median and stddev across runs.
#
# Usage:
#   scripts/bench.sh [--reps N] [--scenario NAME]
#
# Environment:
#   CTRK_BENCH_TASKSET  cpu list for `taskset -c` pinning experiments
#                       (e.g. "0-3" = P-cores on the i5-1335U); default unpinned.
#   CTRK_BENCH_BUILD    build directory to take binaries from (default: build).
#
# Methodology (docs/RESULTS.md "Speed optimization"): AC power, machine otherwise idle,
# report the median p50; a change is real only if it clears 2x the stddev
# measured here. Governor/pinning are recorded in the report line.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD="${CTRK_BENCH_BUILD:-$ROOT/build}"
OUT="$ROOT/results/bench"
REPS=3
ONLY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --reps) REPS="$2"; shift 2 ;;
    --scenario) ONLY="$2"; shift 2 ;;
    *) echo "unknown arg: $1" >&2; exit 1 ;;
  esac
done

mkdir -p "$OUT"
PIN=()
[[ -n "${CTRK_BENCH_TASKSET:-}" ]] && PIN=(taskset -c "$CTRK_BENCH_TASKSET")

# name | headline stage | command (bench json path appended per run)
run_scenario() {
  local name="$1" stage="$2"; shift 2
  local jsons=()
  for ((r = 1; r <= REPS; r++)); do
    local json="$OUT/${name}_run${r}.json"
    "${PIN[@]}" "$@" --bench-json="$json" > /dev/null 2>&1
    jsons+=("$json")
  done
  python3 - "$name" "$stage" "${jsons[@]}" <<'EOF'
import json, statistics, sys
name, stage, paths = sys.argv[1], sys.argv[2], sys.argv[3:]
p50s = []
for p in paths:
    with open(p) as f:
        p50s.append(json.load(f)["stages"][stage]["p50_ms"])
med = statistics.median(p50s)
sd = statistics.pstdev(p50s) if len(p50s) > 1 else 0.0
runs = " ".join(f"{v:.3f}" for v in p50s)
fps = 1000.0 / med if med > 0 else 0.0
print(f"| {name} | {stage} | {runs} | {med:.3f} | {sd:.3f} | {fps:.0f} |")
EOF
}

GOV="$(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo unknown)"
echo "governor: $GOV · nproc: $(nproc) · pin: ${CTRK_BENCH_TASKSET:-none} · reps: $REPS"
echo
echo "| scenario | stage | p50 per run (ms) | median | stddev | FPS |"
echo "|---|---|---|---|---|---|"

want() { [[ -z "$ONLY" || "$ONLY" == "$1" ]]; }

# TBD: full MOT16-04 (1050 frames, 1920x1080), person class.
if want tbd_mot16; then
  run_scenario tbd_mot16 "detect+track" \
    "$BUILD/apps/track_tbd" -i="$ROOT/data/mot/MOT16-04/img1/%06d.jpg" --classes=0 -o=
fi

# TBD: MOT16-13 (750 frames, 1920x1080, moving camera) — the GMC eval scenario.
if want tbd_mot16_13; then
  run_scenario tbd_mot16_13 "detect+track" \
    "$BUILD/apps/track_tbd" -i="$ROOT/data/mot/MOT16-13/img1/%06d.jpg" --classes=0 -o=
fi

# TBD, host-tuned engine (RESULTS.md S3.0 sweep verdict: 8 threads = the
# physical core count on the Ryzen 7 7700; SMT oversubscription regresses.
# The i5-1335U verdict was 10 = nproc−2 — re-sweep per host, S2.2/S3.0).
if want tbd_tuned; then
  run_scenario tbd_tuned "detect+track" \
    "$BUILD/apps/track_tbd" -i="$ROOT/data/mot/MOT16-04/img1/%06d.jpg" --classes=0 -o= \
    --threads=8
fi

# SOT nano: OTB Car4 (659 frames, 360x240), gt init box.
if want sot_nano_car4; then
  run_scenario sot_nano_car4 "sot" \
    "$BUILD/apps/track_sot" -i="$ROOT/data/otb/Car4/img/%04d.jpg" -b=70,51,107,87 -o=
fi

# SOT nano at 1080p: MOT16-04 frames, pedestrian id=1 gt box from frame 1.
# Latency scenario only (no GT scoring) - exposes full-frame subwindow costs.
if want sot_nano_1080p; then
  run_scenario sot_nano_1080p "sot" \
    "$BUILD/apps/track_sot" -i="$ROOT/data/mot/MOT16-04/img1/%06d.jpg" -b=1363,569,103,241 -o=
fi

# SOT MOSSE fallback: Car4.
if want sot_mosse_car4; then
  run_scenario sot_mosse_car4 "sot" \
    "$BUILD/apps/track_sot" -i="$ROOT/data/otb/Car4/img/%04d.jpg" -b=70,51,107,87 \
    --backend=mosse -o=
fi
