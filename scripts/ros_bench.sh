#!/usr/bin/env bash
# S4.2 composition study: container flavor x intra-process comms, on the
# composed player -> tbd tracker -> draw pipeline (pipeline.launch.py).
#
# Per config, per rep:
#   run A (latency, rate 30 Hz):  tracker bench table (cb latency +
#     stamp->tracks e2e), stamp->annotated delay (ros2 topic delay),
#     container CPU%% (/proc utime+stime delta).
#   run B (throughput, rate 200 Hz over best-effort depth-1): sustained
#     tracks rate (ros2 topic hz) — the latest-frame-wins flight posture;
#     drops are (offered - sustained).
# Plus the SOT node-overhead run (H-OVERHEAD's honest half): nano backend,
# 30 Hz Car4, e2e vs the 1.72 ms CLI core time.
#
# Median over CTRK_ROS_BENCH_REPS (default 3) reps, bench.sh discipline.
# Results: results/bench/ros_matrix.csv + printed markdown.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=results/bench
mkdir -p "$OUT"
REPS=${CTRK_ROS_BENCH_REPS:-3}
DUR=${CTRK_ROS_BENCH_DUR:-20}
PY=tools/.venv/bin/python
CSV="$OUT/ros_matrix.csv"

set +u
# shellcheck disable=SC1091
source /opt/ros/lyrical/setup.bash
# shellcheck disable=SC1091
source ros2/install/setup.bash
set -u

# A crashed run leaves a container squatting on the node names and the next
# lifecycle command talks to the zombie — sweep before and after.
sweep() { pkill -f ctrk_pipeline 2>/dev/null || true; pkill -f ctrk_frames_player 2>/dev/null || true; pkill -f ctrk_sot_node 2>/dev/null || true; sleep 1; }
trap sweep EXIT
sweep

echo "config,rep,cb_p50,cb_p95,e2e_p50,e2e_p95,annotated_avg_s,cpu_pct,thru_hz" >"$CSV"

cpu_ticks() { awk '{print $14+$15}' "/proc/$1/stat"; }

wait_active() { # <node name>
  local up=0
  for _ in $(seq 1 150); do
    if timeout 10 ros2 lifecycle set "$1" configure >/dev/null 2>&1; then up=1; break; fi
    sleep 0.2
  done
  [[ $up == 1 ]] || { echo "ros_bench: $1 never came up" >&2; return 1; }
  timeout 10 ros2 lifecycle set "$1" activate >/dev/null
}

launch_pipeline() { # <container> <ipc> <rate> <bench> <log>
  ros2 launch ctrk_ros pipeline.launch.py \
    "container:=$1" "intra_process:=$2" "rate_hz:=$3" "bench:=$4" \
    images:="$ROOT/data/mot/MOT16-04/img1/%06d.jpg" \
    model:="$ROOT/models/cache/yolov8n_640.onnx" \
    threads:=8 loop:=true >"$5" 2>&1 &
  LAUNCH_PID=$!
  wait_active /ctrk_tbd_node
}

stop_pipeline() { # <log> <expect_table 0|1>
  timeout 10 ros2 lifecycle set /ctrk_tbd_node deactivate >/dev/null 2>&1 || true
  # The bench table travels container-stdout -> launch relay -> log file;
  # killing launch too early races the flush and parse_stage reads nothing
  # (what killed rep 2 of the first full run). Wait for it before teardown.
  if [[ ${2:-0} == 1 ]]; then
    for _ in $(seq 1 20); do
      grep -q 'e2e.stamp_to_pub' "$1" 2>/dev/null && break
      sleep 0.5
    done
  fi
  sleep 1
  # ros2 launch ignores a lone SIGINT when backgrounded without a tty
  # (learned the hard way: the first shakedown hung on `wait` overnight).
  # Escalate, then make sure the container itself is gone.
  kill -INT "$LAUNCH_PID" 2>/dev/null || true
  for _ in $(seq 1 10); do
    kill -0 "$LAUNCH_PID" 2>/dev/null || break
    sleep 1
  done
  if kill -0 "$LAUNCH_PID" 2>/dev/null; then
    kill -TERM "$LAUNCH_PID" 2>/dev/null || true
    sleep 2
    kill -KILL "$LAUNCH_PID" 2>/dev/null || true
  fi
  pkill -f component_container 2>/dev/null || true
  wait "$LAUNCH_PID" 2>/dev/null || true
  sleep 1
}

parse_stage() { # <log> <stage>  -> "p50 p95" (fields NF-1, NF)
  grep -E "$2 [0-9]+ " "$1" | tail -1 | awk '{print $(NF-1), $NF}'
}

for container in component_container component_container_mt component_container_isolated; do
  for ipc in true false; do
    cfg="${container#component_container}"
    cfg="ct${cfg:-_st}_ipc_${ipc}"
    for rep in $(seq 1 "$REPS"); do
      log="$OUT/ros_${cfg}_r${rep}.log"

      # --- run A: latency at 30 Hz ---
      launch_pipeline "$container" "$ipc" 30.0 true "$log"
      # match the container binary path, not the launch wrapper (whose
      # cmdline also contains the container name as a launch argument)
      cpid=$(pgrep -n -f "rclcpp_components/$container" || true)
      t0=$(cpu_ticks "$cpid" 2>/dev/null || echo 0)
      timeout 12 ros2 topic delay /image_annotated >"$OUT/delay_${cfg}_r${rep}.log" 2>&1 || true
      remaining=$((DUR - 12)); [[ $remaining -gt 0 ]] && sleep "$remaining"
      t1=$(cpu_ticks "$cpid" 2>/dev/null || echo 0)
      cpu=$(awk -v d="$((t1 - t0))" -v s="$DUR" 'BEGIN{printf "%.0f", d/100/s*100}')
      stop_pipeline "$log" 1
      read -r cb_p50 cb_p95 <<<"$(parse_stage "$log" 'cb.update\+publish')" || true
      read -r e2e_p50 e2e_p95 <<<"$(parse_stage "$log" 'e2e.stamp_to_pub')" || true
      ann=$(grep -oE 'average delay: [0-9.]+' "$OUT/delay_${cfg}_r${rep}.log" | tail -1 | awk '{print $3}')

      # --- run B: throughput at 200 Hz offered, best-effort depth 1 ---
      logb="$OUT/ros_${cfg}_r${rep}_thru.log"
      launch_pipeline "$container" "$ipc" 200.0 false "$logb"
      timeout 15 ros2 topic hz /tracks --window 300 >"$OUT/hz_${cfg}_r${rep}.log" 2>&1 || true
      stop_pipeline "$logb" 0
      thru=$(grep -oE 'average rate: [0-9.]+' "$OUT/hz_${cfg}_r${rep}.log" | tail -1 | awk '{print $3}')

      echo "$cfg,$rep,${cb_p50:-},${cb_p95:-},${e2e_p50:-},${e2e_p95:-},${ann:-},${cpu:-},${thru:-}" | tee -a "$CSV"
    done
  done
done

# --- SOT node overhead (nano, 30 Hz, Car4) ---
for rep in $(seq 1 "$REPS"); do
  log="$OUT/ros_sot_overhead_r${rep}.log"
  ros2 run ctrk_ros ctrk_sot_node --ros-args \
    -p backbone_z_path:="$ROOT/models/cache/nanotrack_backbone_z.onnx" \
    -p backbone_x_path:="$ROOT/models/cache/nanotrack_backbone_x.onnx" \
    -p head_path:="$ROOT/models/cache/nanotrack_head.onnx" \
    -p "init_bbox:=[70.0,51.0,107.0,87.0]" -p bench:=true >"$log" 2>&1 &
  NODE_PID=$!
  wait_active /ctrk_sot_node
  ros2 run ctrk_ros ctrk_frames_player --ros-args \
    -p path:="$ROOT/data/otb/Car4/img/%04d.jpg" -p mode:=rate -p rate_hz:=30.0 \
    -p stamp_source:=clock -p loop:=true >/dev/null 2>&1 &
  PLAYER_PID=$!
  sleep "$DUR"
  timeout 10 ros2 lifecycle set /ctrk_sot_node deactivate >/dev/null
  kill -INT $PLAYER_PID $NODE_PID 2>/dev/null || true
  sleep 2
  kill -TERM $PLAYER_PID $NODE_PID 2>/dev/null || true
  wait $PLAYER_PID $NODE_PID 2>/dev/null || true
  read -r cb_p50 cb_p95 <<<"$(parse_stage "$log" 'cb.update\+publish')" || true
  read -r e2e_p50 e2e_p95 <<<"$(parse_stage "$log" 'e2e.stamp_to_pub')" || true
  echo "sot_overhead,$rep,${cb_p50:-},${cb_p95:-},${e2e_p50:-},${e2e_p95:-},,," | tee -a "$CSV"
done

echo
echo "== medians over $REPS reps =="
$PY - "$CSV" <<'EOF'
import csv, statistics, sys
rows = list(csv.DictReader(open(sys.argv[1])))
cols = ["cb_p50", "cb_p95", "e2e_p50", "e2e_p95", "annotated_avg_s", "cpu_pct", "thru_hz"]
configs = []
for r in rows:
    if r["config"] not in configs:
        configs.append(r["config"])
print("| config | " + " | ".join(cols) + " |")
print("|" + "---|" * (len(cols) + 1))
for cfg in configs:
    med = []
    for c in cols:
        vals = [float(r[c]) for r in rows if r["config"] == cfg and r[c]]
        med.append(f"{statistics.median(vals):.2f}" if vals else "—")
    print(f"| {cfg} | " + " | ".join(med) + " |")
EOF
echo "ros_bench: OK (governor: $(cat /sys/devices/system/cpu/cpu0/cpufreq/scaling_governor 2>/dev/null || echo n/a))"
