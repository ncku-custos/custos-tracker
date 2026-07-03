#!/usr/bin/env bash
# S4.1 parity gate: the ROS2 node path must reproduce the CLI dumps
# DIGIT-IDENTICALLY on the anchor datasets. Transport is drop-proof by
# construction (lockstep player + reliable QoS) and the stamps replicate the
# CLI VideoSource math bit-exactly, so any diff is a real behavior change.
#
# Gating:   TBD on MOT16-04 (classes=0, defaults) — diff + line count + anchor
#           re-score (MOTA 31.3 / IDSW 23); SOT nano on the 6 mini-OTB seqs —
#           per-seq diff + identical AUC re-score (mean 0.631).
# Non-gating: drone profile (gmc sparse_flow, detect_interval 2) on MOT16-13 —
#           determinism finding, reported either way (RESULTS.md S4.1).
#
# Prereqs: scripts/ros_check.sh green, models fetched, datasets fetched.
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT=$PWD
OUT=results/parity
mkdir -p "$OUT"
PY=tools/.venv/bin/python

set +u
# shellcheck disable=SC1091
source /opt/ros/lyrical/setup.bash
# shellcheck disable=SC1091
source ros2/install/setup.bash
set -u

fail() {
  echo "ros_parity: FAIL — $*" >&2
  exit 1
}

# run_pair <node_exec> <node_name> -- <node -p args...> -- <player -p args...>
# Tracker up -> configure -> activate -> player runs to EOF (lockstep,
# exit_on_done) -> deactivate (flushes dump) -> tracker down.
run_pair() {
  local node_exec=$1 node_name=$2
  shift 3 # exec, name, first "--"
  local -a node_params=() player_params=()
  local dest=node
  local a
  for a in "$@"; do
    if [[ $a == "--" ]]; then dest=player; continue; fi
    if [[ $dest == node ]]; then node_params+=("$a"); else player_params+=("$a"); fi
  done
  ros2 run ctrk_ros "$node_exec" --ros-args "${node_params[@]}" >"$OUT/${node_name#/}.log" 2>&1 &
  local pid=$!
  local up=0
  for _ in $(seq 1 150); do
    if ros2 lifecycle set "$node_name" configure >/dev/null 2>&1; then up=1; break; fi
    sleep 0.2
  done
  [[ $up == 1 ]] || fail "$node_name never reached configure"
  ros2 lifecycle set "$node_name" activate >/dev/null
  ros2 run ctrk_ros ctrk_frames_player --ros-args -p exit_on_done:=true \
    "${player_params[@]}" >>"$OUT/${node_name#/}.log" 2>&1
  ros2 lifecycle set "$node_name" deactivate >/dev/null
  kill -INT $pid 2>/dev/null || true
  wait $pid 2>/dev/null || true
}

check_pair() { # <cli dump> <ros dump> <label>
  diff -q "$1" "$2" >/dev/null || fail "$3: dumps differ ($1 vs $2)"
  local n_cli n_ros
  n_cli=$(wc -l <"$1")
  n_ros=$(wc -l <"$2")
  [[ $n_cli == "$n_ros" ]] || fail "$3: line counts differ"
  echo "  $3: digit-identical ($n_ros lines)"
}

# The CLI derives nominal_fps from the source: OpenCV's FFMPEG backend claims
# %d image patterns and reports 25 fps (the image2 default), so the anchors
# were produced with stamps AND nominal_fps at 25. The node cannot probe the
# source — tell it explicitly, and pin the player to the same value.
SEQ_FPS=25.0

# ---------------------------------------------------------------- TBD gate
echo "== TBD parity: MOT16-04, defaults + classes=0 =="
build/apps/track_tbd --input=data/mot/MOT16-04/img1/%06d.jpg --classes=0 \
  --output= --dump="$OUT/cli_tbd_mot16_04.txt" >/dev/null
run_pair ctrk_tbd_node /ctrk_tbd_node -- \
  -p detector.model_path:="$ROOT/models/cache/yolov8n_640.onnx" \
  -p "detector.keep_classes:=[0]" -p nominal_fps:=$SEQ_FPS \
  -p dump_path:="$ROOT/$OUT/ros_tbd_mot16_04.txt" \
  -p image_qos_reliability:=reliable -p image_qos_depth:=10 -- \
  -p path:="$ROOT/data/mot/MOT16-04/img1/%06d.jpg" -p mode:=lockstep \
  -p fps:=$SEQ_FPS \
  -p lockstep_topic:=tracks -p lockstep_type:=vision_msgs/msg/Detection2DArray \
  -p stamp_source:=index
check_pair "$OUT/cli_tbd_mot16_04.txt" "$OUT/ros_tbd_mot16_04.txt" "MOT16-04"
[[ $(wc -l <"$OUT/ros_tbd_mot16_04.txt") -gt 0 ]] || fail "MOT16-04: empty dump"

echo "-- anchor re-score (expect MOTA 31.3 / IDSW 23) --"
$PY tools/eval/mot_eval.py --gt data/mot/MOT16-04/gt/gt_eval.txt \
  --res "$OUT/ros_tbd_mot16_04.txt" | tee "$OUT/tbd_rescore.txt"
grep -q "31.3%" "$OUT/tbd_rescore.txt" || fail "MOTA anchor drifted from 31.3"

# ---------------------------------------------------------------- SOT gate
echo "== SOT parity: mini-OTB, nano backend, defaults =="
SEQS=(Car4 CarDark BlurCar2 Jogging Girl2 Woman)
for seq in "${SEQS[@]}"; do
  gt=data/otb/$seq/groundtruth_rect.txt
  [[ -f $gt ]] || gt=data/otb/$seq/groundtruth_rect.1.txt
  bbox=$($PY -c "
import sys; sys.path.insert(0, 'tools/eval')
from sot_eval import load_boxes
x, y, w, h = load_boxes('$gt')[0]
print(f'{float(x)},{float(y)},{float(w)},{float(h)}')")
  build/apps/track_sot --input="data/otb/$seq/img/%04d.jpg" --bbox="$bbox" \
    --output= --dump="$OUT/cli_sot_$seq.txt" >/dev/null
  run_pair ctrk_sot_node /ctrk_sot_node -- \
    -p backbone_z_path:="$ROOT/models/cache/nanotrack_backbone_z.onnx" \
    -p backbone_x_path:="$ROOT/models/cache/nanotrack_backbone_x.onnx" \
    -p head_path:="$ROOT/models/cache/nanotrack_head.onnx" \
    -p "init_bbox:=[$bbox]" \
    -p dump_path:="$ROOT/$OUT/ros_sot_$seq.txt" \
    -p image_qos_reliability:=reliable -p image_qos_depth:=10 -- \
    -p path:="$ROOT/data/otb/$seq/img/%04d.jpg" -p mode:=lockstep \
    -p lockstep_topic:=target -p lockstep_type:=ctrk_interfaces/msg/SotStatus \
    -p stamp_source:=index
  check_pair "$OUT/cli_sot_$seq.txt" "$OUT/ros_sot_$seq.txt" "$seq"
done

echo "-- AUC re-score from the ROS dumps (expect mean 0.631) --"
$PY -c "
import sys; sys.path.insert(0, 'tools/eval')
import numpy as np
from sot_eval import evaluate, load_boxes
from pathlib import Path
seqs = ['Car4', 'CarDark', 'BlurCar2', 'Jogging', 'Girl2', 'Woman']
aucs = []
for seq in seqs:
    gt = Path(f'data/otb/{seq}/groundtruth_rect.txt')
    if not gt.exists():
        gt = Path(f'data/otb/{seq}/groundtruth_rect.1.txt')
    r = evaluate(load_boxes(gt), load_boxes(f'$OUT/ros_sot_{seq}.txt'))
    aucs.append(r['auc'])
    print(f'{seq:<12} AUC {r[\"auc\"]:.3f}')
mean = np.mean(aucs)
print(f'{\"MEAN\":<12} AUC {mean:.3f}')
assert f'{mean:.3f}' == '0.631', f'AUC anchor drifted: {mean:.3f}'
" || fail "SOT AUC anchor drifted"

# --------------------------------------- drone profile (NON-gating finding)
echo "== drone-profile determinism (MOT16-13, gmc+N=2) — non-gating =="
build/apps/track_tbd --input=data/mot/MOT16-13/img1/%06d.jpg --classes=0 \
  --gmc --detect-every=2 --output= --dump="$OUT/cli_tbd_mot16_13_drone.txt" >/dev/null
run_pair ctrk_tbd_node /ctrk_tbd_node -- \
  -p detector.model_path:="$ROOT/models/cache/yolov8n_640.onnx" \
  -p "detector.keep_classes:=[0]" -p gmc:=sparse_flow -p detect_interval:=2 \
  -p nominal_fps:=$SEQ_FPS \
  -p dump_path:="$ROOT/$OUT/ros_tbd_mot16_13_drone.txt" \
  -p image_qos_reliability:=reliable -p image_qos_depth:=10 -- \
  -p path:="$ROOT/data/mot/MOT16-13/img1/%06d.jpg" -p mode:=lockstep \
  -p fps:=$SEQ_FPS \
  -p lockstep_topic:=tracks -p lockstep_type:=vision_msgs/msg/Detection2DArray \
  -p stamp_source:=index
if diff -q "$OUT/cli_tbd_mot16_13_drone.txt" "$OUT/ros_tbd_mot16_13_drone.txt" >/dev/null; then
  echo "  drone profile: digit-identical ($(wc -l <"$OUT/ros_tbd_mot16_13_drone.txt") lines)"
else
  echo "  drone profile: DIFFERS (GMC nondeterminism? document in S4.1, not a gate)"
fi

echo "ros_parity: OK"
