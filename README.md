# custos-tracking

Visual tracking for a drone companion computer: **SOT** (single-object tracking, the main
goal) and **tracking-by-detection** (TBD), built as ROS-free C++20 core libraries with thin
CLI apps and **ROS2 Lyrical composable lifecycle nodes** (`ros2/`). NPU deployment
comes later — all models are chosen to be NPU-portable (conv-only, static shapes,
INT8-friendly, opset 12).

## Subsystems

| | Main line | Fallback / baseline |
|---|---|---|
| **SOT** | NanoTrack v2/v3 (conv-only siamese, 3 static ONNX graphs) | hand-written MOSSE correlation filter |
| **TBD** | YOLOv8n (ONNX, decode+NMS in C++) + ByteTrack association | SORT (same machinery, low-score stage off) |

Both sit behind stable interfaces (`ctrk::SotTracker`, `ctrk::MultiTracker`); inference goes
through a backend seam (`ctrk::IEngine`, ONNX Runtime on host, vendor NPU SDK later).

## Build

Requires: Ubuntu 26.04 system packages `libopencv-dev` and `libonnxruntime-dev`, CMake >= 3.28.

```sh
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build --output-on-failure
```

`scripts/check.sh` runs build + tests + format check (the pre-commit gate).
`scripts/bench.sh` reproduces the latency tables in `docs/RESULTS.md`
(multi-run medians; see the noise methodology there).

Models: `models/get_models.sh` (SHA256-verified, see `models/manifest.json`);
`tools/export/export_yolo.py --imgsz 640 512 ...` for detector variants and
`tools/export/quantize.py` for the INT8 QDQ variants (docs/DECISIONS.md D-0008).
Datasets: `data/fetch_otb.sh`, `data/fetch_mot.sh`.

## Speed & quality knobs (measured in docs/RESULTS.md)

- `track_tbd --threads=N` — detector intra-op threads. Host-best is
  hardware-specific: physical-core count on homogeneous CPUs (8 on the
  Ryzen 7700), nproc−2 on the hybrid i5 — re-sweep per host (D-0011).
- `track_tbd --detect-every=N` — detector cadence with Kalman coasting
  between. The tentative-gate fix (S3.2) makes intervals safe for
  fast movers; N=2 improves identity metrics on both eval sequences.
- `--model models/cache/yolov8n_640_int8.onnx` — INT8 detector,
  metric-parity-plus at ~1.4x speed; composes with every knob below.
- `--gmc` — camera-motion compensation (S3.4): the moving-camera eval
  jumps +5.4 MOTA / +9.7 IDF1 for ~2 ms/frame. Recommended whenever the
  camera moves (i.e. on the drone); off by default for static mounts.
- `--no-spin` (TBD) — ~40% less CPU for a few % latency; the SoC power
  posture. SOT defaults to no-spin already.
- `track_sot --reacquire` — detector-assisted re-acquisition, now guarded
  by an HSV appearance veto (S3.3; `--reid=none` for pure geometry). Add
  `--reid=nanoz --drift-every=5` to catch confident drift (the Girl2 mode).
- NanoTrack v3 (`--backbone-z/x/--head` at the `nanotrackv3_*` exports +
  `--penalty-k=0.138 --size-lr=0.348`) — big vehicle-tracking gains, person
  regressions; v2 stays default (S3.7).

## ROS2

The core is consumed strictly via `find_package(ctrk)`; `ros2/` holds two ament packages:
`ctrk_interfaces` (SotStatus msg + SetTarget srv — the one place vision_msgs wasn't enough)
and `ctrk_ros` (composable **lifecycle** components: `ctrk_tbd_node`, `ctrk_sot_node`, plus
`ctrk_frames_player`/`ctrk_draw_node` for playback and debug overlay).

```sh
scripts/ros_check.sh        # core install to ros2/install-ctrk + colcon build + test
ros2 launch ctrk_ros tbd.launch.py params_file:=ros2/ctrk_ros/params/tbd_drone.yaml
ros2 launch ctrk_ros pipeline.launch.py container:=component_container_isolated  # player->tracker->draw
```

- Topics: `image` (bgr8, zero-copy into the core; best-effort depth 1 = latest-frame-wins)
  -> `tracks` (vision_msgs/Detection2DArray) / `target` (SotStatus); SOT init via the
  `set_target` service or the `init_bbox` param.
- Parameters map 1:1 onto the config structs (defaults = struct defaults; see
  `ros2/ctrk_ros/params/*_default.yaml` for the annotated reference, `*_drone.yaml` for the
  D-0013 flight profiles). **`nominal_fps` must match the camera rate** or KF dt is scaled.
- `scripts/ros_parity.sh` proves the node path digit-identical to the CLI (RESULTS.md S4.1);
  `scripts/ros_bench.sh` runs the container/intra-process matrix (S4.2).

## Layout

```
core/include/ctrk/   public API (plain structs, no OpenCV/ROS in headers)
core/src/common      geometry, Kalman, Munkres, subwindow crop, timing, logging
core/src/infer       IEngine seam + ONNX Runtime backend
core/src/sot         NanoTrack siamese pipeline + MOSSE fallback
core/src/tbd         detector wrapper + SORT/ByteTrack association
apps/                track_sot, track_tbd CLIs
ros2/                ctrk_interfaces + ctrk_ros ament packages (lifecycle components)
tools/               Python tooling only (model export, metric evaluation)
tests/               gtest unit tests + synthetic-video e2e
docs/                PLAN.md, DECISIONS.md, RESULTS.md, ROADMAP.md (SoC handoff)
```

## License

Apache-2.0 (see `LICENSE`). Note: the default detector weights derive from Ultralytics
YOLOv8 (AGPL-3.0) — see `docs/DECISIONS.md` D-0003 before shipping.
