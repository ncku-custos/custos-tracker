# custos-tracking

Visual tracking for a drone companion computer: **SOT** (single-object tracking, the main
goal) and **tracking-by-detection** (TBD), built as ROS-free C++20 core libraries with thin
CLI apps. ROS2 Lyrical node wrappers come later (step 4); NPU deployment comes later — all
models are chosen to be NPU-portable (conv-only, static shapes, INT8-friendly, opset 12).

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

Models: `models/get_models.sh` (SHA256-verified, see `models/manifest.json`).
Datasets: `data/fetch_otb.sh`, `data/fetch_mot.sh`.

## Layout

```
core/include/ctrk/   public API (plain structs, no OpenCV/ROS in headers)
core/src/common      geometry, Kalman, Munkres, subwindow crop, timing, logging
core/src/infer       IEngine seam + ONNX Runtime backend
core/src/sot         NanoTrack siamese pipeline + MOSSE fallback
core/src/tbd         detector wrapper + SORT/ByteTrack association
apps/                track_sot, track_tbd CLIs
tools/               Python tooling only (model export, metric evaluation)
tests/               gtest unit tests + synthetic-video e2e
docs/                PLAN.md, DECISIONS.md, RESULTS.md
```

## License

Apache-2.0 (see `LICENSE`). Note: the default detector weights derive from Ultralytics
YOLOv8 (AGPL-3.0) — see `docs/DECISIONS.md` D-0003 before shipping.
