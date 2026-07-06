#!/usr/bin/env python3
"""Export YOLOv8n to a static, NPU-portable ONNX.

Contract (docs/PLAN.md, D-0004/D-0007): opset 12, static [1,3,SZ,SZ], no
in-graph NMS (decode + NMS live in C++), simplified graph. Output lands in
models/cache/<stem>_<SZ>.onnx (stem defaults to yolov8n).

The C++ detector reads input size, anchor count AND class count from the
graph, so both the resolution ladder (512/448/416, RESULTS.md S2.4) and fine-tuned
class counts (e.g. the 10-class VisDrone model, RESULTS.md S3.6) are
export-only: pass --imgsz / --weights.

Usage:
  tools/.venv/bin/python tools/export/export_yolo.py [--imgsz 640 512 ...]
      [--weights runs/.../best.pt --stem yolov8n_visdrone]
"""

import argparse
from pathlib import Path

import onnx
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]


def anchors_for(size: int) -> int:
    # YOLOv8 heads at strides 8/16/32.
    return (size // 8) ** 2 + (size // 16) ** 2 + (size // 32) ** 2


def export_one(weights: str, stem: str, size: int) -> None:
    assert size % 32 == 0, f"imgsz {size} must be a multiple of 32"
    out = ROOT / "models" / "cache" / f"{stem}_{size}.onnx"
    out.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO(weights)
    nc = len(model.names)  # channel dim is 4 box coords + one score per class
    exported = model.export(
        format="onnx", opset=12, imgsz=size, dynamic=False, simplify=True, nms=False
    )

    graph = onnx.load(exported)
    opset = max(op.version for op in graph.opset_import if op.domain in ("", "ai.onnx"))
    assert opset <= 12, f"opset {opset} breaks the NPU-portability contract (<= 12)"
    for vi in graph.graph.input:
        dims = [d.dim_value for d in vi.type.tensor_type.shape.dim]
        assert all(d > 0 for d in dims), f"dynamic dim on input {vi.name}"
    out_shapes = [
        [d.dim_value for d in vo.type.tensor_type.shape.dim] for vo in graph.graph.output
    ]
    expected = [[1, 4 + nc, anchors_for(size)]]
    assert out_shapes == expected, f"unexpected outputs {out_shapes} (want {expected})"

    Path(exported).replace(out)
    print(
        f"exported {out} (opset {opset}, static 1x3x{size}x{size} -> "
        f"1x{4 + nc}x{anchors_for(size)}, {nc} classes)"
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--imgsz", type=int, nargs="+", default=[640])
    ap.add_argument("--weights", default="yolov8n.pt")
    ap.add_argument("--stem", default=None, help="output name stem (default: weights stem)")
    args = ap.parse_args()
    stem = args.stem or Path(args.weights).stem
    for size in args.imgsz:
        export_one(args.weights, stem, size)


if __name__ == "__main__":
    main()
