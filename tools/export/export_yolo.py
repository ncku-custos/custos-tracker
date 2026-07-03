#!/usr/bin/env python3
"""Export YOLOv8n to a static, NPU-portable ONNX.

Contract (docs/PLAN.md, D-0004/D-0007): opset 12, static [1,3,640,640], no
in-graph NMS (decode + NMS live in C++), simplified graph. Output lands in
models/cache/yolov8n_640.onnx.

Usage: tools/.venv/bin/python tools/export/export_yolo.py
"""

from pathlib import Path

import onnx
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "models" / "cache" / "yolov8n_640.onnx"


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)
    model = YOLO("yolov8n.pt")  # cached under ~/.config/Ultralytics after first run
    exported = model.export(
        format="onnx", opset=12, imgsz=640, dynamic=False, simplify=True, nms=False
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
    assert out_shapes == [[1, 84, 8400]], f"unexpected outputs {out_shapes}"

    Path(exported).replace(OUT)
    print(f"exported {OUT} (opset {opset}, static 1x3x640x640 -> 1x84x8400)")


if __name__ == "__main__":
    main()
