#!/usr/bin/env python3
"""Generate the golden-tensor fixture for the C++ YOLOv8 decoder test.

Writes to tests/fixtures/yolo/ (committed):
  raw_output.bin   float32 [1,84,8400] raw model output on bus.jpg,
                   produced with the exact preprocessing the C++ side uses
  expected.csv     ultralytics' own predictions (the trusted reference the
                   C++ decode must reproduce): class_id,conf,x,y,w,h
                   header line carries "# src_w src_h conf_thr nms_iou"

The C++ test decodes raw_output.bin and must match expected.csv — a
differential test against the reference implementation, not against our own
python port of the same bugs.

Usage: tools/.venv/bin/python tools/export/make_yolo_fixture.py
"""

from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from ultralytics import YOLO
from ultralytics.utils import ASSETS

ROOT = Path(__file__).resolve().parents[2]
MODEL = ROOT / "models" / "cache" / "yolov8n_640.onnx"
OUTDIR = ROOT / "tests" / "fixtures" / "yolo"
CONF, NMS_IOU, SIZE = 0.25, 0.45, 640


def letterbox_chw(img: np.ndarray) -> np.ndarray:
    """Mirror of the C++ preprocessing: letterbox to 640, pad 114, RGB, /255, CHW."""
    h, w = img.shape[:2]
    scale = min(SIZE / w, SIZE / h)
    nw, nh = round(w * scale), round(h * scale)
    px, py = (SIZE - nw) // 2, (SIZE - nh) // 2
    canvas = np.full((SIZE, SIZE, 3), 114, np.uint8)
    canvas[py : py + nh, px : px + nw] = cv2.resize(img, (nw, nh))
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]


def main() -> None:
    assert MODEL.exists(), "run tools/export/export_yolo.py first"
    OUTDIR.mkdir(parents=True, exist_ok=True)

    img = cv2.imread(str(ASSETS / "bus.jpg"))
    h, w = img.shape[:2]

    sess = ort.InferenceSession(str(MODEL), providers=["CPUExecutionProvider"])
    (raw,) = sess.run(None, {sess.get_inputs()[0].name: letterbox_chw(img)})
    raw.astype(np.float32).tofile(OUTDIR / "raw_output.bin")

    # Reference predictions from ultralytics' own pipeline on the same image.
    result = YOLO(MODEL, task="detect")(ASSETS / "bus.jpg", conf=CONF, iou=NMS_IOU, imgsz=SIZE,
                                        verbose=False)[0]
    lines = [f"# {w} {h} {CONF} {NMS_IOU}"]
    for box in result.boxes:
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        lines.append(
            f"{int(box.cls)},{float(box.conf):.4f},{x1:.1f},{y1:.1f},{x2 - x1:.1f},{y2 - y1:.1f}"
        )
    (OUTDIR / "expected.csv").write_text("\n".join(lines) + "\n")
    print(f"fixture: {len(lines) - 1} reference boxes -> {OUTDIR}")


if __name__ == "__main__":
    main()
