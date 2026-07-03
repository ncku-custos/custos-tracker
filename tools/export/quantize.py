#!/usr/bin/env python3
"""Static INT8 QDQ quantization dry run — the NPU rehearsal (step-2 P5, D-0008).

Produces parallel INT8 variants next to the fp32 models (which stay
canonical): QDQ format, per-channel INT8 weights, UINT8 activations — the
combination MLAS accelerates via AVX-VNNI on the host and the closest
portable stand-in for vendor NPU INT8 pipelines.

Calibration uses data already on disk:
  yolo  — MOT16-04 frames through an exact replica of the C++ letterbox
          (114 pad, BGR->RGB, /255, CHW).
  nano  — OTB ground-truth-centered context crops through a replica of the
          C++ subwindow geometry (raw 0-255 RGB blobs). Backbones only: the
          depthwise-correlation head stays fp32 by design — it is the graph
          we already plan to keep on CPU if the NPU cannot run it (risk #1),
          so rehearsing its quantization buys nothing.

Usage:
  tools/.venv/bin/python tools/export/quantize.py --model yolo [--imgsz 640]
  tools/.venv/bin/python tools/export/quantize.py --model nano
"""

import argparse
import math
import tempfile
from pathlib import Path

import cv2
import numpy as np
from onnxruntime.quantization import (
    CalibrationDataReader,
    QuantFormat,
    QuantType,
    quantize_static,
)
from onnxruntime.quantization.shape_inference import quant_pre_process

ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "models" / "cache"
MOT_IMG = ROOT / "data" / "mot" / "MOT16-04" / "img1"
OTB = ROOT / "data" / "otb"
CAL_FRAMES = 300


def letterbox_blob(img: np.ndarray, size: int) -> np.ndarray:
    """Replica of Yolov8Detector preprocessing: 114-grey letterbox, RGB, /255, CHW."""
    h, w = img.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = round(w * scale), round(h * scale)
    canvas = np.full((size, size, 3), 114, np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    canvas[py : py + nh, px : px + nw] = cv2.resize(img, (nw, nh))
    rgb = canvas[:, :, ::-1].astype(np.float32) / 255.0
    return rgb.transpose(2, 0, 1)[None]


def subwindow_blob(img: np.ndarray, cx: float, cy: float, original_sz: int,
                   model_sz: int) -> np.ndarray:
    """Replica of nano_subwindow + blob_rgb: mean-pad crop, raw 0-255 RGB CHW."""
    c = (original_sz + 1) // 2
    xmin, ymin = int(cx) - c, int(cy) - c
    pad = np.array(cv2.mean(img)[:3])
    patch = np.empty((original_sz, original_sz, 3), np.uint8)
    patch[:] = pad
    x0, y0 = max(0, xmin), max(0, ymin)
    x1, y1 = min(img.shape[1], xmin + original_sz), min(img.shape[0], ymin + original_sz)
    if x1 > x0 and y1 > y0:
        patch[y0 - ymin : y1 - ymin, x0 - xmin : x1 - xmin] = img[y0:y1, x0:x1]
    crop = cv2.resize(patch, (model_sz, model_sz))
    return crop[:, :, ::-1].astype(np.float32).transpose(2, 0, 1)[None]


class BlobReader(CalibrationDataReader):
    def __init__(self, input_name: str, blobs):
        self.input_name = input_name
        self.it = iter(blobs)

    def get_next(self):
        blob = next(self.it, None)
        return None if blob is None else {self.input_name: blob}


def input_name_of(model: Path) -> str:
    import onnx

    return onnx.load(model).graph.input[0].name


def quantize(model: Path, reader: CalibrationDataReader, exclude=None) -> Path:
    import onnx
    from onnx import version_converter

    out = model.with_name(model.stem + "_int8.onnx")
    with tempfile.NamedTemporaryFile(suffix=".onnx") as src, \
         tempfile.NamedTemporaryFile(suffix=".onnx") as tmp:
        # Per-channel QDQ emits DequantizeLinear with an `axis` attribute,
        # which entered ONNX at opset 13 — older runtimes (system ORT 1.23)
        # reject it on an opset-12 graph. Only this INT8 *variant* is bumped;
        # the canonical fp32 opset-12 artifact is untouched (D-0008: NPU
        # toolchains quantize from fp32 themselves).
        m = onnx.load(str(model))
        opset = max(op.version for op in m.opset_import if op.domain in ("", "ai.onnx"))
        if opset < 13:
            m = version_converter.convert_version(m, 13)
        onnx.save(m, src.name)
        quant_pre_process(input_model=src.name, output_model_path=tmp.name)
        # Exclusions are matched on the PREPROCESSED graph (optimization can
        # rename/fuse nodes).
        nodes_to_exclude = []
        if exclude is not None:
            pre = onnx.load(tmp.name)
            nodes_to_exclude = [n.name for n in pre.graph.node if exclude(n.name)]
        quantize_static(
            tmp.name,
            str(out),
            reader,
            quant_format=QuantFormat.QDQ,
            per_channel=True,
            activation_type=QuantType.QUInt8,
            weight_type=QuantType.QInt8,
            nodes_to_exclude=nodes_to_exclude,
        )
    print(f"quantized {out.name} ({out.stat().st_size / 1e6:.1f} MB)")
    return out


def yolo_blobs(size: int):
    frames = sorted(MOT_IMG.glob("*.jpg"))
    step = max(1, len(frames) // CAL_FRAMES)
    for p in frames[::step][:CAL_FRAMES]:
        yield letterbox_blob(cv2.imread(str(p)), size)


def otb_boxes():
    """(image path, cx, cy, w, h) across all local OTB sequences, subsampled."""
    for seq in sorted(OTB.iterdir()):
        gt = seq / "groundtruth_rect.txt"
        if not gt.exists():
            gt = seq / "groundtruth_rect.1.txt"
        if not gt.exists():
            continue
        imgs = sorted((seq / "img").glob("*.jpg"))
        lines = gt.read_text().replace("\r", "").replace("\t", ",").splitlines()
        for img, line in list(zip(imgs, lines))[::10]:
            x, y, w, h = (float(v) for v in line.split(",")[:4])
            yield img, x + w / 2, y + h / 2, w, h


def nano_blobs(exemplar: bool):
    model_sz = 127 if exemplar else 255
    for img_path, cx, cy, w, h in otb_boxes():
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        # Context geometry as in NanoTracker (context_amount = 0.5).
        w_ext, h_ext = w + 0.5 * (w + h), h + 0.5 * (w + h)
        s = math.sqrt(w_ext * h_ext)
        original = int(s) if exemplar else int(s) * 2  # sx = sz * (255//127)
        if original < 2:
            continue
        yield subwindow_blob(img, cx, cy, original, model_sz)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["yolo", "nano"], required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    args = ap.parse_args()

    if args.model == "yolo":
        model = CACHE / f"yolov8n_{args.imgsz}.onnx"
        # The Detect head (model.22) concatenates box coordinates (0..imgsz)
        # and class scores (0..1) into one tensor: a single activation scale
        # crushes the scores to zero (measured: 0 detections). Keep the head
        # fp32; backbone+neck carry the FLOPs and the INT8 speedup.
        quantize(model, BlobReader(input_name_of(model), yolo_blobs(args.imgsz)),
                 exclude=lambda name: "/model.22/" in name)
    else:
        for graph, exemplar in (("nanotrack_backbone_z.onnx", True),
                                ("nanotrack_backbone_x.onnx", False)):
            model = CACHE / graph
            quantize(model, BlobReader(input_name_of(model), nano_blobs(exemplar)))
        print("head left fp32 by design (CPU-resident graph in the NPU split plan)")


if __name__ == "__main__":
    main()
