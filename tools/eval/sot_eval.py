#!/usr/bin/env python3
"""OTB-style SOT metrics: success-curve AUC, precision@20px, mean IoU.

Usage:
  tools/.venv/bin/python tools/eval/sot_eval.py \
      --gt data/otb/Car4/groundtruth_rect.txt --res results/Car4.txt
"""

import argparse
import re

import numpy as np


def load_boxes(path: str) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            vals = [float(v) for v in re.split(r"[,\s]+", line.strip()) if v]
            if len(vals) >= 4:
                rows.append(vals[:4])
    return np.asarray(rows, dtype=np.float64)


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x1 = np.maximum(a[:, 0], b[:, 0])
    y1 = np.maximum(a[:, 1], b[:, 1])
    x2 = np.minimum(a[:, 0] + a[:, 2], b[:, 0] + b[:, 2])
    y2 = np.minimum(a[:, 1] + a[:, 3], b[:, 1] + b[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    union = a[:, 2] * a[:, 3] + b[:, 2] * b[:, 3] - inter
    return np.where(union > 0, inter / union, 0.0)


def evaluate(gt: np.ndarray, res: np.ndarray) -> dict:
    n = min(len(gt), len(res))
    gt, res = gt[:n], res[:n]
    overlaps = iou(res, gt)
    thresholds = np.arange(0, 1.05, 0.05)
    success = [(overlaps > t).mean() for t in thresholds]
    centers_res = res[:, :2] + res[:, 2:] / 2
    centers_gt = gt[:, :2] + gt[:, 2:] / 2
    dist = np.linalg.norm(centers_res - centers_gt, axis=1)
    return {
        "auc": float(np.mean(success)),
        "prec20": float((dist <= 20).mean()),
        "mean_iou": float(overlaps.mean()),
        "frames": n,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--name", default="")
    args = ap.parse_args()

    m = evaluate(load_boxes(args.gt), load_boxes(args.res))
    name = args.name or args.res
    print(f"{name:<16} AUC {m['auc']:.3f}  prec@20 {m['prec20']:.3f}  "
          f"mIoU {m['mean_iou']:.3f}  ({m['frames']} frames)")


if __name__ == "__main__":
    main()
