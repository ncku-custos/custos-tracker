#!/usr/bin/env python3
"""Run track_sot over the mini-OTB set and print the metric table.

Usage: tools/.venv/bin/python tools/eval/run_otb.py [--binary build/apps/track_sot]
"""

import argparse
import re
import subprocess
from pathlib import Path

import numpy as np

from sot_eval import evaluate, load_boxes

ROOT = Path(__file__).resolve().parents[2]
SEQS = ["Car4", "CarDark", "BlurCar2", "Jogging", "Girl2", "Woman"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(ROOT / "build/apps/track_sot"))
    ap.add_argument("--backend", default="nano", choices=["nano", "mosse"])
    ap.add_argument("--otb", default=str(ROOT / "data/otb"))
    ap.add_argument("--out", default=str(ROOT / "results/otb"))
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for seq in SEQS:
        seq_dir = Path(args.otb) / seq
        if not seq_dir.exists():
            print(f"{seq:<16} MISSING (run data/fetch_otb.sh)")
            continue
        gt_path = seq_dir / "groundtruth_rect.txt"
        if not gt_path.exists():  # two-target sequences (Jogging): track target 1
            gt_path = seq_dir / "groundtruth_rect.1.txt"
        gt = load_boxes(gt_path)
        x, y, w, h = gt[0]
        res_path = out_dir / f"{seq}_{args.backend}.txt"
        run = subprocess.run(
            [args.binary, f"--input={seq_dir}/img/%04d.jpg", f"--bbox={x},{y},{w},{h}",
             f"--backend={args.backend}", f"--dump={res_path}", "--output="],
            capture_output=True, text=True, check=True,
            cwd=ROOT)  # app model-path defaults are repo-relative
        fps = 0.0
        m = re.search(r"sot\s+\d+\s+([\d.]+)", run.stdout)
        if m:
            fps = 1000.0 / float(m.group(1)) if float(m.group(1)) > 0 else 0.0
        r = evaluate(gt, load_boxes(res_path))
        r.update(seq=seq, fps=fps)
        rows.append(r)
        print(f"{seq:<16} AUC {r['auc']:.3f}  prec@20 {r['prec20']:.3f}  "
              f"mIoU {r['mean_iou']:.3f}  {fps:6.1f} fps  ({r['frames']} frames)")

    if rows:
        print(f"{'MEAN':<16} AUC {np.mean([r['auc'] for r in rows]):.3f}  "
              f"prec@20 {np.mean([r['prec20'] for r in rows]):.3f}  "
              f"mIoU {np.mean([r['mean_iou'] for r in rows]):.3f}")


if __name__ == "__main__":
    main()
