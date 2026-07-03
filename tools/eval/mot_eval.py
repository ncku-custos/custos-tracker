#!/usr/bin/env python3
"""CLEAR metrics (MOTA, FP, FN, IDSW, IDF1) for a MOT-format result file.

Usage:
  tools/.venv/bin/python tools/eval/mot_eval.py \
      --gt data/mot/MOT16-04/gt/gt_mot17.txt --res results/mot16-04_byte.txt

Matching is delegated entirely to py-motmetrics — never reimplement CLEAR.
GT preprocessing: keep rows with consider-flag 1 and pedestrian class (1).
"""

import argparse

import motmetrics as mm

METRICS = ["num_frames", "mota", "motp", "num_false_positives", "num_misses",
           "num_switches", "idf1", "num_objects"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gt", required=True)
    ap.add_argument("--res", required=True)
    ap.add_argument("--iou", type=float, default=0.5)
    args = ap.parse_args()

    gt = mm.io.loadtxt(args.gt, fmt="mot16")
    if "ClassId" in gt.columns:
        gt = gt[gt["ClassId"] == 1]
    if "Confidence" in gt.columns:
        gt = gt[gt["Confidence"] == 1]
    res = mm.io.loadtxt(args.res, fmt="mot15-2D")

    acc = mm.utils.compare_to_groundtruth(gt, res, "iou", distth=args.iou)
    mh = mm.metrics.create()
    summary = mh.compute(acc, metrics=METRICS, name=args.res)
    print(mm.io.render_summary(summary, formatters=mh.formatters,
                               namemap=mm.io.motchallenge_metric_names))


if __name__ == "__main__":
    main()
