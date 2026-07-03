# Results

Metric and latency tables per milestone. Populated as milestones land; this file is the
regression baseline for step 2 (speed) and step 3 (quality improvements).

## M1 — TBD on MOT16-04 (pending)

| Mode | MOTA | FP | FN | IDSW | FPS (e2e) |
|---|---|---|---|---|---|
| SORT | | | | | |
| ByteTrack | | | | | |

Acceptance: MOTA > 25%, ByteTrack IDSW reduction >= 20% vs SORT, >= 10 FPS.

## M2 — SOT on mini-OTB (pending)

| Sequence | Success AUC | Precision@20px | FPS |
|---|---|---|---|

Acceptance: differential vs cv::TrackerNano median IoU >= 0.9; mean AUC >= 0.55; >= 80 FPS.

## M3 — MOSSE fallback (pending)

Acceptance: >= 200 FPS; IoU >= 0.5 mean on Car4/CarDark; PSR drop on occlusion.

## M4 — latency baseline (pending)

Per-stage p50/p95 from `--bench-json`.
