# Results

Metric and latency tables per milestone. Populated as milestones land; this file is the
regression baseline for step 2 (speed) and step 3 (quality improvements).

## M1 — TBD on MOT16-04 (2026-07-03, host CPU, 1920x1080, YOLOv8n@640 person class)

| Mode | MOTA | MOTP(dist) | FP | FN | IDSW | IDF1 | e2e p50 | FPS |
|---|---|---|---|---|---|---|---|---|
| SORT | 25.3% | 0.163 | 744 | 34768 | 29 | 39.0% | 44.9 ms | ~22 |
| ByteTrack | **31.0%** | 0.176 | 1132 | 31665 | **23** | **43.7%** | 44.6 ms | ~22 |

Acceptance: MOTA > 25% ✓ (31.0%) · ByteTrack IDSW cut >= 20% vs SORT ✓ (29 -> 23,
-20.7%) · >= 10 FPS ✓ (~22). GT: MOT16 native (motchallenge.net unreachable; MOT17
re-annotation swap pending network access — flagged in fetch_mot.sh). High FN is the
COCO-pretrained nano detector on small/occluded pedestrians — expected at this stage;
VisDrone/finetune is step 3.

## M2 — SOT on mini-OTB (pending)

| Sequence | Success AUC | Precision@20px | FPS |
|---|---|---|---|

Acceptance: differential vs cv::TrackerNano median IoU >= 0.9; mean AUC >= 0.55; >= 80 FPS.

## M3 — MOSSE fallback (pending)

Acceptance: >= 200 FPS; IoU >= 0.5 mean on Car4/CarDark; PSR drop on occlusion.

## M4 — latency baseline (pending)

Per-stage p50/p95 from `--bench-json`.
