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

## M2 — NanoTrack v2 on mini-OTB (2026-07-03, host CPU, 3 static ORT graphs)

| Sequence | AUC | prec@20 | mIoU | FPS |
|---|---|---|---|---|
| Car4 | 0.719 | 0.974 | 0.731 | 485 |
| CarDark | 0.502 | 0.659 | 0.511 | 412 |
| BlurCar2 | 0.809 | 1.000 | 0.825 | 386 |
| Jogging | 0.684 | 0.984 | 0.694 | 386 |
| Girl2 | 0.437 | 0.627 | 0.437 | 411 |
| Woman | 0.636 | 0.998 | 0.642 | 432 |
| **MEAN** | **0.631** | **0.874** | 0.640 | ~420 |

Acceptance: differential vs cv::TrackerNano median IoU >= 0.9 ✓ (test green) ·
mean AUC >= 0.55 ✓ (0.631) · >= 80 FPS ✓ (~420). Sequence notes: DragonBaby was
never archived and Human3's zip does not survive Wayback transfer — replaced by
Woman and Jogging (documented in fetch_otb.sh). Girl2 (1500 frames, repeated full
occlusion) is the known weak spot — re-acquisition (M4) and step-3 improvements
target exactly this.

## M3 — MOSSE fallback on mini-OTB (same run)

| Sequence | AUC | prec@20 | mIoU | FPS |
|---|---|---|---|---|
| Car4 | 0.481 | 1.000 | 0.478 | 2375 |
| CarDark | 0.752 | 1.000 | 0.768 | 2119 |
| BlurCar2 | 0.129 | 0.132 | 0.116 | 1060 |
| Jogging | 0.558 | 0.974 | 0.561 | 1789 |
| Girl2 | 0.059 | 0.075 | 0.060 | 964 |
| Woman | 0.102 | 0.198 | 0.101 | 2024 |

Acceptance: >= 200 FPS ✓ (964+) · Car4/CarDark mean IoU >= 0.5 ✓ (0.623) · PSR
collapse under occlusion ✓ (unit test). Blur and long occlusion defeat a 64x64
correlation filter as expected — MOSSE is the NN-free fallback, not the main line.

## M4 — latency baseline (pending)

Per-stage p50/p95 from `--bench-json`.
