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

## M4 — lost detection, re-acquisition, latency baseline (2026-07-03)

**State machine + re-acquisition (mechanism)**: verified by unit/e2e tests —
synthetic occlusion drives Tracking -> Unstable -> Lost; a gated (class,
position-radius growing with time-lost, size-similarity) nearest candidate from
the detector re-seeds the template under Unstable probation; a wrong-class
distractor never re-locks.

**Real-data behaviour (honest picture)**:

| Case | Without re-acquire | With re-acquire | What it shows |
|---|---|---|---|
| Woman / MOSSE | AUC 0.102, prec@20 0.198 | AUC 0.198, **prec@20 0.454** | The rescue loop works on real footage (1 re-lock fired) |
| Jogging / MOSSE | AUC 0.558 | AUC 0.155 | Two adjacent same-class targets: gates re-lock the WRONG jogger — needs appearance verification (step 3 re-ID) |
| Girl2 / nano | AUC 0.437 | AUC 0.437 (never fires) | NanoTrack fails by CONFIDENT drift to a distractor; score never collapses, so confidence-gated loss detection is blind to it (step 3: re-ID verification, GMC) |

Consequence for defaults: `--reacquire` is opt-in per scenario; for a
single-target-class scene with real occlusions it is a clear win, in
multi-instance scenes it needs step-3 appearance verification. Both failure
modes are exactly the top of the step-3 improvement list.

## Latency baseline (step-2 optimization reference, host: 12-core x86_64)

| Pipeline | Input | n | mean | p50 | p95 | FPS(p50) |
|---|---|---|---|---|---|---|
| TBD detect+track (YOLOv8n@640 + ByteTrack) | 1920x1080 | 1050 | 44.26 ms | 44.29 ms | 45.37 ms | ~23 |
| SOT NanoTrack (3 ORT graphs) | 360x240 (Car4) | 659 | 2.84 ms | 2.74 ms | 3.26 ms | ~365 |
| SOT MOSSE (cv::dft, 64x64) | 360x240 (Car4) | 659 | 0.43 ms | 0.38 ms | 0.65 ms | ~2600 |

JSON artifacts: `results/bench_*.json` (regenerate with `--bench-json`). Demo
overlays: `results/*.mp4` (gitignored; regenerate with the commands in this file's
history). Known cosmetic noise: duplicate ONNX schema warnings at startup when
OpenCV-dnn and ONNX Runtime coexist in one process (each carries a libonnx copy) —
harmless, filtered in scripts.

# Step 2 — speed optimization (started 2026-07-03)

Method: every change is hypothesis -> `scripts/bench.sh` before/after -> keep only
if the p50 delta clears 2x the baseline stddev below; reverted experiments get
their negative result recorded here. Quality gates for default-config changes:
all tests green (incl. the cv::TrackerNano oracle and the YOLO golden tensor),
mini-OTB mean AUC within 0.005 of 0.631, MOT16-04 MOTA within 0.2 pt of 31.0%.

## S2.0 — fine-grained baseline (5 reps, powersave governor, unpinned)

| scenario | p50 median | stddev | 2σ noise floor | FPS |
|---|---|---|---|---|
| tbd_mot16 (1080p, YOLOv8n@640+ByteTrack) | 43.78 ms | 0.66 | 1.3 ms | ~23 |
| sot_nano_car4 (360x240) | 2.73 ms | 0.27 | 0.5 ms | ~366 |
| sot_nano_1080p (MOT16-04 frames) | 3.65 ms | 0.06 | 0.12 ms | ~274 |
| sot_mosse_car4 | 0.37 ms | 0.002 | — | ~2700 |

Stage attribution (p50, representative run):

| stage | tbd_mot16 | | stage | nano@240p | nano@1080p |
|---|---|---|---|---|---|
| det.preprocess | 1.33 ms | | sot.crop | 0.15 ms | **1.44 ms** |
| det.infer | **41.09 ms** | | sot.blob | 0.07 ms | 0.07 ms |
| det.decode | 0.46 ms | | sot.backbone_x | 1.47 ms | 1.59 ms |
| assoc | 0.04 ms | | sot.head | 0.49 ms | 0.54 ms |
| total | 43.46 ms | | sot.postproc | 0.004 ms | 0.004 ms |

What the numbers dictate:
- **TBD is 96% ORT session time.** Everything outside `det.infer` totals ~1.8 ms,
  so quality-neutral work on preprocess/decode is bounded there; the ≤30 ms bar
  depends on engine/thread tuning (S2.2), and the 30+ FPS targets on the
  tradeoff ladder (resolution / detect-interval / INT8).
- **SOT at 1080p pays 1.44 ms (39%) in `sot.crop`** — the full-frame `cv::mean`
  for pad color plus whole-frame `copyMakeBorder`; the S2.1 subwindow rework
  targets exactly this.
- **nano_car4 is bimodal across runs (2.18 vs 2.73 ms)** — hybrid P/E-core
  scheduling. SOT keep-or-revert decisions use P-core-pinned runs
  (`CTRK_BENCH_TASKSET=0-3 scripts/bench.sh`); headline numbers stay unpinned.
- `perf record` is unavailable on this host (perf_event_paranoid=4); stage
  attribution comes from the in-process profile sink (`ctrk/profile.hpp`),
  whose scopes cover >99% of each pipeline's fused-stage time.
- Capture (sequential JPEG decode, ~5.6 ms @1080p) and encode are outside the
  compute path but reported by `--bench-json` for completeness.

## S2.1 — semantic-preserving CPU work (all outputs proven identical)

| change | stage | before | after | proof of equivalence |
|---|---|---|---|---|
| cached canvas + planar preprocess | det.preprocess | 1.30 ms | 0.81 ms | MOT16-04 dump byte-identical |
| row-contiguous decode + scratch | det.decode | 0.46 ms | 0.32 ms | golden tensor exact + dump |
| pad-only-when-needed subwindow | sot.crop @1080p | 1.44 ms | 0.13 ms | bit-exact test + oracle green |
| — effect on nano @1080p | sot total | 3.65 ms | 2.34 ms | mini-OTB AUC 0.631 reproduced exactly |
| — effect on nano Car4 | sot total | 2.73 ms | 2.01 ms | (same run set) |
| lazy pad mean at MOSSE call site | sot total | 0.370 ms | 0.223 ms | crop_subwindow bit-exact test |

Negative result (D-0009): `-DCTRK_MARCH_NATIVE=ON -DCTRK_LTO=ON` moved no owned
stage (preprocess 0.80 vs 0.81, decode 0.315 vs 0.32); the hand loops are
memory-bound at SSE2 already and ORT dispatches its own SIMD. Both options
stay OFF.

## S2.2 — ORT engine tuning (sweep verdicts, 2026-07-03)

Detector thread sweep (det.infer p50, 1080p MOT16-04; 1 rep, best confirmed 3-rep):

| intra-op threads | 1 | 2 | 4 (old default) | 6 | 8 | **10** | 12 | 0=auto |
|---|---|---|---|---|---|---|---|---|
| det.infer (ms) | 84.2 | 48.5 | 40.9 | 36.8 | 33.6 | **31.0** | 35.9 | 34.0 |

- **10 threads (nproc−2) is host-best**: 3-rep median infer 30.96 / total
  32.27 ms (~31 FPS). 12 oversubscribes (main thread contention); ORT auto
  picks worse. Exposed as `--threads`; `tbd_tuned` bench scenario pins it.
- **Spin control**: TBD t10 `--no-spin` costs ~1 ms (33.4 vs 32.3) but cuts
  process CPU from **1002% to 597%** — the recommended SoC posture; default
  stays spinning for the host latency headline.
- **SOT nano: 2 threads + no-spin strictly dominates** — fastest of every
  measured config (Car4 p50 **1.926 ms**, 3 reps, was 2.73 at S2.0) and 355%
  → 213% CPU. Now the SotConfig default (`--spin` restores the old
  behavior). Three tiny sessions busy-waiting between the two sequential
  runs per frame only fight each other. 1 thread: ~156% CPU, ~0.6 ms slower
  — the minimal-CPU option.
- **oneDNN EP: rejected** — det.infer 79.5 ms, ~2.6x slower than the default
  CPU EP on this graph; dropped (D-0010).
- Engine plumbing (shared env, cached MemoryInfo, prebuilt name arrays,
  reused input vector, load-time warmup) removed the first-frame spike;
  steady-state effect within noise, kept for allocation hygiene.

## S2.3 — tradeoff ladder knob: detect-every-N + KF coasting

MOT16-04 @1080p, YOLOv8n@640, `--threads=10`; coasted predictions are scored
against the per-frame GT (an honest measure of coasting quality). Effective
latency = mean over all frames (detect frames ~29-31 ms, coast frames ~4 µs).

| N | effective mean | eff. FPS | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---|---|---|---|---|---|
| 1 (default) | 32.3 ms | 31 | **31.0%** | 43.7% | 23 | 1132 | 31665 |
| 2 | **15.0 ms** | 67 | 29.2% | **44.4%** | **18** | 1123 | 32510 |
| 3 | 10.5 ms | 95 | 28.4% | 41.3% | 20 | 1075 | 32957 |
| 5 | 6.0 ms | 166 | 25.0% | 38.4% | 13 | 1021 | 34613 |

Findings:
- **N=2 halves compute for −1.8 MOTA while IDENTITY metrics improve**
  (IDF1 +0.7, IDSW 23→18): the KF bridging detector flicker beats
  re-associating noisy per-frame detections. The MOTA cost is coasted-box
  drift counted as FN. N=2 is the recommended drone operating point when
  detector budget matters.
- Even N=5 (166 FPS effective) still meets the original M1 MOTA bar (25%).
- Caveat for fast targets: a track still inside n_init confirmation has no
  learned velocity, and the tentative-stage IoU gate (0.7) sees N× the
  inter-detection motion — track birth churns for objects moving faster
  than ~⅓ box-width per detect interval. Step-3 re-ID/GMC work also lands
  here.
- Side observation: detect-frame p50 is *lower* at N≥2 (28-30 vs 31 ms) —
  the idle coast frames let the hybrid cores recover boost headroom.

## S2.4 — tradeoff ladder knob: detector input resolution

Static exports (opset 12, same contract; `export_yolo.py --imgsz`), MOT16-04
@1080p, `--threads=10`. The C++ detector is shape-driven, so this is
export-only.

| input | total p50 | FPS | MOTA | IDF1 | IDSW | FP | FN |
|---|---|---|---|---|---|---|---|
| 640 (default) | 32.3 ms | 31 | **31.0%** | 43.7% | 23 | 1132 | 31665 |
| 512 | 21.5 ms | 47 | 24.0% | 37.5% | 11 | 844 | 35284 |
| 448 | 16.8 ms | 60 | 18.1% | 28.7% | 11 | 744 | 38171 |
| 416 | 15.4 ms | 65 | 16.9% | 27.4% | 11 | 612 | 38913 |

Verdict: **on this footage, resolution is the WORST rung of the ladder** —
MOT16-04 pedestrians are already small at 1080p, and every step down
explodes FN. Head-to-head at equal compute: 416 costs 15.4 ms for 16.9
MOTA, while detect-every-2 at 640 costs 15.0 ms for **29.2** MOTA. Prefer
the interval knob whenever targets are small; revisit resolution only for
close-range/large-object drone scenarios (and re-measure there — this
ranking is scene-dependent, not universal).

## S2.5 — tradeoff ladder knob: INT8 static QDQ (the NPU rehearsal, D-0008)

`tools/export/quantize.py`: QDQ, per-channel INT8 weights, UINT8
activations, calibrated on 300 MOT16-04 frames (YOLO) / OTB gt-centered
crops (nano). Two rehearsal lessons were earned the hard way:
1. Per-channel DQ needs opset ≥ 13 (`axis`); the INT8 *variant* is bumped,
   the canonical fp32 opset-12 artifact is untouched.
2. Quantizing the whole graph produced **zero detections**: the Detect head
   concatenates box coords (0..640) and class scores (0..1) into one tensor
   and a single activation scale crushes the scores. Head (`model.22`)
   stays fp32; backbone+neck carry the speedup.

| config (@640, t10) | total p50 | FPS | MOTA | IDF1 | IDSW |
|---|---|---|---|---|---|
| fp32 (reference) | 32.3 ms | 31 | 31.0% | 43.7% | 23 |
| INT8 bb+neck, head fp32 | **24.0 ms** | 42 | **31.3%** | **45.1%** | 25 |
| INT8 + detect-every-2 | **10.1 ms eff.** | ~99 | 29.9% | 44.0% | 20 |

- **Detector INT8 is metric-parity at 1.35x speed** (AVX-VNNI; 12.3 → 6.3 MB
  with the fp32 head). The +0.3 MOTA is threshold jitter, not a real gain.
- **INT8 + N=2 is the recommended drone operating point**: 3.2x less
  compute than the fp32 baseline for −1.1 MOTA, identity metrics better
  than baseline.
- **nano backbones: INT8 rejected on host** — mean AUC 0.631 → 0.555
  (Car4 collapses 0.719 → 0.287; depthwise sensitivity is
  sequence-dependent) AND it is *slower* (~430 vs ~500 FPS: per-layer QDQ
  overhead beats VNNI gains on ~1 MB graphs). The correlation head was
  never quantized by design (it is the stay-on-CPU graph in the NPU split
  plan). Re-run this experiment with the vendor toolchain once the NPU is
  known — the balance is different there (D-0008).

## S2.6 — step-2 rollup (v0.2.0, 2026-07-03)

Final `scripts/bench.sh --reps 3` (powersave governor, unpinned; note the
no-spin SOT default also collapsed run-to-run variance, σ 0.27 → 0.005):

| scenario | S2.0 baseline | v0.2.0 | Δ | quality proof |
|---|---|---|---|---|
| TBD default (fp32@640, 4 thr) | 43.78 ms | 43.08 ms | −2% | dump byte-identical |
| TBD `--threads=10` | — | 32.73 ms (31 FPS) | −25% | MOTA/FP/FN/IDSW digit-identical |
| TBD INT8 (head fp32) | — | 23.96 ms (42 FPS) | −45% | MOTA 31.3 vs 31.0 (parity) |
| TBD INT8 + `--detect-every=2` | — | **10.1 ms eff. (~99 FPS)** | −77% | MOTA 29.9, IDF1 44.0 (> baseline) |
| SOT nano Car4 | 2.73 ms | **1.94 ms (515 FPS)** | −29% | oracle green, AUC 0.631 exact |
| SOT nano @1080p | 3.65 ms | **2.12 ms (472 FPS)** | −42% | (same code path) |
| SOT MOSSE Car4 | 0.370 ms | 0.221 ms | −40% | bit-exact subwindow test |

Acceptance-bar scorecard (plan: draft-a-plan step 2):
- SOT Car4 ≤ 2.2 ms ✓ (1.94) · SOT@1080p ≥ 40% cut ✓ (−42%) · quality
  gates ✓ (strongest form: byte-identical / digit-identical / bit-exact) ·
  ladder ≥ 30 FPS with quantified cost ✓ (three rungs: 42 / 67 / ~99 FPS) ·
  bench.sh reproducibility ✓.
- **TBD quality-neutral ≤ 30 ms: MISSED on the strict definition** — the
  best quality-*identical* config is 32.7 ms (1.34×). 96% of the pipeline
  is the fp32 ORT session, and threads were the only quality-identical
  lever left; the plan's risk section predicted exactly this. The INT8 rung
  (24.0 ms, metric-parity) crosses the bar in practice and is one flag away.

Recommended operating points (host; re-validate on the SoC in step 4):
- **Drone TBD**: `--threads=<nproc-2> --model=..._int8.onnx --detect-every=2`
  → ~99 FPS effective, MOTA −1.1 vs baseline, identity metrics better than
  baseline. Add `--no-spin` when power matters (−40% CPU, +~1 ms).
- **Drone SOT**: defaults (2 threads, no-spin) — 1.9-2.1 ms/frame leaves
  >90% of a 30 FPS frame budget to the rest of the system.
- Step-3 quality work (re-ID, GMC) should re-run this ladder; the knobs
  compose with it unchanged.
