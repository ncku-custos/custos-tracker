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

# Step 3 — quality improvements (started 2026-07-03)

Method unchanged from step 2: hypothesis -> eval -> keep-or-revert; negative
results recorded. Quality gates for default-config changes (anchors reproduced
digit-identical on this host, see S3.0 — no re-anchor needed): all tests green
with the oracle and golden tensor actually RUNNING (they GTEST_SKIP without
fetched models — a fresh checkout is "green" without exercising them),
mini-OTB mean AUC within 0.005 of 0.631, MOT16-04 MOTA within 0.2 pt of
31.0%, IDSW within 2 of 23 (adopting ROADMAP.md's stricter form as canonical).
Latency claims clear 2x the S3.0 stddev. Oracle-breaking features stay
config-gated until metrics prove them.

## S3.0 — step-2 host retired; new-host bring-up + baseline (2026-07-03)

Host: AMD Ryzen 7 7700 (8C/16T homogeneous, AVX-512), 14 GB RAM, RTX 5060
8 GB (driver 580.159.03, CUDA 13.0 — unblocks the step-3 VisDrone item),
Ubuntu 26.04, gcc 15.2, OpenCV 4.10, ORT 1.23, powersave governor. Python
tooling: torch 2.11.0+cu128, ultralytics resolved to **8.4.86 — the exact
step-2 pin**, so YOLO exports are the same graphs.

Gate reproduction (fresh local exports, MOT16 native GT — motchallenge.net
is connection-refused from this network too, so the MOT17-04 GT upgrade
stays pending):

| gate | step-2 anchor | this host | verdict |
|---|---|---|---|
| mini-OTB mean AUC | 0.631 | 0.631 (all 6 per-seq AUCs identical) | exact |
| MOT16-04 MOTA / IDSW / IDF1 | 31.0% / 23 / 43.7% | 31.0% / 23 / 43.7% | digit-identical |
| tests | 68 pass | 68 pass, oracle + golden tensor running | green |

Anchors carry over unchanged (D-0011). Noise floor, 5 reps, unpinned:

| scenario | p50 median | stddev | 2σ floor | FPS | old host |
|---|---|---|---|---|---|
| tbd_mot16 (fp32@640, 4 thr) | 24.981 ms | 0.055 | 0.11 ms | 40 | 43.78 |
| tbd_mot16_13 (new: moving cam) | 24.819 ms | 0.070 | 0.14 ms | 40 | — |
| sot_nano_car4 | 1.713 ms | 0.005 | 0.01 ms | 584 | 1.94 |
| sot_nano_1080p | 1.778 ms | 0.004 | 0.008 ms | 562 | 2.12 |
| sot_mosse_car4 | 0.255 ms | 0.000 | — | 3922 | 0.221 |

Homogeneous cores collapse the run-to-run variance (no P/E bimodality —
no pinning caveat needed on this host). Host-tuning re-sweep (D-0010 numbers
do NOT transfer, as that entry warned):

| detector intra-op threads | 4 | 6 | 7 | **8** | 9 | 10 | 12 | 14 | 16 | 0=auto |
|---|---|---|---|---|---|---|---|---|---|---|
| det.infer p50 (ms) | 23.8 | 18.8 | 17.1 | **15.9** | 19.1 | 20.7 | 19.6 | 18.7 | 17.7 | 18.8 |

- **8 threads = the physical core count is a sharp optimum** (3-rep medians:
  7 -> 17.12, 8 -> 15.89, 9 -> 19.07; total 17.05 ms, ~59 FPS). The i5's
  nproc−2 heuristic would pick 14 (18.7 ms, −18%). SMT siblings actively
  hurt MLAS here; `tbd_tuned` now pins `--threads=8`.
- **SOT spin verdict flips on this host**: 2t+no-spin (the SotConfig
  default) is 1.709 ms / 208% CPU, but 2t+spin is 1.512 ms / 358% and
  4t+spin 1.226 ms / 726%. On the i5, no-spin was fastest AND cheapest; here
  no-spin costs ~12% latency for −40% CPU. The default stays no-spin (it is
  the SoC power posture, and the absolute cost is 0.2 ms), but the "strictly
  dominates" claim is i5-specific — re-measure on the SoC in step 4.

MOT16-13 baseline (new second eval scenario, bus-mounted moving camera,
750 frames; MOT16 native GT, pedestrian class):

| config | MOTA | MOTP | FP | FN | IDSW | IDF1 |
|---|---|---|---|---|---|---|
| defaults, N=1 | 13.2% | 0.228 | 78 | 9856 | 6 | 21.3% |
| `--detect-every=2` | 9.3% | 0.240 | 24 | 10356 | 2 | 16.0% |

FN-dominated (9856 of 11450 GT boxes missed — small pedestrians from a
moving platform are exactly the COCO-nano failure mode, M1) and nearly
IDSW-free at this detection density, so GMC/appearance work (S3.4/S3.5)
will be read primarily through MOTA/FN/IDF1 deltas here, not IDSW. Unlike
MOT16-04, N=2 clearly hurts on this sequence — camera motion breaks the
constant-velocity coast — which is itself the GMC hypothesis in miniature.

## S3.1 — NSA-Kalman: kept, default ON (2026-07-03)

Hypothesis: scaling measurement noise R by (1 − det score) (NSA form,
GIAOTracker/BoT-SORT lineage) lets confident detections correct the filter
harder -> tighter boxes, fewer FP/FN at match time. One-line filter change
(`KalmanBox::update(z, r_scale)`), wired as `AssocConfig::nsa`, single call
site in `mark_matched`. Deterministic evals — deltas are exact, not noise:

| config | MOTA | FP | FN | IDSW | IDF1 |
|---|---|---|---|---|---|
| MOT16-04 N=1 | 31.0 -> **31.3** | 1132 -> 1069 | 31665 -> 31593 | 23 | 43.7 -> **43.9** |
| MOT16-04 N=2 | 29.2 -> **29.7** | 1123 -> 1008 | 32510 -> 32391 | 18 | 44.4 -> **44.7** |
| MOT16-13 N=1 | 13.2 -> **13.7** | 78 -> 75 | 9856 -> 9800 | 6 -> **5** | 21.3 -> **22.3** |
| MOT16-13 N=2 | 9.3 (flat) | 24 -> 25 | 10356 -> 10355 | 2 | 16.0 (flat) |

Improves or ties every measured config; nothing regresses. **Default flipped
to ON** (`--no-nsa` restores the classic filter and is byte-identical to the
pre-change default; the new default is byte-identical to the measured `--nsa`
runs). Gates: 69 tests green (oracle running), mini-OTB 0.631 exact (SOT
untouched). **MOT16-04 default-config anchors move to MOTA 31.3 / IDSW 23 /
IDF1 43.9** — quality-work anchor updates are recorded here; the ±0.2 MOTA /
±2 IDSW tolerances now apply to these values.

## S3.2 — tentative-gate churn fix: relax + patience kept, velocity seeding rejected (2026-07-03)

The S2.3 pathology, now reproduced in a unit test
(`TentativeChurn.FastMoverAtIntervalTwoNeverConfirmsWithoutTheS32Knobs`): at
detect-interval N a newborn track has no learned velocity, its prediction
stays put while the target moves N frames, the 0.7 stage-3 gate kills it on
the first miss, and the target churns fresh tentative ids forever — it
NEVER confirms, i.e. a permanent FN, worst exactly where coasting is most
needed. Three independent knobs measured (all default-inert at the time of
measurement; deterministic evals, post-NSA baselines):

| MOTA / IDF1 | 04 N=2 | 04 N=3 | 04 N=5 | 13 N=2 | 13 N=3 | 13 N=5 |
|---|---|---|---|---|---|---|
| baseline (S3.1) | 29.7/44.7 | 28.9/42.2 | 25.6/38.8 | 9.3/16.0 | 6.6/12.1 | 1.6/3.5 |
| relax=0.15 | 29.8/44.8 | 28.9/42.3 | 26.3/39.4 | 11.3/18.5 | 10.4/18.2 | 7.7/14.2 |
| relax=0.3 | 29.8/44.8 | 28.9/42.3 | 26.3/39.4 | 14.7/23.5 | 12.8/21.9 | 7.7/14.2 |
| patience=1 | 29.8/44.8 | 29.1/42.5 | 25.9/39.2 | 9.3/16.0 | 6.6/12.1 | 1.6/3.5 |
| velocity seed | 26.2/40.2 | 25.8/38.4 | 24.6/38.2 | 6.8/14.2 | 6.1/11.7 | 1.5/3.3 |
| relax=0.3+seed | 26.7/40.8 | 25.0/38.4 | 20.3/34.7 | 13.5/23.6 | 10.0/18.9 | 6.1/13.1 |
| **relax=0.3+patience=1** | **29.9/44.9** | **29.2/42.6** | **26.8/40.0** | **14.7/23.5** | **13.2/22.5** | **7.7/14.2** |

- **Kept, default ON: `tentative_relax_per_coast=0.3` (floor 0.2) +
  `tentative_patience=1`.** Best or tied on every MOTA/IDF1 cell; the
  moving-camera sequence is transformed (N=2 MOTA +5.4, N=5 rescued from
  collapse 1.6 -> 7.7). IDSW rises a few counts at deep intervals (04 N=5:
  13 -> 18) because targets that previously never confirmed now exist to be
  switched — IDF1 (up everywhere) is the honest identity read.
- **Negative result, kept OFF: velocity seeding.** Seeding the newborn KF
  velocity from its first re-match delta wins on noiseless synthetic motion
  (unit test) but LOSES on real detections: one noisy displacement sampled
  at interval N flings the prediction (MOTP 0.176 -> 0.20+, FP +70%, MOTA
  −3 to −5 on 04). The Kalman blend from the zero-velocity prior is the
  better velocity estimator once the relaxed gate keeps the track alive
  long enough to learn.
- relax15 ≡ relax30 digit-identical on 04 (candidate IoUs there cluster
  outside the 0.4-0.55 band) but relax30 clearly better on 13 where
  apparent motion is larger; 0.3 with the 0.2 floor adopted.
- **N=1 caveat (gates)**: the relax knob is provably inert at N=1
  (coasted=0), but patience=1 also lets a newborn survive one detector
  flicker at N=1 — measured: MOT16-04 N=1 anchors digit-identical (31.3 /
  23 / 43.9, FN one box lower); MOT16-13 N=1 +0.1 MOTA/IDF1. Within gates;
  anchors unchanged.

## S3.3 — SOT re-ID appearance verification: HSV veto default in --reacquire, NanoZ drift check opt-in (2026-07-03)

Both M4 failure modes attacked with one `similarity(img, box)` seam and two
zero-new-model embedders — (a) HSV H-S histogram (1 − Bhattacharyya,
backend-free) and (b) the NanoTrack template branch re-run on candidate
crops (`NanoTracker::embed`, cosine vs the frozen init `zf_`; nano only).
Three insertion points: re-lock candidates below `accept` are vetoed (and
the survivor is picked by similarity, not proximity); a periodic check on
the *tracked* box latches Lost when appearance collapses even though the
score never does; the veto also guards every later re-lock. References are
frozen at init so verification always compares against the original target.
The synthetic lookalike pair is a unit test (`ReidVerify.*`: gates alone
re-lock the wrong target; the veto stays Lost).

**Wrong re-lock (Jogging, MOSSE + --reacquire)** — accept sweep, HSV:

| accept | 0 (off) | 0.2 | 0.3 | **0.4** | 0.5 |
|---|---|---|---|---|---|
| Jogging AUC | 0.155 | 0.155 | 0.155 | **0.558** | 0.558 |
| Woman AUC / prec@20 | 0.198/0.454 | 0.198/0.454 | 0.198/0.454 | **0.235/0.563** | 0.102/0.198 |

accept=0.4 (the HSV auto) removes the wrong-jogger re-lock entirely
(0.155 -> 0.558, prec@20 0.228 -> 0.974 — the M4 hypothesis target was
>= 0.5) and IMPROVES the Woman rescue (a bad early re-lock is now vetoed, a
better later one taken). Honest caveat: the separating bands are narrow —
wrong-jogger sim lands in [0.3, 0.4), the Woman rescue in [0.4, 0.5).

**Confident drift (Girl2, nano)** — the drift check fires at score 1.00
(log: "drift detected (similarity 0.59 < …, score 1.00)"), which is exactly
the blindness M4 documented. Full nano mini-OTB per embedder (reacquire on):

| config | Girl2 | Jogging | other 4 | MEAN AUC / prec@20 |
|---|---|---|---|---|
| baseline / reacq-only / hsv-veto-only | 0.437/0.616 | 0.684 | unchanged | 0.631/0.872 |
| hsv drift K=5 thr 0.4 | 0.441/**0.793** | **0.657 (regression)** | unchanged | 0.627/0.901 |
| **nanoz drift K=5 thr 0.55 acc 0.65** | **0.455/0.687** | 0.684 | unchanged | **0.634/0.884** |

- nano + reacquire alone (and the veto alone) change NOTHING on nano — the
  score-driven Lost never fires, confirming the M4 analysis.
- HSV as the *drift* embedder false-fires on Jogging's partial occlusions
  (-0.027 AUC there) — rejected for the drift role.
- NanoZ as the drift embedder is strictly >= baseline on every sequence.
  Its similarity scale is compressed (~0.5-0.65 for any person-ish crop —
  the backbone was trained for localization, not identity), so thresholds
  are tight: thr 0.5 never fires (= baseline), 0.6 over-fires. 0.55 adopted.
- Ladder stops at (b): OSNet-lite (c) not needed — (a) won the veto role,
  (b) won the drift role.

**Defaults**: within `--reacquire` the HSV veto is now ON (`--reid=hsv`,
auto accept 0.4; `--reid=none` restores pure geometric gating). The drift
check stays opt-in (`--drift-every=5 --reid=nanoz` is the measured Girl2
configuration). Default-config mini-OTB is untouched: 0.631 exact, 75
tests green (oracle + embed test running). Costs: NanoZ embed ≈ one extra
z-graph pass per checked frame (FPS column above shows the drift-check
runs slower on Lost-heavy sequences); HSV is ~free.

## S3.4 — GMC (BoT-SORT sparse-flow), opt-in: transformative on the moving camera (2026-07-03)

`--gmc`: downscaled-gray LK flow + RANSAC partial affine per frame (coast
frames included — the previous-gray chain must not skip), applied to every
track's KF state (position full affine, velocity linear part, height the
isotropic scale, covariance congruence) via the new
`KalmanBox::apply_affine`. Last frame's boxes are masked out of the corner
detector. Estimator unit-tested on synthetic pans incl. the downscale path.

| MOTA / IDSW / IDF1 | defaults | `--gmc` |
|---|---|---|
| MOT16-04 N=1 (static cam) | 31.3 / 23 / 43.9 | 31.2 / 23 / **45.2** |
| MOT16-04 N=2 | 29.9 / 18 / 44.9 | 29.9 / 19 / 44.1 |
| MOT16-13 N=1 (moving cam) | 13.8 / 5 / 22.4 | **19.2** / 6 / **32.1** |
| MOT16-13 N=2 | 14.7 / 7 / 23.5 | **18.5** / 5 / **30.7** |

- **Moving camera: the largest single quality jump of step 3** (+5.4 MOTA,
  +9.7 IDF1 at N=1; FN −687). Static camera: within gates at both
  intervals (the neutrality safety check).
- **Cost**: `gmc` stage 2.05 ms p50 at 1080p — coast frames go from ~4 us
  to ~2 ms, so the N=2 effective-latency story changes when it is on.
  Kept OPT-IN this step (host latency headline unchanged); D-0013 argues
  default-on for the drone deployment where the camera always moves.
- **Post-mortem worth recording**: the first GMC runs collapsed MOT16-04
  N=2 (IDSW 18 -> 56, IDF1 44.9 -> 29.4) ONLY when NSA was also on, with
  sub-pixel warps (max |t| 1.2 px, inlier ratio ~1.0 — measured, so not
  estimation noise). Root cause was numerical, not algorithmic: repeated
  float32 `J P J^T` congruences slowly de-symmetrize the covariance, and
  once `S = H P H' + R` loses PSD-ness the DECOMP_CHOLESKY solve returns
  garbage gains — classic-R runs never noticed because the larger R
  re-regularizes S, NSA's small R does not. One-line fix: re-symmetrize P
  after the congruence (`P = (P + P')/2`). Two hypotheses were tested and
  rejected on the way (crowd corners hijacking RANSAC — masking changed
  nothing; NSA near-deadbeat gain oscillation — flooring r_scale at 0.1
  and 0.3 changed nothing); dead ends recorded per the process rule.
- Not taken: feeding the warp into the SOT search window. The SOT crop is
  target-centered (camera motion mostly cancels through the tracked
  position) and mini-OTB has no scenario isolating the effect; revisit on
  drone footage in step 4 if search-window loss shows up.

## S3.5 — appearance cost in TBD association: null result with HSV, seam kept (2026-07-03)

`Detection` gained an optional `embedding` (public plain-struct field,
D-0001-compliant), `STrack` an EMA copy, and the stage-1 cost fuses
`(1-w)(1-IoU) + w(1-cos)` for gate-passing pairs
(`--appearance=<w>`, off default). The mechanism is proven by a unit test
(orthogonal embeddings keep ids straight through a complete crossing that
geometry alone swap-flips) and by wiring probes (embed stage 1.36 ms p50
for ~50 dets; w=0.99 changes outputs). The measured verdict on real data
is a clean NULL:

- w ∈ {0.25, 0.5}: **digit-identical to baseline on every config**
  (MOT16-04 and MOT16-13, N ∈ {1,2}, GMC on and off — 8 runs). The HSV
  histograms of small street-scale pedestrians are too similar to flip a
  single IoU-gated Munkres assignment.
- w = 0.99 (appearance-dominant): slightly WORSE (MOT16-04 N=2 IDSW
  18 -> 20, IDF1 44.9 -> 43.9) — when colour does overrule geometry here,
  it is wrong more often than right.

Kept: the association seam and the `Detection::embedding` contract (a
future discriminative embedder — OSNet-class, or detector-derived features
on the NPU — plugs in without touching the tracker); the HSV filler stays
opt-in. Rejected: making HSV appearance a default association term. This
mirrors D-0012's split verdict: colour separates two specific adjacent
lookalikes (the SOT veto) but carries no marginal identity signal over
IoU+GMC in crowd association at these scales.

## S3.6 — VisDrone fine-tune: trained and exported; not for MOT16 (2026-07-03)

The GPU unblock (D-0011) made ROADMAP item 9 cheap: yolov8n on VisDrone,
80 epochs / imgsz 640 / batch 16 on the RTX 5060 took **~65 min**
(vs "days of CPU training" on the step-2 host). Val: mAP50 0.309 /
mAP50-95 0.174 (pedestrian 0.324, car 0.733 — in line with published
yolov8n-VisDrone numbers). `export_yolo.py` now takes `--weights/--stem`
(output-shape assert generalized to the class count; the default yolov8n
path re-verified byte-identical tracking) ->
`models/cache/yolov8n_visdrone_640.onnx` (1x14x8400, 10 classes; local
export, never fetched, never default; AGPL D-0003 extends to fine-tuned
weights).

Measured honestly on the street-level eval (VisDrone class ids 0/0,1 =
pedestrian/people; `--classes` ids are VisDrone's, not COCO's):

| model @640 fp32, N=1 | MOT16-04 | MOT16-13 |
|---|---|---|
| COCO yolov8n (default) | **31.3** / 23 / 43.9, MOTP 0.176 | **13.8** / 5 / 22.4 |
| VisDrone fine-tune | 28.7 / 71 / 37.5, MOTP 0.217 | 10.1 / 5 / 18.5 |

The aerial fine-tune LOSES on eye-level street footage — worse
localization (MOTP +0.04), 2x FP, IDSW 23 -> 71 on MOT16-04. Expected
(VisDrone is small-object aerial imagery; the caveat was recorded before
training) and the model's actual customer is step-4 drone footage, where
it should be re-evaluated against COCO-yolov8n before any default choice.
COCO stays the default everywhere. INT8/VisDrone-calibration deferred to
step 4 alongside that evaluation (no MOT16 eval exists that it could win).

## S3.7 — NanoTrack v3: vehicles up sharply, persons down; v2 stays default (2026-07-03)

Two-commit path per plan. (1) The postproc grid is now derived from the
loaded head graph (`score_size_` from [1,2,S,S]) — v2 proven bit-identical
(oracle green, AUC 0.631 exact); the shared formula is parity-correct
because upstream's points are zero-centered and the kInstance/2 offset
cancels in diff_xs. (2) `export_nanotrack.py --version v3` emits the three
static v3 graphs (96-ch features, 15x15 head, opset 14 HardSwish per
D-0005a) from the SHA-pinned weights; run-verified shapes. v3 selects
purely via `--backbone-z/x/--head`; no oracle exists (D-0005a) so a synth
absolute-tracking smoke test guards the path. The upstream v3 constants
(penalty_k 0.138, lr 0.348 — configv3.yaml) are NOT cv::TrackerNano's
(0.055/0.37) and matter: v3 with v2 constants scores 0.622 (worse than
v2). New `--penalty-k/--window-influence/--size-lr` flags expose them.

| AUC (prec@20) | v2 default | v3 + upstream constants |
|---|---|---|
| Car4 | 0.719 (0.974) | **0.827** (1.000) |
| CarDark | 0.502 (0.659) | **0.713** (1.000) |
| BlurCar2 | 0.809 (1.000) | 0.778 (0.995) |
| Jogging | 0.684 (0.984) | 0.619 (0.997) |
| Girl2 | 0.437 (0.616) | 0.392 (0.557) |
| Woman | 0.636 (0.998) | 0.545 (0.940) |
| **MEAN** | 0.631 (0.872) | **0.646 (0.915)** |

Verdict: **v2 stays the default.** The +0.015 mean is below the +0.02
adoption bar and composed entirely of car-sequence gains (+0.11/+0.21)
against regressions on all four person sequences (Woman −0.09). For the
drone use-case the car result is genuinely interesting — vehicle-centric
step-4 scenarios should try the v3 model paths + constants — but a default
must not trade person tracking away. Latency: v3 ≈ 430 fps vs v2 ≈ 580 on
Car4 (96-ch backbone), both far inside budget. Published VOT2018 EAO
0.352 -> 0.449 did not transfer to mini-OTB persons; possible causes
(training-set composition vs our person-heavy slice, not postproc — the
math is verified against upstream) noted for the record.
