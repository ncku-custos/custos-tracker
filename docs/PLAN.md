# Custos-Tracking — Visual tracking system (SOT + Tracking-by-Detection)

## Context

Drone-mounted visual tracking on a companion-computer SoC (Ubuntu 26.04 + NPU, vendor undecided). Two subsystems: **SOT is the main goal** (operator gives an initial bbox, system follows that target), TBD (detector + association → stable track IDs) is the second system and also serves SOT re-acquisition. Four-step roadmap; **this plan executes step 1 only** (build both systems, confirm working on host x86_64). Steps 2 (speed) and 3 (improvements) get their own plan-mode sessions later; step 4 (ROS2 Lyrical packaging) gets an architecture outline now so step-1 code anticipates it. Work lives in `/space/drone/custos-tracking` (currently empty; semi-separate from the custos workspace — do not couple to it).

## Locked decisions (from user Q&A)

- **Language**: C++20. C-style only in hot loops *if profiling proves it matters* (it rarely will). OpenCV allowed.
- **SOT**: deep siamese (NanoTrack v2, conv-only, ~2 MB, NPU-friendly) as main line + hand-written **MOSSE** correlation filter as CPU fallback/sanity baseline, both behind one `SotTracker` facade.
- **TBD**: YOLOv8n ONNX (COCO) + SORT baseline, upgraded to ByteTrack two-stage association (a config flag — gives the SORT-vs-ByteTrack ablation for free). No re-ID in step 1 (step 3).
- **Inference seam**: `libinfer` (`IEngine` over named tensors, static shapes) with ONNX Runtime backend on host; NPU backend slots in later. **Per-engine backend selection** from day 1 (see risk #1).
- **NPU portability rules**: conv-only graphs, static shapes, export opset **12**, no in-graph NMS, INT8-friendly.
- **Validation bar**: demo overlay videos + light metrics (OTB-style success/AUC on ~6 sequences, MOTA/IDSW on one MOT16 sequence, per-stage FPS/latency). Not a full benchmark harness.
- Python = tooling only (export/eval scripts); never in the runtime path. Core libs have **zero ROS2 dependency**.

## Host facts (verified)

Ubuntu 26.04, gcc 15.2, CMake 4.3.2, 12 cores. `libopencv-dev 4.10.0` **installed** — includes `cv::TrackerNano` in system headers → differential-test oracle for our NanoTrack pipeline. `libonnxruntime-dev 1.23.2` available in apt (runtime libs already installed) → M0 does `sudo apt install libonnxruntime-dev`; `find_package(onnxruntime CONFIG)` works. Python 3.13 + venv for tooling.

## Architecture

```
apps (CLI)          track_sot / track_tbd  — video io, overlay, --dump-results, --bench-json
core/ctrk_sot       NanoTrack 3-graph pipeline + postproc │ MOSSE fallback │ lost-detection facade
core/ctrk_tbd       IDetector → yolov8 decode+NMS │ Kalman + Munkres + ByteTrack/SORT lifecycle
core/ctrk_infer     IEngine, TensorDesc, TensorView │ OrtEngine (per-engine backend choice)
core/ctrk_common    FrameView, geometry (IoU/NMS/letterbox), kalman, munkres, subwindow, timer, log
```

### Public API (ROS2-ready by construction)

- Plain structs in public headers, **no cv::Mat**: `FrameView{data,w,h,stride,fmt,t_ns}`, `BBox`, `SotResult{box,score,state}`, `Track{id,box,score,class_id,state,age,hits}`.
- `SotTracker::init(frame, bbox)` / `update(frame) → SotResult`; `MultiTracker::update(frame) → vector<Track>`.
- Config = aggregate structs with defaults (maps 1:1 to ROS params). Time passed in via `t_ns` (KF dt from timestamps → replay-deterministic, sim-time-compatible). Injectable log callback. No globals; objects single-threaded/synchronous, one instance per thread is safe.

### Models (fetched by script, never committed; `models/manifest.json` pins URL+SHA256)

- **NanoTrack v2/v3** ONNX from HonglinChu/SiamTrackers — v2 is the exact model `cv::TrackerNano` consumes (the oracle). **v3 also ships ONNX** and is a large quality jump (GOT-10k-Val AO 0.680→0.719, VOT2018 EAO 0.352→0.449, DTB70 success +4.4pt) with the same conv-only architecture family; whether `cv::TrackerNano` postproc accepts v3's wider tensors is untested. **M2 therefore starts with a half-day v3 spike**: if v3 works against the oracle path, v3 becomes primary; else ship v2 (zero loss). Export script re-emits **three static graphs**: `backbone_z` [1,3,127,127] (run once at init), `backbone_x` [1,3,255,255] (per frame), `head` (zf,xf → cls [1,2,16,16] + reg [1,4,16,16]) via `onnxsim --overwrite-input-shape`. Step-3 quality upgrade target: **LightFC** (best conv-only tracker as of mid-2026 — beats MixFormerV2-S, UAV123 64.8 AUC, 3.16M params; costs the oracle property, so it lands only after our own harness is trusted). LightTrack demoted — stale and outclassed.
- **YOLOv8n**: `yolo export format=onnx opset=12 imgsz=640 dynamic=False simplify=True nms=False` → [1,3,640,640] → [1,84,8400]; decode+NMS in C++. **Ultralytics is AGPL** — record in DECISIONS.md; Apache fallback = YOLOX-Nano/Tiny (best NPU-zoo evidence) or NanoDet-Plus behind `IDetector` (swap = one decode function).

### Component-selection rationale (researched July 2026, sources in DECISIONS.md)

Selection criterion is **NPU portability + verifiability first, quality second** — models are swappable behind `IEngine`/`IDetector`, so step 1 optimizes for a provably-correct bring-up; quality upgrades are step 3.
- **YOLO26n evaluated and rejected for now**: better vendor-claimed mAP/CPU speed than v8n, but its flagship NMS-free one-to-one head does **not** survive real NPU export as of mid-2026 (RKNN INT8 segfault/empty-detection issues #23340/#23753 unresolved through mid-2026; independent Hailo port needed a manual CPU/NPU graph split; zero TI/NXP model-zoo presence), and it is still AGPL. YOLOv8n retains the widest vendor-zoo conversion coverage (RKNN, Hailo, TI, NXP). Revisit at step 3 once the NPU vendor is known.
- **DETR-class (RT-DETR/D-FINE/DEIM/RF-DETR)**: Apache-licensed and strong, but transformer decoders have zero official support in any of the four vendor NPU zoos — ruled out while the vendor is unknown.
- **SOT**: every 2024–2026 tracker that beats LightFC does it by adding transformer attention back in (AsymTrack, SMAT, HiT, LiteTrack). NanoTrack is the only tracker that is simultaneously conv-only, tiny, and has a maintained C++ reference implementation (`cv::TrackerNano`) to diff against. AsymTrack-T (single attention layer, UAV123 66.5) is the noted runner-up if the eventual NPU proves transformer-tolerant.

### Locked algorithm constants

- **NanoTrack postproc** (constants from OpenCV `tracker_nano.cpp`, our oracle): softmax cls → penalty `exp(−0.055·(sc·rc−1))` → Hanning window mix 0.455 → argmax → size smoothing lr `= penalty·score·0.37` → clamp. Raw peak score = confidence; lost if < 0.30 for 5 frames (tune on clips). Crop geometry (context 0.5, template 127 / search 255, mean-pad) shared with MOSSE via one `subwindow` primitive built in M0.
- **Kalman**: 8-dim `[cx,cy,a,h] + velocities` (ByteTrack convention), `std_weight_pos=1/20`, `std_weight_vel=1/160`; clamp `a∈[0.05,20]`, `h≥1` after update.
- **ByteTrack**: high ≥ 0.5, low [0.1,0.5); stage-1 IoU ≥ 0.2, stage-2 IoU ≥ 0.5; new tracks from unmatched high dets ≥ 0.6. Lifecycle: Tentative→Confirmed at 3 hits, Lost coasts 30 frames, then Removed. IoU-only cost (genuine ByteTrack); `gate_fn` hook reserved for step-3 re-ID.
- **MOSSE**: grayscale log-preproc, Hann, Gaussian target σ=2, lr=0.125, PSR confidence (lost < ~8), 3-scale {0.985, 1, 1.015}, FFT via `cv::dft` (no kissfft dep).

## Repo layout

```
custos-tracking/
├── CMakeLists.txt            # C++20, options CTRK_BUILD_TESTS, CTRK_ORT_ROOT (tarball fallback, default off)
├── cmake/ort.cmake
├── core/include/ctrk/        # types.hpp sot.hpp tbd.hpp infer.hpp version.hpp  (ROS-free, OpenCV-free)
├── core/src/{common,infer,sot,tbd}/            # per architecture table above
├── apps/                     # app_common + track_sot.cpp + track_tbd.cpp (cv::CommandLineParser)
├── models/                   # manifest.json (committed), get_models.sh, *.onnx (gitignored)
├── tools/                    # requirements.txt; export/{export_yolo,export_nanotrack}.py; eval/{sot_eval,mot_eval}.py
├── data/                     # fetch_otb.sh fetch_mot.sh fetch_clips.sh (payloads gitignored)
├── tests/                    # gtest 1.17 via FetchContent; support/synth.hpp (synthetic-video generator)
└── docs/                     # PLAN.md DECISIONS.md RESULTS.md
```

Install as CMake package (`find_package(ctrk)`) — the artifact step-4 ROS packages consume.

## Git workflow & commit points

**State**: repo already `git init`ed on `main` (zero commits); identity + `commit.gpgsign=true` configured. No LFS — models/datasets fetched-by-script + SHA256 manifest, `.gitignore` enforces (weights, dataset payloads, output videos, build dirs never enter history).

**Rules**: Conventional Commits (`type(scope): subject`) + DCO sign-off (`-s`) + GPG signing (already configured). One commit = smallest unit that builds clean with its tests green — `scripts/check.sh` (build + ctest + format check) gates every commit. Linear history on `main` during solo bring-up. Tags `m0`…`m4` at milestone acceptance, `v0.1.0` at step-1 completion.

**Commit points** (≈22 commits across step 1):

| # | Commit | Milestone |
|---|---|---|
| 1 | `chore: repository scaffold` — .gitignore, .clang-format, LICENSE, README, docs/ (DECISIONS.md initial entries, PLAN.md, RESULTS.md stub) | M0 |
| 2 | `build: cmake superbuild, system OpenCV/ORT wiring, gtest, check.sh` | M0 |
| 3 | `feat(common): FrameView + geometry (IoU, NMS, letterbox) + tests` | M0 |
| 4 | `feat(common): ByteTrack-convention Kalman filter + tests` | M0 |
| 5 | `feat(common): Munkres assignment + brute-force differential tests` | M0 |
| 6 | `feat(common): mean-pad subwindow crop, timers, logging + tests` | M0 |
| 7 | `feat(apps): video I/O + overlay harness skeleton` | M0 |
| 8 | `test: synthetic-video generator + headless e2e` | M0 |
| 9 | `chore: model manifest + data fetch scripts` → **tag `m0`** | M0 |
| 10 | `feat(infer): IEngine/TensorDesc seam + ONNX Runtime backend` | M1 |
| 11 | `feat(tbd): YOLOv8n decode + NMS + golden-tensor test` (+ export_yolo.py) | M1 |
| 12 | `feat(tbd): SORT — KF track lifecycle + IoU association + tests` | M1 |
| 13 | `feat(tbd): ByteTrack two-stage association flag + lifecycle tests` | M1 |
| 14 | `feat(apps): track_tbd MOT dump + mot_eval.py; docs: M1 results` → **tag `m1`** | M1 |
| 15 | `tools(export): nanotrack 3-graph export; DECISIONS.md: v3-vs-v2 spike verdict` | M2 |
| 16 | `feat(sot): siamese pipeline (crop/infer/postproc) + cv::TrackerNano differential test` | M2 |
| 17 | `feat(apps): track_sot + sot_eval.py; docs: M2 results` → **tag `m2`** | M2 |
| 18 | `feat(sot): MOSSE fallback behind facade + synthetic tests; docs: M3 results` → **tag `m3`** | M3 |
| 19 | `feat(sot): Tracking→Unstable→Lost state machine` | M4 |
| 20 | `feat(sot): detector-assisted re-acquisition (class/position/size gating)` | M4 |
| 21 | `feat(apps): --bench-json per-stage p50/p95 latency` | M4 |
| 22 | `docs: RESULTS.md full metrics + latency baseline` → **tag `m4`, `v0.1.0`** | M4 |

Deviation policy: extra fix-up commits are fine (`fix(scope):`), but never fold unrelated components into one commit, and never commit with red tests except `wip:` on a branch (not on `main`).

## Milestones & acceptance (~5–6 weeks)

| M | Scope | Accept |
|---|---|---|
| **M0** (2–3 d) | apt deps; CMake superbuild; FrameView/geometry/subwindow/timer; video-io+overlay app skeleton; gtest; synth generator; fetch scripts + SHA256 manifest; pick OTB mirror | clean `-Wall -Wextra -Wpedantic` build; synth e2e test green headless; annotated mp4 out with per-stage ms |
| **M1** (1–1.5 wk) | OrtEngine; YOLOv8n export/decode/NMS (golden-tensor test); KF+Munkres+lifecycle; SORT→ByteTrack flag | unit tests green; MOT16-04 (+MOT17 GT): MOTA > 25%, ByteTrack cuts IDSW ≥ 20% vs SORT; ≥ 10 FPS e2e |
| **M2** (1–1.5 wk) | v3-vs-v2 spike (half-day: does v3 ONNX pass the oracle path?) → 3-graph NanoTrack export (opset ≤ 12 checked by script); crop+postproc; confidence out | **differential vs `cv::TrackerNano` median IoU ≥ 0.9** (on whichever version wins the spike); mini-OTB mean AUC ≥ 0.55; ≥ 80 FPS CPU |
| **M3** (3–4 d) | MOSSE behind same facade (config switch) | synthetic-blob test (peak within 1 px/100 frames); ≥ 200 FPS; IoU ≥ 0.5 on Car4/CarDark; PSR drops on occlusion |
| **M4** (1 wk) | SOT state machine Tracking→Unstable→Lost; detector-assisted re-acquisition (class + KF-position + size gate → re-init template); `--bench-json` p50/p95; RESULTS.md | scripted occlusion clip: loss detected + correct re-lock; all metric/latency tables committed; tag `v0.1.0` |

**Test strategy**: unit-test pure logic (IoU/NMS/letterbox round-trip, KF convergence + coasting covariance + degenerate-box clamps, Munkres vs brute force n≤7, ByteTrack lifecycle on scripted det streams incl. crossing boxes, MOSSE on synth blob, YOLO decode on a recorded golden tensor). Differential-test NanoTrack vs `cv::TrackerNano`. E2E on synth video in CI (zero downloads); real-clip metrics run manually per milestone. C++ apps **dump, never score** (MOT-txt / OTB-txt); Python `motmetrics` + ~60-line numpy script do the arithmetic.

**Data**: OTB mini-set (Car4, CarDark, BlurCar2, Human3, Girl2, DragonBaby, ~150 MB; mirror-listed + SHA256) ; MOT16.zip one-time 1.9 GB → keep only MOT16-04 (+MOT16-05 optional) with MOT17 GT ≈ 400 MB on disk. VisDrone opt-in secondary. Plus any user-provided drone clips in `data/clips/`.

## Key risks

1. **Depthwise cross-correlation head may not run/quantize on the NPU** → 3-graph split + per-engine backend choice means backbones go NPU, head stays CPU (~1% of FLOPs). This is why libinfer's per-engine backend selection is non-negotiable now.
2. **Ultralytics AGPL** → structural swap path via `IDetector`; DECISIONS.md entry day 1.
3. **Model link rot** (personal GitHub repos) → SHA256 manifest + private mirror after first fetch.
4. **COCO→aerial domain gap** → expect degradation on nadir views; note in RESULTS.md; VisDrone fine-tune is step 3.
5. **Drone ego-motion breaks search-region locality** → measure on a drone clip in step 1; GMC is step 3.
6. **KF blowup on degenerate boxes** → clamps + unit test.
7. **Headless CI** → `cv::imshow` only behind `--display`.

## Roadmap stubs (future plan-mode sessions)

- **Step 2 — speed**: profile-first from M4 bench JSON; ORT session/thread tuning; pre-allocated tensors + letterbox-into-input-buffer; capture∥infer∥post pipeline threading around the still-synchronous core; detect-every-N with KF coasting; host INT8 static-quant dry run as NPU rehearsal; SIMD only where the profiler proves it.
- **Step 3 — improvements**: BoT-SORT-style camera-motion compensation (critical on drone) feeding KF + SOT search placement; cheap re-ID (color-hist → OSNet-lite INT8) for the `gate_fn` hook + re-acquisition verification; NSA-Kalman; dual-template SOT update (frozen init + slow EMA); VisDrone detector fine-tune; SOT model upgrade to **LightFC** (or AsymTrack-T if the by-then-known NPU tolerates its single attention layer) / drone-domain retrain; detector re-evaluation (YOLO26n if its NPU-export story has matured, DETR-class if vendor supports it).
- **Step 4 — ROS2 Lyrical**: composable **lifecycle** nodes `ctrk_sot_node` / `ctrk_tbd_node`; prefer standard `vision_msgs` (own msgs only if insufficient); `on_configure` builds core objects from params (1:1 with config structs); image callback wraps ROS buffer in `FrameView` zero-copy; `SetTarget` service for SOT init; sensor-data QoS; core consumed via `find_package(ctrk)` inside ament_cmake — zero core changes if the API rules above hold.

## Verification (end of step 1)

1. `scripts/check.sh` → clean build + all unit tests + format check.
2. `./track_tbd --input data/mot/MOT16-04 --dump-results out.txt` → `tools/eval/mot_eval.py` prints MOTA/IDSW meeting M1 bars (both SORT and ByteTrack modes).
3. `./track_sot --input <OTB seq> --bbox <gt0>` → `tools/eval/sot_eval.py` prints AUC ≥ 0.55; differential test vs `cv::TrackerNano` ≥ 0.9 median IoU.
4. Occlusion clip demo: SOT reports Lost, re-acquires via detector, overlay video shows re-lock.
5. `--bench-json` latency tables (p50/p95 per stage) committed to `docs/RESULTS.md` as the step-2 baseline.
