# Roadmap — steps 3 & 4 (handoff, written 2026-07-03 at v0.2.0)

Steps 1 (host bring-up, v0.1.0) and 2 (speed, v0.2.0) are DONE — see `RESULTS.md` for
every number and `DECISIONS.md` D-0001..D-0010 for every verdict. This file carries the
remaining work and the pre-planning recon so work can resume on another machine.

**Process rule (user-locked):** steps 3 and 4 each get their own plan-mode session before
any code. Measurement discipline from step 2 stays in force: hypothesis → `scripts/bench.sh`
/ eval → keep-or-revert; negative results get documented anyway.

**Quality gates for any default-config change** (RESULTS.md step-2 header): all tests green
incl. the `cv::TrackerNano` oracle + YOLO golden tensor; mini-OTB mean AUC within 0.005 of
0.631; MOT16-04 MOTA within 0.2 pt of 31.0 / IDSW within 2 of 23.

---

## Step 3 — quality improvements (pending; plan-mode session first)

### The two SOT failure modes driving the top items (RESULTS.md M4)

1. **Jogging / wrong re-lock**: two adjacent same-class targets; the geometric re-acquisition
   gates (class + growing radius + size ratio) cannot distinguish them → re-locks the wrong
   jogger (AUC 0.558 → 0.155 with `--reacquire`). Needs appearance verification.
2. **Girl2 / confident drift**: NanoTrack drifts to a distractor **without the score ever
   collapsing** (AUC 0.437, re-acquire never fires). The score-threshold loss detector is
   structurally blind to this — verification must run *proactively on the tracked box*,
   not only in the Lost state.

### Work items, ranked (insertion points verified at v0.2.0 / commit 0daa02a)

1. **SOT re-ID appearance verification** — three insertion points in
   `core/src/sot/sot_tracker.cpp`: inside `pick_candidate` (~L46-64, replace the
   nearest-distance tie-break), before the re-lock commit (`backend_init`, ~L127, reject
   non-matching candidates and stay Lost), and in the Tracking branch (~L109, proactive
   drift check for the Girl2 mode). Embedder ladder to measure (cheapest first):
   (a) HSV color histogram — zero new models; (b) **reuse `backbone_z` as embedder** —
   run the existing 127×127 z-graph on candidate crops, cosine vs stored `zf_`
   (zero new models, conv-only, already NPU-exported); (c) OSNet-lite ONNX INT8 — new
   model decision under D-0005 portability rules, only if (a)/(b) fail.
2. **GMC (BoT-SORT-style camera-motion compensation)** — estimate the warp in
   `core/src/tbd/multi_tracker.cpp` (`Impl` needs a prev-frame/feature cache; ~L16, L25-46);
   pass into `ByteTracker::update`/`coast` and apply to the `predicted[]` boxes
   (`byte_tracker.cpp` ~L75-81) — full BoT-SORT form also warps KF state via a new
   `KalmanBox::apply_affine`. Also feed the SOT search-window position.
   **Evaluation trap: MOT16-04 is a static camera — GMC shows ~zero there.** The
   already-downloaded `data/downloads/MOT16.zip` contains moving-camera train sequences
   with GT (MOT16-05/10/11/13); extract one as a second eval scenario first.
3. **NSA-Kalman** — confidence-adaptive measurement noise. Two-line change:
   `core/src/common/kalman.cpp` ~L56-59 builds R height-scaled only; add a `conf` param
   to `update()` and scale R (NSA: `(1-conf)` form); pass `det.score` at the single call
   site `byte_tracker.cpp` ~L52.
4. **Tentative-gate churn fix** — documented in RESULTS.md S2.3: at detect-interval N a
   track inside `n_init` has no learned velocity and the 0.7 tentative IoU gate
   (`tbd.hpp` `tentative_match_thresh`, applied `byte_tracker.cpp` ~L114-120; unmatched
   tentative dies on a single miss ~L126) sees N× the motion → birth churn for targets
   faster than ~⅓ box-width per detect interval. Candidates: interval-aware gate,
   tentative coasting, velocity seeding from first two detections.
5. **Appearance cost in TBD association** — fuse cosine distance into the IoU cost at
   `byte_tracker.cpp` ~L27 (`match_by_iou`); needs an embedding field on `STrack`
   (`byte_tracker.hpp` ~L31-40) and `Detection` (`types.hpp` ~L50-54) + EMA update in
   `mark_matched`. Reuse whatever embedder wins item 1. Note IDSW is already low (23) on
   static-camera MOT16-04 — measure on the moving-camera sequence.
6. **NanoTrack v3 with bespoke 15×15 postproc** — v3 weights are already fetched +
   SHA-pinned in `models/manifest.json`; the v2→v3 blocker (DECISIONS D-0005a) is that
   `cv::TrackerNano` hardcodes the 16×16 grid, so v3 loses the differential oracle and
   needs its own grid/window/penalty math (`nanotrack.cpp` postproc is fully documented;
   `kScore=16` constant). Needs a new static-export path in
   `tools/export/export_nanotrack.py` (currently v2-only). Config-gated; v2 stays default
   until mini-OTB proves v3 (published jump: VOT2018 EAO 0.352→0.449).
7. **Dual-template** — `zf_` is frozen at init (`nanotrack.hpp` ~L44); add a second
   periodically-refreshed template buffer + head fusion (two head passes or feature blend).
   Breaks the oracle → config-gated experiment; keep-or-revert on mini-OTB.
8. **LightFC swap** — best conv-only tracker (UAV123 AUC 64.8, 3.16M params, D-0005), but
   weights sourcing/export from a research repo is a schedule risk and it also loses the
   oracle. Attempt only after 1–7 land, if at all.
9. **VisDrone fine-tune of YOLOv8n** — addresses the dominant FN floor (COCO-pretrained
   nano on small pedestrians, RESULTS.md M1). **Blocked on the step-2 host: no CUDA GPU
   (torch CPU-only, 12 cores, ~9 GB free RAM) → days of training.** If the new machine
   has a GPU this unblocks. Recipe: venv has `ultralytics 8.4.86`; `VisDrone.yaml` ships
   with the package (auto-downloads ~2 GB); train → re-export via
   `tools/export/export_yolo.py` (detector C++ is shape-driven, 10-class model adapts
   automatically — but `keep_classes` ids then mean VisDrone classes, not COCO). AGPL
   D-0003 still applies to fine-tuned weights.
10. **After the quality work**: re-run the step-2 tradeoff ladder (quality changes shift
    the operating points); revisit the resolution rung at close range. **YOLO26 re-eval
    stays gated on the NPU vendor decision** (D-0004), as does any attention-based tracker.

### Open scope decisions — ask the user at the next plan session (unanswered as of handoff)

1. **VisDrone**: defer-to-GPU-box (recommended on the old host — moot if the new machine
   has CUDA) vs overnight CPU mini-run vs skip.
2. **SOT model upgrades**: v3-only (recommended) vs v3+LightFC vs robustness-only.
3. **TBD scope**: full set (NSA + GMC + churn fix + appearance cost; recommended) vs
   motion-only vs minimal (NSA + churn fix).

---

## Step 4 — ROS2 packaging (pending; plan-mode session first)

- ROS2 **Lyrical** composable lifecycle nodes wrapping `ctrk`; the core stays ROS-free and
  is consumed via `find_package(ctrk)` (D-0001 plain-struct public headers exist for this).
- App-level pipelining (capture/infer/draw overlap) was **deliberately deferred from step 2
  to here** — threading belongs to the ROS2 executor/composition design, not the apps.
- On the actual SoC: re-sweep intra-op threads and spin (D-0010 defaults are host-measured:
  detector 10 = nproc−2 on i5-1335U, nano 2+no-spin — both may differ on ARM); re-run the
  INT8 ladder with the vendor toolchain instead of ORT QDQ (D-0008 lessons: per-channel DQ
  needs opset ≥13, keep the YOLO Detect head fp32).
- Open upstream question: uXRCE-DDS vs MAVROS on the flight side (owned by custos-control,
  not this repo — integration is by topic contract only).

---

## New-machine bring-up notes

- **System deps** (Ubuntu 26.04): `libopencv-dev`, `libonnxruntime-dev`, CMake ≥ 3.28;
  Python tooling: `python3 -m venv tools/.venv && tools/.venv/bin/pip install -r tools/requirements.txt`.
- **Models**: `models/get_models.sh` (SHA-pinned fetches). YOLO ONNX files are **local
  exports, never fetched** — regenerate with `tools/export/export_yolo.py` (opset 12,
  static, no in-graph NMS) and `tools/export/quantize.py` for INT8 variants.
- **Data**: `data/fetch_otb.sh`, `data/fetch_mot.sh`. From the step-1/2 network,
  `motchallenge.net` was connection-refused and the Hanyang OTB server 404'd — both
  scripts have working mirrors baked in (PaddleDetection bcebos for MOT16; Wayback with
  forced HTTP/1.1 + resume for OTB). Retry motchallenge.net from the new network: if
  reachable, `fetch_mot.sh` prefers the MOT17-04 GT re-annotation (better GT than MOT16).
- **`results/` is gitignored by design** — the tables in `docs/RESULTS.md` are the record;
  re-baseline with `scripts/bench.sh` (5 reps first, to establish the new host's noise
  floor) before trusting any before/after comparison.
- **Host-specific numbers do not transfer**: thread counts, spin verdicts, and all
  latencies in RESULTS.md are i5-1335U (2P+8E, AVX-VNNI, powersave governor) measurements.
- **Gate discipline**: run `scripts/check.sh` UNPIPED (a zsh pipe masks its exit status —
  this bit twice in step 2); commits are Conventional Commits + DCO (`-s`) + GPG-signed.
