# Decisions

Append-only log of decisions that are expensive to revisit. Newest last.

## D-0001 — C++20 core, C only where profiling demands it

Full plan in `docs/PLAN.md`. C++20 with system OpenCV 4.10 for I/O and image ops. Hot loops
drop to C-style only when a profile proves it matters. Public headers (`core/include/ctrk/`)
expose plain structs only — no OpenCV, no ROS types — so step-4 ROS2 wrapping and a future
NPU-vendor build stay mechanical.

## D-0002 — Models and datasets are fetched, never committed

No git-LFS. `models/manifest.json` pins URL + SHA256 for every weight file;
`models/get_models.sh` fetches and verifies. Dataset payloads live under gitignored `data/`
subdirs with fetch scripts. Rationale: weights are rehostable artifacts any machine can
re-fetch and verify; LFS adds clone friction and server cost for no integrity gain over a
pinned SHA256.

## D-0003 — Detector: YOLOv8n (AGPL flag), Apache fallback ready

YOLOv8n chosen for the widest NPU-toolchain conversion coverage verified in vendor model
zoos as of 2026-07 (Rockchip rknn_model_zoo, Hailo Model Zoo, TI edgeai-modelzoo, NXP eIQ).
**Ultralytics is AGPL-3.0** — weights exported through their pipeline carry license risk for
a proprietary product. Mitigation is structural: the detector sits behind `IDetector`; the
Apache-2.0 swap is YOLOX-Nano/Tiny (best vendor-zoo evidence among permissive options) or
NanoDet-Plus. Swapping is one decode function. Resolve before any commercial ship.

## D-0004 — YOLO26n evaluated and rejected (for now)

Evaluated 2026-07. Better vendor-claimed numbers than v8n (mAP 40.9 vs 37.3, ~2x CPU speed),
but its NMS-free one-to-one head does not survive real NPU export today: RKNN INT8
segfault/empty-detection issues (ultralytics#23340, #23753) unresolved through mid-2026; an
independent Hailo port required a manual CPU/NPU graph split; zero TI/NXP model-zoo
presence. Still AGPL-3.0, so it does not even resolve D-0003. Re-evaluate at step 3 once the
NPU vendor is known.

## D-0005 — SOT: NanoTrack v2/v3 with cv::TrackerNano as differential oracle

NanoTrack is the only lightweight tracker (2026-07 survey) that is simultaneously (a)
genuinely conv-only — NPU-safe for an unknown vendor, (b) tiny/static-shape/INT8-friendly,
and (c) implemented in a maintained C++ reference (`cv::TrackerNano` in system OpenCV),
giving a mechanical correctness oracle: our pipeline must match it at median IoU >= 0.9.
NanoTrack v3 ships ONNX and is a large quality jump over v2 (GOT-10k-Val AO 0.680 -> 0.719,
VOT2018 EAO 0.352 -> 0.449); M2 opens with a spike testing v3 against the oracle path —
adopt v3 if compatible, else v2. Step-3 upgrade target: LightFC (best conv-only tracker,
UAV123 AUC 64.8, 3.16M params) — it costs the oracle property, so it lands only after our
own harness is trusted. Every 2024-2026 tracker that beats LightFC does so by re-adding
transformer attention (AsymTrack, SMAT, HiT, LiteTrack) — deferred until the NPU is known.

## D-0005a — M2 spike verdict (2026-07-03): NanoTrack v2 ships, v3 deferred

The v3 head emits 15x15 score maps; cv::TrackerNano hardcodes the 16x16 grid
((255-127)/16+8), so v3 cannot ride the differential-oracle path. v2 ships for
step 1. v3 (needs bespoke 15x15 postproc, unverifiable against the oracle)
moves to the step-3 upgrade list next to LightFC. Second finding: the
published NanoTrack graphs use HardSwish, an op that ENTERED ONNX at opset 14
— opset 12 is unreachable for them without decomposing HardSwish
(x * HardSigmoid(x)); most NPU toolchains support HardSwish natively, so we
carry opset 14 for the SOT graphs and revisit only if the chosen vendor
objects. The YOLOv8n export remains opset 12.

## D-0006 — Repository license: Apache-2.0

Consistent with the wider project's licensing posture. The NanoTrack/SiamTrackers upstream
license must be confirmed from its LICENSE file before any public release of derived weights
(tracked in M2).

## D-0007 — System packages over vendored deps

Host distro == deployment distro (Ubuntu 26.04), so system `libopencv-dev` 4.10 and
`libonnxruntime-dev` 1.23.2 are the reproducibility story; the distro maintains ABI for the
LTS lifetime. Escape hatch for other hosts: `-DCTRK_ORT_ROOT=<unpacked official tarball>`.
Export opset is pinned at 12 — NPU toolchains lag at opset ~12-17, so portability is
enforced at export time, not runtime.

## D-0009 — Host build-flag policy: portable default, opt-in native tuning (2026-07-03)

The shipped build stays plain `-O3` (baseline x86-64 today, whatever the SoC's
compiler defaults to later): the deployment target is an unknown ARM SoC, so
ISA-specific host flags would optimize the wrong machine and mask the numbers
that transfer. `-DCTRK_MARCH_NATIVE=ON` and `-DCTRK_LTO=ON` exist as opt-in
CMake options for benchmarking how far compiler-side vectorization can carry
the hand-written loops (blob conversion, decode) — measured against the same
scenarios in RESULTS.md before/after. Note ORT's MLAS kernels dispatch
AVX2/VNNI at runtime regardless of these flags, so they can only move the
~4% of TBD time outside `det.infer` (and the SOT crop/blob stages).

## D-0010 — ORT engine tuning verdicts on the reference host (2026-07-03)

Full sweep in RESULTS.md S2.2. Adopted: 10 intra-op threads (nproc−2) as the
documented host-tuned detector setting (default stays portable; `--threads`
exposes it); nano default becomes 2 threads + `allow_spinning=false`, which
measured fastest AND cheapest (three small sessions' pools busy-waiting
between sequential per-frame runs only fight each other on the hybrid
2P+8E part). Rejected: the oneDNN execution provider (~2.6x slower than the
default CPU EP for YOLOv8n despite shipping in the Ubuntu package — not
worth a config surface) and ORT auto thread selection (worse than explicit
nproc−2). Spinning-off is the recommended power posture for the SoC: on TBD
it trades ~3% latency for ~40% process-CPU; re-measure both on the real
target before freezing step-4 defaults.

## D-0008 — INT8 rehearsal verdict: quantize the detector, not the tiny tracker (2026-07-03)

Static QDQ dry run on the host (full data in RESULTS.md S2.5). YOLOv8n with
backbone+neck INT8 and the Detect head fp32 is metric-parity (MOTA 31.3 vs
31.0) at 1.35x speed — the INT8 path is real and the ladder's best rung;
combined with detect-every-2 it is the recommended drone operating point
(10.1 ms effective, MOTA 29.9). Two portable lessons for the eventual NPU
bring-up: (1) the mixed-magnitude Detect-head output must never share one
activation scale (whole-graph quantization yields ZERO detections — split
or keep the head fp: exactly the per-engine backend freedom libinfer was
built for); (2) per-channel DequantizeLinear requires opset >= 13, so INT8
variants carry opset 13 while fp32 opset 12 remains the canonical NPU
input. NanoTrack backbones: INT8 rejected on host — accuracy collapse on
some sequences (Car4 AUC 0.719 -> 0.287, depthwise sensitivity) and no
speed win (QDQ overhead beats VNNI on ~1 MB graphs). fp32 stays canonical
everywhere; INT8 files are parallel *_int8.onnx variants, regenerated by
tools/export/quantize.py, never fetched.

## D-0011 — new reference host; anchors carry over, tuning verdicts do not (2026-07-03)

Step-3 development moves to a Ryzen 7 7700 + RTX 5060 host (full spec in
RESULTS.md S3.0). Verdict on what transfers: **quality anchors carry over
unchanged** — mini-OTB mean AUC 0.631 and MOT16-04 MOTA 31.0 / IDSW 23 /
IDF1 43.7 reproduced digit-identical from fresh local exports (same
ultralytics 8.4.86 pin, SHA-pinned tracker weights, deterministic C++
pipeline), so the step-2 gate numbers remain canonical and no re-anchor was
needed. **Host tuning verdicts do not transfer and must be re-swept per
host** (confirming D-0010's warning): detector-best is 8 intra-op threads
(= physical cores; SMT oversubscription regresses ~20%) where the i5 wanted
nproc−2 = 10, and the i5's "2t+no-spin strictly dominates" SOT verdict
flips — spin buys ~12% latency here. Portable defaults stay as-is (4-thread
detector, 2t+no-spin SOT); host-best settings live in `--threads` and
`scripts/bench.sh tbd_tuned`. Same discipline applies to the step-4 SoC:
re-sweep, don't port numbers. The GPU unblocks ROADMAP item 9 (VisDrone
fine-tune); training runs on this host with torch cu128.
