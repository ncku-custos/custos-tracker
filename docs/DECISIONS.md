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
