# Decisions

Append-only log of decisions that are expensive to revisit. Newest last.

## D-0001 — C++20 core, C only where profiling demands it

Full plan in `docs/PLAN.md`. C++20 with system OpenCV 4.10 for I/O and image ops. Hot loops
drop to C-style only when a profile proves it matters. Public headers (`core/include/ctrk/`)
expose plain structs only — no OpenCV, no ROS types — so ROS2 wrapping and a future
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
presence. Still AGPL-3.0, so it does not even resolve D-0003. Re-evaluate once the
NPU vendor is known.

## D-0005 — SOT: NanoTrack v2/v3 with cv::TrackerNano as differential oracle

NanoTrack is the only lightweight tracker (2026-07 survey) that is simultaneously (a)
genuinely conv-only — NPU-safe for an unknown vendor, (b) tiny/static-shape/INT8-friendly,
and (c) implemented in a maintained C++ reference (`cv::TrackerNano` in system OpenCV),
giving a mechanical correctness oracle: our pipeline must match it at median IoU >= 0.9.
NanoTrack v3 ships ONNX and is a large quality jump over v2 (GOT-10k-Val AO 0.680 -> 0.719,
VOT2018 EAO 0.352 -> 0.449); M2 opens with a spike testing v3 against the oracle path —
adopt v3 if compatible, else v2. Planned upgrade target: LightFC (best conv-only tracker,
UAV123 AUC 64.8, 3.16M params) — it costs the oracle property, so it lands only after our
own harness is trusted. Every 2024-2026 tracker that beats LightFC does so by re-adding
transformer attention (AsymTrack, SMAT, HiT, LiteTrack) — deferred until the NPU is known.

## D-0005a — M2 spike verdict (2026-07-03): NanoTrack v2 ships, v3 deferred

The v3 head emits 15x15 score maps; cv::TrackerNano hardcodes the 16x16 grid
((255-127)/16+8), so v3 cannot ride the differential-oracle path. v2 ships.
v3 (needs bespoke 15x15 postproc, unverifiable against the oracle)
moves to the quality-upgrade list next to LightFC. Second finding: the
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
target before freezing deployment defaults.

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

Development moves to a Ryzen 7 7700 + RTX 5060 host (full spec in
RESULTS.md S3.0). Verdict on what transfers: **quality anchors carry over
unchanged** — mini-OTB mean AUC 0.631 and MOT16-04 MOTA 31.0 / IDSW 23 /
IDF1 43.7 reproduced digit-identical from fresh local exports (same
ultralytics 8.4.86 pin, SHA-pinned tracker weights, deterministic C++
pipeline), so the v0.2.0 gate numbers remain canonical and no re-anchor was
needed. **Host tuning verdicts do not transfer and must be re-swept per
host** (confirming D-0010's warning): detector-best is 8 intra-op threads
(= physical cores; SMT oversubscription regresses ~20%) where the i5 wanted
nproc−2 = 10, and the i5's "2t+no-spin strictly dominates" SOT verdict
flips — spin buys ~12% latency here. Portable defaults stay as-is (4-thread
detector, 2t+no-spin SOT); host-best settings live in `--threads` and
`scripts/bench.sh tbd_tuned`. Same discipline applies to the SoC:
re-sweep, don't port numbers. The GPU unblocks ROADMAP item 9 (VisDrone
fine-tune); training runs on this host with torch cu128.

## D-0012 — re-ID embedder ladder verdict: HSV for the re-lock veto, NanoZ reuse for drift, no new model (2026-07-03)

Full data in RESULTS.md S3.3. The ladder stopped before OSNet: the HSV
H-S histogram (zero models, backend-free) is the re-lock verifier — it
removes the M4 wrong-jogger re-lock at accept 0.4 and improves the Woman
rescue — and the NanoTrack template branch reused as an embedder
(`NanoTracker::embed`, zero NEW models, conv-only and already
NPU-exported) is the proactive drift detector for the nano backend
(strictly >= baseline everywhere, Girl2 +0.018 AUC at thr 0.55 / K=5).
The two failure roles want different embedders: colour separates adjacent
lookalikes but false-fires as a drift signal under partial occlusion,
while the siamese features are too identity-compressed (~0.5-0.65 cosine
for any pedestrian) to veto lookalikes but are exactly calibrated to
"does this still look like my template" drift. OSNet-lite would add a
model, an INT8 decision and an NPU-portability review (D-0005) for a
niche neither zero-cost embedder failed to cover — not taken; revisit
only if a real drone scenario shows both embedders failing (that evidence
would also justify the model cost). References freeze at init by design:
re-ID must answer "is this the ORIGINAL target", and template adaptation
is the separate dual-template experiment (RESULTS.md S3.8).

## D-0013 — GMC: sparse-flow partial affine, opt-in on host, default-on posture for the drone (2026-07-03)

Full data in RESULTS.md S3.4. Method: BoT-SORT-style sparse LK flow on
a ~480 px gray working copy + RANSAC partial affine, applied to the KF
state of every track on every frame (coast frames included) — full
BoT-SORT form with covariance congruence via `KalmanBox::apply_affine`.
Verdict: kept, `--gmc`, OFF by default on the host — the moving-camera
gain is decisive (MOT16-13 MOTA +5.4 / IDF1 +9.7 at N=1) and the
static-camera check is within gates, but it prices every coast frame at
~2 ms (vs ~4 us), which inverts the detect-every-N latency story (S2.3)
on a machine where the camera may not move. For the drone
deployment the camera ALWAYS moves: plan to flip GMC on as the SoC
default operating point and re-measure there. Engineering lesson pinned
in S3.4: covariance congruences in float32 need re-symmetrization or
NSA-small R exposes the Cholesky solve — any future state-transform
feature (e.g. homography GMC) must keep the `(P+P')/2` line.

## D-0014 — LightFC deferred with sourcing de-risked (2026-07-03)

Full data in RESULTS.md S3.9. The scheduled risk — weights sourcing
from a research repo — is retired: MIT license confirmed, checkpoint
downloaded and SHA-pinned (gdown id + sha256 in S3.9), architecture
verified export-friendly (two-graph split matching our engine seam,
conv-only backbone). Implementation deferred anyway: the remaining work
(RepVGG-style head fusion, bespoke crop/normalization/center-head
postproc, no differential oracle) is a focused half-day-plus that would
have shipped numerically unvalidated — against the keep-or-revert
discipline. The concrete resume recipe lives in S3.9;
the decision point for actually doing it is when drone-footage SOT evals
exist or when mini-OTB person robustness (currently v2+re-ID's win over
v3) becomes the binding constraint.

## D-0015 — ROS2 packaging: config-file CMake package + in-repo ament packages, core not a rosdep key (2026-07-04)

Packaging structure (RESULTS.md S4.0). The core stays a plain CMake package
consumed via `find_package(ctrk)`: a generated `ctrkConfig.cmake` carries
`find_dependency(OpenCV ...)` and the onnxruntime branch (system package
re-found; tarball builds bake absolute dirs — same-machine only), and
every target's EXPORT_NAME matches its in-tree alias. The ROS side lives
in `ros2/` as two ament packages (`ctrk_interfaces`, `ctrk_ros`);
`scripts/ros_check.sh` installs the core into `ros2/install-ctrk` and
runs colcon with cwd `ros2/`. `ctrk` is deliberately absent from
package.xml — rosdep cannot resolve it and colcon needs no ordering, the
prefix step provides it. Rejected alternatives: repo root as a colcon
"plain cmake" package (package identification stops directory recursion,
hiding `ros2/`, and colcon would rebuild core — including the gtest
FetchContent network hit — inside its own build dir); root COLCON_IGNORE
(hides ros2/ too); a symlinked overlay workspace (pure indirection).
`scripts/check.sh` is untouched and core-only checkouts never see ROS.

## D-0016 — topic/param contract: vision_msgs where it fits, one custom msg where it does not, params 1:1 with config structs (2026-07-04)

TBD publishes standard `vision_msgs/Detection2DArray` (Confirmed only by
default; center-convention bbox, `detection.id` = track id). SOT state
(IDLE/TRACKING/UNSTABLE/LOST) has no vision_msgs home, so
`ctrk_interfaces/SotStatus` carries it in ctrk's top-left pixel
convention, and `SetTarget`/`reset` services own target init — the
PLAN.md "own msgs only if insufficient" clause exercised once. Parameters
map 1:1 onto the config structs with struct defaults; the three
deliberate exceptions mirror CLI operating defaults (reid.embedder=hsv,
reacquire class_id=0, reacquire conf_thr=0.25 promoted from a hardcode).
Image input is bgr8-only, wrapped zero-copy into FrameView — **no
cv_bridge, no image_transport yet**: both are seams isolated behind
frame_view.hpp/draw_node, decided when the SoC vendor image's package set
is known. The tracker dump lives in the node (`dump_path`, the CLI's
exact snprintf) rather than a sink node: same code path as the CLI, no
QoS in the measurement loop, doubles as a flight black-box. Process-wide
ctrk log/profile sinks bridge once per process (single components .so;
sink_bridge registry) — one rclcpp logger "ctrk", per-node profile
collectors. Deployment note from the S4.1 post-mortem: `nominal_fps`
must match the camera rate or KF dt is scaled.

## D-0017 — executor/composition verdict: single-threaded container + intra-process comms; MT container rejected (2026-07-04)

The deferred pipelining question, answered by measurement
(RESULTS.md S4.2). At the camera's 30 Hz, the plain single-threaded
`component_container` with intra-process comms on gives the best e2e
latency (19.31 ms stamp→tracks p50, +2.3 ms over the CLI), the lowest
CPU, and comfortable headroom — it is the launch default.
`component_container_isolated` (one executor per node) is the documented
high-rate option: +23% sustained throughput when the input outruns
~35 Hz. `component_container_mt` is rejected outright — its worker pool
fights ORT's spinning intra-op threads and it loses on every axis.
Intra-process comms stays on for latency (0.3–1.1 ms, cheap 6 MB
loopback on desktop bandwidth) with a recorded overload caveat: under a
firehose the ST executor absorbs the drops itself; DDS-side dropping
(IPC off) sustains more. All verdicts are 16-thread-desktop verdicts —
D-0011 applies in full: re-run `scripts/ros_bench.sh` on the SoC before
believing any of it there.
