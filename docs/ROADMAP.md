# Roadmap — step 4 (handoff, updated 2026-07-03 at v0.3.0)

Steps 1 (host bring-up, v0.1.0), 2 (speed, v0.2.0) and 3 (quality, v0.3.0)
are DONE — `RESULTS.md` has every number (step 3: S3.0–S3.10),
`DECISIONS.md` D-0001..D-0014 every verdict. This file carries the
remaining work.

**Process rule (user-locked):** step 4 gets its own plan-mode session before
any code. Measurement discipline stays in force: hypothesis →
`scripts/bench.sh` / eval → keep-or-revert; negative results get documented
anyway.

**Quality gates for any default-config change** (RESULTS.md step-3 header):
all tests green with the `cv::TrackerNano` oracle + YOLO golden tensor
actually RUNNING (they skip without fetched models); mini-OTB mean AUC
within 0.005 of 0.631; MOT16-04 MOTA within 0.2 pt of 31.3 / IDSW within 2
of 23 (anchors moved by S3.1).

---

## Step 3 — DONE (v0.3.0). Rollup: RESULTS.md S3.10

One-line summary: both M4 failure modes fixed (Jogging wrong-relock
0.155→0.558 via the HSV re-ID veto; Girl2 confident drift caught by the
opt-in NanoZ drift check), the detect-interval churn removed (MOT16-13 N=2
MOTA 9.3→14.7 at defaults), and GMC lands the step's largest win
(moving-camera MOTA +5.4 / IDF1 +9.7, opt-in at ~2 ms/frame). NSA-Kalman
default-on. Negative results recorded: velocity seeding, HSV drift/
appearance costs, dual-template (do not enable), v3 person regression,
VisDrone-on-street. v3 exports + constants are the vehicle-scenario
option; LightFC is de-risked and deferred (D-0014, resume recipe in S3.9).

## Step 4 — ROS2 packaging (pending; plan-mode session first)

- ROS2 **Lyrical** composable lifecycle nodes wrapping `ctrk`; the core
  stays ROS-free and is consumed via `find_package(ctrk)` (D-0001
  plain-struct public headers exist for this).
- App-level pipelining (capture/infer/draw overlap) was **deliberately
  deferred from step 2 to here** — threading belongs to the ROS2
  executor/composition design, not the apps.
- On the actual SoC:
  - Re-sweep intra-op threads and spin. D-0011: tuning verdicts do NOT
    transfer between hosts — the Ryzen wanted 8 (= physical cores, SMT
    hurts) where the i5 wanted nproc−2, and the i5's "no-spin is free"
    verdict flipped. Measure, don't port.
  - Re-run the INT8 ladder with the vendor toolchain instead of ORT QDQ
    (D-0008 lessons: per-channel DQ needs opset ≥13, keep the YOLO Detect
    head fp32). Note the SOT graphs are opset 14 (HardSwish, D-0005a).
  - **Flip `--gmc` on as the drone operating point** and re-measure — the
    D-0013 posture: the camera always moves in flight; host default stays
    off only because of the 2 ms coast-frame price on static scenes.
  - Any state-transform feature must keep the covariance re-symmetrization
    line in `KalmanBox::apply_affine` (S3.4 post-mortem).
- Drone-footage evals unlock three parked options: the VisDrone fine-tune
  (S3.6 — never default until it beats COCO on OUR footage), NanoTrack v3
  for vehicle-centric tracking (S3.7 constants in README), and the LightFC
  attempt (S3.9 recipe).
- Open upstream question: uXRCE-DDS vs MAVROS on the flight side (owned by
  custos-control, not this repo — integration is by topic contract only).

---

## Machine notes

- The step-3 host (Ryzen 7 7700 + RTX 5060, S3.0) has everything brought
  up: system deps, venv (torch cu128 + ultralytics 8.4.86), SHA-verified
  models + local exports, OTB/MOT16-04/MOT16-13 datasets, all gates green.
- `data/fetch_mot.sh` now takes `MOT_SEQS`; motchallenge.net was
  unreachable from both networks so far — MOT16 native GT is the anchor GT;
  the MOT17-04 re-annotation upgrade stays pending (re-anchor at a section
  boundary if it ever lands).
- `results/` stays gitignored; `runs/` (ultralytics) is gitignored too.
- Gate discipline: run `scripts/check.sh` UNPIPED; commits are
  Conventional + DCO (`-s`) + SSH-signed. clang-format touches C++ ONLY
  (it silently destroys Python — learned the hard way).
