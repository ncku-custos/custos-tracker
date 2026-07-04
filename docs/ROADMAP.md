# Roadmap — SoC deployment (handoff, updated 2026-07-04 at v0.4.0)

Steps 1 (host bring-up, v0.1.0), 2 (speed, v0.2.0), 3 (quality, v0.3.0)
and 4 (ROS2 packaging, v0.4.0) are DONE — `RESULTS.md` has every number
(step 4: S4.0–S4.3), `DECISIONS.md` D-0001..D-0017 every verdict. This
file carries what remains: putting it on the vehicle.

**Process rule (user-locked):** each step gets its own plan-mode session
before any code. Measurement discipline stays in force: hypothesis →
eval → keep-or-revert; negative results get documented anyway.

**Quality gates for any default-config change:** all tests green with the
`cv::TrackerNano` oracle + YOLO golden tensor actually RUNNING, plus the
ROS-side suite (`scripts/ros_check.sh`, 22 cases, MOSSE e2e is the
model-free anchor); mini-OTB mean AUC within 0.005 of 0.631; MOT16-04
MOTA within 0.2 pt of 31.3 / IDSW within 2 of 23; **and the S4.1 parity
gate** (`scripts/ros_parity.sh`) — the ROS node path must stay
digit-identical to the CLI.

---

## Step 4 — DONE (v0.4.0). Rollup: RESULTS.md S4.3

One-line summary: the core is consumed via `find_package(ctrk)` (D-0015),
both trackers run as composable lifecycle components with params 1:1 to
the config structs (D-0016), the node path is proven digit-identical to
the CLI incl. the GMC drone profile (S4.1), and the deferred
capture/infer/draw pipelining question is answered with measurements
(S4.2, D-0017). Two parity landmines are on the record: imread-vs-
VideoCapture JPEG decode divergence, and FFmpeg's 25 fps default for
image sequences (nominal_fps must match the camera rate).

## Step 5 — SoC deployment (pending; plan-mode session first)

- **Bring-up on the vendor SoC image**: build core + `ros2/` there
  (ros_check.sh path); decide cv_bridge/image_transport now that the
  vendor package set is known (D-0016 left both out deliberately —
  compressed debug streams over the drone link live behind draw_node).
- **Camera driver integration**: real driver replaces frames_player;
  stamps come from the driver — verify `nominal_fps` matches the true
  rate (S4.1 deployment note) and the encoding is bgr8 (or add the
  conversion at the frame_view seam).
- **Re-sweep intra-op threads and spin on the SoC** (D-0011: tuning
  verdicts do NOT transfer — the Ryzen wanted 8 = physical cores where
  the i5 wanted nproc−2, and the spin verdict flipped between them).
  `ros2/ctrk_ros/params/*_drone.yaml` deliberately omit these knobs.
- **Vendor INT8 ladder** instead of ORT QDQ (D-0008 lessons: per-channel
  DQ needs opset ≥13, keep the YOLO Detect head fp32; SOT graphs are
  opset 14 HardSwish, D-0005a). Swap via `detector.model_path`.
- **Flip GMC on for flight** — it already IS on in tbd_drone.yaml
  (D-0013); re-measure on drone footage and re-run the S4.2 matrix on
  SoC silicon before trusting any latency number from the Ryzen.
- **Executor choice on the SoC**: D-0017 picked the container flavor on
  a 16-thread desktop; a 4–8 core SoC shifts the ST-vs-isolated
  trade — re-run `scripts/ros_bench.sh`, keep-or-revert the launch
  default.
- Drone-footage evals unlock the parked options: VisDrone fine-tune
  (S3.6 — never default until it beats COCO on OUR footage), NanoTrack
  v3 for vehicle-centric tracking (S3.7 constants in README), LightFC
  (S3.9 recipe, D-0014).
- Open upstream question: uXRCE-DDS vs MAVROS on the flight side (owned
  by custos-control; integration is by topic contract — the table in
  README/D-0016 is the interface).

---

## Machine notes

- Reference host (Ryzen 7 7700 + RTX 5060, S3.0): everything brought up —
  system deps, ROS2 Lyrical at /opt/ros/lyrical (+ ros-lyrical-vision-msgs),
  venv (torch cu128 + ultralytics 8.4.86), SHA-verified models + local
  exports, OTB/MOT16-04/MOT16-13 datasets, all gates green incl. parity.
- Gate discipline: `scripts/check.sh` UNPIPED (core), `scripts/ros_check.sh`
  (colcon side), `scripts/ros_parity.sh` per milestone. Commits are
  Conventional + DCO (`-s`) + SSH-signed. clang-format touches C++ ONLY
  (it silently destroys Python — learned the hard way). Launch/params
  Python and YAML are not auto-formatted.
- Long unattended ROS runs: `ros2 launch` backgrounded without a tty can
  ignore a lone SIGINT — ros_bench.sh escalates INT→TERM→KILL and sweeps
  `component_container` zombies (a stale container squats on node names
  and hijacks lifecycle CLI calls).
- `results/` and `runs/` stay gitignored; colcon dirs live under `ros2/`
  (`build*/`, `install*/`, `log/` ignored).
- motchallenge.net was unreachable from every network so far — MOT16
  native GT via `gt_eval.txt` is the anchor GT (data/fetch_mot.sh,
  MOT_SEQS env for more sequences).
