#pragma once

#include <memory>
#include <vector>

#include "ctrk/detector.hpp"
#include "ctrk/types.hpp"

namespace ctrk {

// Association configuration. ByteTrack paper defaults; use_byte=false
// degrades to plain SORT (the same machinery minus the low-score second
// stage) — the M1 ablation is this one boolean.
struct AssocConfig {
  bool use_byte = true;
  float track_thresh = 0.5f;            // high/low detection score split
  float match_thresh_high = 0.8f;       // stage 1: max (1 - IoU) cost
  float match_thresh_low = 0.5f;        // stage 2: min IoU vs low-score dets
  float tentative_match_thresh = 0.7f;  // stage 3: min IoU for unconfirmed tracks
  float new_track_thresh = 0.6f;        // min score to spawn a track
  int n_init = 3;                       // hits to confirm
  int max_age = 30;                     // frames a lost track coasts
  // NSA-Kalman (RESULTS.md S3.1): scale measurement noise by (1 - det score)
  // so confident detections correct the filter harder. Default on — improved
  // or tied MOTA/IDF1 on every measured config; false = classic filter.
  bool nsa = true;
  // Tentative churn at detect-interval N (RESULTS.md S2.3/S3.2): a newborn
  // track has no learned velocity, so its prediction stays put while the
  // target moves N frames per detect — the stage-3 gate then kills it and it
  // respawns with a fresh id forever (never confirms). Three remedies; all
  // are inert when no frames were coasted, so N=1 stays bit-identical.
  // Defaults per the S3.2 matrix: relax 0.3 + patience 1 improved MOTA/IDF1
  // at every measured interval on both eval sequences (MOT16-13 N=2 MOTA
  // 9.3 -> 14.7); velocity seeding HURT on real detections (noisy first
  // deltas fling the prediction, MOTP +0.03, FP +70%) and stays off.
  float tentative_relax_per_coast = 0.3f;  // subtracted from the stage-3 gate per coasted frame
  float tentative_gate_floor = 0.2f;       // stage-3 gate never relaxes below this IoU
  int tentative_patience = 1;              // extra detect-frame misses a tentative survives
  bool velocity_seed = false;              // seed newborn KF velocity at its first re-match
};

// Camera-motion compensation (RESULTS.md S3.4): estimate the inter-frame
// camera warp (grayscale sparse LK flow + RANSAC partial affine, BoT-SORT
// style) and apply it to every track's motion state on every frame —
// coast frames included. Off = static-camera behavior.
enum class GmcMethod : uint8_t { Off, SparseFlow };

struct TbdConfig {
  Yolov8Config detector;
  AssocConfig assoc;
  GmcMethod gmc = GmcMethod::Off;
  double nominal_fps = 30.0;  // converts FrameView timestamps to KF dt
  // Run the detector every Nth frame; the frames between coast on the
  // Kalman prediction alone. Association state (misses, max_age, the
  // ByteTrack stage-2 window) counts in detect-frame units, so track
  // lifecycle does not decay faster at higher N. 1 = detect every frame.
  int detect_interval = 1;
};

// Tracking-by-detection facade: frames in, identity-stable tracks out.
// Single-threaded, synchronous; one instance per thread.
class MultiTracker {
 public:
  explicit MultiTracker(const TbdConfig& config);
  ~MultiTracker();
  MultiTracker(MultiTracker&&) noexcept;
  MultiTracker& operator=(MultiTracker&&) noexcept;

  // Returns all live tracks: Confirmed (matched or briefly coasting) and
  // Lost (coasting on prediction). Display layers usually keep Confirmed.
  std::vector<Track> update(const FrameView& frame);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ctrk
