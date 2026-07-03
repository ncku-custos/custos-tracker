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
};

struct TbdConfig {
  Yolov8Config detector;
  AssocConfig assoc;
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
