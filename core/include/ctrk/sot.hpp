#pragma once

#include <memory>
#include <string>

#include "ctrk/types.hpp"

namespace ctrk {

struct SotConfig {
  // NanoTrack v2 as three static graphs (tools/export/export_nanotrack.py).
  std::string backbone_z_path;  // template branch, run once at init
  std::string backbone_x_path;  // search branch, per frame
  std::string head_path;        // correlation head, per frame
  int intra_op_threads = 2;

  // Post-processing constants — cv::TrackerNano parity (do not tune before
  // the oracle differential test passes; tuning is step 3).
  float penalty_k = 0.055f;
  float window_influence = 0.455f;
  float size_lr = 0.37f;
  float context_amount = 0.5f;
};

// Single-object tracker facade. init() with a target box, then update() per
// frame. Single-threaded, synchronous; one instance per thread.
class SotTracker {
 public:
  explicit SotTracker(const SotConfig& config);
  ~SotTracker();
  SotTracker(SotTracker&&) noexcept;
  SotTracker& operator=(SotTracker&&) noexcept;

  void init(const FrameView& frame, const BBox& target);
  // score is the raw (pre-window) classifier peak — the lost-detection
  // signal for M4. state is always Tracking until the M4 state machine.
  SotResult update(const FrameView& frame);

 private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

}  // namespace ctrk
