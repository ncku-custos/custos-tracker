#pragma once

#include <vector>

#include "common/kalman.hpp"
#include "ctrk/tbd.hpp"
#include "ctrk/types.hpp"

namespace ctrk {

// Detection-stream association core (detector-free, unit-testable):
// Kalman prediction + staged IoU association + track lifecycle.
// Stage 1: high-score dets vs confirmed+lost tracks.
// Stage 2 (ByteTrack only): low-score dets vs still-unmatched tracks that
//          were matched last frame.
// Stage 3: remaining high-score dets vs tentative tracks.
class ByteTracker {
 public:
  explicit ByteTracker(const AssocConfig& config) : cfg_(config) {}

  // dt in frame units (1.0 = one nominal frame).
  std::vector<Track> update(const std::vector<Detection>& detections, float dt = 1.f);

 private:
  struct STrack {
    KalmanBox kf;
    int id = -1;
    TrackState state = TrackState::Tentative;
    int hits = 1;
    int age = 1;
    int time_since_update = 0;
    float score = 0.f;
    int class_id = -1;
  };

  void mark_matched(STrack& track, const Detection& det);

  AssocConfig cfg_;
  std::vector<STrack> tracks_;
  int next_id_ = 1;
};

}  // namespace ctrk
