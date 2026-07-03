#pragma once

#include <opencv2/core.hpp>

#include "common/geometry.hpp"
#include "ctrk/types.hpp"

namespace ctrk {

// Constant-velocity Kalman filter over [cx, cy, a, h] with a = w/h aspect,
// ByteTrack/DeepSORT convention: measurement and process noise scale with box
// height h (the best distance proxy from an aerial camera). dt is in frame
// units (1.0 = one frame at nominal rate); callers derive it from timestamps.
class KalmanBox {
 public:
  void initiate(const BBox& z);
  void predict(float dt = 1.f);
  // r_scale multiplies the measurement-noise variance R. 1 = classic filter;
  // NSA-Kalman (GIAOTracker) passes (1 - detection score) so confident
  // detections correct harder. Clamped to a small floor so conf=1 cannot
  // make R singular.
  void update(const BBox& z, float r_scale = 1.f);
  // Overwrite the velocity state from the measurement delta over dt frame
  // units. Call BEFORE update(): the current mean must still be the prior
  // (for a newborn track, its birth position). Seeds a track's velocity from
  // its first two detections instead of blending from the zero-velocity
  // prior — the detect-interval churn remedy (RESULTS.md S3.2).
  void seed_velocity(const BBox& z, float dt);
  // Warp the state into camera-motion-compensated coordinates (BoT-SORT
  // GMC, RESULTS.md S3.4): position takes the full affine, velocity the
  // linear part, height the isotropic scale (aspect is similarity-
  // invariant), and the covariance position/velocity blocks rotate along.
  void apply_affine(const Affine23& w);

  bool initialized() const { return init_; }
  BBox box() const;
  float covariance_trace() const;  // uncertainty proxy, grows while coasting

 private:
  static constexpr float kStdWeightPos = 1.f / 20.f;
  static constexpr float kStdWeightVel = 1.f / 160.f;

  void clamp_state();

  cv::Matx<float, 8, 1> mean_;
  cv::Matx<float, 8, 8> cov_;
  bool init_ = false;
};

}  // namespace ctrk
