#pragma once

#include <opencv2/core.hpp>

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
  void update(const BBox& z);

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
