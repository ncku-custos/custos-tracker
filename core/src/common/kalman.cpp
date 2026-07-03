#include "common/kalman.hpp"

#include <algorithm>

namespace ctrk {

namespace {

cv::Matx<float, 4, 1> to_measurement(const BBox& b) {
  const float h = std::max(b.h, 1e-3f);
  return {b.cx(), b.cy(), b.w / h, b.h};
}

}  // namespace

void KalmanBox::initiate(const BBox& z) {
  const auto m = to_measurement(z);
  mean_ = cv::Matx<float, 8, 1>::zeros();
  for (int i = 0; i < 4; ++i) mean_(i) = m(i);

  const float h = m(3);
  const float std[8] = {
      2 * kStdWeightPos * h,  2 * kStdWeightPos * h,  1e-2f, 2 * kStdWeightPos * h,
      10 * kStdWeightVel * h, 10 * kStdWeightVel * h, 1e-5f, 10 * kStdWeightVel * h};
  cov_ = cv::Matx<float, 8, 8>::zeros();
  for (int i = 0; i < 8; ++i) cov_(i, i) = std[i] * std[i];

  init_ = true;
  clamp_state();
}

void KalmanBox::predict(float dt) {
  CV_Assert(init_);
  auto F = cv::Matx<float, 8, 8>::eye();
  for (int i = 0; i < 4; ++i) F(i, i + 4) = dt;

  const float h = mean_(3);
  const float std[8] = {kStdWeightPos * h, kStdWeightPos * h, 1e-2f, kStdWeightPos * h,
                        kStdWeightVel * h, kStdWeightVel * h, 1e-5f, kStdWeightVel * h};
  auto Q = cv::Matx<float, 8, 8>::zeros();
  for (int i = 0; i < 8; ++i) Q(i, i) = std[i] * std[i];

  mean_ = F * mean_;
  cov_ = F * cov_ * F.t() + Q;
  clamp_state();
}

void KalmanBox::update(const BBox& z, float r_scale) {
  CV_Assert(init_);
  const auto m = to_measurement(z);

  // H = [I4 | 0]
  auto H = cv::Matx<float, 4, 8>::zeros();
  for (int i = 0; i < 4; ++i) H(i, i) = 1.f;

  const float h = mean_(3);
  const float scale = std::max(r_scale, 1e-4f);
  const float std[4] = {kStdWeightPos * h, kStdWeightPos * h, 1e-1f, kStdWeightPos * h};
  auto R = cv::Matx<float, 4, 4>::zeros();
  for (int i = 0; i < 4; ++i) R(i, i) = std[i] * std[i] * scale;

  const cv::Matx<float, 4, 4> S = H * cov_ * H.t() + R;
  const cv::Matx<float, 8, 4> K = cov_ * H.t() * S.inv(cv::DECOMP_CHOLESKY);
  const cv::Matx<float, 4, 1> innovation = m - H * mean_;

  mean_ += K * innovation;
  cov_ = (cv::Matx<float, 8, 8>::eye() - K * H) * cov_;
  clamp_state();
}

void KalmanBox::apply_affine(const Affine23& w) {
  CV_Assert(init_);
  const float x = mean_(0), y = mean_(1);
  mean_(0) = w.a11 * x + w.a12 * y + w.tx;
  mean_(1) = w.a21 * x + w.a22 * y + w.ty;
  const float vx = mean_(4), vy = mean_(5);
  mean_(4) = w.a11 * vx + w.a12 * vy;
  mean_(5) = w.a21 * vx + w.a22 * vy;
  const float s = std::sqrt(std::abs(w.a11 * w.a22 - w.a12 * w.a21));
  mean_(3) *= s;
  mean_(7) *= s;

  // J over [x y a h | vx vy va vh]: linear part on the position and
  // velocity pairs, s on the heights, aspect untouched.
  auto J = cv::Matx<float, 8, 8>::eye();
  J(0, 0) = w.a11;
  J(0, 1) = w.a12;
  J(1, 0) = w.a21;
  J(1, 1) = w.a22;
  J(4, 4) = w.a11;
  J(4, 5) = w.a12;
  J(5, 4) = w.a21;
  J(5, 5) = w.a22;
  J(3, 3) = s;
  J(7, 7) = s;
  cov_ = J * cov_ * J.t();
  // Re-symmetrize: repeated float32 congruences slowly de-symmetrize P, and
  // once S = HPH' + R loses PSD-ness the Cholesky solve in update() returns
  // garbage gains — with NSA's small R there is nothing left to regularize
  // it (S3.4 post-mortem: fragmentation on MOT16-04 N=2 under sub-pixel
  // warps).
  cov_ = 0.5f * (cov_ + cov_.t());
  clamp_state();
}

void KalmanBox::seed_velocity(const BBox& z, float dt) {
  CV_Assert(init_);
  if (dt <= 0.f) return;
  const auto m = to_measurement(z);
  for (int i = 0; i < 4; ++i) mean_(i + 4) = (m(i) - mean_(i)) / dt;
}

void KalmanBox::clamp_state() {
  mean_(2) = std::clamp(mean_(2), 0.05f, 20.f);  // aspect
  mean_(3) = std::max(mean_(3), 1.f);            // height
}

BBox KalmanBox::box() const {
  CV_Assert(init_);
  const float h = mean_(3);
  const float w = mean_(2) * h;
  return {mean_(0) - 0.5f * w, mean_(1) - 0.5f * h, w, h};
}

float KalmanBox::covariance_trace() const {
  float t = 0.f;
  for (int i = 0; i < 8; ++i) t += cov_(i, i);
  return t;
}

}  // namespace ctrk
