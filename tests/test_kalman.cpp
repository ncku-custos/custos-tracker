#include <gtest/gtest.h>

#include <cmath>
#include <random>

#include "common/geometry.hpp"
#include "common/kalman.hpp"

namespace ctrk {
namespace {

TEST(Kalman, InitiateReproducesMeasurement) {
  KalmanBox kf;
  const BBox z{100, 50, 40, 80};
  kf.initiate(z);
  const BBox b = kf.box();
  EXPECT_NEAR(b.x, z.x, 1e-3f);
  EXPECT_NEAR(b.y, z.y, 1e-3f);
  EXPECT_NEAR(b.w, z.w, 1e-3f);
  EXPECT_NEAR(b.h, z.h, 1e-3f);
}

TEST(Kalman, ConvergesOnNoisyConstantVelocityTrack) {
  std::mt19937 rng(7);
  std::normal_distribution<float> noise(0.f, 1.5f);

  const float vx = 4.f, vy = -2.f;
  auto gt = [&](int t) { return BBox{200 + vx * t, 300 + vy * t, 60, 120}; };

  KalmanBox kf;
  kf.initiate(gt(0));
  for (int t = 1; t <= 60; ++t) {
    kf.predict(1.f);
    BBox z = gt(t);
    z.x += noise(rng);
    z.y += noise(rng);
    kf.update(z);
  }
  // After convergence a pure predict step should land close to ground truth.
  kf.predict(1.f);
  const BBox pred = kf.box();
  const BBox expect = gt(61);
  EXPECT_GT(iou(pred, expect), 0.85f);
  EXPECT_NEAR(pred.cx(), expect.cx(), 4.f);
  EXPECT_NEAR(pred.cy(), expect.cy(), 4.f);
}

TEST(Kalman, CovarianceGrowsWhileCoasting) {
  KalmanBox kf;
  kf.initiate({100, 100, 50, 100});
  kf.predict(1.f);
  kf.update({104, 100, 50, 100});
  const float before = kf.covariance_trace();
  for (int i = 0; i < 10; ++i) kf.predict(1.f);
  EXPECT_GT(kf.covariance_trace(), before);
}

TEST(Kalman, DegenerateBoxesDoNotProduceNan) {
  KalmanBox kf;
  kf.initiate({10, 10, 0.f, 0.f});  // zero-size measurement
  for (int i = 0; i < 20; ++i) {
    kf.predict(1.f);
    kf.update({10, 10, 0.001f, 0.0001f});
  }
  const BBox b = kf.box();
  EXPECT_TRUE(std::isfinite(b.x) && std::isfinite(b.y) && std::isfinite(b.w) && std::isfinite(b.h));
  EXPECT_GE(b.h, 1.f);                  // height clamp
  EXPECT_GE(b.w / b.h, 0.05f - 1e-4f);  // aspect clamp
  EXPECT_LE(b.w / b.h, 20.f + 1e-4f);
}

TEST(Kalman, ExtremeAspectIsClamped) {
  KalmanBox kf;
  kf.initiate({0, 0, 1000, 2});  // aspect 500 -> clamped to 20
  const BBox b = kf.box();
  EXPECT_LE(b.w / b.h, 20.f + 1e-4f);
}

}  // namespace
}  // namespace ctrk
