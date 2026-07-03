#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "tbd/gmc.hpp"

namespace ctrk {
namespace {

// Textured frame with enough corner structure for the flow to latch onto.
cv::Mat speckle(int w, int h, uint32_t seed) {
  cv::Mat img(h, w, CV_8UC3);
  cv::RNG rng(seed);
  rng.fill(img, cv::RNG::UNIFORM, 0, 255);
  cv::GaussianBlur(img, img, {5, 5}, 0);
  return img;
}

cv::Mat shifted(const cv::Mat& src, float dx, float dy) {
  const cv::Matx23f m(1, 0, dx, 0, 1, dy);
  cv::Mat out;
  cv::warpAffine(src, out, m, src.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
  return out;
}

TEST(Gmc, FirstFrameIsIdentity) {
  GmcEstimator gmc;
  EXPECT_TRUE(gmc.estimate(speckle(640, 480, 1)).identity());
}

TEST(Gmc, RecoversPureTranslation) {
  GmcEstimator gmc;
  const cv::Mat f0 = speckle(640, 480, 2);
  gmc.estimate(f0);
  const Affine23 w = gmc.estimate(shifted(f0, 7.f, -4.f));
  EXPECT_NEAR(w.tx, 7.f, 0.5f);
  EXPECT_NEAR(w.ty, -4.f, 0.5f);
  EXPECT_NEAR(w.a11, 1.f, 0.02f);
  EXPECT_NEAR(w.a22, 1.f, 0.02f);
}

TEST(Gmc, RecoversTranslationThroughTheDownscalePath) {
  // 1920-wide input exercises the ~480 px working-copy rescale of the
  // translation back to full resolution.
  GmcEstimator gmc;
  const cv::Mat f0 = speckle(1920, 1080, 3);
  gmc.estimate(f0);
  const Affine23 w = gmc.estimate(shifted(f0, 24.f, 12.f));
  EXPECT_NEAR(w.tx, 24.f, 2.f);
  EXPECT_NEAR(w.ty, 12.f, 2.f);
}

TEST(Gmc, StaticSceneStaysNearIdentity) {
  GmcEstimator gmc;
  const cv::Mat f0 = speckle(640, 480, 4);
  gmc.estimate(f0);
  const Affine23 w = gmc.estimate(f0.clone());
  EXPECT_NEAR(w.tx, 0.f, 0.1f);
  EXPECT_NEAR(w.ty, 0.f, 0.1f);
}

}  // namespace
}  // namespace ctrk
