#include <gtest/gtest.h>

#include <cmath>

#include "common/geometry.hpp"
#include "common/mat_view.hpp"
#include "ctrk/sot.hpp"
#include "support/synth.hpp"

namespace ctrk {
namespace {

SotConfig mosse_cfg() {
  SotConfig cfg;
  cfg.backend = SotBackend::Mosse;
  return cfg;
}

TEST(Mosse, FollowsTranslatingTargetWithNoise) {
  synth::Options opt;
  opt.frames = 100;
  opt.noise_sigma = 4.f;
  synth::Sequence seq(opt, {{.box0 = {100, 150, 60, 60}, .vx = 2.f, .vy = 1.f,
                             .color = {0, 60, 220}}});

  SotTracker tracker(mosse_cfg());
  tracker.init(as_frame_view(seq.frame(0), 0), seq.gt(0)[0]);

  float err_sum = 0.f;
  float iou_sum = 0.f;
  for (int t = 1; t < seq.frames(); ++t) {
    const cv::Mat f = seq.frame(t);
    const SotResult r = tracker.update(as_frame_view(f, t));
    const BBox gt = seq.gt(t)[0];
    err_sum += std::hypot(r.box.cx() - gt.cx(), r.box.cy() - gt.cy());
    iou_sum += iou(r.box, gt);
  }
  const float n = static_cast<float>(seq.frames() - 1);
  EXPECT_LT(err_sum / n, 3.f) << "mean center error too high";
  EXPECT_GT(iou_sum / n, 0.5f);
}

TEST(Mosse, PsrHealthyOnLockCollapsesUnderOcclusion) {
  synth::Options opt;
  opt.frames = 60;
  opt.occluder = true;
  opt.occluder_rect = {60, 110, 160, 160};  // fully covers the target
  opt.occluder_from = 30;
  synth::Sequence seq(opt, {{.box0 = {100, 150, 60, 60}, .vx = 0.f, .vy = 0.f,
                             .color = {0, 60, 220}}});

  SotTracker tracker(mosse_cfg());
  tracker.init(as_frame_view(seq.frame(0), 0), seq.gt(0)[0]);

  float psr_before = 0.f, psr_after = 0.f;
  for (int t = 1; t < seq.frames(); ++t) {
    const cv::Mat f = seq.frame(t);
    const SotResult r = tracker.update(as_frame_view(f, t));
    if (t >= 10 && t < 30) psr_before += r.score / 20.f;
    if (t >= 40) psr_after += r.score / 20.f;
  }
  EXPECT_GT(psr_before, 10.f) << "healthy lock should have strong PSR";
  EXPECT_LT(psr_after, 8.f) << "occluded target should collapse the PSR";
  EXPECT_LT(psr_after, psr_before * 0.6f);
}

TEST(Mosse, SurvivesTargetNearImageEdge) {
  synth::Options opt;
  opt.frames = 40;
  synth::Sequence seq(opt, {{.box0 = {5, 5, 50, 50}, .vx = 1.f, .vy = 1.f,
                             .color = {200, 40, 40}}});
  SotTracker tracker(mosse_cfg());
  tracker.init(as_frame_view(seq.frame(0), 0), seq.gt(0)[0]);
  for (int t = 1; t < seq.frames(); ++t) {
    const cv::Mat f = seq.frame(t);
    const SotResult r = tracker.update(as_frame_view(f, t));
    ASSERT_TRUE(std::isfinite(r.box.x) && std::isfinite(r.box.y));
  }
}

}  // namespace
}  // namespace ctrk
