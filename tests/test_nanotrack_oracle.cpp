#include <gtest/gtest.h>

#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <filesystem>
#include <string>
#include <vector>

#include "common/geometry.hpp"
#include "common/mat_view.hpp"
#include "ctrk/sot.hpp"
#include "support/synth.hpp"

namespace ctrk {
namespace {

// Differential oracle: our 3-engine pipeline vs cv::TrackerNano running the
// SAME v2 weights on the SAME frames. This does not require good tracking —
// it requires IDENTICAL behaviour (median per-frame IoU >= 0.9; residual gap
// is OpenCV-dnn vs ONNX Runtime numerics).
std::string cache(const std::string& f) {
  return std::string(CTRK_SOURCE_DIR) + "/models/cache/" + f;
}

bool models_present() {
  for (const char* f :
       {"nanotrack_backbone_z.onnx", "nanotrack_backbone_x.onnx", "nanotrack_head.onnx",
        "nanotrackv2_nanotrack_backbone_sim.onnx", "nanotrackv2_nanotrack_head_sim.onnx"})
    if (!std::filesystem::exists(cache(f))) return false;
  return true;
}

TEST(NanotrackOracle, MatchesCvTrackerNano) {
  if (!models_present())
    GTEST_SKIP() << "run models/get_models.sh && tools/export/export_nanotrack.py";

  synth::Options opt;
  opt.frames = 80;
  synth::Sequence seq(opt, {{.box0 = {80, 140, 70, 110}, .vx = 3.5f, .vy = 1.2f,
                             .color = {30, 60, 200}}});

  SotConfig cfg;
  cfg.backbone_z_path = cache("nanotrack_backbone_z.onnx");
  cfg.backbone_x_path = cache("nanotrack_backbone_x.onnx");
  cfg.head_path = cache("nanotrack_head.onnx");
  SotTracker ours(cfg);

  cv::TrackerNano::Params params;
  params.backbone = cache("nanotrackv2_nanotrack_backbone_sim.onnx");
  params.neckhead = cache("nanotrackv2_nanotrack_head_sim.onnx");
  const auto oracle = cv::TrackerNano::create(params);

  const BBox b0 = seq.gt(0)[0];
  const cv::Mat f0 = seq.frame(0);
  ours.init(as_frame_view(f0, 0), b0);
  oracle->init(f0, cv::Rect(static_cast<int>(b0.x), static_cast<int>(b0.y),
                            static_cast<int>(b0.w), static_cast<int>(b0.h)));

  std::vector<float> agreement;
  for (int t = 1; t < seq.frames(); ++t) {
    const cv::Mat f = seq.frame(t);
    const SotResult r = ours.update(as_frame_view(f, t));
    cv::Rect ref;
    ASSERT_TRUE(oracle->update(f, ref));
    const BBox ref_box{static_cast<float>(ref.x), static_cast<float>(ref.y),
                       static_cast<float>(ref.width), static_cast<float>(ref.height)};
    agreement.push_back(iou(r.box, ref_box));
  }

  std::sort(agreement.begin(), agreement.end());
  const float median = agreement[agreement.size() / 2];
  EXPECT_GE(median, 0.9f) << "pipelines diverged (median IoU vs cv::TrackerNano)";
  // Weakest frame should still broadly agree — catches single-frame blowups.
  EXPECT_GE(agreement.front(), 0.5f);
}

TEST(NanotrackOracle, TracksSynthTargetAbsolutely) {
  if (!models_present())
    GTEST_SKIP() << "run models/get_models.sh && tools/export/export_nanotrack.py";

  synth::Options opt;
  opt.frames = 60;
  synth::Sequence seq(opt, {{.box0 = {100, 150, 60, 90}, .vx = 3.f, .vy = 0.5f,
                             .color = {0, 80, 220}}});

  SotConfig cfg;
  cfg.backbone_z_path = cache("nanotrack_backbone_z.onnx");
  cfg.backbone_x_path = cache("nanotrack_backbone_x.onnx");
  cfg.head_path = cache("nanotrack_head.onnx");
  SotTracker tracker(cfg);
  tracker.init(as_frame_view(seq.frame(0), 0), seq.gt(0)[0]);

  float mean_iou = 0.f;
  for (int t = 1; t < seq.frames(); ++t) {
    const cv::Mat f = seq.frame(t);
    const SotResult r = tracker.update(as_frame_view(f, t));
    mean_iou += iou(r.box, seq.gt(t)[0]);
  }
  mean_iou /= static_cast<float>(seq.frames() - 1);
  EXPECT_GT(mean_iou, 0.5f) << "tracker lost the synthetic target";
}

}  // namespace
}  // namespace ctrk
