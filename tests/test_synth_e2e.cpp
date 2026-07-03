#include <gtest/gtest.h>

#include <opencv2/videoio.hpp>

#include <string>

#include "common/geometry.hpp"
#include "common/mat_view.hpp"
#include "support/synth.hpp"

namespace ctrk {
namespace {

synth::Sequence make_two_target_scene() {
  synth::Options opt;
  opt.frames = 60;
  return synth::Sequence(opt, {{.box0 = {50, 100, 60, 90}, .vx = 4.f, .vy = 1.f,
                                .color = {0, 0, 220}},
                               {.box0 = {400, 300, 80, 60}, .vx = -3.f, .vy = -2.f,
                                .color = {220, 120, 0}}});
}

TEST(Synth, GroundTruthMovesLinearly) {
  const auto seq = make_two_target_scene();
  const auto g0 = seq.gt(0);
  const auto g10 = seq.gt(10);
  EXPECT_FLOAT_EQ(g10[0].x, g0[0].x + 40.f);
  EXPECT_FLOAT_EQ(g10[0].y, g0[0].y + 10.f);
  EXPECT_FLOAT_EQ(g10[1].x, g0[1].x - 30.f);
}

TEST(Synth, RenderingIsDeterministic) {
  const auto a = make_two_target_scene();
  const auto b = make_two_target_scene();
  EXPECT_EQ(cv::norm(a.frame(17), b.frame(17), cv::NORM_INF), 0.0);
}

TEST(Synth, OccluderCoversTarget) {
  synth::Options opt;
  opt.frames = 30;
  opt.occluder = true;
  opt.occluder_rect = {100, 100, 100, 100};
  opt.occluder_from = 10;
  synth::Sequence seq(opt, {{.box0 = {120, 120, 40, 40}, .vx = 0.f, .vy = 0.f,
                             .color = {0, 0, 255}}});
  // Sample inside the fill but off the diagonal line and inner block:
  // (row 155, col 125) is plain outer fill. Red before occlusion, grey after.
  const cv::Vec3b before = seq.frame(5).at<cv::Vec3b>(155, 125);
  const cv::Vec3b after = seq.frame(15).at<cv::Vec3b>(155, 125);
  EXPECT_EQ(before, cv::Vec3b(0, 0, 255));
  EXPECT_EQ(after, cv::Vec3b(90, 90, 90));
}

// Headless CI end-to-end: synth -> FrameView -> mp4 -> reopen -> count.
TEST(SynthE2E, WriteAndReadBackVideo) {
  const auto seq = make_two_target_scene();
  const std::string path = testing::TempDir() + "/synth_e2e.mp4";

  cv::VideoWriter writer(path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), 30.0, seq.size());
  ASSERT_TRUE(writer.isOpened());
  for (int t = 0; t < seq.frames(); ++t) {
    const cv::Mat f = seq.frame(t);
    const FrameView view = as_frame_view(f, t);  // exercise the app-side wrap
    writer.write(as_mat(view));
  }
  writer.release();

  cv::VideoCapture cap(path);
  ASSERT_TRUE(cap.isOpened());
  int n = 0;
  for (cv::Mat f; cap.read(f) && !f.empty(); ++n) {
    ASSERT_EQ(f.size(), seq.size());
  }
  EXPECT_EQ(n, seq.frames());
}

}  // namespace
}  // namespace ctrk
