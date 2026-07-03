#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>
#include <string>
#include <vector>

#include "common/subwindow.hpp"
#include "common/timer.hpp"
#include "ctrk/log.hpp"

namespace ctrk {
namespace {

TEST(Subwindow, InteriorCropNeedsNoPadding) {
  cv::Mat img(200, 200, CV_8UC3);
  cv::randu(img, 0, 255);
  const cv::Mat out = crop_subwindow(img, 100.f, 100.f, 64, 64, cv::Scalar(0, 0, 0));
  ASSERT_EQ(out.size(), cv::Size(64, 64));
  // original_sz == model_sz: pixels are a straight copy of the source ROI.
  const cv::Mat roi = img(cv::Rect(68, 68, 64, 64));  // floor(100 - 32.5 + 0.5) = 68
  EXPECT_EQ(cv::norm(out, roi, cv::NORM_INF), 0.0);
}

TEST(Subwindow, EdgeCropIsPaddedWithGivenValue) {
  cv::Mat img(100, 100, CV_8UC3, cv::Scalar(50, 60, 70));
  const cv::Scalar pad(1, 2, 3);
  const cv::Mat out = crop_subwindow(img, 0.f, 0.f, 80, 80, pad);
  ASSERT_EQ(out.size(), cv::Size(80, 80));
  // Top-left corner lies outside the image -> pad value.
  const auto tl = out.at<cv::Vec3b>(0, 0);
  EXPECT_EQ(tl, cv::Vec3b(1, 2, 3));
  // Bottom-right of the crop is inside the image -> image value.
  const auto br = out.at<cv::Vec3b>(79, 79);
  EXPECT_EQ(br, cv::Vec3b(50, 60, 70));
}

TEST(Subwindow, ResizesToModelSize) {
  cv::Mat img(300, 300, CV_8UC3, cv::Scalar(9, 9, 9));
  const cv::Mat out = crop_subwindow(img, 150.f, 150.f, 255, 127, cv::Scalar(0, 0, 0));
  EXPECT_EQ(out.size(), cv::Size(127, 127));
}

TEST(StageStats, PercentilesNearestRank) {
  StageStats s;
  for (int i = 1; i <= 100; ++i) s.add_ns(i * 1000000LL);  // 1..100 ms
  EXPECT_DOUBLE_EQ(s.p50_ms(), 50.0);
  EXPECT_DOUBLE_EQ(s.p95_ms(), 95.0);
  EXPECT_DOUBLE_EQ(s.mean_ms(), 50.5);
}

TEST(StageTimer, ScopeRecordsSample) {
  StageTimer timer;
  { auto scope = timer.scope("stage_a"); }
  ASSERT_EQ(timer.stats().count("stage_a"), 1u);
  EXPECT_EQ(timer.stats().at("stage_a").count(), 1u);
}

TEST(Log, CustomSinkReceivesMessages) {
  std::vector<std::pair<LogLevel, std::string>> captured;
  set_log_sink([&](LogLevel lvl, std::string_view msg) { captured.emplace_back(lvl, std::string(msg)); });
  log(LogLevel::Warn, "hello");
  set_log_sink(nullptr);  // restore default
  ASSERT_EQ(captured.size(), 1u);
  EXPECT_EQ(captured[0].first, LogLevel::Warn);
  EXPECT_EQ(captured[0].second, "hello");
}

}  // namespace
}  // namespace ctrk
