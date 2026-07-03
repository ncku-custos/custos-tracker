#include <gtest/gtest.h>

#include <opencv2/core.hpp>

#include "common/geometry.hpp"
#include "common/mat_view.hpp"

namespace ctrk {
namespace {

TEST(Iou, IdenticalBoxes) { EXPECT_FLOAT_EQ(iou({0, 0, 10, 10}, {0, 0, 10, 10}), 1.f); }

TEST(Iou, DisjointBoxes) { EXPECT_FLOAT_EQ(iou({0, 0, 10, 10}, {20, 20, 10, 10}), 0.f); }

TEST(Iou, TouchingEdgesIsZero) { EXPECT_FLOAT_EQ(iou({0, 0, 10, 10}, {10, 0, 10, 10}), 0.f); }

TEST(Iou, HalfOverlap) {
  // Boxes [0,10]x[0,10] and [5,15]x[0,10]: inter 50, union 150.
  EXPECT_NEAR(iou({0, 0, 10, 10}, {5, 0, 10, 10}), 50.f / 150.f, 1e-6f);
}

TEST(Iou, DegenerateBoxIsZero) {
  EXPECT_FLOAT_EQ(iou({0, 0, 0, 10}, {0, 0, 10, 10}), 0.f);
  EXPECT_FLOAT_EQ(iou({0, 0, -5, 10}, {0, 0, 10, 10}), 0.f);
}

TEST(Nms, SuppressesOverlapsKeepsBestFirst) {
  std::vector<BBox> boxes = {{0, 0, 10, 10}, {1, 1, 10, 10}, {30, 30, 10, 10}};
  std::vector<float> scores = {0.8f, 0.9f, 0.5f};
  auto keep = nms(boxes, scores, 0.5f);
  ASSERT_EQ(keep.size(), 2u);
  EXPECT_EQ(keep[0], 1);  // highest score survives
  EXPECT_EQ(keep[1], 2);  // disjoint box survives
}

TEST(Nms, ThresholdBoundary) {
  // IoU of these two is ~0.33; below thr 0.5 both survive, above 0.3 one is cut.
  std::vector<BBox> boxes = {{0, 0, 10, 10}, {5, 0, 10, 10}};
  std::vector<float> scores = {0.9f, 0.8f};
  EXPECT_EQ(nms(boxes, scores, 0.5f).size(), 2u);
  EXPECT_EQ(nms(boxes, scores, 0.3f).size(), 1u);
}

TEST(Letterbox, WideSourceScalesByWidth) {
  auto m = letterbox_map(1280, 720, 640, 640);
  EXPECT_FLOAT_EQ(m.scale, 0.5f);
  EXPECT_FLOAT_EQ(m.pad_x, 0.f);
  EXPECT_FLOAT_EQ(m.pad_y, (640.f - 360.f) / 2.f);
}

TEST(Letterbox, RoundTripIsIdentity) {
  const auto m = letterbox_map(1920, 1080, 640, 640);
  const BBox b{123.4f, 56.7f, 210.f, 89.f};
  const BBox r = from_letterbox(to_letterbox(b, m), m);
  EXPECT_NEAR(r.x, b.x, 1e-3f);
  EXPECT_NEAR(r.y, b.y, 1e-3f);
  EXPECT_NEAR(r.w, b.w, 1e-3f);
  EXPECT_NEAR(r.h, b.h, 1e-3f);
}

TEST(MatView, RoundTripIsZeroCopy) {
  cv::Mat img(480, 640, CV_8UC3, cv::Scalar(1, 2, 3));
  FrameView f = as_frame_view(img, 42);
  EXPECT_EQ(f.width, 640);
  EXPECT_EQ(f.height, 480);
  EXPECT_EQ(f.t_ns, 42);
  cv::Mat back = as_mat(f);
  EXPECT_EQ(back.data, img.data);  // same pixels, no copy
  EXPECT_EQ(back.step, img.step);
}

TEST(MatView, RespectsStrideOfSubmatrix) {
  cv::Mat img(480, 640, CV_8UC3, cv::Scalar(0, 0, 0));
  cv::Mat roi = img(cv::Rect(10, 10, 320, 240));  // non-tight stride
  FrameView f = as_frame_view(roi, 0);
  EXPECT_EQ(f.stride_bytes, static_cast<int>(img.step));
  cv::Mat back = as_mat(f);
  EXPECT_EQ(back.ptr(1), roi.ptr(1));
}

}  // namespace
}  // namespace ctrk
