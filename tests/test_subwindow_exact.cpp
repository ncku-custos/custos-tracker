#include <gtest/gtest.h>

#include <opencv2/imgproc.hpp>

#include "common/subwindow.hpp"
#include "sot/nanotrack.hpp"

// The subwindow rework (pad only the crop window, compute the pad color
// lazily — RESULTS.md S2.1) must be pixel-identical to the original pad-the-whole-frame
// implementations — the cv::TrackerNano differential oracle depends on the
// nano variant, MOSSE on the shared one. These references are verbatim copies
// of the pre-rework code.

namespace ctrk {
namespace {

cv::Mat ref_crop_subwindow(const cv::Mat& img, float cx, float cy, int original_sz, int model_sz,
                           const cv::Scalar& pad_value) {
  const float c = static_cast<float>(original_sz + 1) / 2.f;
  int xmin = static_cast<int>(std::floor(cx - c + 0.5f));
  int ymin = static_cast<int>(std::floor(cy - c + 0.5f));
  const int xmax = xmin + original_sz - 1;
  const int ymax = ymin + original_sz - 1;
  const int left_pad = std::max(0, -xmin);
  const int top_pad = std::max(0, -ymin);
  const int right_pad = std::max(0, xmax - img.cols + 1);
  const int bottom_pad = std::max(0, ymax - img.rows + 1);

  cv::Mat patch;
  if (left_pad || top_pad || right_pad || bottom_pad) {
    cv::Mat padded;
    cv::copyMakeBorder(img, padded, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT,
                       pad_value);
    patch = padded(cv::Rect(xmin + left_pad, ymin + top_pad, original_sz, original_sz));
  } else {
    patch = img(cv::Rect(xmin, ymin, original_sz, original_sz));
  }
  cv::Mat out;
  if (original_sz == model_sz) {
    patch.copyTo(out);
  } else {
    cv::resize(patch, out, cv::Size(model_sz, model_sz));
  }
  return out;
}

cv::Mat ref_nano_subwindow(const cv::Mat& img, float pos_x, float pos_y, int original_sz,
                           int model_sz) {
  const cv::Scalar avg = cv::mean(img);
  const int c = (original_sz + 1) / 2;
  int xmin = static_cast<int>(pos_x) - c;
  int ymin = static_cast<int>(pos_y) - c;
  const int xmax = xmin + original_sz - 1;
  const int ymax = ymin + original_sz - 1;
  const int left_pad = std::max(0, -xmin);
  const int top_pad = std::max(0, -ymin);
  const int right_pad = std::max(0, xmax - img.cols + 1);
  const int bottom_pad = std::max(0, ymax - img.rows + 1);
  xmin += left_pad;
  ymin += top_pad;

  cv::Mat crop;
  if (left_pad || top_pad || right_pad || bottom_pad) {
    cv::Mat padded;
    cv::copyMakeBorder(img, padded, top_pad, bottom_pad, left_pad, right_pad, cv::BORDER_CONSTANT,
                       avg);
    crop = padded(cv::Rect(xmin, ymin, original_sz, original_sz));
  } else {
    crop = img(cv::Rect(xmin, ymin, original_sz, original_sz));
  }
  cv::Mat out;
  cv::resize(crop, out, {model_sz, model_sz});
  return out;
}

cv::Mat random_image(cv::RNG& rng, int rows, int cols) {
  cv::Mat img(rows, cols, CV_8UC3);
  rng.fill(img, cv::RNG::UNIFORM, 0, 256);
  return img;
}

TEST(SubwindowExact, CropSubwindowMatchesReferenceEverywhere) {
  cv::RNG rng(42);
  const cv::Mat img = random_image(rng, 57, 83);
  const cv::Scalar pad(11, 22, 33);
  for (int original_sz : {3, 20, 55, 90, 200}) {
    for (int model_sz : {20, 64, original_sz}) {
      for (int i = 0; i < 40; ++i) {
        const float cx = rng.uniform(-120.f, 200.f);
        const float cy = rng.uniform(-120.f, 180.f);
        const cv::Mat ours = crop_subwindow(img, cx, cy, original_sz, model_sz, pad);
        const cv::Mat ref = ref_crop_subwindow(img, cx, cy, original_sz, model_sz, pad);
        ASSERT_EQ(cv::norm(ours, ref, cv::NORM_INF), 0.0)
            << "cx=" << cx << " cy=" << cy << " sz=" << original_sz << " model=" << model_sz;
      }
    }
  }
}

TEST(SubwindowExact, NeedsPaddingAgreesWithReferenceGeometry) {
  cv::RNG rng(7);
  const cv::Mat img = random_image(rng, 40, 60);
  for (int i = 0; i < 300; ++i) {
    const float cx = rng.uniform(-80.f, 140.f);
    const float cy = rng.uniform(-80.f, 120.f);
    const int sz = rng.uniform(2, 100);
    // Padding is needed exactly when a non-pad pixel would change: compare
    // crops made with two different pad colors.
    const cv::Mat a = ref_crop_subwindow(img, cx, cy, sz, sz, cv::Scalar(0, 0, 0));
    const cv::Mat b = ref_crop_subwindow(img, cx, cy, sz, sz, cv::Scalar(255, 255, 255));
    const bool pad_visible = cv::norm(a, b, cv::NORM_INF) > 0.0;
    if (pad_visible) {
      EXPECT_TRUE(subwindow_needs_padding(img, cx, cy, sz));
    }
    if (!subwindow_needs_padding(img, cx, cy, sz)) {
      EXPECT_FALSE(pad_visible);
    }
  }
}

TEST(SubwindowExact, NanoSubwindowMatchesReferenceEverywhere) {
  cv::RNG rng(1234);
  const cv::Mat img = random_image(rng, 61, 47);
  cv::Mat out, scratch;
  for (int original_sz : {5, 31, 60, 120}) {
    for (int model_sz : {127, 255, original_sz}) {
      for (int i = 0; i < 40; ++i) {
        const float px = rng.uniform(-90.f, 140.f);
        const float py = rng.uniform(-90.f, 150.f);
        nano_subwindow(img, px, py, original_sz, model_sz, out, scratch);
        const cv::Mat ref = ref_nano_subwindow(img, px, py, original_sz, model_sz);
        ASSERT_EQ(cv::norm(out, ref, cv::NORM_INF), 0.0)
            << "px=" << px << " py=" << py << " sz=" << original_sz << " model=" << model_sz;
      }
    }
  }
}

TEST(SubwindowExact, NanoSubwindowReusesBuffersAcrossCalls) {
  cv::RNG rng(5);
  const cv::Mat img = random_image(rng, 100, 100);
  cv::Mat out, scratch;
  nano_subwindow(img, -20.f, -20.f, 80, 64, out, scratch);  // padded call dirties scratch
  const void* scratch_data = scratch.data;
  nano_subwindow(img, 50.f, 50.f, 40, 64, out, scratch);  // interior call, scratch untouched
  const cv::Mat ref = ref_nano_subwindow(img, 50.f, 50.f, 40, 64);
  EXPECT_EQ(cv::norm(out, ref, cv::NORM_INF), 0.0);
  nano_subwindow(img, -10.f, 60.f, 80, 64, out, scratch);  // padded again, same buffer
  EXPECT_EQ(scratch.data, scratch_data);
  EXPECT_EQ(cv::norm(out, ref_nano_subwindow(img, -10.f, 60.f, 80, 64), cv::NORM_INF), 0.0);
}

}  // namespace
}  // namespace ctrk
