#include "common/subwindow.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace ctrk {

cv::Mat crop_subwindow(const cv::Mat& img, float cx, float cy, int original_sz, int model_sz,
                       const cv::Scalar& pad_value) {
  CV_Assert(!img.empty() && original_sz > 0 && model_sz > 0);

  const float c = static_cast<float>(original_sz + 1) / 2.f;
  int xmin = static_cast<int>(std::floor(cx - c + 0.5f));
  int ymin = static_cast<int>(std::floor(cy - c + 0.5f));
  int xmax = xmin + original_sz - 1;
  int ymax = ymin + original_sz - 1;

  const int left_pad = std::max(0, -xmin);
  const int top_pad = std::max(0, -ymin);
  const int right_pad = std::max(0, xmax - img.cols + 1);
  const int bottom_pad = std::max(0, ymax - img.rows + 1);

  cv::Mat patch;
  if (left_pad || top_pad || right_pad || bottom_pad) {
    cv::Mat padded;
    cv::copyMakeBorder(img, padded, top_pad, bottom_pad, left_pad, right_pad,
                       cv::BORDER_CONSTANT, pad_value);
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

}  // namespace ctrk
