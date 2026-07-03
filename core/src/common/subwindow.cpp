#include "common/subwindow.hpp"

#include <algorithm>
#include <cmath>

#include <opencv2/imgproc.hpp>

namespace ctrk {

namespace {

struct CropGeometry {
  int xmin, ymin;
  int left_pad, top_pad, right_pad, bottom_pad;
  bool padded() const { return left_pad || top_pad || right_pad || bottom_pad; }
};

CropGeometry crop_geometry(const cv::Mat& img, float cx, float cy, int original_sz) {
  const float c = static_cast<float>(original_sz + 1) / 2.f;
  const int xmin = static_cast<int>(std::floor(cx - c + 0.5f));
  const int ymin = static_cast<int>(std::floor(cy - c + 0.5f));
  const int xmax = xmin + original_sz - 1;
  const int ymax = ymin + original_sz - 1;
  return {xmin,
          ymin,
          std::max(0, -xmin),
          std::max(0, -ymin),
          std::max(0, xmax - img.cols + 1),
          std::max(0, ymax - img.rows + 1)};
}

}  // namespace

bool subwindow_needs_padding(const cv::Mat& img, float cx, float cy, int original_sz) {
  return crop_geometry(img, cx, cy, original_sz).padded();
}

cv::Mat crop_subwindow(const cv::Mat& img, float cx, float cy, int original_sz, int model_sz,
                       const cv::Scalar& pad_value) {
  CV_Assert(!img.empty() && original_sz > 0 && model_sz > 0);

  const CropGeometry g = crop_geometry(img, cx, cy, original_sz);

  cv::Mat patch;
  if (g.padded()) {
    // Build only the original_sz^2 padded window instead of padding the whole
    // frame; pixel-identical to copyMakeBorder-then-ROI (setTo and
    // BORDER_CONSTANT share the same scalar-to-pixel conversion).
    patch.create(original_sz, original_sz, img.type());
    patch.setTo(pad_value);
    const int vw = original_sz - g.left_pad - g.right_pad;
    const int vh = original_sz - g.top_pad - g.bottom_pad;
    if (vw > 0 && vh > 0) {
      img(cv::Rect(g.xmin + g.left_pad, g.ymin + g.top_pad, vw, vh))
          .copyTo(patch(cv::Rect(g.left_pad, g.top_pad, vw, vh)));
    }
  } else {
    patch = img(cv::Rect(g.xmin, g.ymin, original_sz, original_sz));
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
