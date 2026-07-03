#pragma once

#include <opencv2/core.hpp>

namespace ctrk {

// SiamFC-style subwindow: extract a square `original_sz` crop centered at
// (cx, cy), pad with `pad_value` where the crop leaves the image, resize to
// `model_sz` x `model_sz`. Shared by the siamese template/search crops and
// MOSSE; must stay numerically aligned with cv::TrackerNano's getSubwindow
// (the M2 differential oracle).
cv::Mat crop_subwindow(const cv::Mat& img, float cx, float cy, int original_sz, int model_sz,
                       const cv::Scalar& pad_value);

// True if the crop above would extend beyond the image. Lets callers compute
// an expensive pad value (e.g. the whole-frame mean) only when it is used.
bool subwindow_needs_padding(const cv::Mat& img, float cx, float cy, int original_sz);

}  // namespace ctrk
