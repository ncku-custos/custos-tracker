#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "common/geometry.hpp"

namespace ctrk {

// BoT-SORT-style sparse-flow camera-motion estimator (RESULTS.md S3.4):
// grayscale downscale to ~480 px wide, goodFeaturesToTrack on the previous
// frame, pyramidal LK into the current one, RANSAC partial affine. Returns
// the previous->current warp in full-resolution pixel coordinates; identity
// on the first frame and on degenerate estimates (too few corners, LK
// dropout, RANSAC failure). Stateful — keeps the previous downscaled gray;
// call exactly once per frame in display order (coast frames included, or
// the warp chain breaks).
//
// `exclude` are full-resolution boxes of tracked/detected OBJECTS in the
// previous frame: their corners follow object motion, not camera motion, and
// in a dense scene they can dominate the corner set and hijack the estimate
// (measured on MOT16-04, S3.4). They are masked out of the corner detector.
class GmcEstimator {
 public:
  Affine23 estimate(const cv::Mat& bgr, const std::vector<BBox>& exclude = {});

 private:
  cv::Mat prev_gray_;
  std::vector<BBox> prev_exclude_;  // boxes live in the PREVIOUS frame
};

}  // namespace ctrk
