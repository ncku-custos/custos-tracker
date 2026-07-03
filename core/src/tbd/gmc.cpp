#include "tbd/gmc.hpp"

#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/video/tracking.hpp>

#include <algorithm>
#include <vector>

namespace ctrk {

namespace {

constexpr int kWorkWidth = 480;    // flow runs on a downscaled gray copy
constexpr int kMaxCorners = 200;   // plenty for a 6-dof partial affine
constexpr size_t kMinPoints = 12;  // margin over the RANSAC minimum

}  // namespace

Affine23 GmcEstimator::estimate(const cv::Mat& bgr, const std::vector<BBox>& exclude) {
  cv::Mat gray;
  cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
  const float k = std::min(1.f, static_cast<float>(kWorkWidth) / static_cast<float>(gray.cols));
  if (k < 1.f) cv::resize(gray, gray, {}, k, k, cv::INTER_AREA);

  Affine23 out;  // identity fallback on every failure path
  if (!prev_gray_.empty() && prev_gray_.size() == gray.size()) {
    // Corners must come from the static background: mask out the previous
    // frame's object boxes (in downscaled coordinates).
    cv::Mat mask(prev_gray_.size(), CV_8UC1, cv::Scalar(255));
    for (const auto& b : prev_exclude_) {
      const cv::Rect r(static_cast<int>(b.x * k), static_cast<int>(b.y * k),
                       static_cast<int>(b.w * k) + 1, static_cast<int>(b.h * k) + 1);
      mask(r & cv::Rect(0, 0, mask.cols, mask.rows)).setTo(0);
    }
    std::vector<cv::Point2f> prev_pts;
    cv::goodFeaturesToTrack(prev_gray_, prev_pts, kMaxCorners, 0.01, 8, mask);
    if (prev_pts.size() >= kMinPoints) {
      std::vector<cv::Point2f> cur_pts;
      std::vector<uchar> status;
      cv::calcOpticalFlowPyrLK(prev_gray_, gray, prev_pts, cur_pts, status, cv::noArray());
      std::vector<cv::Point2f> p0, p1;
      p0.reserve(prev_pts.size());
      p1.reserve(prev_pts.size());
      for (size_t i = 0; i < status.size(); ++i) {
        if (!status[i]) continue;
        p0.push_back(prev_pts[i]);
        p1.push_back(cur_pts[i]);
      }
      if (p0.size() >= kMinPoints) {
        std::vector<uchar> inliers;
        const cv::Mat m = cv::estimateAffinePartial2D(p0, p1, inliers, cv::RANSAC);
        if (!m.empty()) {
          out.a11 = static_cast<float>(m.at<double>(0, 0));
          out.a12 = static_cast<float>(m.at<double>(0, 1));
          out.a21 = static_cast<float>(m.at<double>(1, 0));
          out.a22 = static_cast<float>(m.at<double>(1, 1));
          // Flow ran in downscaled coordinates: the linear part is
          // scale-free, the translation maps back with 1/k.
          out.tx = static_cast<float>(m.at<double>(0, 2)) / k;
          out.ty = static_cast<float>(m.at<double>(1, 2)) / k;
        }
      }
    }
  }
  prev_gray_ = std::move(gray);
  prev_exclude_ = exclude;
  return out;
}

}  // namespace ctrk
