#include "common/appearance.hpp"

#include <opencv2/imgproc.hpp>

#include <cmath>

namespace ctrk {

std::vector<float> hsv_embedding(const cv::Mat& img, const BBox& box) {
  const cv::Rect roi = cv::Rect(static_cast<int>(box.x), static_cast<int>(box.y),
                                static_cast<int>(box.w), static_cast<int>(box.h)) &
                       cv::Rect(0, 0, img.cols, img.rows);
  if (roi.width < 2 || roi.height < 2) return {};
  cv::Mat hsv;
  cv::cvtColor(img(roi), hsv, cv::COLOR_BGR2HSV);
  const int channels[] = {0, 1};
  const int bins[] = {30, 32};
  const float h_range[] = {0, 180}, s_range[] = {0, 256};
  const float* ranges[] = {h_range, s_range};
  cv::Mat hist;
  cv::calcHist(&hsv, 1, channels, cv::Mat(), hist, 2, bins, ranges);
  cv::normalize(hist, hist, 1.0, 0.0, cv::NORM_L2);
  return {hist.begin<float>(), hist.end<float>()};
}

float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b) {
  if (a.empty() || a.size() != b.size()) return 0.f;
  float dot = 0.f, na = 0.f, nb = 0.f;
  for (size_t i = 0; i < a.size(); ++i) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const float denom = std::sqrt(na) * std::sqrt(nb);
  return denom > 0.f ? dot / denom : 0.f;
}

}  // namespace ctrk
