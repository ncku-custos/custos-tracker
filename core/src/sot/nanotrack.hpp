#pragma once

#include <opencv2/core.hpp>

#include <memory>
#include <vector>

#include "ctrk/infer.hpp"
#include "ctrk/sot.hpp"

namespace ctrk {

// TrackerNanoImpl::getSubwindow replica as a free function (unit-testable
// without engines): integer-truncated center, integer half-size, whole-image
// mean as padding — but the mean is computed and the padded window built
// (crop-sized, not frame-sized) only when the crop leaves the image. `out`
// and `scratch` are caller-owned so repeated calls do not allocate.
void nano_subwindow(const cv::Mat& img, float pos_x, float pos_y, int original_sz, int model_sz,
                    cv::Mat& out, cv::Mat& scratch);

// NanoTrack v2 siamese pipeline. The numerics deliberately replicate
// OpenCV's TrackerNanoImpl bit-for-bit (including its integer-division and
// sizeCal(targetPos) quirks) — the differential oracle test depends on it.
// Deviations from the reference live behind config knobs and were added only
// after the oracle proved the baseline correct.
class NanoTracker {
 public:
  explicit NanoTracker(const SotConfig& config);

  void init(const cv::Mat& image, const BBox& target);
  SotResult update(const cv::Mat& image);

  // Appearance embedding for re-ID verification (RESULTS.md S3.3): run the
  // template branch on a 127x127 context crop centered on `box` (the same
  // geometry init() uses) and return the raw feature vector. Does not touch
  // tracker state.
  std::vector<float> embed(const cv::Mat& image, const BBox& box) const;
  // Frozen init-time template features — the re-ID reference.
  const std::vector<float>& zf() const { return zf_; }

 private:
  static constexpr int kExemplar = 127;
  static constexpr int kInstance = 255;
  static constexpr int kStride = 16;

  void blob_rgb(const cv::Mat& crop, std::vector<float>& out) const;

  SotConfig cfg_;
  std::unique_ptr<IEngine> backbone_z_, backbone_x_, head_;

  // Score-map side, read from the loaded head graph ([1,2,S,S]) so v2
  // (16x16) and v3 (15x15) share one postproc path (RESULTS.md S3.7).
  int score_size_ = 0;

  std::vector<float> zf_;      // template features, frozen at init
  std::vector<float> zf_mix_;  // dual-template blend (S3.8); empty = use zf_
  int since_refresh_ = 0;
  std::vector<float> x_blob_;  // reusable search-branch input
  float pos_x_ = 0, pos_y_ = 0, sz_w_ = 0, sz_h_ = 0;
  cv::Size img_size_;
  cv::Mat hann_, grid_x_, grid_y_;  // score_size_^2 CV_32F
  cv::Mat crop_, crop_scratch_;     // nano_subwindow reusable buffers
};

}  // namespace ctrk
