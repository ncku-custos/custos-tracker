#pragma once

#include <opencv2/core.hpp>

#include "ctrk/sot.hpp"
#include "ctrk/types.hpp"

namespace ctrk {

// MOSSE correlation filter (Bolme et al. 2010): the NN-free CPU fallback.
// Grayscale log-preproc, Hann window, Gaussian target (sigma 2), online
// numerator/denominator update (lr 0.125), 3-scale search, PSR confidence.
// Filter runs at a fixed 64x64 resolution; the tracked box keeps its own
// scale via the pyramid. score in SotResult is the raw PSR (healthy lock is
// typically > 15; < ~8 means the peak has collapsed — M4's lost signal).
class MosseTracker {
 public:
  explicit MosseTracker(const SotConfig& config) : cfg_(config) {}

  void init(const cv::Mat& image, const BBox& target);
  SotResult update(const cv::Mat& image);

 private:
  static constexpr int kSize = 64;         // filter resolution
  static constexpr float kSigma = 2.f;     // gaussian target
  static constexpr float kLr = 0.125f;     // online update rate
  static constexpr float kEps = 1e-5f;

  cv::Mat preprocess(const cv::Mat& patch) const;  // gray, log, norm, Hann
  cv::Mat grab_patch(const cv::Mat& image, float scale) const;
  void train(const cv::Mat& processed, float lr);

  SotConfig cfg_;
  cv::Mat hann_, gaussian_fft_;  // fixed windows
  cv::Mat num_, den_;            // filter state (complex spectra)
  float pos_x_ = 0, pos_y_ = 0, sz_w_ = 0, sz_h_ = 0;
  cv::Size img_size_;
};

}  // namespace ctrk
