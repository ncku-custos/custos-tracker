#include "sot/mosse.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

#include "common/subwindow.hpp"

namespace ctrk {

namespace {

// Complex spectrum ops on 2-channel CV_32FC2 mats.
cv::Mat spec_mul(const cv::Mat& a, const cv::Mat& b, bool conj_b) {
  cv::Mat out;
  cv::mulSpectrums(a, b, out, 0, conj_b);
  return out;
}

}  // namespace

cv::Mat MosseTracker::preprocess(const cv::Mat& patch) const {
  cv::Mat gray;
  if (patch.channels() == 3) {
    cv::cvtColor(patch, gray, cv::COLOR_BGR2GRAY);
  } else {
    gray = patch;
  }
  gray.convertTo(gray, CV_32F);
  cv::log(gray + 1.f, gray);
  cv::Scalar mean, stddev;
  cv::meanStdDev(gray, mean, stddev);
  gray = (gray - mean[0]) / (stddev[0] + kEps);
  return gray.mul(hann_);
}

cv::Mat MosseTracker::grab_patch(const cv::Mat& image, float scale) const {
  // Square context window around the target (its larger side, scaled),
  // resized to the fixed filter resolution.
  const int side = std::max(2, static_cast<int>(std::lround(std::max(sz_w_, sz_h_) * scale)));
  return crop_subwindow(image, pos_x_, pos_y_, side, kSize, cv::mean(image));
}

void MosseTracker::train(const cv::Mat& processed, float lr) {
  cv::Mat f_fft;
  cv::dft(processed, f_fft, cv::DFT_COMPLEX_OUTPUT);
  const cv::Mat num_new = spec_mul(gaussian_fft_, f_fft, /*conj_b=*/true);
  const cv::Mat den_new = spec_mul(f_fft, f_fft, /*conj_b=*/true);
  if (num_.empty()) {
    num_ = num_new;
    den_ = den_new;
  } else {
    num_ = num_ * (1.f - lr) + num_new * lr;
    den_ = den_ * (1.f - lr) + den_new * lr;
  }
}

void MosseTracker::init(const cv::Mat& image, const BBox& target) {
  pos_x_ = target.x + target.w * 0.5f;
  pos_y_ = target.y + target.h * 0.5f;
  sz_w_ = target.w;
  sz_h_ = target.h;
  img_size_ = image.size();

  cv::createHanningWindow(hann_, {kSize, kSize}, CV_32F);

  cv::Mat gauss(kSize, kSize, CV_32F);
  for (int r = 0; r < kSize; ++r)
    for (int c = 0; c < kSize; ++c) {
      const float dy = static_cast<float>(r - kSize / 2);
      const float dx = static_cast<float>(c - kSize / 2);
      gauss.at<float>(r, c) = std::exp(-(dx * dx + dy * dy) / (2.f * kSigma * kSigma));
    }
  cv::dft(gauss, gaussian_fft_, cv::DFT_COMPLEX_OUTPUT);

  num_.release();
  den_.release();

  // Train on the exact patch plus small deterministic rotations — the classic
  // MOSSE augmentation, keeping the first filter from overfitting one frame.
  const cv::Mat base = grab_patch(image, 1.f);
  train(preprocess(base), 1.f);
  for (int k = -2; k <= 2; ++k) {
    if (k == 0) continue;
    const cv::Mat rot =
        cv::getRotationMatrix2D({kSize / 2.f, kSize / 2.f}, static_cast<double>(k) * 2.0, 1.0);
    cv::Mat warped;
    cv::warpAffine(base, warped, rot, base.size(), cv::INTER_LINEAR, cv::BORDER_REFLECT);
    train(preprocess(warped), 0.5f);
  }
}

SotResult MosseTracker::update(const cv::Mat& image) {
  static constexpr float kScales[] = {0.985f, 1.f, 1.015f};

  cv::Mat filter;
  cv::Mat den_reg = den_.clone();
  den_reg.forEach<cv::Vec2f>([](cv::Vec2f& v, const int*) { v[0] += kEps; });
  cv::divSpectrums(num_, den_reg, filter, 0);

  float best_peak = -1.f, best_psr = 0.f, best_scale = 1.f;
  cv::Point best_loc;
  cv::Mat best_patch;

  for (const float s : kScales) {
    const cv::Mat patch = grab_patch(image, s);
    cv::Mat f_fft;
    cv::dft(preprocess(patch), f_fft, cv::DFT_COMPLEX_OUTPUT);
    cv::Mat resp;
    cv::idft(spec_mul(filter, f_fft, /*conj_b=*/false), resp, cv::DFT_SCALE | cv::DFT_REAL_OUTPUT);

    double peak;
    cv::Point loc;
    cv::minMaxLoc(resp, nullptr, &peak, nullptr, &loc);

    // PSR: peak vs sidelobe (response excluding 11x11 around the peak).
    cv::Mat mask = cv::Mat::ones(resp.size(), CV_8U);
    const cv::Rect exclusion(std::max(0, loc.x - 5), std::max(0, loc.y - 5), 11, 11);
    mask(exclusion & cv::Rect(0, 0, kSize, kSize)) = 0;
    cv::Scalar mean, stddev;
    cv::meanStdDev(resp, mean, stddev, mask);
    const float psr =
        static_cast<float>((peak - mean[0]) / (stddev[0] + static_cast<double>(kEps)));

    if (static_cast<float>(peak) > best_peak) {
      best_peak = static_cast<float>(peak);
      best_psr = psr;
      best_scale = s;
      best_loc = loc;
      best_patch = patch;
    }
  }

  // Peak offset in filter pixels -> image pixels at the chosen scale.
  const float side = std::max(sz_w_, sz_h_) * best_scale;
  const float to_img = side / static_cast<float>(kSize);
  pos_x_ += (static_cast<float>(best_loc.x) - kSize / 2.f) * to_img;
  pos_y_ += (static_cast<float>(best_loc.y) - kSize / 2.f) * to_img;
  pos_x_ = std::clamp(pos_x_, 0.f, static_cast<float>(img_size_.width));
  pos_y_ = std::clamp(pos_y_, 0.f, static_cast<float>(img_size_.height));
  sz_w_ = std::clamp(sz_w_ * best_scale, 10.f, static_cast<float>(img_size_.width));
  sz_h_ = std::clamp(sz_h_ * best_scale, 10.f, static_cast<float>(img_size_.height));

  // Only adapt the filter while the lock is credible — updating on a
  // collapsed peak burns the template (classic MOSSE failure mode).
  if (best_psr >= 8.f) train(preprocess(grab_patch(image, 1.f)), kLr);

  SotResult result;
  result.box = {pos_x_ - sz_w_ * 0.5f, pos_y_ - sz_h_ * 0.5f, sz_w_, sz_h_};
  result.score = best_psr;
  result.state = SotState::Tracking;
  return result;
}

}  // namespace ctrk
