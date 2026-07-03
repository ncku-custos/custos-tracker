#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <random>
#include <vector>

#include "ctrk/types.hpp"

namespace ctrk::synth {

// One constant-velocity rectangular target.
struct Target {
  BBox box0;
  float vx = 0.f, vy = 0.f;  // px/frame
  cv::Scalar color{0, 0, 255};
};

struct Options {
  int width = 640;
  int height = 480;
  int frames = 120;
  float noise_sigma = 0.f;  // per-pixel gaussian noise
  // Optional static occluder drawn over the scene for a frame range.
  bool occluder = false;
  cv::Rect occluder_rect{280, 0, 80, 480};
  int occluder_from = 0, occluder_to = -1;
  uint32_t seed = 1234;
};

// Deterministic scripted scene: textured background + moving textured
// rectangles + optional occluder + optional noise. Ground truth is exact and
// available even while a target is occluded. Renders any frame on demand —
// no state, safe to use from multiple tests.
class Sequence {
 public:
  Sequence(Options opt, std::vector<Target> targets)
      : opt_(opt), targets_(std::move(targets)) {
    // Fixed speckle background so correlation trackers have structure to
    // reject, and detectors see non-uniform context.
    cv::Mat bg(opt_.height, opt_.width, CV_8UC3);
    cv::RNG rng(opt_.seed);
    rng.fill(bg, cv::RNG::UNIFORM, 40, 120);
    cv::GaussianBlur(bg, bg, {5, 5}, 0);
    background_ = bg;
  }

  int frames() const { return opt_.frames; }
  cv::Size size() const { return {opt_.width, opt_.height}; }

  std::vector<BBox> gt(int t) const {
    std::vector<BBox> boxes;
    boxes.reserve(targets_.size());
    for (const auto& tgt : targets_)
      boxes.push_back({tgt.box0.x + tgt.vx * static_cast<float>(t),
                       tgt.box0.y + tgt.vy * static_cast<float>(t), tgt.box0.w, tgt.box0.h});
    return boxes;
  }

  cv::Mat frame(int t) const {
    cv::Mat f = background_.clone();
    const auto boxes = gt(t);
    for (size_t i = 0; i < boxes.size(); ++i) {
      const auto& b = boxes[i];
      const cv::Rect r(static_cast<int>(b.x), static_cast<int>(b.y), static_cast<int>(b.w),
                       static_cast<int>(b.h));
      cv::rectangle(f, r, targets_[i].color, cv::FILLED);
      // Interior texture: contrasting inner block + diagonal, so the target
      // is not a flat patch (correlation trackers need gradients).
      const cv::Rect inner(r.x + r.width / 4, r.y + r.height / 4, r.width / 2, r.height / 2);
      cv::rectangle(f, inner & cv::Rect(0, 0, f.cols, f.rows), targets_[i].color * 0.4,
                    cv::FILLED);
      cv::line(f, r.tl(), r.br(), {255, 255, 255}, 2);
    }
    if (opt_.occluder && t >= opt_.occluder_from &&
        (opt_.occluder_to < 0 || t <= opt_.occluder_to)) {
      cv::rectangle(f, opt_.occluder_rect, {90, 90, 90}, cv::FILLED);
    }
    if (opt_.noise_sigma > 0.f) {
      cv::Mat noise(f.size(), CV_16SC3);
      cv::RNG rng(opt_.seed + static_cast<uint32_t>(t) + 1);
      rng.fill(noise, cv::RNG::NORMAL, 0, opt_.noise_sigma);
      cv::Mat f16;
      f.convertTo(f16, CV_16SC3);
      f16 += noise;
      f16.convertTo(f, CV_8UC3);
    }
    return f;
  }

 private:
  Options opt_;
  std::vector<Target> targets_;
  cv::Mat background_;
};

}  // namespace ctrk::synth
