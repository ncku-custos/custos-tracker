#pragma once

#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>

#include <memory>
#include <string>

#include "common/timer.hpp"
#include "ctrk/types.hpp"

namespace ctrk::app {

// Video input: a file path, or an integer string ("0") for a camera index.
// Timestamps: files get frame_index/fps (replay-deterministic); cameras get
// the steady clock. Core never reads a clock — the app layer owns time.
class VideoSource {
 public:
  static std::unique_ptr<VideoSource> open(const std::string& input);

  bool read(cv::Mat& frame, int64_t& t_ns);
  double fps() const { return fps_; }
  cv::Size size() const { return size_; }
  int frames_read() const { return idx_; }

 private:
  VideoSource() = default;
  cv::VideoCapture cap_;
  double fps_ = 30.0;
  cv::Size size_;
  int idx_ = 0;
  bool is_camera_ = false;
};

// Video output: always writes an annotated mp4; optionally also displays a
// window (must stay opt-in — CI and the drone are headless).
class VideoSink {
 public:
  VideoSink(const std::string& out_path, double fps, cv::Size size, bool display);
  ~VideoSink();

  void write(const cv::Mat& frame);
  bool wants_more() const { return !stop_requested_; }  // user pressed q/ESC

 private:
  cv::VideoWriter writer_;
  bool display_ = false;
  bool stop_requested_ = false;
};

void draw_box(cv::Mat& frame, const BBox& box, const cv::Scalar& color, const std::string& label);

// Top-left HUD: per-stage p50 latency plus instantaneous FPS.
void draw_hud(cv::Mat& frame, const StageTimer& timer, double fps_now);

// Print the end-of-run stage table to stdout.
void print_stage_summary(const StageTimer& timer);

// Parse "x,y,w,h" into a BBox; returns false on malformed input.
bool parse_bbox(const std::string& s, BBox& out);

}  // namespace ctrk::app
