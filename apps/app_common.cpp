#include "app_common.hpp"

#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>

#include <charconv>
#include <chrono>
#include <cstdio>

#include "ctrk/log.hpp"

namespace ctrk::app {

namespace {

bool is_integer(const std::string& s) {
  int value = 0;
  const auto [ptr, ec] = std::from_chars(s.data(), s.data() + s.size(), value);
  return ec == std::errc() && ptr == s.data() + s.size();
}

int64_t steady_now_ns() {
  return std::chrono::duration_cast<std::chrono::nanoseconds>(
             std::chrono::steady_clock::now().time_since_epoch())
      .count();
}

}  // namespace

std::unique_ptr<VideoSource> VideoSource::open(const std::string& input) {
  auto src = std::unique_ptr<VideoSource>(new VideoSource());
  if (is_integer(input)) {
    src->is_camera_ = true;
    src->cap_.open(std::stoi(input));
  } else {
    src->cap_.open(input);
  }
  if (!src->cap_.isOpened()) {
    log(LogLevel::Error, "failed to open input: " + input);
    return nullptr;
  }
  const double fps = src->cap_.get(cv::CAP_PROP_FPS);
  src->fps_ = (fps > 1.0 && fps < 1000.0) ? fps : 30.0;
  src->size_ = cv::Size(static_cast<int>(src->cap_.get(cv::CAP_PROP_FRAME_WIDTH)),
                        static_cast<int>(src->cap_.get(cv::CAP_PROP_FRAME_HEIGHT)));
  return src;
}

bool VideoSource::read(cv::Mat& frame, int64_t& t_ns) {
  if (!cap_.read(frame) || frame.empty()) return false;
  if (is_camera_) {
    t_ns = steady_now_ns();
  } else {
    t_ns = static_cast<int64_t>(static_cast<double>(idx_) / fps_ * 1e9);
  }
  ++idx_;
  return true;
}

VideoSink::VideoSink(const std::string& out_path, double fps, cv::Size size, bool display)
    : display_(display) {
  if (!out_path.empty()) {
    writer_.open(out_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, size);
    if (!writer_.isOpened()) log(LogLevel::Error, "failed to open output: " + out_path);
  }
}

VideoSink::~VideoSink() {
  if (display_) cv::destroyAllWindows();
}

void VideoSink::write(const cv::Mat& frame) {
  if (writer_.isOpened()) writer_.write(frame);
  if (display_) {
    cv::imshow("ctrk", frame);
    const int key = cv::waitKey(1) & 0xFF;
    if (key == 'q' || key == 27) stop_requested_ = true;
  }
}

void draw_box(cv::Mat& frame, const BBox& box, const cv::Scalar& color, const std::string& label) {
  const cv::Rect r(static_cast<int>(box.x), static_cast<int>(box.y), static_cast<int>(box.w),
                   static_cast<int>(box.h));
  cv::rectangle(frame, r, color, 2);
  if (!label.empty()) {
    const int baseline_y = std::max(r.y - 6, 12);
    cv::putText(frame, label, {r.x, baseline_y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1,
                cv::LINE_AA);
  }
}

void draw_hud(cv::Mat& frame, const StageTimer& timer, double fps_now) {
  int y = 18;
  char line[128];
  std::snprintf(line, sizeof(line), "%.1f fps", fps_now);
  cv::putText(frame, line, {8, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1, cv::LINE_AA);
  for (const auto& [stage, stats] : timer.stats()) {
    y += 18;
    std::snprintf(line, sizeof(line), "%s p50 %.2f ms", stage.c_str(), stats.p50_ms());
    cv::putText(frame, line, {8, y}, cv::FONT_HERSHEY_SIMPLEX, 0.5, {0, 255, 0}, 1, cv::LINE_AA);
  }
}

void print_stage_summary(const StageTimer& timer) {
  std::printf("%-16s %8s %10s %10s %10s\n", "stage", "n", "mean_ms", "p50_ms", "p95_ms");
  for (const auto& [stage, stats] : timer.stats()) {
    std::printf("%-16s %8zu %10.3f %10.3f %10.3f\n", stage.c_str(), stats.count(), stats.mean_ms(),
                stats.p50_ms(), stats.p95_ms());
  }
}

bool write_bench_json(const StageTimer& timer, const std::string& path, int frames) {
  FILE* f = std::fopen(path.c_str(), "w");
  if (!f) {
    log(LogLevel::Error, "failed to open bench json: " + path);
    return false;
  }
  std::fprintf(f, "{\n  \"frames\": %d,\n  \"stages\": {", frames);
  bool first = true;
  for (const auto& [stage, stats] : timer.stats()) {
    std::fprintf(
        f, "%s\n    \"%s\": {\"n\": %zu, \"mean_ms\": %.3f, \"p50_ms\": %.3f, \"p95_ms\": %.3f}",
        first ? "" : ",", stage.c_str(), stats.count(), stats.mean_ms(), stats.p50_ms(),
        stats.p95_ms());
    first = false;
  }
  std::fprintf(f, "\n  }\n}\n");
  std::fclose(f);
  return true;
}

bool parse_bbox(const std::string& s, BBox& out) {
  float v[4];
  int consumed = 0;
  if (std::sscanf(s.c_str(), "%f,%f,%f,%f%n", &v[0], &v[1], &v[2], &v[3], &consumed) != 4 ||
      consumed != static_cast<int>(s.size()))
    return false;
  out = {v[0], v[1], v[2], v[3]};
  return out.w > 0.f && out.h > 0.f;
}

}  // namespace ctrk::app
