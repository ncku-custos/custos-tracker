#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <fstream>
#include <sstream>
#include <string>

#include "app_common.hpp"
#include "common/mat_view.hpp"
#include "ctrk/tbd.hpp"
#include "ctrk/types.hpp"

namespace {

const char* kUsage =
    "{help h   |       | show this message}"
    "{input i  |       | video file, image-sequence pattern, or camera index}"
    "{model m  | models/cache/yolov8n_640.onnx | detector ONNX}"
    "{mode     | byte  | association mode: byte | sort}"
    "{classes  |       | comma-separated COCO class ids to keep (empty = all, 0 = person)}"
    "{conf     | 0.1   | detector confidence floor}"
    "{output o | tbd_out.mp4 | annotated output video ('' to disable)}"
    "{dump     |       | write MOT-format results (frame,id,x,y,w,h,score,-1,-1,-1)}"
    "{display  |       | show a live window (never default: headless CI/drone)}";

std::vector<int> parse_classes(const std::string& s) {
  std::vector<int> out;
  std::stringstream ss(s);
  for (std::string item; std::getline(ss, item, ',');)
    if (!item.empty()) out.push_back(std::stoi(item));
  return out;
}

cv::Scalar id_color(int id) {
  // Stable, well-separated hues per id.
  const int h = (id * 47) % 180;
  cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar(h, 200, 255)), bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const auto v = bgr.at<cv::Vec3b>(0, 0);
  return {static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

}  // namespace

int main(int argc, char** argv) {
  cv::CommandLineParser cli(argc, argv, kUsage);
  cli.about("ctrk tracking-by-detection: YOLOv8n + SORT/ByteTrack");
  if (cli.has("help") || !cli.has("input")) {
    cli.printMessage();
    return cli.has("help") ? 0 : 1;
  }

  auto src = ctrk::app::VideoSource::open(cli.get<std::string>("input"));
  if (!src) return 1;
  ctrk::app::VideoSink sink(cli.get<std::string>("output"), src->fps(), src->size(),
                            cli.has("display"));

  ctrk::TbdConfig cfg;
  cfg.detector.model_path = cli.get<std::string>("model");
  cfg.detector.conf_thr = cli.get<float>("conf");
  cfg.detector.keep_classes = parse_classes(cli.get<std::string>("classes"));
  cfg.assoc.use_byte = cli.get<std::string>("mode") != "sort";
  cfg.nominal_fps = src->fps();
  ctrk::MultiTracker tracker(cfg);

  std::ofstream dump;
  if (cli.has("dump") && !cli.get<std::string>("dump").empty())
    dump.open(cli.get<std::string>("dump"));

  ctrk::StageTimer timer;
  cv::Mat frame;
  int64_t t_ns = 0;
  int frame_no = 0;

  while (src->read(frame, t_ns) && sink.wants_more()) {
    ++frame_no;  // MOT frame numbers are 1-based
    std::vector<ctrk::Track> tracks;
    {
      const auto scope = timer.scope("detect+track");
      tracks = tracker.update(ctrk::as_frame_view(frame, t_ns));
    }
    for (const auto& t : tracks) {
      if (t.state != ctrk::TrackState::Confirmed) continue;
      char label[64];
      std::snprintf(label, sizeof(label), "#%d %.2f", t.id, t.score);
      ctrk::app::draw_box(frame, t.box, id_color(t.id), label);
      if (dump.is_open()) {
        char line[128];
        std::snprintf(line, sizeof(line), "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1\n", frame_no,
                      t.id, t.box.x, t.box.y, t.box.w, t.box.h, t.score);
        dump << line;
      }
    }
    const auto& stats = timer.stats().at("detect+track");
    ctrk::app::draw_hud(frame, timer, stats.p50_ms() > 0 ? 1000.0 / stats.p50_ms() : 0.0);
    sink.write(frame);
  }

  std::printf("frames: %d\n", src->frames_read());
  ctrk::app::print_stage_summary(timer);
  return 0;
}
