#include <opencv2/core.hpp>

#include <cstdio>
#include <fstream>
#include <optional>
#include <string>

#include "app_common.hpp"
#include "common/mat_view.hpp"
#include "ctrk/detector.hpp"
#include "ctrk/profile.hpp"
#include "ctrk/sot.hpp"
#include "ctrk/types.hpp"

namespace {

const char* kUsage =
    "{help h      |       | show this message}"
    "{input i     |       | video file, image-sequence pattern, or camera index}"
    "{bbox b      |       | initial target box as x,y,w,h (pixels, first frame)}"
    "{backend     | nano  | tracker backend: nano | mosse}"
    "{backbone-z  | models/cache/nanotrack_backbone_z.onnx | template branch ONNX}"
    "{backbone-x  | models/cache/nanotrack_backbone_x.onnx | search branch ONNX}"
    "{head        | models/cache/nanotrack_head.onnx | correlation head ONNX}"
    "{reacquire   |       | on Lost, re-acquire via detector (class/position/size gated)}"
    "{det-model   | models/cache/yolov8n_640.onnx | detector ONNX for --reacquire}"
    "{class       | 0     | required detector class for --reacquire (-1 = any)}"
    "{output o    | sot_out.mp4 | annotated output video ('' to disable)}"
    "{dump        |       | write per-frame 'x,y,w,h' results (OTB format, incl. init frame)}"
    "{bench-json  |       | write per-stage latency stats as JSON}"
    "{display     |       | show a live window (never default: headless CI/drone)}";

}  // namespace

int main(int argc, char** argv) {
  cv::CommandLineParser cli(argc, argv, kUsage);
  cli.about("ctrk single-object tracking: NanoTrack v2 (3 static graphs)");
  if (cli.has("help") || !cli.has("input") || !cli.has("bbox")) {
    cli.printMessage();
    return cli.has("help") ? 0 : 1;
  }

  ctrk::BBox box;
  if (!ctrk::app::parse_bbox(cli.get<std::string>("bbox"), box)) {
    std::fprintf(stderr, "bad --bbox, expected x,y,w,h\n");
    return 1;
  }

  auto src = ctrk::app::VideoSource::open(cli.get<std::string>("input"));
  if (!src) return 1;
  ctrk::app::VideoSink sink(cli.get<std::string>("output"), src->fps(), src->size(),
                            cli.has("display"));

  ctrk::SotConfig cfg;
  cfg.backend = cli.get<std::string>("backend") == "mosse" ? ctrk::SotBackend::Mosse
                                                           : ctrk::SotBackend::NanoTrack;
  cfg.backbone_z_path = cli.get<std::string>("backbone-z");
  cfg.backbone_x_path = cli.get<std::string>("backbone-x");
  cfg.head_path = cli.get<std::string>("head");
  std::optional<ctrk::SotTracker> tracker;
  try {
    tracker.emplace(cfg);
    if (cli.has("reacquire")) {
      ctrk::Yolov8Config det;
      det.model_path = cli.get<std::string>("det-model");
      det.conf_thr = 0.25f;
      ctrk::ReacquireConfig rc;
      rc.class_id = cli.get<int>("class");
      tracker->enable_reacquire(ctrk::make_yolov8_detector(det), rc);
    }
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tracker init failed: %s\n", e.what());
    return 1;
  }

  std::ofstream dump;
  if (cli.has("dump") && !cli.get<std::string>("dump").empty())
    dump.open(cli.get<std::string>("dump"));
  const auto dump_box = [&](const ctrk::BBox& b) {
    if (dump.is_open()) {
      char line[96];
      std::snprintf(line, sizeof(line), "%.2f,%.2f,%.2f,%.2f\n", b.x, b.y, b.w, b.h);
      dump << line;
    }
  };

  ctrk::StageTimer timer;
  ctrk::set_profile_sink(
      [&timer](std::string_view stage, double ms) { timer.add_ms(std::string(stage), ms); });
  cv::Mat frame;
  int64_t t_ns = 0;
  bool initialized = false;

  while (sink.wants_more()) {
    bool got = false;
    {
      const auto scope = timer.scope("capture");
      got = src->read(frame, t_ns);
    }
    if (!got) break;
    ctrk::SotResult result;
    {
      const auto scope = timer.scope("sot");
      const ctrk::FrameView view = ctrk::as_frame_view(frame, t_ns);
      if (!initialized) {
        tracker->init(view, box);
        result = {box, 1.f, ctrk::SotState::Tracking};
        initialized = true;
      } else {
        result = tracker->update(view);
      }
    }
    dump_box(result.box);
    {
      const auto scope = timer.scope("draw");
      char label[64];
      std::snprintf(label, sizeof(label), "target %.2f", result.score);
      ctrk::app::draw_box(frame, result.box, {0, 200, 255}, label);
      const auto& stats = timer.stats().at("sot");
      ctrk::app::draw_hud(frame, timer, stats.p50_ms() > 0 ? 1000.0 / stats.p50_ms() : 0.0);
    }
    {
      const auto scope = timer.scope("encode");
      sink.write(frame);
    }
  }
  ctrk::set_profile_sink({});

  std::printf("frames: %d\n", src->frames_read());
  ctrk::app::print_stage_summary(timer);
  if (cli.has("bench-json") && !cli.get<std::string>("bench-json").empty())
    ctrk::app::write_bench_json(timer, cli.get<std::string>("bench-json"), src->frames_read());
  return 0;
}
