#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include <cstdio>
#include <fstream>
#include <optional>
#include <sstream>
#include <string>

#include "app_common.hpp"
#include "common/mat_view.hpp"
#include "ctrk/profile.hpp"
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
    "{detect-every | 1 | run the detector every Nth frame (KF coasting between)}"
    "{no-nsa   |       | classic Kalman R (disable NSA det-score scaling, S3.1)}"
    "{tentative-relax | 0.3 | relax the stage-3 IoU gate per coasted frame (S3.2)}"
    "{tentative-patience | 1 | detect-frame misses a tentative track survives (S3.2)}"
    "{velocity-seed |   | seed newborn KF velocity from its first re-match (S3.2)}"
    "{gmc      |       | camera-motion compensation via sparse flow (S3.4)}"
    "{threads  | 4     | detector intra-op threads}"
    "{no-spin  |       | stop the ORT pool busy-waiting between runs (lower idle CPU)}"
    "{dnnl     |       | experimental oneDNN execution provider}"
    "{output o | tbd_out.mp4 | annotated output video ('' to disable)}"
    "{dump     |       | write MOT-format results (frame,id,x,y,w,h,score,-1,-1,-1)}"
    "{bench-json |     | write per-stage latency stats as JSON}"
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
  cfg.detector.engine.intra_op_threads = cli.get<int>("threads");
  cfg.detector.engine.allow_spinning = !cli.has("no-spin");
  cfg.detector.engine.use_dnnl = cli.has("dnnl");
  cfg.assoc.use_byte = cli.get<std::string>("mode") != "sort";
  cfg.assoc.nsa = !cli.has("no-nsa");
  cfg.assoc.tentative_relax_per_coast = cli.get<float>("tentative-relax");
  cfg.assoc.tentative_patience = cli.get<int>("tentative-patience");
  cfg.assoc.velocity_seed = cli.has("velocity-seed");
  cfg.gmc = cli.has("gmc") ? ctrk::GmcMethod::SparseFlow : ctrk::GmcMethod::Off;
  cfg.detect_interval = cli.get<int>("detect-every");
  cfg.nominal_fps = src->fps();
  std::optional<ctrk::MultiTracker> tracker;
  try {
    tracker.emplace(cfg);
  } catch (const std::exception& e) {
    std::fprintf(stderr, "tracker init failed: %s\n", e.what());
    return 1;
  }

  std::ofstream dump;
  if (cli.has("dump") && !cli.get<std::string>("dump").empty())
    dump.open(cli.get<std::string>("dump"));

  ctrk::StageTimer timer;
  ctrk::set_profile_sink(
      [&timer](std::string_view stage, double ms) { timer.add_ms(std::string(stage), ms); });
  cv::Mat frame;
  int64_t t_ns = 0;
  int frame_no = 0;

  while (sink.wants_more()) {
    bool got = false;
    {
      const auto scope = timer.scope("capture");
      got = src->read(frame, t_ns);
    }
    if (!got) break;
    ++frame_no;  // MOT frame numbers are 1-based
    std::vector<ctrk::Track> tracks;
    {
      const auto scope = timer.scope("detect+track");
      tracks = tracker->update(ctrk::as_frame_view(frame, t_ns));
    }
    {
      const auto scope = timer.scope("draw");
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
