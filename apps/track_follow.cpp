#include <opencv2/core.hpp>

#include <algorithm>
#include <cmath>
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

constexpr float kPi = 3.14159265358979323846f;

enum FollowState : uint8_t {
  kUnable = 0,
  kFollowing = 1,
  kUnstable = 2,
  kLost = 3,
};

struct FollowConfig {
  float init_distance = 0.f;
  ctrk::BBox init_box;
  bool has_init_box = false;
  float cam_pitch_rad = 0.f;
  float hfov_rad = 0.f;
  float vfov_rad = 0.f;
  float lpf_alpha = 0.35f;
  float min_score = 0.f;
  bool allow_unstable = true;
  float kp_x = 0.25f;
  float kp_y = 0.60f;
  float kp_z = 0.35f;
  float kp_yaw = 1.2f;
  float deadband_x = 0.05f;
  float deadband_y = 0.10f;
  float deadband_z = 0.05f;
  float deadband_yaw_rad = 2.f * kPi / 180.f;
  float max_vx = 0.6f;
  float max_vy = 1.0f;
  float max_vz = 0.5f;
  float max_yaw_rate = 1.0f;
};

struct FollowStatus {
  int64_t t_ns = 0;
  FollowState state = kUnable;
  float target_cx = 0.f;
  float target_cy = 0.f;
  float target_w = 0.f;
  float target_h = 0.f;
  float score = 0.f;
  float estimated_distance = 0.f;
  float error_x = 0.f;
  float error_y = 0.f;
  float error_z = 0.f;
  float distance_error = 0.f;
  float vx = 0.f;
  float vy = 0.f;
  float vz = 0.f;
  float yaw_rate = 0.f;
  bool command_valid = false;
  bool command_saturated = false;
  std::string message;
};

struct FilteredBox {
  bool ready = false;
  ctrk::BBox box;

  ctrk::BBox update(const ctrk::BBox& in, float alpha) {
    const float a = std::clamp(alpha, 0.f, 1.f);
    if (!ready) {
      box = in;
      ready = true;
      return box;
    }
    box.x = a * in.x + (1.f - a) * box.x;
    box.y = a * in.y + (1.f - a) * box.y;
    box.w = a * in.w + (1.f - a) * box.w;
    box.h = a * in.h + (1.f - a) * box.h;
    return box;
  }
};

const char* kUsage =
    "{help h      |       | show this message}"
    "{input i     |       | video file, image-sequence pattern, or camera index}"
    "{bbox b      |       | initial SOT target box as x,y,w,h (pixels, first frame)}"
    "{init-distance |     | initial target distance in meters (required)}"
    "{init-box    |       | distance reference box as x,y,w,h; defaults to first SOT box}"
    "{cam-pitch   |       | camera pitch in degrees, positive downward from drone horizon (required)}"
    "{hfov        | 69.0  | horizontal camera field-of-view in degrees}"
    "{vfov        | 42.0  | vertical camera field-of-view in degrees}"
    "{lpf-alpha   | 0.35  | low-pass alpha for tracked box, 1 = no smoothing}"
    "{min-score   | 0.0   | command invalid below this tracker confidence}"
    "{no-unstable |       | do not emit commands while SOT state is Unstable}"
    "{kp-x        | 0.25  | lateral velocity gain from horizontal error}"
    "{kp-y        | 0.60  | forward velocity gain from distance error}"
    "{kp-z        | 0.35  | vertical velocity gain from vertical error}"
    "{kp-yaw      | 1.20  | yaw-rate gain from horizontal image angle}"
    "{deadband-x  | 0.05  | lateral deadband in meters}"
    "{deadband-y  | 0.10  | forward/depth deadband in meters}"
    "{deadband-z  | 0.05  | vertical deadband in meters}"
    "{deadband-yaw | 2.0  | yaw deadband in degrees}"
    "{max-vx      | 0.60  | lateral velocity limit}"
    "{max-vy      | 1.00  | forward velocity limit}"
    "{max-vz      | 0.50  | vertical velocity limit}"
    "{max-yaw-rate | 1.00 | yaw-rate limit in rad/s}"
    "{status-json | follow_status.jsonl | per-frame FollowStatus JSONL output ('' to disable)}"
    "{backend     | nano  | tracker backend: nano | mosse}"
    "{backbone-z  | models/cache/nanotrack_backbone_z.onnx | template branch ONNX}"
    "{backbone-x  | models/cache/nanotrack_backbone_x.onnx | search branch ONNX}"
    "{head        | models/cache/nanotrack_head.onnx | correlation head ONNX}"
    "{threads     | 2     | intra-op threads per graph}"
    "{penalty-k   | 0.055 | postproc size/ratio penalty}"
    "{window-influence | 0.455 | Hann window blend weight}"
    "{size-lr     | 0.37  | size update learning rate}"
    "{template-every | 0 | dual-template refresh interval, frames (0 = off)}"
    "{template-blend | 0.5 | weight of the frozen template in the dual blend}"
    "{spin        |       | re-enable ORT pool busy-waiting}"
    "{dnnl        |       | experimental oneDNN execution provider}"
    "{reacquire   |       | on Lost, re-acquire via detector}"
    "{det-model   | models/cache/yolov8n_640.onnx | detector ONNX for --reacquire}"
    "{class       | 0     | required detector class for --reacquire (-1 = any)}"
    "{reid        | hsv   | re-lock appearance verification: hsv | nanoz | none}"
    "{reid-accept | -1    | min re-ID similarity for a re-lock candidate (-1 = auto)}"
    "{drift-every | 0     | verify the tracked box every K frames (0 = off)}"
    "{drift-thr   | -1    | tracked-box similarity below this -> Lost (-1 = auto)}"
    "{output o    | follow_out.mp4 | annotated output video ('' to disable)}"
    "{bench-json  |       | write per-stage latency stats as JSON}"
    "{display     |       | show a live window}";

float deg_to_rad(float deg) { return deg * kPi / 180.f; }

float apply_deadband(float v, float deadband) {
  return std::abs(v) < deadband ? 0.f : v;
}

float limit(float v, float max_abs, bool& saturated) {
  const float m = std::max(0.f, max_abs);
  const float out = std::clamp(v, -m, m);
  saturated = saturated || out != v;
  return out;
}

FollowStatus unable_status(const ctrk::SotResult& sot, int64_t t_ns, FollowState state,
                           std::string message) {
  FollowStatus status;
  status.t_ns = t_ns;
  status.state = state;
  status.target_cx = sot.box.cx();
  status.target_cy = sot.box.cy();
  status.target_w = sot.box.w;
  status.target_h = sot.box.h;
  status.score = sot.score;
  status.command_valid = false;
  status.message = std::move(message);
  return status;
}

FollowStatus compute_follow_status(const ctrk::SotResult& sot, const ctrk::BBox& box,
                                   const ctrk::BBox& init_box, const FollowConfig& cfg,
                                   const cv::Size& image_size, int64_t t_ns) {
  FollowStatus status;
  status.t_ns = t_ns;
  status.state = sot.state == ctrk::SotState::Unstable ? kUnstable : kFollowing;
  status.target_cx = box.cx();
  status.target_cy = box.cy();
  status.target_w = box.w;
  status.target_h = box.h;
  status.score = sot.score;

  if (box.w <= 0.f || box.h <= 0.f || init_box.w <= 0.f || init_box.h <= 0.f) {
    status.message = "invalid box geometry";
    return status;
  }
  if (sot.score < cfg.min_score) {
    status.message = "score below min-score";
    return status;
  }
  if (sot.state == ctrk::SotState::Unstable && !cfg.allow_unstable) {
    status.message = "unstable tracking disabled";
    status.state = kUnstable;
    return status;
  }

  const float image_cx = 0.5f * static_cast<float>(image_size.width);
  const float image_cy = 0.5f * static_cast<float>(image_size.height);
  const float nx = (status.target_cx - image_cx) / image_cx;
  const float ny = (image_cy - status.target_cy) / image_cy;
  const float yaw_angle = std::atan(nx * std::tan(0.5f * cfg.hfov_rad));
  const float image_elevation = std::atan(ny * std::tan(0.5f * cfg.vfov_rad));
  const float elevation = image_elevation - cfg.cam_pitch_rad;

  status.estimated_distance = cfg.init_distance * (init_box.h / box.h);
  status.error_x = status.estimated_distance * std::tan(yaw_angle);
  status.error_z = status.estimated_distance * std::tan(elevation);
  status.distance_error = status.estimated_distance - cfg.init_distance;
  status.error_y = status.distance_error;

  bool saturated = false;
  status.vx = limit(cfg.kp_x * apply_deadband(status.error_x, cfg.deadband_x), cfg.max_vx,
                    saturated);
  status.vy = limit(cfg.kp_y * apply_deadband(status.error_y, cfg.deadband_y), cfg.max_vy,
                    saturated);
  status.vz = limit(cfg.kp_z * apply_deadband(status.error_z, cfg.deadband_z), cfg.max_vz,
                    saturated);
  status.yaw_rate =
      limit(cfg.kp_yaw * apply_deadband(yaw_angle, cfg.deadband_yaw_rad), cfg.max_yaw_rate,
            saturated);
  status.command_valid = true;
  status.command_saturated = saturated;
  status.message = sot.state == ctrk::SotState::Unstable ? "following unstable target" : "following";
  return status;
}

void write_status(std::ostream& out, const FollowStatus& s) {
  out << "{\"t_ns\":" << s.t_ns << ",\"state\":" << static_cast<int>(s.state)
      << ",\"target_cx\":" << s.target_cx << ",\"target_cy\":" << s.target_cy
      << ",\"target_w\":" << s.target_w << ",\"target_h\":" << s.target_h
      << ",\"score\":" << s.score << ",\"estimated_distance\":" << s.estimated_distance
      << ",\"error_x\":" << s.error_x << ",\"error_y\":" << s.error_y
      << ",\"error_z\":" << s.error_z << ",\"distance_error\":" << s.distance_error
      << ",\"vx\":" << s.vx << ",\"vy\":" << s.vy << ",\"vz\":" << s.vz
      << ",\"yaw_rate\":" << s.yaw_rate
      << ",\"command_valid\":" << (s.command_valid ? "true" : "false")
      << ",\"command_saturated\":" << (s.command_saturated ? "true" : "false")
      << ",\"message\":\"" << s.message << "\"}\n";
}

}  // namespace

int main(int argc, char** argv) {
  cv::CommandLineParser cli(argc, argv, kUsage);
  cli.about("ctrk drone follow prototype: SOT target -> structured follow status");
  if (cli.has("help") || !cli.has("input") || !cli.has("bbox") || !cli.has("init-distance") ||
      !cli.has("cam-pitch")) {
    cli.printMessage();
    return cli.has("help") ? 0 : 1;
  }

  ctrk::BBox sot_init_box;
  if (!ctrk::app::parse_bbox(cli.get<std::string>("bbox"), sot_init_box)) {
    std::fprintf(stderr, "bad --bbox, expected x,y,w,h\n");
    return 1;
  }

  FollowConfig follow;
  follow.init_distance = cli.get<float>("init-distance");
  if (follow.init_distance <= 0.f) {
    std::fprintf(stderr, "bad --init-distance, expected a positive distance in meters\n");
    return 1;
  }
  if (cli.has("init-box") && !cli.get<std::string>("init-box").empty()) {
    if (!ctrk::app::parse_bbox(cli.get<std::string>("init-box"), follow.init_box)) {
      std::fprintf(stderr, "bad --init-box, expected x,y,w,h\n");
      return 1;
    }
    follow.has_init_box = true;
  }
  follow.cam_pitch_rad = deg_to_rad(cli.get<float>("cam-pitch"));
  follow.hfov_rad = deg_to_rad(cli.get<float>("hfov"));
  follow.vfov_rad = deg_to_rad(cli.get<float>("vfov"));
  follow.lpf_alpha = cli.get<float>("lpf-alpha");
  follow.min_score = cli.get<float>("min-score");
  follow.allow_unstable = !cli.has("no-unstable");
  follow.kp_x = cli.get<float>("kp-x");
  follow.kp_y = cli.get<float>("kp-y");
  follow.kp_z = cli.get<float>("kp-z");
  follow.kp_yaw = cli.get<float>("kp-yaw");
  follow.deadband_x = cli.get<float>("deadband-x");
  follow.deadband_y = cli.get<float>("deadband-y");
  follow.deadband_z = cli.get<float>("deadband-z");
  follow.deadband_yaw_rad = deg_to_rad(cli.get<float>("deadband-yaw"));
  follow.max_vx = cli.get<float>("max-vx");
  follow.max_vy = cli.get<float>("max-vy");
  follow.max_vz = cli.get<float>("max-vz");
  follow.max_yaw_rate = cli.get<float>("max-yaw-rate");
  if (follow.hfov_rad <= 0.f || follow.vfov_rad <= 0.f) {
    std::fprintf(stderr, "bad --hfov/--vfov, expected positive degrees\n");
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
  cfg.engine.intra_op_threads = cli.get<int>("threads");
  cfg.penalty_k = cli.get<float>("penalty-k");
  cfg.window_influence = cli.get<float>("window-influence");
  cfg.size_lr = cli.get<float>("size-lr");
  cfg.template_update_interval = cli.get<int>("template-every");
  cfg.template_blend = cli.get<float>("template-blend");
  if (cli.has("spin")) cfg.engine.allow_spinning = true;
  cfg.engine.use_dnnl = cli.has("dnnl");
  const std::string reid = cli.get<std::string>("reid");
  if (reid == "hsv") {
    cfg.reid.embedder = ctrk::SotConfig::Reid::Embedder::HsvHist;
  } else if (reid == "nanoz") {
    cfg.reid.embedder = ctrk::SotConfig::Reid::Embedder::NanoZ;
  } else if (reid != "none" && !reid.empty()) {
    std::fprintf(stderr, "bad --reid, expected hsv, nanoz or none\n");
    return 1;
  }
  cfg.reid.accept = cli.get<float>("reid-accept");
  cfg.reid.drift_check_every = cli.get<int>("drift-every");
  cfg.reid.drift_thr = cli.get<float>("drift-thr");

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

  std::ofstream status_file;
  const std::string status_path = cli.get<std::string>("status-json");
  if (!status_path.empty()) {
    status_file.open(status_path);
    if (!status_file.is_open()) {
      std::fprintf(stderr, "failed to open --status-json: %s\n", status_path.c_str());
      return 1;
    }
  }

  ctrk::StageTimer timer;
  ctrk::set_profile_sink(
      [&timer](std::string_view stage, double ms) { timer.add_ms(std::string(stage), ms); });

  cv::Mat frame;
  int64_t t_ns = 0;
  bool initialized = false;
  FilteredBox filtered;

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
        tracker->init(view, sot_init_box);
        result = {sot_init_box, 1.f, ctrk::SotState::Tracking};
        initialized = true;
        if (!follow.has_init_box) {
          follow.init_box = result.box;
          follow.has_init_box = true;
        }
      } else {
        result = tracker->update(view);
      }
    }

    FollowStatus status;
    if (result.state == ctrk::SotState::Lost) {
      status = unable_status(result, t_ns, kLost, "sot lost");
      filtered.ready = false;
    } else if (!follow.has_init_box) {
      status = unable_status(result, t_ns, kUnable, "missing init box");
    } else {
      const ctrk::BBox smooth_box = filtered.update(result.box, follow.lpf_alpha);
      status = compute_follow_status(result, smooth_box, follow.init_box, follow, src->size(), t_ns);
    }

    if (status_file.is_open()) write_status(status_file, status);

    {
      const auto scope = timer.scope("draw");
      char label[96];
      std::snprintf(label, sizeof(label), "follow s=%d %.2f d=%.2f",
                    static_cast<int>(status.state), status.score, status.estimated_distance);
      const cv::Scalar color = status.command_valid ? cv::Scalar(0, 220, 80) : cv::Scalar(0, 0, 255);
      ctrk::app::draw_box(frame, result.box, color, label);
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
