#include "param_config.hpp"

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace ctrk_ros {

namespace {

using Node = rclcpp_lifecycle::LifecycleNode;

double get_double(Node& n, const std::string& name, double def) {
  return n.declare_parameter<double>(name, def);
}
float get_float(Node& n, const std::string& name, float def) {
  return static_cast<float>(get_double(n, name, static_cast<double>(def)));
}
int get_int(Node& n, const std::string& name, int def) {
  return static_cast<int>(n.declare_parameter<int64_t>(name, def));
}
bool get_bool(Node& n, const std::string& name, bool def) {
  return n.declare_parameter<bool>(name, def);
}
std::string get_string(Node& n, const std::string& name, const std::string& def) {
  return n.declare_parameter<std::string>(name, def);
}

[[noreturn]] void bad_enum(const std::string& name, const std::string& value) {
  throw std::invalid_argument("parameter '" + name + "': unknown value '" + value + "'");
}

ctrk::EngineOptions declare_engine(Node& n, const std::string& prefix,
                                   const ctrk::EngineOptions& def) {
  ctrk::EngineOptions e;
  e.intra_op_threads = get_int(n, prefix + ".intra_op_threads", def.intra_op_threads);
  e.allow_spinning = get_bool(n, prefix + ".allow_spinning", def.allow_spinning);
  e.warmup = get_bool(n, prefix + ".warmup", def.warmup);
  e.use_dnnl = get_bool(n, prefix + ".use_dnnl", def.use_dnnl);
  return e;
}

ctrk::Yolov8Config declare_detector(Node& n, const std::string& prefix,
                                    const ctrk::Yolov8Config& def) {
  ctrk::Yolov8Config d;
  d.model_path = get_string(n, prefix + ".model_path", def.model_path);
  d.conf_thr = get_float(n, prefix + ".conf_thr", def.conf_thr);
  d.nms_iou = get_float(n, prefix + ".nms_iou", def.nms_iou);
  const auto classes =
      n.declare_parameter<std::vector<int64_t>>(prefix + ".keep_classes", std::vector<int64_t>{});
  for (int64_t c : classes) d.keep_classes.push_back(static_cast<int>(c));
  d.engine = declare_engine(n, prefix + ".engine", def.engine);
  return d;
}

CommonParams declare_common(Node& n) {
  CommonParams c;
  c.dump_path = get_string(n, "dump_path", "");
  c.qos_reliability = get_string(n, "image_qos_reliability", c.qos_reliability);
  if (c.qos_reliability != "best_effort" && c.qos_reliability != "reliable")
    bad_enum("image_qos_reliability", c.qos_reliability);
  c.qos_depth = get_int(n, "image_qos_depth", c.qos_depth);
  c.bench = get_bool(n, "bench", c.bench);
  return c;
}

}  // namespace

TbdNodeParams declare_tbd_params(rclcpp_lifecycle::LifecycleNode& node) {
  TbdNodeParams p;
  ctrk::TbdConfig def;
  def.detector.model_path = "models/cache/yolov8n_640.onnx";  // the CLI --model default

  p.cfg.detector = declare_detector(node, "detector", def.detector);

  auto& a = p.cfg.assoc;
  const ctrk::AssocConfig ad;
  a.use_byte = get_bool(node, "assoc.use_byte", ad.use_byte);
  a.track_thresh = get_float(node, "assoc.track_thresh", ad.track_thresh);
  a.match_thresh_high = get_float(node, "assoc.match_thresh_high", ad.match_thresh_high);
  a.match_thresh_low = get_float(node, "assoc.match_thresh_low", ad.match_thresh_low);
  a.tentative_match_thresh =
      get_float(node, "assoc.tentative_match_thresh", ad.tentative_match_thresh);
  a.new_track_thresh = get_float(node, "assoc.new_track_thresh", ad.new_track_thresh);
  a.n_init = get_int(node, "assoc.n_init", ad.n_init);
  a.max_age = get_int(node, "assoc.max_age", ad.max_age);
  a.nsa = get_bool(node, "assoc.nsa", ad.nsa);
  a.tentative_relax_per_coast =
      get_float(node, "assoc.tentative_relax_per_coast", ad.tentative_relax_per_coast);
  a.tentative_gate_floor = get_float(node, "assoc.tentative_gate_floor", ad.tentative_gate_floor);
  a.tentative_patience = get_int(node, "assoc.tentative_patience", ad.tentative_patience);
  a.velocity_seed = get_bool(node, "assoc.velocity_seed", ad.velocity_seed);
  a.appearance_weight = get_float(node, "assoc.appearance_weight", ad.appearance_weight);
  a.embedding_ema = get_float(node, "assoc.embedding_ema", ad.embedding_ema);

  const auto gmc = get_string(node, "gmc", "off");
  if (gmc == "off")
    p.cfg.gmc = ctrk::GmcMethod::Off;
  else if (gmc == "sparse_flow")
    p.cfg.gmc = ctrk::GmcMethod::SparseFlow;
  else
    bad_enum("gmc", gmc);

  p.cfg.nominal_fps = get_double(node, "nominal_fps", def.nominal_fps);
  p.cfg.detect_interval = get_int(node, "detect_interval", def.detect_interval);

  p.publish_lost = get_bool(node, "publish_lost", false);
  p.common = declare_common(node);
  return p;
}

SotNodeParams declare_sot_params(rclcpp_lifecycle::LifecycleNode& node) {
  SotNodeParams p;
  const ctrk::SotConfig def;

  const auto backend = get_string(node, "backend", "nano");
  if (backend == "nano")
    p.cfg.backend = ctrk::SotBackend::NanoTrack;
  else if (backend == "mosse")
    p.cfg.backend = ctrk::SotBackend::Mosse;
  else
    bad_enum("backend", backend);

  // The CLI --backbone-z/--backbone-x/--head defaults.
  p.cfg.backbone_z_path =
      get_string(node, "backbone_z_path", "models/cache/nanotrack_backbone_z.onnx");
  p.cfg.backbone_x_path =
      get_string(node, "backbone_x_path", "models/cache/nanotrack_backbone_x.onnx");
  p.cfg.head_path = get_string(node, "head_path", "models/cache/nanotrack_head.onnx");
  p.cfg.engine = declare_engine(node, "engine", def.engine);

  p.cfg.penalty_k = get_float(node, "penalty_k", def.penalty_k);
  p.cfg.window_influence = get_float(node, "window_influence", def.window_influence);
  p.cfg.size_lr = get_float(node, "size_lr", def.size_lr);
  p.cfg.context_amount = get_float(node, "context_amount", def.context_amount);
  p.cfg.lost_score_thr = get_float(node, "lost_score_thr", def.lost_score_thr);
  p.cfg.lost_patience = get_int(node, "lost_patience", def.lost_patience);
  p.cfg.template_update_interval =
      get_int(node, "template_update_interval", def.template_update_interval);
  p.cfg.template_refresh_thr = get_float(node, "template_refresh_thr", def.template_refresh_thr);
  p.cfg.template_blend = get_float(node, "template_blend", def.template_blend);

  // Default "hsv" diverges from the struct default None on purpose: it mirrors
  // the CLI --reid operating default (RESULTS.md S3.3 — the re-lock veto is
  // a measured keep). "none" restores pure geometry.
  const auto reid = get_string(node, "reid.embedder", "hsv");
  if (reid == "hsv")
    p.cfg.reid.embedder = ctrk::SotConfig::Reid::Embedder::HsvHist;
  else if (reid == "nanoz")
    p.cfg.reid.embedder = ctrk::SotConfig::Reid::Embedder::NanoZ;
  else if (reid == "none")
    p.cfg.reid.embedder = ctrk::SotConfig::Reid::Embedder::None;
  else
    bad_enum("reid.embedder", reid);
  p.cfg.reid.accept = get_float(node, "reid.accept", def.reid.accept);
  p.cfg.reid.drift_check_every =
      get_int(node, "reid.drift_check_every", def.reid.drift_check_every);
  p.cfg.reid.drift_thr = get_float(node, "reid.drift_thr", def.reid.drift_thr);

  p.reacquire.enable = get_bool(node, "reacquire.enable", false);
  {
    ctrk::Yolov8Config det_def;
    det_def.model_path = "models/cache/yolov8n_640.onnx";  // the CLI --det-model default
    det_def.conf_thr = 0.25f;  // the CLI's (formerly hardcoded) reacquire detector floor
    p.reacquire.detector = declare_detector(node, "reacquire.detector", det_def);

    auto& r = p.reacquire.config;
    const ctrk::ReacquireConfig rd;
    r.class_id = get_int(node, "reacquire.class_id", 0);  // CLI --class default (person), not -1
    r.detect_every = get_int(node, "reacquire.detect_every", rd.detect_every);
    r.min_score = get_float(node, "reacquire.min_score", rd.min_score);
    r.size_low = get_float(node, "reacquire.size_low", rd.size_low);
    r.size_high = get_float(node, "reacquire.size_high", rd.size_high);
    r.base_radius_frac = get_float(node, "reacquire.base_radius_frac", rd.base_radius_frac);
    r.growth_per_frame = get_float(node, "reacquire.growth_per_frame", rd.growth_per_frame);
  }

  p.init_bbox = node.declare_parameter<std::vector<double>>("init_bbox", std::vector<double>{});
  if (!p.init_bbox.empty() &&
      (p.init_bbox.size() != 4 || p.init_bbox[2] <= 0.0 || p.init_bbox[3] <= 0.0))
    throw std::invalid_argument("parameter 'init_bbox': expected [x,y,w,h] with w,h > 0");

  p.common = declare_common(node);
  return p;
}

}  // namespace ctrk_ros
