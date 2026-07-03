#pragma once

#include <string>
#include <vector>

#include <ctrk/sot.hpp>
#include <ctrk/tbd.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>

namespace ctrk_ros {

// Node-level knobs shared by both tracker nodes (not part of the core configs).
struct CommonParams {
  std::string dump_path;                        // per-frame csv, CLI --dump format; "" = off
  std::string qos_reliability = "best_effort";  // image sub: "best_effort" | "reliable"
  int qos_depth = 1;                            // image sub queue: 1 = latest frame wins
  bool bench = false;                           // collect per-stage latency, log on deactivate
};

struct TbdNodeParams {
  ctrk::TbdConfig cfg;
  bool publish_lost = false;  // false = Confirmed only (matches the CLI draw/dump loop)
  CommonParams common;
};

// The app-side reacquire wiring: detector construction + gates
// (SotTracker::enable_reacquire takes ownership of the detector).
struct ReacquireParams {
  bool enable = false;
  ctrk::Yolov8Config detector;
  ctrk::ReacquireConfig config;
};

struct SotNodeParams {
  ctrk::SotConfig cfg;
  ReacquireParams reacquire;
  std::vector<double> init_bbox;  // [x,y,w,h] arms init on the first frame; empty = SetTarget
  CommonParams common;
};

// Declare every parameter on `node` (names mirror the config-struct paths,
// defaults equal the struct defaults; deliberate exceptions are documented at
// the declaration site) and return the assembled configs. Throws
// std::invalid_argument on a bad enum string — on_configure turns that into a
// FAILURE transition.
TbdNodeParams declare_tbd_params(rclcpp_lifecycle::LifecycleNode& node);
SotNodeParams declare_sot_params(rclcpp_lifecycle::LifecycleNode& node);

}  // namespace ctrk_ros
