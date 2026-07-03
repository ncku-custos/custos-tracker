#pragma once

#include <fstream>
#include <memory>
#include <optional>

#include <ctrk/sot.hpp>
#include <ctrk_interfaces/msg/sot_status.hpp>
#include <ctrk_interfaces/srv/set_target.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "bench_stats.hpp"
#include "param_config.hpp"
#include "sink_bridge.hpp"

namespace ctrk_ros {

// Single-object-tracker lifecycle component: `image` (bgr8) in, `target`
// (ctrk_interfaces/SotStatus) out, target (re-)init via the `set_target`
// service (or the init_bbox param for CLI-exact replay), `reset` back to
// IDLE. While ACTIVE with no target set, every frame publishes STATE_IDLE
// (downstream liveness) and is cached so SetTarget can init immediately on
// the last seen frame. All callbacks share the node's mutually-exclusive
// group — the single-threaded core object needs no further locking.
class SotNode : public rclcpp_lifecycle::LifecycleNode {
 public:
  explicit SotNode(const rclcpp::NodeOptions& options);

  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) override;

 private:
  void on_image(sensor_msgs::msg::Image::ConstSharedPtr msg);
  void on_set_target(const ctrk_interfaces::srv::SetTarget::Request::SharedPtr req,
                     ctrk_interfaces::srv::SetTarget::Response::SharedPtr res);
  void on_reset(const std_srvs::srv::Trigger::Request::SharedPtr req,
                std_srvs::srv::Trigger::Response::SharedPtr res);
  void dump_box(const ctrk::BBox& b);

  SotNodeParams params_;
  std::optional<ctrk::SotTracker> tracker_;
  rclcpp_lifecycle::LifecyclePublisher<ctrk_interfaces::msg::SotStatus>::SharedPtr target_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Service<ctrk_interfaces::srv::SetTarget>::SharedPtr set_target_srv_;
  rclcpp::Service<std_srvs::srv::Trigger>::SharedPtr reset_srv_;

  sensor_msgs::msg::Image::ConstSharedPtr last_frame_;  // cached while IDLE
  std::optional<ctrk::BBox> pending_init_;              // armed by init_bbox or early SetTarget
  bool target_set_ = false;
  std::ofstream dump_;
  int frame_no_ = 0;
  BenchStats bench_;
  std::optional<ProfileCollector> profile_collector_;
};

}  // namespace ctrk_ros
