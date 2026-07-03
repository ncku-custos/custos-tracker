#pragma once

#include <fstream>
#include <memory>
#include <optional>

#include <ctrk/tbd.hpp>
#include <rclcpp_lifecycle/lifecycle_node.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "bench_stats.hpp"
#include "param_config.hpp"
#include "sink_bridge.hpp"

namespace ctrk_ros {

// Tracking-by-detection lifecycle component: `image` (bgr8) in, `tracks`
// (vision_msgs/Detection2DArray, Confirmed only by default) out.
// on_configure constructs the core tracker from parameters (ORT load +
// warmup happens there, deliberately not in the constructor); the image
// subscription only exists while ACTIVE. All callbacks run in the node's
// default mutually-exclusive group — the single-threaded core object needs
// no further locking.
class TbdNode : public rclcpp_lifecycle::LifecycleNode {
 public:
  explicit TbdNode(const rclcpp::NodeOptions& options);

  using CallbackReturn = rclcpp_lifecycle::node_interfaces::LifecycleNodeInterface::CallbackReturn;
  CallbackReturn on_configure(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_activate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_deactivate(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_cleanup(const rclcpp_lifecycle::State& state) override;
  CallbackReturn on_shutdown(const rclcpp_lifecycle::State& state) override;

 private:
  void on_image(sensor_msgs::msg::Image::ConstSharedPtr msg);

  TbdNodeParams params_;
  std::optional<ctrk::MultiTracker> tracker_;
  rclcpp_lifecycle::LifecyclePublisher<vision_msgs::msg::Detection2DArray>::SharedPtr tracks_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  std::ofstream dump_;
  int frame_no_ = 0;  // 1-based on the wire, MOT convention
  BenchStats bench_;
  std::optional<ProfileCollector> profile_collector_;
};

}  // namespace ctrk_ros
