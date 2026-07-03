#include "tbd_node.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <string>
#include <utility>

#include <rclcpp_components/register_node_macro.hpp>

#include "frame_view.hpp"
#include "msg_convert.hpp"

namespace ctrk_ros {

TbdNode::TbdNode(const rclcpp::NodeOptions& options) : LifecycleNode("ctrk_tbd_node", options) {}

TbdNode::CallbackReturn TbdNode::on_configure(const rclcpp_lifecycle::State&) {
  install_process_sinks();
  try {
    params_ = declare_tbd_params(*this);
    tracker_.emplace(params_.cfg);  // ORT session load + warmup
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "configure failed: %s", e.what());
    return CallbackReturn::FAILURE;
  }
  tracks_pub_ =
      create_publisher<vision_msgs::msg::Detection2DArray>("tracks", rclcpp::QoS(10).reliable());
  RCLCPP_INFO(get_logger(), "configured: model=%s detect_interval=%d gmc=%s",
              params_.cfg.detector.model_path.c_str(), params_.cfg.detect_interval,
              params_.cfg.gmc == ctrk::GmcMethod::Off ? "off" : "sparse_flow");
  return CallbackReturn::SUCCESS;
}

TbdNode::CallbackReturn TbdNode::on_activate(const rclcpp_lifecycle::State& state) {
  LifecycleNode::on_activate(state);  // activates the managed lifecycle publisher
  frame_no_ = 0;
  if (!params_.common.dump_path.empty()) {
    dump_.open(params_.common.dump_path);
    if (!dump_.is_open()) {
      RCLCPP_ERROR(get_logger(), "cannot open dump_path %s", params_.common.dump_path.c_str());
      return CallbackReturn::FAILURE;
    }
  }
  if (params_.common.bench) {
    bench_ = BenchStats{};
    profile_collector_.emplace(
        [this](std::string_view stage, double ms) { bench_.add(stage, ms); });
  }
  rclcpp::QoS qos(params_.common.qos_depth);
  if (params_.common.qos_reliability == "reliable")
    qos.reliable();
  else
    qos.best_effort();
  image_sub_ = create_subscription<sensor_msgs::msg::Image>(
      "image", qos, [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { on_image(msg); });
  return CallbackReturn::SUCCESS;
}

TbdNode::CallbackReturn TbdNode::on_deactivate(const rclcpp_lifecycle::State& state) {
  image_sub_.reset();
  LifecycleNode::on_deactivate(state);
  if (dump_.is_open()) dump_.close();
  profile_collector_.reset();
  if (params_.common.bench && !bench_.empty())
    RCLCPP_INFO(get_logger(), "bench (%d frames)\n%s", frame_no_, bench_.table().c_str());
  return CallbackReturn::SUCCESS;
}

TbdNode::CallbackReturn TbdNode::on_cleanup(const rclcpp_lifecycle::State&) {
  tracker_.reset();
  tracks_pub_.reset();
  return CallbackReturn::SUCCESS;
}

TbdNode::CallbackReturn TbdNode::on_shutdown(const rclcpp_lifecycle::State&) {
  image_sub_.reset();
  tracker_.reset();
  tracks_pub_.reset();
  return CallbackReturn::SUCCESS;
}

void TbdNode::on_image(sensor_msgs::msg::Image::ConstSharedPtr msg) {
  const auto view = to_frame_view(*msg);
  if (!view) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "dropping frame: encoding '%s' (bgr8 only, D-0016)",
                         msg->encoding.c_str());
    return;
  }
  ++frame_no_;  // MOT frame numbers are 1-based

  const auto t0 = std::chrono::steady_clock::now();
  const auto tracks = tracker_->update(*view);
  tracks_pub_->publish(to_detections(tracks, msg->header, params_.publish_lost));

  if (dump_.is_open()) {
    for (const auto& t : tracks) {
      if (t.state != ctrk::TrackState::Confirmed) continue;
      char line[128];
      std::snprintf(line, sizeof(line), "%d,%d,%.2f,%.2f,%.2f,%.2f,%.2f,-1,-1,-1\n", frame_no_,
                    t.id, t.box.x, t.box.y, t.box.w, t.box.h, t.score);
      dump_ << line;
    }
  }
  if (params_.common.bench) {
    const auto dt = std::chrono::steady_clock::now() - t0;
    bench_.add("cb.update+publish", std::chrono::duration<double, std::milli>(dt).count());
    // Meaningful only when the producer stamps with the same system clock
    // (frames_player stamp_source=clock); index-stamped parity runs ignore it.
    bench_.add("e2e.stamp_to_pub", (now() - rclcpp::Time(msg->header.stamp)).seconds() * 1e3);
  }
}

}  // namespace ctrk_ros

RCLCPP_COMPONENTS_REGISTER_NODE(ctrk_ros::TbdNode)
