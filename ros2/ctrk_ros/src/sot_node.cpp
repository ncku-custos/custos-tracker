#include "sot_node.hpp"

#include <chrono>
#include <cstdio>
#include <exception>
#include <utility>

#include <ctrk/detector.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp_components/register_node_macro.hpp>

#include "frame_view.hpp"
#include "msg_convert.hpp"

namespace ctrk_ros {

SotNode::SotNode(const rclcpp::NodeOptions& options) : LifecycleNode("ctrk_sot_node", options) {}

SotNode::CallbackReturn SotNode::on_configure(const rclcpp_lifecycle::State&) {
  install_process_sinks();
  try {
    params_ = declare_sot_params(*this);
    tracker_.emplace(params_.cfg);  // ORT session loads + warmup (nano backend)
    if (params_.reacquire.enable)
      tracker_->enable_reacquire(ctrk::make_yolov8_detector(params_.reacquire.detector),
                                 params_.reacquire.config);
  } catch (const std::exception& e) {
    RCLCPP_ERROR(get_logger(), "configure failed: %s", e.what());
    return CallbackReturn::FAILURE;
  }
  target_pub_ =
      create_publisher<ctrk_interfaces::msg::SotStatus>("target", rclcpp::QoS(10).reliable());
  set_target_srv_ = create_service<ctrk_interfaces::srv::SetTarget>(
      "set_target", [this](ctrk_interfaces::srv::SetTarget::Request::SharedPtr req,
                           ctrk_interfaces::srv::SetTarget::Response::SharedPtr res) {
        on_set_target(req, res);
      });
  reset_srv_ = create_service<std_srvs::srv::Trigger>(
      "reset", [this](std_srvs::srv::Trigger::Request::SharedPtr req,
                      std_srvs::srv::Trigger::Response::SharedPtr res) { on_reset(req, res); });
  RCLCPP_INFO(get_logger(), "configured: backend=%s reacquire=%s",
              params_.cfg.backend == ctrk::SotBackend::Mosse ? "mosse" : "nano",
              params_.reacquire.enable ? "on" : "off");
  return CallbackReturn::SUCCESS;
}

SotNode::CallbackReturn SotNode::on_activate(const rclcpp_lifecycle::State& state) {
  LifecycleNode::on_activate(state);
  frame_no_ = 0;
  target_set_ = false;
  last_frame_.reset();
  pending_init_.reset();
  if (!params_.init_bbox.empty()) {
    // CLI-exact replay path: behaves as a SetTarget armed before frame one.
    pending_init_ = ctrk::BBox{
        static_cast<float>(params_.init_bbox[0]), static_cast<float>(params_.init_bbox[1]),
        static_cast<float>(params_.init_bbox[2]), static_cast<float>(params_.init_bbox[3])};
  }
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

SotNode::CallbackReturn SotNode::on_deactivate(const rclcpp_lifecycle::State& state) {
  image_sub_.reset();
  LifecycleNode::on_deactivate(state);
  last_frame_.reset();
  if (dump_.is_open()) dump_.close();
  profile_collector_.reset();
  if (params_.common.bench && !bench_.empty())
    RCLCPP_INFO(get_logger(), "bench (%d frames)\n%s", frame_no_, bench_.table().c_str());
  return CallbackReturn::SUCCESS;
}

SotNode::CallbackReturn SotNode::on_cleanup(const rclcpp_lifecycle::State&) {
  tracker_.reset();
  target_pub_.reset();
  set_target_srv_.reset();
  reset_srv_.reset();
  return CallbackReturn::SUCCESS;
}

SotNode::CallbackReturn SotNode::on_shutdown(const rclcpp_lifecycle::State&) {
  image_sub_.reset();
  tracker_.reset();
  target_pub_.reset();
  set_target_srv_.reset();
  reset_srv_.reset();
  return CallbackReturn::SUCCESS;
}

void SotNode::dump_box(const ctrk::BBox& b) {
  if (!dump_.is_open()) return;
  char line[96];
  std::snprintf(line, sizeof(line), "%.2f,%.2f,%.2f,%.2f\n", b.x, b.y, b.w, b.h);
  dump_ << line;
}

void SotNode::on_image(sensor_msgs::msg::Image::ConstSharedPtr msg) {
  const auto view = to_frame_view(*msg);
  if (!view) {
    RCLCPP_WARN_THROTTLE(get_logger(), *get_clock(), 5000,
                         "dropping frame: encoding '%s' (bgr8 only, D-0016)",
                         msg->encoding.c_str());
    return;
  }
  ++frame_no_;

  const auto t0 = std::chrono::steady_clock::now();
  ctrk::SotResult result;
  if (!target_set_) {
    if (pending_init_) {
      tracker_->init(*view, *pending_init_);
      result = {*pending_init_, 1.f, ctrk::SotState::Tracking};  // the CLI's init-frame synthesis
      pending_init_.reset();
      target_set_ = true;
    } else {
      last_frame_ = msg;
      target_pub_->publish(idle_status(msg->header));
      return;
    }
  } else {
    result = tracker_->update(*view);
  }
  dump_box(result.box);
  target_pub_->publish(to_sot_status(result, msg->header));

  if (params_.common.bench) {
    const auto dt = std::chrono::steady_clock::now() - t0;
    bench_.add("cb.update+publish", std::chrono::duration<double, std::milli>(dt).count());
    bench_.add("e2e.stamp_to_pub", (now() - rclcpp::Time(msg->header.stamp)).seconds() * 1e3);
  }
}

void SotNode::on_set_target(const ctrk_interfaces::srv::SetTarget::Request::SharedPtr req,
                            ctrk_interfaces::srv::SetTarget::Response::SharedPtr res) {
  if (get_current_state().id() != lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE) {
    res->accepted = false;
    res->message = "node not active";
    return;
  }
  if (req->w <= 0.f || req->h <= 0.f) {
    res->accepted = false;
    res->message = "w and h must be positive";
    return;
  }
  const ctrk::BBox box{req->x, req->y, req->w, req->h};
  if (last_frame_) {
    // Init immediately on the cached last frame; its next update starts there.
    const auto view = to_frame_view(*last_frame_);
    tracker_->init(*view, box);
    dump_box(box);  // the init frame dumps its box, like the CLI
    target_set_ = true;
    pending_init_.reset();
    res->message = "initialized on cached frame";
  } else {
    pending_init_ = box;
    target_set_ = false;
    res->message = "armed; will initialize on the next frame";
  }
  res->accepted = true;
}

void SotNode::on_reset(const std_srvs::srv::Trigger::Request::SharedPtr,
                       std_srvs::srv::Trigger::Response::SharedPtr res) {
  target_set_ = false;
  pending_init_.reset();
  last_frame_.reset();
  res->success = true;
  res->message = "idle";
}

}  // namespace ctrk_ros

RCLCPP_COMPONENTS_REGISTER_NODE(ctrk_ros::SotNode)
