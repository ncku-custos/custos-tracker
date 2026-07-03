// The CI anchor: a full pub/sub -> track -> publish e2e through the SOT
// lifecycle component with the model-free MOSSE backend — no downloads, no
// skips. Covers SetTarget (armed and cached-frame paths), state pass-through,
// reset -> IDLE, deactivate stops publishing, SetTarget refused off-ACTIVE.
#include <gtest/gtest.h>

#include <chrono>
#include <memory>
#include <vector>

#include <ctrk_interfaces/msg/sot_status.hpp>
#include <ctrk_interfaces/srv/set_target.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <std_srvs/srv/trigger.hpp>

#include "sot_node.hpp"
#include "support/synth_frames.hpp"

namespace {

using ctrk_interfaces::msg::SotStatus;

void ensure_rcl() {
  if (!rclcpp::ok()) rclcpp::init(0, nullptr);
}

}  // namespace

TEST(SotLifecycleMosse, FullPipelineTracksResetsAndStopsCleanly) {
  ensure_rcl();
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(
      {{"backend", "mosse"}, {"image_qos_reliability", "reliable"}, {"image_qos_depth", 50}});
  auto node = std::make_shared<ctrk_ros::SotNode>(opts);
  auto helper = std::make_shared<rclcpp::Node>("sot_test_helper");

  std::vector<SotStatus> statuses;
  auto sub = helper->create_subscription<SotStatus>(
      "target", rclcpp::QoS(50).reliable(),
      [&statuses](SotStatus::ConstSharedPtr m) { statuses.push_back(*m); });
  auto image_pub =
      helper->create_publisher<sensor_msgs::msg::Image>("image", rclcpp::QoS(50).reliable());
  auto set_target = helper->create_client<ctrk_interfaces::srv::SetTarget>("set_target");
  auto reset = helper->create_client<std_srvs::srv::Trigger>("reset");

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper);

  const ctrk_ros::test::SynthScene scene;
  int frame_idx = 0;
  const auto push_frame = [&](size_t expect_statuses) {
    image_pub->publish(scene.frame(frame_idx++));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (statuses.size() < expect_statuses && std::chrono::steady_clock::now() < deadline)
      exec.spin_some(std::chrono::milliseconds(20));
    ASSERT_GE(statuses.size(), expect_statuses);
  };

  // SetTarget while unconfigured/inactive is refused.
  ASSERT_EQ(node->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  {
    auto req = std::make_shared<ctrk_interfaces::srv::SetTarget::Request>();
    req->x = scene.x0;
    req->y = scene.y0;
    req->w = static_cast<float>(scene.target_size);
    req->h = static_cast<float>(scene.target_size);
    auto fut = set_target->async_send_request(req);
    ASSERT_EQ(exec.spin_until_future_complete(fut, std::chrono::seconds(10)),
              rclcpp::FutureReturnCode::SUCCESS);
    EXPECT_FALSE(fut.get()->accepted);
  }

  ASSERT_EQ(node->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  // No target yet: frames come back IDLE (and get cached for SetTarget).
  push_frame(1);
  EXPECT_EQ(statuses.back().state, SotStatus::STATE_IDLE);

  // Degenerate box refused even while active.
  {
    auto req = std::make_shared<ctrk_interfaces::srv::SetTarget::Request>();
    req->w = -1.f;
    req->h = 10.f;
    auto fut = set_target->async_send_request(req);
    ASSERT_EQ(exec.spin_until_future_complete(fut, std::chrono::seconds(10)),
              rclcpp::FutureReturnCode::SUCCESS);
    EXPECT_FALSE(fut.get()->accepted);
  }

  // Real target: inits on the cached frame (frame 0 geometry).
  {
    auto req = std::make_shared<ctrk_interfaces::srv::SetTarget::Request>();
    req->x = scene.target_x(0);
    req->y = scene.target_y(0);
    req->w = static_cast<float>(scene.target_size);
    req->h = static_cast<float>(scene.target_size);
    auto fut = set_target->async_send_request(req);
    ASSERT_EQ(exec.spin_until_future_complete(fut, std::chrono::seconds(10)),
              rclcpp::FutureReturnCode::SUCCESS);
    EXPECT_TRUE(fut.get()->accepted);
  }

  // Track 20 frames: state passes through as TRACKING and the box follows
  // the +2 px/frame translation.
  constexpr int kFrames = 20;
  int tracking = 0;
  for (int i = 0; i < kFrames; ++i) {
    push_frame(statuses.size() + 1);
    if (statuses.back().state == SotStatus::STATE_TRACKING) ++tracking;
  }
  EXPECT_GE(tracking, kFrames * 8 / 10) << "MOSSE lost a rigid textured square";
  const auto& final_status = statuses.back();
  const float expected_x = scene.target_x(frame_idx - 1);
  EXPECT_NEAR(final_status.x, expected_x, static_cast<float>(scene.target_size) / 2.f)
      << "final box did not follow the target";
  EXPECT_GT(final_status.x, scene.x0 + 10.f) << "box never moved off the init position";

  // reset -> IDLE again.
  {
    auto req = std::make_shared<std_srvs::srv::Trigger::Request>();
    auto fut = reset->async_send_request(req);
    ASSERT_EQ(exec.spin_until_future_complete(fut, std::chrono::seconds(10)),
              rclcpp::FutureReturnCode::SUCCESS);
    EXPECT_TRUE(fut.get()->success);
  }
  push_frame(statuses.size() + 1);
  EXPECT_EQ(statuses.back().state, SotStatus::STATE_IDLE);

  // Deactivate: frames are dropped, nothing more publishes.
  ASSERT_EQ(node->deactivate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  const auto count_before = statuses.size();
  image_pub->publish(scene.frame(frame_idx++));
  exec.spin_some(std::chrono::milliseconds(100));
  EXPECT_EQ(statuses.size(), count_before);

  ASSERT_EQ(node->cleanup().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST(SotLifecycleMosse, InitBboxParamArmsInitOnFirstFrame) {
  ensure_rcl();
  const ctrk_ros::test::SynthScene scene;
  rclcpp::NodeOptions opts;
  opts.parameter_overrides(
      {{"backend", "mosse"},
       {"image_qos_reliability", "reliable"},
       {"image_qos_depth", 50},
       {"init_bbox",
        std::vector<double>{static_cast<double>(scene.x0), static_cast<double>(scene.y0),
                            static_cast<double>(scene.target_size),
                            static_cast<double>(scene.target_size)}}});
  auto node = std::make_shared<ctrk_ros::SotNode>(opts);
  auto helper = std::make_shared<rclcpp::Node>("sot_test_helper2");

  std::vector<SotStatus> statuses;
  auto sub = helper->create_subscription<SotStatus>(
      "target", rclcpp::QoS(50).reliable(),
      [&statuses](SotStatus::ConstSharedPtr m) { statuses.push_back(*m); });
  auto image_pub =
      helper->create_publisher<sensor_msgs::msg::Image>("image", rclcpp::QoS(50).reliable());

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper);

  ASSERT_EQ(node->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  // First frame: no IDLE — the armed init consumes it and reports TRACKING
  // with the exact init box (the CLI's synthesized init-frame result).
  image_pub->publish(scene.frame(0));
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
  while (statuses.empty() && std::chrono::steady_clock::now() < deadline)
    exec.spin_some(std::chrono::milliseconds(20));
  ASSERT_EQ(statuses.size(), 1u);
  EXPECT_EQ(statuses[0].state, SotStatus::STATE_TRACKING);
  EXPECT_FLOAT_EQ(statuses[0].x, scene.x0);
  EXPECT_FLOAT_EQ(statuses[0].score, 1.f);

  node->deactivate();
  node->cleanup();
}
