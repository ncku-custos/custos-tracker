// TBD lifecycle component: configure fails cleanly on a bad model path
// (always runs), and with the fetched YOLO model the full pub/sub pipeline
// answers every image with a Detection2DArray (skips when models absent,
// like the core oracle tests).
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <lifecycle_msgs/msg/state.hpp>
#include <rclcpp/rclcpp.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include "tbd_node.hpp"

namespace {

constexpr const char* kYoloModel = CTRK_REPO_ROOT "/models/cache/yolov8n_640.onnx";

void ensure_rcl() {
  if (!rclcpp::ok()) rclcpp::init(0, nullptr);
}

sensor_msgs::msg::Image make_frame(int idx, int width = 320, int height = 240) {
  sensor_msgs::msg::Image img;
  img.encoding = "bgr8";
  img.width = width;
  img.height = height;
  img.step = static_cast<uint32_t>(width * 3);
  img.data.resize(img.step * height);
  std::iota(img.data.begin(), img.data.end(), static_cast<uint8_t>(idx));  // arbitrary texture
  const auto t_ns = static_cast<int64_t>(static_cast<double>(idx) / 30.0 * 1e9);
  img.header.stamp.sec = static_cast<int32_t>(t_ns / 1000000000LL);
  img.header.stamp.nanosec = static_cast<uint32_t>(t_ns % 1000000000LL);
  return img;
}

}  // namespace

TEST(TbdLifecycle, BadModelPathFailsConfigureAndStaysUnconfigured) {
  ensure_rcl();
  rclcpp::NodeOptions opts;
  opts.parameter_overrides({{"detector.model_path", "/nonexistent/model.onnx"}});
  auto node = std::make_shared<ctrk_ros::TbdNode>(opts);

  const auto state = node->configure();
  EXPECT_EQ(state.id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}

TEST(TbdLifecycle, EveryFrameInProducesOneTracksMessageOut) {
  if (!std::filesystem::exists(kYoloModel))
    GTEST_SKIP() << "model missing - run models/get_models.sh + export_yolo.py";
  ensure_rcl();

  rclcpp::NodeOptions opts;
  opts.parameter_overrides({{"detector.model_path", std::string(kYoloModel)},
                            {"image_qos_reliability", "reliable"},
                            {"image_qos_depth", 10}});
  auto node = std::make_shared<ctrk_ros::TbdNode>(opts);
  auto helper = std::make_shared<rclcpp::Node>("tbd_test_helper");

  int received = 0;
  auto sub = helper->create_subscription<vision_msgs::msg::Detection2DArray>(
      "tracks", rclcpp::QoS(10).reliable(),
      [&received](vision_msgs::msg::Detection2DArray::ConstSharedPtr) { ++received; });
  auto pub = helper->create_publisher<sensor_msgs::msg::Image>("image", rclcpp::QoS(10).reliable());

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(node->get_node_base_interface());
  exec.add_node(helper);

  ASSERT_EQ(node->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(node->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  constexpr int kFrames = 5;
  for (int i = 0; i < kFrames; ++i) {
    pub->publish(make_frame(i));
    const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(10);
    while (received <= i && std::chrono::steady_clock::now() < deadline)
      exec.spin_some(std::chrono::milliseconds(50));
    ASSERT_EQ(received, i + 1) << "no tracks message for frame " << i;
  }

  ASSERT_EQ(node->deactivate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  // Deactivated: frames are dropped, not queued.
  pub->publish(make_frame(kFrames));
  exec.spin_some(std::chrono::milliseconds(100));
  EXPECT_EQ(received, kFrames);

  ASSERT_EQ(node->cleanup().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_UNCONFIGURED);
}
