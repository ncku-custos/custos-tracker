// Composition: the full player -> SOT(mosse) -> draw graph in ONE process
// with intra-process comms ON. The player and draw components are loaded
// through the real plugin path (class_loader on libctrk_ros_components.so,
// exactly what a component_container does); the tracker is constructed
// directly so the test can drive its lifecycle. Model-free, no datasets —
// the frames are synthesized and written to a temp dir for the player.
#include <gtest/gtest.h>

#include <chrono>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include <ament_index_cpp/get_package_prefix.hpp>
#include <class_loader/class_loader.hpp>
#include <ctrk_interfaces/msg/sot_status.hpp>
#include <lifecycle_msgs/msg/state.hpp>
#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/node_factory.hpp>

#include "sot_node.hpp"
#include "support/synth_frames.hpp"

namespace {

using ctrk_interfaces::msg::SotStatus;

void ensure_rcl() {
  if (!rclcpp::ok()) rclcpp::init(0, nullptr);
}

std::string write_scene_frames(const ctrk_ros::test::SynthScene& scene, int n_frames) {
  const auto dir = std::filesystem::temp_directory_path() / "ctrk_ros_composition";
  std::filesystem::remove_all(dir);
  std::filesystem::create_directories(dir);
  for (int i = 0; i < n_frames; ++i) {
    const auto img = scene.frame(i);
    const cv::Mat m(static_cast<int>(img.height), static_cast<int>(img.width), CV_8UC3,
                    const_cast<uint8_t*>(img.data.data()), img.step);
    char name[32];
    std::snprintf(name, sizeof(name), "%04d.png", i);
    EXPECT_TRUE(cv::imwrite((dir / name).string(), m));
  }
  return (dir / "%04d.png").string();
}

}  // namespace

TEST(Composition, PlayerTrackerDrawIntraProcessLockstep) {
  ensure_rcl();
  constexpr int kFrames = 10;
  const ctrk_ros::test::SynthScene scene;
  const auto pattern = write_scene_frames(scene, kFrames);
  const auto dump_path =
      (std::filesystem::temp_directory_path() / "ctrk_ros_composition" / "sot.txt").string();

  // Tracker: direct construction (lifecycle control), IPC on.
  rclcpp::NodeOptions sot_opts;
  sot_opts.use_intra_process_comms(true);
  sot_opts.parameter_overrides(
      {{"backend", "mosse"},
       {"image_qos_reliability", "reliable"},
       {"image_qos_depth", 10},
       {"dump_path", dump_path},
       {"init_bbox",
        std::vector<double>{static_cast<double>(scene.x0), static_cast<double>(scene.y0),
                            static_cast<double>(scene.target_size),
                            static_cast<double>(scene.target_size)}}});
  auto sot = std::make_shared<ctrk_ros::SotNode>(sot_opts);

  // Player + draw: loaded through the component plugin path.
  const auto lib_path =
      ament_index_cpp::get_package_prefix("ctrk_ros") + "/lib/libctrk_ros_components.so";
  class_loader::ClassLoader loader(lib_path);

  rclcpp::NodeOptions player_opts;
  player_opts.use_intra_process_comms(true);
  player_opts.parameter_overrides({{"path", pattern},
                                   {"mode", "lockstep"},
                                   {"lockstep_topic", "target"},
                                   {"lockstep_type", "ctrk_interfaces/msg/SotStatus"},
                                   {"stamp_source", "index"},
                                   {"start_delay_s", 0.0}});
  auto player_factory = loader.createUniqueInstance<rclcpp_components::NodeFactory>(
      "rclcpp_components::NodeFactoryTemplate<ctrk_ros::FramesPlayerNode>");
  auto player = player_factory->create_node_instance(player_opts);

  rclcpp::NodeOptions draw_opts;
  draw_opts.use_intra_process_comms(true);
  auto draw_factory = loader.createUniqueInstance<rclcpp_components::NodeFactory>(
      "rclcpp_components::NodeFactoryTemplate<ctrk_ros::DrawNode>");
  auto draw = draw_factory->create_node_instance(draw_opts);

  // Observer for the end of the graph.
  auto helper = std::make_shared<rclcpp::Node>("composition_observer");
  int statuses = 0, annotated = 0;
  auto status_sub = helper->create_subscription<SotStatus>(
      "target", rclcpp::QoS(50).reliable(), [&statuses](SotStatus::ConstSharedPtr) { ++statuses; });
  auto annotated_sub = helper->create_subscription<sensor_msgs::msg::Image>(
      "image_annotated", rclcpp::QoS(rclcpp::KeepLast(50)).best_effort(),
      [&annotated](sensor_msgs::msg::Image::ConstSharedPtr) { ++annotated; });

  rclcpp::executors::SingleThreadedExecutor exec;
  exec.add_node(sot->get_node_base_interface());
  exec.add_node(player.get_node_base_interface());
  exec.add_node(draw.get_node_base_interface());
  exec.add_node(helper);

  ASSERT_EQ(sot->configure().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);
  ASSERT_EQ(sot->activate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_ACTIVE);

  // Lockstep drains the whole sequence: one status per frame, no drops.
  const auto deadline = std::chrono::steady_clock::now() + std::chrono::seconds(60);
  while (statuses < kFrames && std::chrono::steady_clock::now() < deadline)
    exec.spin_some(std::chrono::milliseconds(20));
  EXPECT_EQ(statuses, kFrames);
  EXPECT_GE(annotated, 1) << "draw node published nothing";

  ASSERT_EQ(sot->deactivate().id(), lifecycle_msgs::msg::State::PRIMARY_STATE_INACTIVE);

  // The dump (flushed by deactivate) has one line per frame, init box first.
  std::ifstream dump(dump_path);
  ASSERT_TRUE(dump.is_open());
  std::vector<std::string> lines;
  for (std::string line; std::getline(dump, line);) lines.push_back(line);
  EXPECT_EQ(lines.size(), static_cast<size_t>(kFrames));
  ASSERT_FALSE(lines.empty());
  EXPECT_EQ(lines[0], "40.00,100.00,48.00,48.00");

  sot->cleanup();
}
