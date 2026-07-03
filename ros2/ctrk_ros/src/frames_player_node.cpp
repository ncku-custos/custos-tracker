// Dataset playback component: publishes an image-sequence pattern
// (e.g. data/mot/MOT16-04/img1/%06d.jpg) as bgr8 `image` messages.
//
// Two pacing modes:
//   rate     — wall-clock timer at rate_hz (the live-camera stand-in).
//   lockstep — publish frame N+1 only after the tracker's output for frame N
//              arrives on lockstep_topic; drop-proof by construction, which
//              is what makes the S4.1 parity gate a real digit-identical
//              comparison instead of pacing luck.
// Two stamp sources:
//   index — t = int64(idx/fps*1e9), bit-exactly the CLI VideoSource math
//           (apps/app_common.cpp), so the tracker's KF dt sequence matches
//           the CLI run digit for digit.
//   clock — node clock now() (for latency measurements).
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>

#include <opencv2/imgcodecs.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace ctrk_ros {

class FramesPlayerNode : public rclcpp::Node {
 public:
  explicit FramesPlayerNode(const rclcpp::NodeOptions& options)
      : Node("ctrk_frames_player", options) {
    path_ = declare_parameter<std::string>("path", "");
    if (path_.empty()) throw std::invalid_argument("frames_player: 'path' parameter is required");
    fps_ = declare_parameter<double>("fps", 30.0);
    mode_ = declare_parameter<std::string>("mode", "rate");
    if (mode_ != "rate" && mode_ != "lockstep")
      throw std::invalid_argument("frames_player: mode must be 'rate' or 'lockstep'");
    rate_hz_ = declare_parameter<double>("rate_hz", 30.0);
    lockstep_topic_ = declare_parameter<std::string>("lockstep_topic", "tracks");
    lockstep_type_ =
        declare_parameter<std::string>("lockstep_type", "vision_msgs/msg/Detection2DArray");
    lockstep_timeout_s_ = declare_parameter<double>("lockstep_timeout_s", 5.0);
    stamp_source_ = declare_parameter<std::string>("stamp_source", "index");
    if (stamp_source_ != "index" && stamp_source_ != "clock")
      throw std::invalid_argument("frames_player: stamp_source must be 'index' or 'clock'");
    loop_ = declare_parameter<bool>("loop", false);
    frame_limit_ = static_cast<int>(declare_parameter<int64_t>("frame_limit", 0));
    file_index_ = static_cast<int>(declare_parameter<int64_t>("start_index", -1));
    exit_on_done_ = declare_parameter<bool>("exit_on_done", false);
    const double start_delay = declare_parameter<double>("start_delay_s", 0.5);

    image_pub_ = create_publisher<sensor_msgs::msg::Image>("image", rclcpp::QoS(10).reliable());
    if (file_index_ < 0) file_index_ = probe_start_index();

    if (mode_ == "lockstep") {
      ack_sub_ = create_generic_subscription(
          lockstep_topic_, lockstep_type_, rclcpp::QoS(10).reliable(),
          [this](std::shared_ptr<rclcpp::SerializedMessage>) { on_ack(); });
    }
    start_deadline_ = now() + rclcpp::Duration::from_seconds(start_delay);
    wait_timer_ = create_wall_timer(std::chrono::milliseconds(100), [this] { on_wait_tick(); });
  }

 private:
  int probe_start_index() const {
    for (int idx : {0, 1}) {
      char buf[1024];
      std::snprintf(buf, sizeof(buf), path_.c_str(), idx);
      if (!cv::imread(buf, cv::IMREAD_COLOR).empty()) return idx;
    }
    throw std::invalid_argument("frames_player: no frame at index 0 or 1 for pattern " + path_);
  }

  // Wait until the tracker's subscription is matched (it appears at
  // on_activate) and the settle delay passed, then kick off frame one.
  void on_wait_tick() {
    if (image_pub_->get_subscription_count() == 0 || now() < start_deadline_) return;
    wait_timer_->cancel();
    publish_next();
    if (mode_ == "rate") {
      rate_timer_ = create_wall_timer(std::chrono::duration<double>(1.0 / rate_hz_),
                                      [this] { publish_next(); });
    } else {
      arm_timeout();
    }
  }

  void on_ack() {
    if (done_) return;
    publish_next();
    if (!done_) arm_timeout();
  }

  void arm_timeout() {
    timeout_timer_ = create_wall_timer(std::chrono::duration<double>(lockstep_timeout_s_), [this] {
      RCLCPP_WARN(get_logger(), "lockstep ack timeout after %.1fs — advancing anyway",
                  lockstep_timeout_s_);
      on_ack();
    });
  }

  void publish_next() {
    if (timeout_timer_) timeout_timer_->cancel();
    if (done_) return;
    if (frame_limit_ > 0 && published_ >= frame_limit_) return finish();

    char buf[1024];
    std::snprintf(buf, sizeof(buf), path_.c_str(), file_index_);
    cv::Mat frame = cv::imread(buf, cv::IMREAD_COLOR);
    if (frame.empty()) {
      if (loop_ && published_ > 0) {
        file_index_ = probe_start_index();
        std::snprintf(buf, sizeof(buf), path_.c_str(), file_index_);
        frame = cv::imread(buf, cv::IMREAD_COLOR);
      }
      if (frame.empty()) return finish();
    }

    sensor_msgs::msg::Image msg;
    msg.encoding = "bgr8";
    msg.width = static_cast<uint32_t>(frame.cols);
    msg.height = static_cast<uint32_t>(frame.rows);
    msg.step = static_cast<uint32_t>(frame.cols * 3);
    msg.data.resize(static_cast<size_t>(msg.step) * msg.height);
    for (int y = 0; y < frame.rows; ++y)  // imread rows may be padded; copy row-wise
      std::memcpy(&msg.data[static_cast<size_t>(y) * msg.step], frame.ptr(y), msg.step);
    if (stamp_source_ == "index") {
      // Bit-exact VideoSource math: double division, then int64 truncation.
      const auto t_ns = static_cast<int64_t>(static_cast<double>(published_) / fps_ * 1e9);
      msg.header.stamp.sec = static_cast<int32_t>(t_ns / 1000000000LL);
      msg.header.stamp.nanosec = static_cast<uint32_t>(t_ns % 1000000000LL);
    } else {
      msg.header.stamp = now();
    }
    image_pub_->publish(std::move(msg));
    ++published_;
    ++file_index_;
  }

  void finish() {
    if (done_) return;
    done_ = true;
    if (rate_timer_) rate_timer_->cancel();
    if (timeout_timer_) timeout_timer_->cancel();
    RCLCPP_INFO(get_logger(), "done: %d frames published", published_);
    if (exit_on_done_) rclcpp::shutdown();
  }

  std::string path_, mode_, lockstep_topic_, lockstep_type_, stamp_source_;
  double fps_ = 30.0, rate_hz_ = 30.0, lockstep_timeout_s_ = 5.0;
  bool loop_ = false, exit_on_done_ = false, done_ = false;
  int frame_limit_ = 0, file_index_ = -1, published_ = 0;
  rclcpp::Time start_deadline_;
  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr image_pub_;
  rclcpp::GenericSubscription::SharedPtr ack_sub_;
  rclcpp::TimerBase::SharedPtr wait_timer_, rate_timer_, timeout_timer_;
};

}  // namespace ctrk_ros

RCLCPP_COMPONENTS_REGISTER_NODE(ctrk_ros::FramesPlayerNode)
