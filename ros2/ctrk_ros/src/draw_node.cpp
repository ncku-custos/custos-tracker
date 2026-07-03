// Debug visualization component: `image` + latest `tracks` (TBD) and/or
// `target` (SOT) -> `image_annotated`. The annotated message keeps the source
// image header, so `ros2 topic delay image_annotated` reads the whole
// pipeline's stamp->display latency for free.
#include <cstdio>
#include <memory>
#include <string>
#include <utility>

#include <opencv2/imgproc.hpp>
#include <rclcpp/rclcpp.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <sensor_msgs/msg/image.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

#include <ctrk_interfaces/msg/sot_status.hpp>

namespace ctrk_ros {

namespace {

// The CLI's id palette idea (track_tbd id_color): hue from the id, full
// saturation — stable per identity, distinct between neighbors.
cv::Scalar id_color(int id) {
  const cv::Mat hsv(1, 1, CV_8UC3, cv::Scalar((id * 47) % 180, 220, 255));
  cv::Mat bgr;
  cv::cvtColor(hsv, bgr, cv::COLOR_HSV2BGR);
  const auto v = bgr.at<cv::Vec3b>(0, 0);
  return {static_cast<double>(v[0]), static_cast<double>(v[1]), static_cast<double>(v[2])};
}

void draw_box(cv::Mat& img, float x, float y, float w, float h, const cv::Scalar& color,
              const char* label) {
  const cv::Rect r(static_cast<int>(x), static_cast<int>(y), static_cast<int>(w),
                   static_cast<int>(h));
  cv::rectangle(img, r, color, 2);
  cv::putText(img, label, {r.x, r.y - 6}, cv::FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv::LINE_AA);
}

}  // namespace

class DrawNode : public rclcpp::Node {
 public:
  explicit DrawNode(const rclcpp::NodeOptions& options) : Node("ctrk_draw_node", options) {
    annotated_pub_ =
        create_publisher<sensor_msgs::msg::Image>("image_annotated", rclcpp::SensorDataQoS());
    image_sub_ = create_subscription<sensor_msgs::msg::Image>(
        "image", rclcpp::QoS(1).best_effort(),
        [this](sensor_msgs::msg::Image::ConstSharedPtr msg) { on_image(std::move(msg)); });
    tracks_sub_ = create_subscription<vision_msgs::msg::Detection2DArray>(
        "tracks", rclcpp::QoS(10).reliable(),
        [this](vision_msgs::msg::Detection2DArray::ConstSharedPtr msg) {
          tracks_ = std::move(msg);
        });
    target_sub_ = create_subscription<ctrk_interfaces::msg::SotStatus>(
        "target", rclcpp::QoS(10).reliable(),
        [this](ctrk_interfaces::msg::SotStatus::ConstSharedPtr msg) { target_ = std::move(msg); });
  }

 private:
  void on_image(sensor_msgs::msg::Image::ConstSharedPtr msg) {
    if (msg->encoding != "bgr8") return;
    auto out = *msg;  // one copy to draw into; header (and stamp) pass through
    cv::Mat canvas(static_cast<int>(out.height), static_cast<int>(out.width), CV_8UC3,
                   out.data.data(), out.step);

    if (tracks_) {
      for (const auto& d : tracks_->detections) {
        const int id = std::atoi(d.id.c_str());
        char label[64];
        const double score = d.results.empty() ? 0.0 : d.results[0].hypothesis.score;
        std::snprintf(label, sizeof(label), "#%s %.2f", d.id.c_str(), score);
        draw_box(canvas, static_cast<float>(d.bbox.center.position.x - 0.5 * d.bbox.size_x),
                 static_cast<float>(d.bbox.center.position.y - 0.5 * d.bbox.size_y),
                 static_cast<float>(d.bbox.size_x), static_cast<float>(d.bbox.size_y), id_color(id),
                 label);
      }
    }
    if (target_ && target_->state != ctrk_interfaces::msg::SotStatus::STATE_IDLE) {
      char label[64];
      std::snprintf(label, sizeof(label), "target %.2f", static_cast<double>(target_->score));
      draw_box(canvas, target_->x, target_->y, target_->w, target_->h, {0, 200, 255}, label);
    }
    annotated_pub_->publish(std::move(out));
  }

  rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr annotated_pub_;
  rclcpp::Subscription<sensor_msgs::msg::Image>::SharedPtr image_sub_;
  rclcpp::Subscription<vision_msgs::msg::Detection2DArray>::SharedPtr tracks_sub_;
  rclcpp::Subscription<ctrk_interfaces::msg::SotStatus>::SharedPtr target_sub_;
  vision_msgs::msg::Detection2DArray::ConstSharedPtr tracks_;
  ctrk_interfaces::msg::SotStatus::ConstSharedPtr target_;
};

}  // namespace ctrk_ros

RCLCPP_COMPONENTS_REGISTER_NODE(ctrk_ros::DrawNode)
