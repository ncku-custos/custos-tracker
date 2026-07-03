#pragma once

#include <cstdint>
#include <optional>

#include <ctrk/types.hpp>
#include <sensor_msgs/msg/image.hpp>

namespace ctrk_ros {

inline int64_t stamp_to_ns(const builtin_interfaces::msg::Time& t) {
  return static_cast<int64_t>(t.sec) * 1000000000LL + static_cast<int64_t>(t.nanosec);
}

// Zero-copy view over a bgr8 image message; the message must outlive every
// use of the view (tracker update calls are synchronous, so holding the
// ConstSharedPtr across the call is enough). Any other encoding: nullopt —
// the nodes warn-throttle and drop (conversion is a deliberate non-goal
// until the SoC decides on cv_bridge, docs/DECISIONS.md D-0016).
inline std::optional<ctrk::FrameView> to_frame_view(const sensor_msgs::msg::Image& img) {
  if (img.encoding != "bgr8") return std::nullopt;
  ctrk::FrameView v;
  v.data = img.data.data();
  v.width = static_cast<int>(img.width);
  v.height = static_cast<int>(img.height);
  v.stride_bytes = static_cast<int>(img.step);
  v.fmt = ctrk::PixelFormat::BGR8;
  v.t_ns = stamp_to_ns(img.header.stamp);
  return v;
}

}  // namespace ctrk_ros
