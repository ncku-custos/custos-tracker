#include "msg_convert.hpp"

#include <string>

namespace ctrk_ros {

vision_msgs::msg::Detection2DArray to_detections(const std::vector<ctrk::Track>& tracks,
                                                 const std_msgs::msg::Header& header,
                                                 bool include_lost) {
  vision_msgs::msg::Detection2DArray out;
  out.header = header;
  for (const ctrk::Track& t : tracks) {
    if (t.state != ctrk::TrackState::Confirmed &&
        !(include_lost && t.state == ctrk::TrackState::Lost))
      continue;
    vision_msgs::msg::Detection2D d;
    d.header = header;
    d.id = std::to_string(t.id);
    d.bbox.center.position.x = static_cast<double>(t.box.cx());
    d.bbox.center.position.y = static_cast<double>(t.box.cy());
    d.bbox.size_x = static_cast<double>(t.box.w);
    d.bbox.size_y = static_cast<double>(t.box.h);
    vision_msgs::msg::ObjectHypothesisWithPose hyp;
    hyp.hypothesis.class_id = std::to_string(t.class_id);
    hyp.hypothesis.score = static_cast<double>(t.score);
    d.results.push_back(hyp);
    out.detections.push_back(d);
  }
  return out;
}

ctrk_interfaces::msg::SotStatus to_sot_status(const ctrk::SotResult& result,
                                              const std_msgs::msg::Header& header) {
  using Msg = ctrk_interfaces::msg::SotStatus;
  Msg m;
  m.header = header;
  switch (result.state) {
    case ctrk::SotState::Tracking:
      m.state = Msg::STATE_TRACKING;
      break;
    case ctrk::SotState::Unstable:
      m.state = Msg::STATE_UNSTABLE;
      break;
    case ctrk::SotState::Lost:
      m.state = Msg::STATE_LOST;
      break;
  }
  m.x = result.box.x;
  m.y = result.box.y;
  m.w = result.box.w;
  m.h = result.box.h;
  m.score = result.score;
  return m;
}

ctrk_interfaces::msg::SotStatus idle_status(const std_msgs::msg::Header& header) {
  ctrk_interfaces::msg::SotStatus m;
  m.header = header;
  m.state = ctrk_interfaces::msg::SotStatus::STATE_IDLE;
  return m;
}

}  // namespace ctrk_ros
