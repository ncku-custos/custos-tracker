#pragma once

#include <vector>

#include <ctrk/types.hpp>
#include <ctrk_interfaces/msg/sot_status.hpp>
#include <std_msgs/msg/header.hpp>
#include <vision_msgs/msg/detection2_d_array.hpp>

namespace ctrk_ros {

// Confirmed tracks (plus Lost when include_lost) as a Detection2DArray:
// bbox in the vision_msgs center convention, detection.id = track id,
// hypothesis carries class_id/score. Tentative/Removed never publish.
vision_msgs::msg::Detection2DArray to_detections(const std::vector<ctrk::Track>& tracks,
                                                 const std_msgs::msg::Header& header,
                                                 bool include_lost);

// SotResult in ctrk's top-left pixel convention (SotStatus.msg contract).
ctrk_interfaces::msg::SotStatus to_sot_status(const ctrk::SotResult& result,
                                              const std_msgs::msg::Header& header);

// Active but no target set: STATE_IDLE, zero box/score.
ctrk_interfaces::msg::SotStatus idle_status(const std_msgs::msg::Header& header);

}  // namespace ctrk_ros
