#pragma once

#include <vector>

#include "common/geometry.hpp"
#include "ctrk/types.hpp"

namespace ctrk {

// Decode a raw YOLOv8 head output (layout [1, 4+num_classes, num_anchors],
// channel-major: value(c, a) = out[c * num_anchors + a]; rows 0-3 are
// cx,cy,w,h in letterbox pixels, rows 4+ are post-sigmoid class scores) into
// NMS-filtered detections in source-frame coordinates. Exposed separately
// from the engine so the golden-tensor test covers it without inference.
std::vector<Detection> decode_yolov8(const float* out, int num_classes, int num_anchors,
                                     float conf_thr, float nms_iou, const LetterboxMap& map,
                                     const std::vector<int>& keep_classes);

}  // namespace ctrk
