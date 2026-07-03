#pragma once

#include <memory>
#include <string>
#include <vector>

#include "ctrk/types.hpp"

namespace ctrk {

// Object detector interface. Implementations own their preprocessing; boxes
// come back in source-frame pixel coordinates.
class IDetector {
 public:
  virtual ~IDetector() = default;
  virtual std::vector<Detection> detect(const FrameView& frame) = 0;
};

struct Yolov8Config {
  std::string model_path;        // models/cache/yolov8n_640.onnx
  float conf_thr = 0.1f;         // deliberately low: ByteTrack consumes low-score dets
  float nms_iou = 0.45f;
  std::vector<int> keep_classes; // COCO ids; empty keeps all (0 = person)
  int intra_op_threads = 4;
};

// Throws std::runtime_error on model-load failure.
std::unique_ptr<IDetector> make_yolov8_detector(const Yolov8Config& config);

}  // namespace ctrk
