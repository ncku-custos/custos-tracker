#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

#include "common/geometry.hpp"
#include "common/mat_view.hpp"
#include "ctrk/detector.hpp"
#include "ctrk/infer.hpp"
#include "tbd/yolov8_decode.hpp"

namespace ctrk {

std::vector<Detection> decode_yolov8(const float* out, int num_classes, int num_anchors,
                                     float conf_thr, float nms_iou, const LetterboxMap& map,
                                     const std::vector<int>& keep_classes) {
  const auto value = [&](int c, int a) { return out[c * num_anchors + a]; };
  const auto wanted = [&](int cls) {
    return keep_classes.empty() ||
           std::find(keep_classes.begin(), keep_classes.end(), cls) != keep_classes.end();
  };

  std::vector<Detection> candidates;
  for (int a = 0; a < num_anchors; ++a) {
    int best_cls = -1;
    float best = conf_thr;
    for (int c = 0; c < num_classes; ++c) {
      const float s = value(4 + c, a);
      if (s >= best) {
        best = s;
        best_cls = c;
      }
    }
    if (best_cls < 0 || !wanted(best_cls)) continue;

    const float cx = value(0, a), cy = value(1, a);
    const float w = value(2, a), h = value(3, a);
    BBox box = from_letterbox({cx - 0.5f * w, cy - 0.5f * h, w, h}, map);
    // Clip to the source frame.
    const float x2 = std::min(box.x + box.w, static_cast<float>(map.src_w));
    const float y2 = std::min(box.y + box.h, static_cast<float>(map.src_h));
    box.x = std::max(box.x, 0.f);
    box.y = std::max(box.y, 0.f);
    box.w = x2 - box.x;
    box.h = y2 - box.y;
    if (box.w <= 1.f || box.h <= 1.f) continue;
    candidates.push_back({box, best, best_cls});
  }

  // Class-aware NMS via the offset trick: shift boxes per class so different
  // classes never suppress each other, then run one global pass.
  std::vector<BBox> shifted;
  std::vector<float> scores;
  shifted.reserve(candidates.size());
  for (const auto& d : candidates) {
    const float off = static_cast<float>(d.class_id) * 4096.f;
    shifted.push_back({d.box.x + off, d.box.y + off, d.box.w, d.box.h});
    scores.push_back(d.score);
  }
  std::vector<Detection> result;
  for (int idx : nms(shifted, scores, nms_iou)) result.push_back(candidates[idx]);
  return result;
}

namespace {

class Yolov8Detector final : public IDetector {
 public:
  explicit Yolov8Detector(const Yolov8Config& config)
      : config_(config),
        engine_(make_ort_engine(config.model_path, {.intra_op_threads = config.intra_op_threads})) {
    const auto& in = engine_->input_descs();
    const auto& out = engine_->output_descs();
    if (in.size() != 1 || out.size() != 1 || in[0].shape.size() != 4 ||
        out[0].shape.size() != 3 || in[0].shape[2] != in[0].shape[3])
      throw std::runtime_error("unexpected yolov8 graph layout: " + config.model_path);
    input_size_ = static_cast<int>(in[0].shape[2]);
    num_classes_ = static_cast<int>(out[0].shape[1]) - 4;
    num_anchors_ = static_cast<int>(out[0].shape[2]);
    blob_.resize(static_cast<size_t>(in[0].elements()));
  }

  std::vector<Detection> detect(const FrameView& frame) override {
    const cv::Mat img = as_mat(frame);
    const LetterboxMap map = letterbox_map(img.cols, img.rows, input_size_, input_size_);

    // Letterbox onto a 114-grey canvas, then RGB float CHW in [0,1].
    cv::Mat canvas(input_size_, input_size_, CV_8UC3, cv::Scalar(114, 114, 114));
    const int nw = static_cast<int>(std::round(img.cols * map.scale));
    const int nh = static_cast<int>(std::round(img.rows * map.scale));
    const int px = (input_size_ - nw) / 2, py = (input_size_ - nh) / 2;
    cv::resize(img, canvas(cv::Rect(px, py, nw, nh)), cv::Size(nw, nh));

    const int plane = input_size_ * input_size_;
    for (int y = 0; y < input_size_; ++y) {
      const cv::Vec3b* row = canvas.ptr<cv::Vec3b>(y);
      for (int x = 0; x < input_size_; ++x) {
        const int i = y * input_size_ + x;
        blob_[0 * plane + i] = row[x][2] / 255.f;  // R (canvas is BGR)
        blob_[1 * plane + i] = row[x][1] / 255.f;  // G
        blob_[2 * plane + i] = row[x][0] / 255.f;  // B
      }
    }

    const auto& in = engine_->input_descs()[0];
    const auto out = engine_->run({{in.name, in.shape, blob_.data()}});
    return decode_yolov8(out[0].data, num_classes_, num_anchors_, config_.conf_thr,
                         config_.nms_iou, map, config_.keep_classes);
  }

 private:
  Yolov8Config config_;
  std::unique_ptr<IEngine> engine_;
  int input_size_ = 640, num_classes_ = 80, num_anchors_ = 8400;
  std::vector<float> blob_;
};

}  // namespace

std::unique_ptr<IDetector> make_yolov8_detector(const Yolov8Config& config) {
  return std::make_unique<Yolov8Detector>(config);
}

}  // namespace ctrk
