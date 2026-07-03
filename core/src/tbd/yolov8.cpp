#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

#include "common/geometry.hpp"
#include "common/mat_view.hpp"
#include "ctrk/detector.hpp"
#include "ctrk/infer.hpp"
#include "ctrk/profile.hpp"
#include "tbd/yolov8_decode.hpp"

namespace ctrk {

std::vector<Detection> decode_yolov8(const float* out, int num_classes, int num_anchors,
                                     float conf_thr, float nms_iou, const LetterboxMap& map,
                                     const std::vector<int>& keep_classes, DecodeScratch* scratch) {
  const auto value = [&](int c, int a) { return out[c * num_anchors + a]; };
  const auto wanted = [&](int cls) {
    return keep_classes.empty() ||
           std::find(keep_classes.begin(), keep_classes.end(), cls) != keep_classes.end();
  };

  // Per-anchor class max, scanned per CLASS so every pass is a contiguous
  // row of the [4+C, A] tensor instead of an A-strided walk. Ascending class
  // with >= keeps the original tie-break (highest class id wins) and produces
  // bit-identical results to the former per-anchor inner loop.
  DecodeScratch local;
  DecodeScratch& s = scratch ? *scratch : local;
  s.best.assign(static_cast<size_t>(num_anchors), conf_thr);
  s.best_cls.assign(static_cast<size_t>(num_anchors), -1);
  for (int c = 0; c < num_classes; ++c) {
    const float* row = out + static_cast<size_t>(4 + c) * num_anchors;
    float* best = s.best.data();
    int16_t* best_cls = s.best_cls.data();
    for (int a = 0; a < num_anchors; ++a) {
      if (row[a] >= best[a]) {
        best[a] = row[a];
        best_cls[a] = static_cast<int16_t>(c);
      }
    }
  }

  std::vector<Detection> candidates;
  for (int a = 0; a < num_anchors; ++a) {
    const int best_cls = s.best_cls[a];
    const float best = s.best[a];
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
      : config_(config), engine_(make_ort_engine(config.model_path, config.engine)) {
    const auto& in = engine_->input_descs();
    const auto& out = engine_->output_descs();
    if (in.size() != 1 || out.size() != 1 || in[0].shape.size() != 4 || out[0].shape.size() != 3 ||
        in[0].shape[2] != in[0].shape[3])
      throw std::runtime_error("unexpected yolov8 graph layout: " + config.model_path);
    input_size_ = static_cast<int>(in[0].shape[2]);
    num_classes_ = static_cast<int>(out[0].shape[1]) - 4;
    num_anchors_ = static_cast<int>(out[0].shape[2]);
    blob_.resize(static_cast<size_t>(in[0].elements()));
    canvas_.create(input_size_, input_size_, CV_8UC3);
    for (auto& ch : chan_) ch.create(input_size_, input_size_, CV_8U);
  }

  std::vector<Detection> detect(const FrameView& frame) override {
    const cv::Mat img = as_mat(frame);
    const LetterboxMap map = letterbox_map(img.cols, img.rows, input_size_, input_size_);

    {
      ProfileScope prof("det.preprocess");
      // Letterbox onto a cached 114-grey canvas, then RGB float CHW in [0,1].
      // The grey bands only need refreshing when the scaled size changes;
      // the resize overwrites the interior rect every frame.
      const int nw = static_cast<int>(std::round(img.cols * map.scale));
      const int nh = static_cast<int>(std::round(img.rows * map.scale));
      const int px = (input_size_ - nw) / 2, py = (input_size_ - nh) / 2;
      if (nw != fitted_w_ || nh != fitted_h_) {
        canvas_.setTo(cv::Scalar(114, 114, 114));
        fitted_w_ = nw;
        fitted_h_ = nh;
      }
      cv::resize(img, canvas_(cv::Rect(px, py, nw, nh)), cv::Size(nw, nh));

      // Deinterleave with cv::split (runtime-SIMD dispatched), then scale each
      // contiguous plane; same values as the former per-pixel loop.
      cv::split(canvas_, chan_);
      const int plane = input_size_ * input_size_;
      for (int c = 0; c < 3; ++c) {
        const uchar* src = chan_[2 - c].ptr<uchar>(0);  // BGR planes -> RGB blob
        float* dst = blob_.data() + static_cast<size_t>(c) * plane;
        for (int i = 0; i < plane; ++i) dst[i] = src[i] / 255.f;
      }
    }

    std::vector<TensorView> out;
    {
      ProfileScope prof("det.infer");
      const auto& in = engine_->input_descs()[0];
      out = engine_->run({{in.name, in.shape, blob_.data()}});
    }

    ProfileScope prof("det.decode");
    return decode_yolov8(out[0].data, num_classes_, num_anchors_, config_.conf_thr, config_.nms_iou,
                         map, config_.keep_classes, &scratch_);
  }

 private:
  Yolov8Config config_;
  std::unique_ptr<IEngine> engine_;
  int input_size_ = 640, num_classes_ = 80, num_anchors_ = 8400;
  std::vector<float> blob_;
  cv::Mat canvas_;
  cv::Mat chan_[3];
  int fitted_w_ = -1, fitted_h_ = -1;  // letterboxed interior of canvas_
  DecodeScratch scratch_;
};

}  // namespace

std::unique_ptr<IDetector> make_yolov8_detector(const Yolov8Config& config) {
  return std::make_unique<Yolov8Detector>(config);
}

}  // namespace ctrk
