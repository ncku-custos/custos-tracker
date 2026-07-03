#include "sot/nanotrack.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>

#include "ctrk/profile.hpp"

namespace ctrk {

namespace {

inline float size_cal(float w, float h) {
  const float pad = (w + h) * 0.5f;
  return std::sqrt((w + pad) * (h + pad));
}

inline float recip_max(float v) {
  return std::max(v, 1.f / v);
}

}  // namespace

NanoTracker::NanoTracker(const SotConfig& config)
    : cfg_(config),
      backbone_z_(make_ort_engine(config.backbone_z_path, config.engine)),
      backbone_x_(make_ort_engine(config.backbone_x_path, config.engine)),
      head_(make_ort_engine(config.head_path, config.engine)) {
  cv::createHanningWindow(hann_, {kScore, kScore}, CV_32F);
  grid_x_.create(kScore, kScore, CV_32F);
  grid_y_.create(kScore, kScore, CV_32F);
  for (int r = 0; r < kScore; ++r)
    for (int c = 0; c < kScore; ++c) {
      grid_x_.at<float>(r, c) = static_cast<float>((c - kScore / 2) * kStride + kInstance / 2);
      grid_y_.at<float>(r, c) = static_cast<float>((r - kScore / 2) * kStride + kInstance / 2);
    }
  x_blob_.resize(static_cast<size_t>(3) * kInstance * kInstance);
}

// Exact replica of TrackerNanoImpl::getSubwindow: integer-truncated center,
// integer half-size, whole-image mean as padding value. The mean and the
// padded buffer (crop-sized, not frame-sized) are produced only when the
// crop actually leaves the image — pixel-identical to the reference's
// copyMakeBorder-then-ROI on every path.
void nano_subwindow(const cv::Mat& img, float pos_x, float pos_y, int original_sz, int model_sz,
                    cv::Mat& out, cv::Mat& scratch) {
  const int c = (original_sz + 1) / 2;
  const int xmin = static_cast<int>(pos_x) - c;
  const int ymin = static_cast<int>(pos_y) - c;
  const int xmax = xmin + original_sz - 1;
  const int ymax = ymin + original_sz - 1;

  const int left_pad = std::max(0, -xmin);
  const int top_pad = std::max(0, -ymin);
  const int right_pad = std::max(0, xmax - img.cols + 1);
  const int bottom_pad = std::max(0, ymax - img.rows + 1);

  cv::Mat crop;
  if (left_pad || top_pad || right_pad || bottom_pad) {
    const cv::Scalar avg = cv::mean(img);
    scratch.create(original_sz, original_sz, img.type());
    scratch.setTo(avg);
    const int vw = original_sz - left_pad - right_pad;
    const int vh = original_sz - top_pad - bottom_pad;
    if (vw > 0 && vh > 0) {
      img(cv::Rect(xmin + left_pad, ymin + top_pad, vw, vh))
          .copyTo(scratch(cv::Rect(left_pad, top_pad, vw, vh)));
    }
    crop = scratch;
  } else {
    crop = img(cv::Rect(xmin, ymin, original_sz, original_sz));
  }
  cv::resize(crop, out, {model_sz, model_sz});
}

// blobFromImage(crop, 1.0, Size(), Scalar(), swapRB=true): raw 0-255 floats,
// RGB channel order, CHW.
void NanoTracker::blob_rgb(const cv::Mat& crop, std::vector<float>& out) const {
  const int hw = crop.rows * crop.cols;
  out.resize(static_cast<size_t>(3) * hw);
  for (int y = 0; y < crop.rows; ++y) {
    const cv::Vec3b* row = crop.ptr<cv::Vec3b>(y);
    for (int x = 0; x < crop.cols; ++x) {
      const int i = y * crop.cols + x;
      out[0 * hw + i] = row[x][2];
      out[1 * hw + i] = row[x][1];
      out[2 * hw + i] = row[x][0];
    }
  }
}

void NanoTracker::init(const cv::Mat& image, const BBox& target) {
  pos_x_ = target.x + target.w * 0.5f;
  pos_y_ = target.y + target.h * 0.5f;
  sz_w_ = target.w;
  sz_h_ = target.h;
  img_size_ = image.size();

  const float sum = sz_w_ + sz_h_;
  const float w_ext = sz_w_ + cfg_.context_amount * sum;
  const float h_ext = sz_h_ + cfg_.context_amount * sum;
  const int s = static_cast<int>(std::sqrt(w_ext * h_ext));

  std::vector<float> blob;
  nano_subwindow(image, pos_x_, pos_y_, s, kExemplar, crop_, crop_scratch_);
  blob_rgb(crop_, blob);
  const auto& in = backbone_z_->input_descs()[0];
  const auto out = backbone_z_->run({{in.name, in.shape, blob.data()}});
  zf_.assign(out[0].data, out[0].data + backbone_z_->output_descs()[0].elements());
}

std::vector<float> NanoTracker::embed(const cv::Mat& image, const BBox& box) const {
  const float sum = box.w + box.h;
  const float w_ext = box.w + cfg_.context_amount * sum;
  const float h_ext = box.h + cfg_.context_amount * sum;
  const int s = std::max(1, static_cast<int>(std::sqrt(w_ext * h_ext)));

  cv::Mat crop, scratch;  // local buffers: embed runs at re-ID cadence, not per frame
  nano_subwindow(image, box.cx(), box.cy(), s, kExemplar, crop, scratch);
  std::vector<float> blob;
  blob_rgb(crop, blob);
  const auto& in = backbone_z_->input_descs()[0];
  const auto out = backbone_z_->run({{in.name, in.shape, blob.data()}});
  return {out[0].data, out[0].data + backbone_z_->output_descs()[0].elements()};
}

SotResult NanoTracker::update(const cv::Mat& image) {
  // Reference quirks preserved: int truncation of the size sum, and
  // sx = sz * (255/127) where 255/127 is INTEGER division = 2.
  const int sz_sum = static_cast<int>(sz_w_ + sz_h_);
  const float wc = sz_w_ + cfg_.context_amount * static_cast<float>(sz_sum);
  const float hc = sz_h_ + cfg_.context_amount * static_cast<float>(sz_sum);
  const float sz = std::sqrt(wc * hc);
  const float scale_z = static_cast<float>(kExemplar) / sz;
  const float sx = sz * static_cast<float>(kInstance / kExemplar);
  sz_w_ *= scale_z;
  sz_h_ *= scale_z;

  {
    ProfileScope prof("sot.crop");
    nano_subwindow(image, pos_x_, pos_y_, static_cast<int>(sx), kInstance, crop_, crop_scratch_);
  }
  {
    ProfileScope prof("sot.blob");
    blob_rgb(crop_, x_blob_);
  }
  std::vector<TensorView> xf;
  {
    ProfileScope prof("sot.backbone_x");
    const auto& xin = backbone_x_->input_descs()[0];
    xf = backbone_x_->run({{xin.name, xin.shape, x_blob_.data()}});
  }
  std::vector<TensorView> outs;
  {
    ProfileScope prof("sot.head");
    const auto& hin = head_->input_descs();
    outs = head_->run({{"input1", hin[0].shape, zf_.data()}, {"input2", hin[1].shape, xf[0].data}});
  }
  ProfileScope prof("sot.postproc");
  const float* cls = outs[0].data;  // [1,2,16,16]
  const float* reg = outs[1].data;  // [1,4,16,16]

  constexpr int kCells = kScore * kScore;
  float best_pscore = -1.f;
  int best = 0;
  float best_score = 0.f, best_penalty = 0.f;
  float best_x1 = 0, best_y1 = 0, best_x2 = 0, best_y2 = 0;

  const float* hann = hann_.ptr<float>(0);
  const float* gx = grid_x_.ptr<float>(0);
  const float* gy = grid_y_.ptr<float>(0);
  const float ratio = sz_w_ / sz_h_;
  const float target_ref = size_cal(pos_x_, pos_y_);  // reference quirk (not sz!)

  for (int i = 0; i < kCells; ++i) {
    const float c0 = cls[i], c1 = cls[kCells + i];
    const float m = std::max(c0, c1);
    const float e0 = std::exp(c0 - m), e1 = std::exp(c1 - m);
    const float score = e1 / (e0 + e1);

    const float x1 = gx[i] - reg[0 * kCells + i];
    const float y1 = gy[i] - reg[1 * kCells + i];
    const float x2 = gx[i] + reg[2 * kCells + i];
    const float y2 = gy[i] + reg[3 * kCells + i];

    const float sc = recip_max(size_cal(x2 - x1, y2 - y1) / target_ref);
    const float rc = recip_max(ratio / ((x2 - x1) / (y2 - y1)));
    const float penalty = std::exp(-(rc * sc - 1.f) * cfg_.penalty_k);
    const float pscore =
        penalty * score * (1.f - cfg_.window_influence) + hann[i] * cfg_.window_influence;

    if (pscore > best_pscore) {
      best_pscore = pscore;
      best = i;
      best_score = score;
      best_penalty = penalty;
      best_x1 = x1;
      best_y1 = y1;
      best_x2 = x2;
      best_y2 = y2;
    }
  }
  (void)best;

  const float pred_xs = (best_x1 + best_x2) * 0.5f;
  const float pred_ys = (best_y1 + best_y2) * 0.5f;
  const float pred_w = (best_x2 - best_x1) / scale_z;
  const float pred_h = (best_y2 - best_y1) / scale_z;
  const float diff_xs = (pred_xs - static_cast<float>(kInstance / 2)) / scale_z;
  const float diff_ys = (pred_ys - static_cast<float>(kInstance / 2)) / scale_z;

  sz_w_ /= scale_z;
  sz_h_ /= scale_z;

  const float lr = best_penalty * best_score * cfg_.size_lr;
  const float w_f = static_cast<float>(img_size_.width);
  const float h_f = static_cast<float>(img_size_.height);
  pos_x_ = std::clamp(pos_x_ + diff_xs, 0.f, w_f);
  pos_y_ = std::clamp(pos_y_ + diff_ys, 0.f, h_f);
  sz_w_ = std::clamp(pred_w * lr + (1.f - lr) * sz_w_, 10.f, w_f);
  sz_h_ = std::clamp(pred_h * lr + (1.f - lr) * sz_h_, 10.f, h_f);

  SotResult result;
  result.box = {pos_x_ - sz_w_ * 0.5f, pos_y_ - sz_h_ * 0.5f, sz_w_, sz_h_};
  result.score = best_score;  // raw classifier peak, the lost-detection signal
  result.state = SotState::Tracking;
  return result;
}

}  // namespace ctrk
