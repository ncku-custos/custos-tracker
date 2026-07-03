#include "common/geometry.hpp"

#include <algorithm>
#include <numeric>

namespace ctrk {

float iou(const BBox& a, const BBox& b) {
  if (a.w <= 0.f || a.h <= 0.f || b.w <= 0.f || b.h <= 0.f) return 0.f;
  const float x1 = std::max(a.x, b.x);
  const float y1 = std::max(a.y, b.y);
  const float x2 = std::min(a.x + a.w, b.x + b.w);
  const float y2 = std::min(a.y + a.h, b.y + b.h);
  const float iw = x2 - x1;
  const float ih = y2 - y1;
  if (iw <= 0.f || ih <= 0.f) return 0.f;
  const float inter = iw * ih;
  return inter / (a.area() + b.area() - inter);
}

std::vector<int> nms(const std::vector<BBox>& boxes, const std::vector<float>& scores,
                     float iou_thr) {
  std::vector<int> order(boxes.size());
  std::iota(order.begin(), order.end(), 0);
  std::sort(order.begin(), order.end(), [&](int a, int b) { return scores[a] > scores[b]; });

  std::vector<int> keep;
  std::vector<bool> suppressed(boxes.size(), false);
  for (int i : order) {
    if (suppressed[i]) continue;
    keep.push_back(i);
    for (int j : order) {
      if (j == i || suppressed[j]) continue;
      if (iou(boxes[i], boxes[j]) > iou_thr) suppressed[j] = true;
    }
  }
  return keep;
}

LetterboxMap letterbox_map(int src_w, int src_h, int dst_w, int dst_h) {
  LetterboxMap m;
  m.src_w = src_w;
  m.src_h = src_h;
  m.dst_w = dst_w;
  m.dst_h = dst_h;
  m.scale = std::min(static_cast<float>(dst_w) / static_cast<float>(src_w),
                     static_cast<float>(dst_h) / static_cast<float>(src_h));
  m.pad_x = 0.5f * (static_cast<float>(dst_w) - m.scale * static_cast<float>(src_w));
  m.pad_y = 0.5f * (static_cast<float>(dst_h) - m.scale * static_cast<float>(src_h));
  return m;
}

BBox to_letterbox(const BBox& b, const LetterboxMap& m) {
  return {b.x * m.scale + m.pad_x, b.y * m.scale + m.pad_y, b.w * m.scale, b.h * m.scale};
}

BBox from_letterbox(const BBox& b, const LetterboxMap& m) {
  return {(b.x - m.pad_x) / m.scale, (b.y - m.pad_y) / m.scale, b.w / m.scale, b.h / m.scale};
}

}  // namespace ctrk
