#pragma once

#include <vector>

#include "ctrk/types.hpp"

namespace ctrk {

// Row-major 2x3 affine (a11 a12 tx / a21 a22 ty) mapping previous-frame
// pixel coordinates into current-frame ones — the camera-motion warp
// (RESULTS.md S3.4). Identity by default.
struct Affine23 {
  float a11 = 1.f, a12 = 0.f, tx = 0.f;
  float a21 = 0.f, a22 = 1.f, ty = 0.f;
  bool identity() const {
    return a11 == 1.f && a12 == 0.f && tx == 0.f && a21 == 0.f && a22 == 1.f && ty == 0.f;
  }
};

// Intersection-over-union of two boxes; 0 when either has non-positive area.
float iou(const BBox& a, const BBox& b);

// Greedy IoU-based non-maximum suppression. Returns indices into `boxes`
// ordered by descending score. boxes and scores must be the same length.
std::vector<int> nms(const std::vector<BBox>& boxes, const std::vector<float>& scores,
                     float iou_thr);

// Aspect-preserving resize-with-padding mapping (the YOLO "letterbox").
// Padding is split evenly between the two sides of the short axis.
struct LetterboxMap {
  float scale = 1.f;
  float pad_x = 0.f;
  float pad_y = 0.f;
  int src_w = 0, src_h = 0;
  int dst_w = 0, dst_h = 0;
};

LetterboxMap letterbox_map(int src_w, int src_h, int dst_w, int dst_h);
BBox to_letterbox(const BBox& b, const LetterboxMap& m);
BBox from_letterbox(const BBox& b, const LetterboxMap& m);

}  // namespace ctrk
