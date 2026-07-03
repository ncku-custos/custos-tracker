#pragma once

#include <opencv2/core.hpp>

#include <vector>

#include "ctrk/types.hpp"

namespace ctrk {

// HSV H-S histogram of the box region (clipped to the image), flattened and
// L2-normalized for cosine comparison — the TBD appearance embedding
// (RESULTS.md S3.5). Empty when the clipped region is degenerate.
std::vector<float> hsv_embedding(const cv::Mat& img, const BBox& box);

// Cosine similarity; 0 when either side is empty or sizes mismatch.
float cosine_similarity(const std::vector<float>& a, const std::vector<float>& b);

}  // namespace ctrk
