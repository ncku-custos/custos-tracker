#pragma once

#include <opencv2/core.hpp>

#include "ctrk/types.hpp"

namespace ctrk {

// Zero-copy cv::Mat view of a FrameView. The Mat does not own the pixels.
inline cv::Mat as_mat(const FrameView& f) {
  CV_Assert(f.fmt == PixelFormat::BGR8 && f.data != nullptr && f.width > 0 && f.height > 0);
  const size_t step =
      f.stride_bytes > 0 ? static_cast<size_t>(f.stride_bytes) : static_cast<size_t>(f.width) * 3;
  return cv::Mat(f.height, f.width, CV_8UC3, const_cast<uint8_t*>(f.data), step);
}

// Zero-copy FrameView of a BGR cv::Mat. The Mat must outlive the view.
inline FrameView as_frame_view(const cv::Mat& m, int64_t t_ns) {
  CV_Assert(m.type() == CV_8UC3 && !m.empty());
  FrameView f;
  f.data = m.ptr<uint8_t>(0);
  f.width = m.cols;
  f.height = m.rows;
  f.stride_bytes = static_cast<int>(m.step);
  f.fmt = PixelFormat::BGR8;
  f.t_ns = t_ns;
  return f;
}

}  // namespace ctrk
