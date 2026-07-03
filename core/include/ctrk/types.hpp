#pragma once

#include <cstdint>
#include <vector>

namespace ctrk {

enum class PixelFormat : uint8_t { BGR8 };

// Non-owning view of one video frame. The producer owns the pixels; they must
// stay valid for the duration of the call the view is passed to. Core never
// reads a clock: t_ns is the caller-supplied capture timestamp (monotonic).
struct FrameView {
  const uint8_t* data = nullptr;
  int width = 0;
  int height = 0;
  int stride_bytes = 0;  // 0 => tightly packed (width * bytes-per-pixel)
  PixelFormat fmt = PixelFormat::BGR8;
  int64_t t_ns = 0;
};

// Axis-aligned box, top-left origin, pixel units.
struct BBox {
  float x = 0.f, y = 0.f, w = 0.f, h = 0.f;

  float cx() const { return x + 0.5f * w; }
  float cy() const { return y + 0.5f * h; }
  float area() const { return w * h; }
};

enum class SotState : uint8_t { Tracking, Unstable, Lost };

struct SotResult {
  BBox box;
  float score = 0.f;  // raw model confidence (pre-window peak for siamese, PSR for MOSSE)
  SotState state = SotState::Lost;
};

enum class TrackState : uint8_t { Tentative, Confirmed, Lost, Removed };

struct Track {
  int id = -1;
  BBox box;
  float score = 0.f;
  int class_id = -1;
  TrackState state = TrackState::Tentative;
  int age = 0;   // frames since creation
  int hits = 0;  // total matched detections
};

struct Detection {
  BBox box;
  float score = 0.f;
  int class_id = -1;
  // Optional appearance descriptor for association (RESULTS.md S3.5);
  // empty = geometry-only. Producer-defined contents (the built-in TBD
  // pipeline fills an L2-normalized HSV histogram when enabled).
  std::vector<float> embedding;
};

}  // namespace ctrk
