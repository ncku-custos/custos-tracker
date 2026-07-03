#include <gtest/gtest.h>

#include <cstdio>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "common/geometry.hpp"
#include "ctrk/types.hpp"
#include "tbd/yolov8_decode.hpp"

namespace ctrk {
namespace {

// Golden-tensor differential test: decode a recorded raw YOLOv8 output and
// match ultralytics' own predictions on the same image (fixture produced by
// tools/export/make_yolo_fixture.py). Skips when the fixture is absent.
struct Fixture {
  std::vector<float> raw;
  std::vector<Detection> expected;
  int src_w = 0, src_h = 0;
  float conf = 0.f, nms_iou = 0.f;
};

bool load_fixture(Fixture& f) {
  const std::string dir = std::string(CTRK_SOURCE_DIR) + "/tests/fixtures/yolo";
  const std::string bin = dir + "/raw_output.bin";
  const std::string csv = dir + "/expected.csv";
  if (!std::filesystem::exists(bin) || !std::filesystem::exists(csv)) return false;

  const auto bytes = std::filesystem::file_size(bin);
  f.raw.resize(bytes / sizeof(float));
  std::ifstream(bin, std::ios::binary)
      .read(reinterpret_cast<char*>(f.raw.data()), static_cast<std::streamsize>(bytes));

  std::ifstream in(csv);
  std::string line;
  std::getline(in, line);
  if (std::sscanf(line.c_str(), "# %d %d %f %f", &f.src_w, &f.src_h, &f.conf, &f.nms_iou) != 4)
    return false;
  while (std::getline(in, line)) {
    Detection d;
    if (std::sscanf(line.c_str(), "%d,%f,%f,%f,%f,%f", &d.class_id, &d.score, &d.box.x, &d.box.y,
                    &d.box.w, &d.box.h) == 6)
      f.expected.push_back(d);
  }
  return !f.expected.empty();
}

TEST(YoloDecode, MatchesUltralyticsReference) {
  Fixture f;
  if (!load_fixture(f)) GTEST_SKIP() << "run tools/export/make_yolo_fixture.py";
  ASSERT_EQ(f.raw.size(), 84u * 8400u);

  const LetterboxMap map = letterbox_map(f.src_w, f.src_h, 640, 640);
  const auto dets = decode_yolov8(f.raw.data(), 80, 8400, f.conf, f.nms_iou, map, {});

  // Every reference box must be reproduced: same class, IoU > 0.85, similar
  // confidence. Allow up to 2 extra detections (NMS tie-break differences).
  for (const auto& exp : f.expected) {
    bool matched = false;
    for (const auto& got : dets) {
      if (got.class_id == exp.class_id && iou(got.box, exp.box) > 0.85f &&
          std::abs(got.score - exp.score) < 0.05f) {
        matched = true;
        break;
      }
    }
    EXPECT_TRUE(matched) << "missing: class " << exp.class_id << " conf " << exp.score << " at "
                         << exp.box.x << "," << exp.box.y;
  }
  EXPECT_LE(dets.size(), f.expected.size() + 2);
  EXPECT_GE(dets.size(), f.expected.size());
}

TEST(YoloDecode, ConfThresholdFiltersEverything) {
  Fixture f;
  if (!load_fixture(f)) GTEST_SKIP() << "run tools/export/make_yolo_fixture.py";
  const LetterboxMap map = letterbox_map(f.src_w, f.src_h, 640, 640);
  EXPECT_TRUE(decode_yolov8(f.raw.data(), 80, 8400, 0.999f, f.nms_iou, map, {}).empty());
}

TEST(YoloDecode, ClassFilterKeepsOnlyRequested) {
  Fixture f;
  if (!load_fixture(f)) GTEST_SKIP() << "run tools/export/make_yolo_fixture.py";
  const LetterboxMap map = letterbox_map(f.src_w, f.src_h, 640, 640);
  const auto dets = decode_yolov8(f.raw.data(), 80, 8400, f.conf, f.nms_iou, map, {0});
  ASSERT_FALSE(dets.empty());  // bus.jpg contains persons
  for (const auto& d : dets) EXPECT_EQ(d.class_id, 0);
}

}  // namespace
}  // namespace ctrk
