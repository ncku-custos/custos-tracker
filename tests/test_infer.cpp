#include <gtest/gtest.h>

#include <cmath>
#include <filesystem>
#include <numeric>
#include <vector>

#include "ctrk/infer.hpp"

namespace ctrk {
namespace {

// Uses the fetched NanoTrack v2 backbone (models/get_models.sh). Skipped when
// absent so a network-less checkout still passes; run get_models.sh first for
// full coverage.
std::string v2_backbone_path() {
  return std::string(CTRK_SOURCE_DIR) + "/models/cache/nanotrackv2_nanotrack_backbone_sim.onnx";
}

TEST(OrtEngine, LoadsAndDescribesStaticGraph) {
  if (!std::filesystem::exists(v2_backbone_path())) GTEST_SKIP() << "run models/get_models.sh";
  const auto engine = make_ort_engine(v2_backbone_path());
  ASSERT_EQ(engine->input_descs().size(), 1u);
  const auto& in = engine->input_descs()[0];
  // Published v2 backbone is exported at the 255 search size.
  EXPECT_EQ(in.shape, (std::vector<int64_t>{1, 3, 255, 255}));
  ASSERT_EQ(engine->output_descs().size(), 1u);
}

TEST(OrtEngine, RunsAndIsDeterministic) {
  if (!std::filesystem::exists(v2_backbone_path())) GTEST_SKIP() << "run models/get_models.sh";
  const auto engine = make_ort_engine(v2_backbone_path());
  const auto& in = engine->input_descs()[0];

  std::vector<float> input(static_cast<size_t>(in.elements()));
  for (size_t i = 0; i < input.size(); ++i) input[i] = static_cast<float>(i % 255);

  const auto out1 = engine->run({{in.name, in.shape, input.data()}});
  ASSERT_EQ(out1.size(), 1u);
  const auto n =
      std::accumulate(out1[0].shape.begin(), out1[0].shape.end(), int64_t{1}, std::multiplies<>());
  ASSERT_GT(n, 0);
  const std::vector<float> first(out1[0].data, out1[0].data + n);
  for (float v : first) ASSERT_TRUE(std::isfinite(v));

  const auto out2 = engine->run({{in.name, in.shape, input.data()}});
  for (int64_t i = 0; i < n; ++i) ASSERT_EQ(first[static_cast<size_t>(i)], out2[0].data[i]);
}

TEST(OrtEngine, RejectsMissingAndMisshapenInputs) {
  if (!std::filesystem::exists(v2_backbone_path())) GTEST_SKIP() << "run models/get_models.sh";
  const auto engine = make_ort_engine(v2_backbone_path());
  EXPECT_THROW(engine->run({}), std::runtime_error);
  std::vector<float> tiny(3, 0.f);
  EXPECT_THROW(engine->run({{engine->input_descs()[0].name, {1, 3, 1, 1}, tiny.data()}}),
               std::runtime_error);
}

TEST(OrtEngine, ThrowsOnBadPath) {
  EXPECT_THROW(make_ort_engine("/nonexistent/model.onnx"), std::exception);
}

}  // namespace
}  // namespace ctrk
