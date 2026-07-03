#include <gtest/gtest.h>

#include <vector>

#include "common/mat_view.hpp"
#include "ctrk/sot.hpp"
#include "support/synth.hpp"

namespace ctrk {
namespace {

// State-machine behaviour via the MOSSE backend (deterministic, no model
// files): full occlusion collapses PSR -> Unstable -> Lost. Recovery is NOT
// asserted here — the window may drift during occlusion, which is exactly
// why detector-assisted re-acquisition exists (its own test).
TEST(SotState, OcclusionDrivesUnstableThenLost) {
  synth::Options opt;
  opt.frames = 90;
  opt.occluder = true;
  opt.occluder_rect = {60, 110, 160, 160};
  opt.occluder_from = 30;
  opt.occluder_to = 60;
  synth::Sequence seq(opt, {{.box0 = {100, 150, 60, 60}, .vx = 0.f, .vy = 0.f,
                             .color = {0, 60, 220}}});

  SotConfig cfg;
  cfg.backend = SotBackend::Mosse;
  cfg.lost_patience = 5;
  SotTracker tracker(cfg);
  tracker.init(as_frame_view(seq.frame(0), 0), seq.gt(0)[0]);

  std::vector<SotState> states;
  for (int t = 1; t < seq.frames(); ++t) {
    const cv::Mat f = seq.frame(t);
    states.push_back(tracker.update(as_frame_view(f, t)).state);
  }

  // Healthy lock before the occluder.
  for (int t = 5; t < 25; ++t) EXPECT_EQ(states[t - 1], SotState::Tracking) << "frame " << t;
  // Lost while fully occluded (after the patience window).
  bool saw_unstable = false, saw_lost = false;
  for (int t = 31; t <= 60; ++t) {
    if (states[t - 1] == SotState::Unstable) saw_unstable = true;
    if (states[t - 1] == SotState::Lost) saw_lost = true;
  }
  EXPECT_TRUE(saw_unstable) << "should pass through Unstable during patience window";
  EXPECT_TRUE(saw_lost) << "sustained occlusion must end in Lost";
}

TEST(SotState, InitResetsTheStreak) {
  synth::Options opt;
  opt.frames = 30;
  opt.occluder = true;
  opt.occluder_rect = {60, 110, 160, 160};
  opt.occluder_from = 5;
  synth::Sequence seq(opt, {{.box0 = {100, 150, 60, 60}, .vx = 0.f, .vy = 0.f,
                             .color = {0, 60, 220}}});

  SotConfig cfg;
  cfg.backend = SotBackend::Mosse;
  cfg.lost_patience = 3;
  SotTracker tracker(cfg);
  tracker.init(as_frame_view(seq.frame(0), 0), seq.gt(0)[0]);
  SotState last = SotState::Tracking;
  for (int t = 1; t < 15; ++t) {
    const cv::Mat f = seq.frame(t);
    last = tracker.update(as_frame_view(f, t)).state;
  }
  ASSERT_EQ(last, SotState::Lost);

  // Re-init on a clean frame restores a fresh Tracking streak.
  const cv::Mat clean = seq.frame(0);
  tracker.init(as_frame_view(clean, 100), seq.gt(0)[0]);
  const cv::Mat f = seq.frame(0);
  EXPECT_EQ(tracker.update(as_frame_view(f, 101)).state, SotState::Tracking);
}

}  // namespace
}  // namespace ctrk
