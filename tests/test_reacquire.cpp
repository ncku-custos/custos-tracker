#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common/geometry.hpp"
#include "common/mat_view.hpp"
#include "ctrk/sot.hpp"
#include "support/synth.hpp"

namespace ctrk {
namespace {

// Detector stand-in: reports the moving synthetic target (from ground truth)
// whenever it is not occluded, plus a far-away distractor every frame. Lets
// the re-acquisition path be tested without a real NN or real imagery.
class MockDetector final : public IDetector {
 public:
  MockDetector(const synth::Sequence* seq, int occluded_from, int occluded_to)
      : seq_(seq), from_(occluded_from), to_(occluded_to) {}

  std::vector<Detection> detect(const FrameView& frame) override {
    const int t = static_cast<int>(frame.t_ns);  // tests pass the frame index as t_ns
    std::vector<Detection> dets;
    dets.push_back({{500, 30, 60, 60}, 0.95f, 7});  // distractor, wrong class
    if (t < from_ || t > to_) dets.push_back({seq_->gt(t)[0], 0.9f, 0});
    return dets;
  }

 private:
  const synth::Sequence* seq_;
  int from_, to_;
};

TEST(Reacquire, RelocksMovingTargetAfterOcclusion) {
  synth::Options opt;
  opt.frames = 120;
  opt.occluder = true;
  opt.occluder_rect = {120, 100, 200, 200};
  opt.occluder_from = 30;
  opt.occluder_to = 70;
  synth::Sequence seq(opt,
                      {{.box0 = {100, 150, 60, 60}, .vx = 1.5f, .vy = 0.f, .color = {0, 60, 220}}});

  SotConfig cfg;
  cfg.backend = SotBackend::Mosse;
  cfg.lost_patience = 5;
  SotTracker tracker(cfg);
  ReacquireConfig rc;
  rc.class_id = 0;  // must reject the class-7 distractor
  tracker.enable_reacquire(std::make_unique<MockDetector>(&seq, 30, 70), rc);

  const cv::Mat f0 = seq.frame(0);
  tracker.init(as_frame_view(f0, 0), seq.gt(0)[0]);

  bool lost_seen = false;
  int relock_frame = -1;
  SotResult last;
  for (int t = 1; t < seq.frames(); ++t) {
    const cv::Mat f = seq.frame(t);
    last = tracker.update(as_frame_view(f, t));
    if (last.state == SotState::Lost) lost_seen = true;
    if (lost_seen && relock_frame < 0 && t > 70 && last.state == SotState::Tracking)
      relock_frame = t;
  }

  EXPECT_TRUE(lost_seen) << "occlusion must drive the tracker to Lost";
  ASSERT_GT(relock_frame, 0) << "tracker never re-locked after the occluder lifted";
  EXPECT_LE(relock_frame, 85) << "re-lock took too long";
  EXPECT_GT(iou(last.box, seq.gt(seq.frames() - 1)[0]), 0.5f)
      << "re-locked track should follow the target to the end";
}

TEST(Reacquire, DistractorAloneNeverRelocks) {
  synth::Options opt;
  opt.frames = 60;
  opt.occluder = true;
  opt.occluder_rect = {60, 110, 160, 160};
  opt.occluder_from = 10;  // target never comes back
  synth::Sequence seq(opt,
                      {{.box0 = {100, 150, 60, 60}, .vx = 0.f, .vy = 0.f, .color = {0, 60, 220}}});

  SotConfig cfg;
  cfg.backend = SotBackend::Mosse;
  cfg.lost_patience = 5;
  SotTracker tracker(cfg);
  ReacquireConfig rc;
  rc.class_id = 0;
  tracker.enable_reacquire(std::make_unique<MockDetector>(&seq, 10, 10000), rc);

  const cv::Mat f0 = seq.frame(0);
  tracker.init(as_frame_view(f0, 0), seq.gt(0)[0]);
  SotState final_state = SotState::Tracking;
  for (int t = 1; t < seq.frames(); ++t) {
    const cv::Mat f = seq.frame(t);
    final_state = tracker.update(as_frame_view(f, t)).state;
  }
  EXPECT_EQ(final_state, SotState::Lost)
      << "wrong-class distractor must never satisfy re-acquisition";
}

// Two same-class lookalikes (the M4 Jogging failure): the tracked target is
// occluded for good while a same-class, same-size, differently-coloured
// distractor stays inside the growing search radius. Reports both as class 0.
class TwoTargetDetector final : public IDetector {
 public:
  TwoTargetDetector(const synth::Sequence* seq, int target_occluded_from)
      : seq_(seq), from_(target_occluded_from) {}

  std::vector<Detection> detect(const FrameView& frame) override {
    const int t = static_cast<int>(frame.t_ns);
    std::vector<Detection> dets;
    const auto boxes = seq_->gt(t);
    if (t < from_) dets.push_back({boxes[0], 0.9f, 0});  // the real target
    dets.push_back({boxes[1], 0.9f, 0});                 // the lookalike
    return dets;
  }

 private:
  const synth::Sequence* seq_;
  int from_;
};

// Geometry shared by the wrong-relock pair below: target (red) occluded for
// good at frame 10; distractor (blue) 200 px below — far outside MOSSE's
// local correlation window (so the tracker itself cannot drift onto it) but
// inside the re-acquisition radius once it has grown for ~28 lost frames.
// Same class, size and motion: only appearance separates them.
synth::Sequence wrong_relock_sequence() {
  synth::Options opt;
  opt.frames = 100;
  opt.occluder = true;
  opt.occluder_rect = {0, 130, 640, 90};
  opt.occluder_from = 10;  // the real target never comes back
  return synth::Sequence(opt, {{.box0 = {100, 150, 60, 60}, .vx = 1.5f, .color = {0, 60, 220}},
                               {.box0 = {100, 350, 60, 60}, .vx = 1.5f, .color = {220, 60, 0}}});
}

// Documents the M4 pathology: without appearance verification the geometric
// gates accept the lookalike and re-lock the WRONG target.
TEST(ReidVerify, GatesAloneRelockTheWrongLookalike) {
  synth::Sequence seq = wrong_relock_sequence();
  SotConfig cfg;
  cfg.backend = SotBackend::Mosse;
  cfg.lost_patience = 5;
  SotTracker tracker(cfg);
  ReacquireConfig rc;
  rc.class_id = 0;
  tracker.enable_reacquire(std::make_unique<TwoTargetDetector>(&seq, 10), rc);

  tracker.init(as_frame_view(seq.frame(0), 0), seq.gt(0)[0]);
  bool wrong_relock = false;
  for (int t = 1; t < seq.frames(); ++t) {
    const SotResult r = tracker.update(as_frame_view(seq.frame(t), t));
    if (t > 10 && r.state != SotState::Lost && iou(r.box, seq.gt(t)[1]) > 0.3f) wrong_relock = true;
  }
  EXPECT_TRUE(wrong_relock) << "expected the documented failure the re-ID veto removes";
}

TEST(ReidVerify, HsvVetoRejectsTheLookalikeAndStaysLost) {
  synth::Sequence seq = wrong_relock_sequence();
  SotConfig cfg;
  cfg.backend = SotBackend::Mosse;
  cfg.lost_patience = 5;
  cfg.reid.embedder = SotConfig::Reid::Embedder::HsvHist;
  cfg.reid.accept = 0.5f;
  SotTracker tracker(cfg);
  ReacquireConfig rc;
  rc.class_id = 0;
  tracker.enable_reacquire(std::make_unique<TwoTargetDetector>(&seq, 10), rc);

  tracker.init(as_frame_view(seq.frame(0), 0), seq.gt(0)[0]);
  bool wrong_relock = false;
  SotState final_state = SotState::Tracking;
  for (int t = 1; t < seq.frames(); ++t) {
    const SotResult r = tracker.update(as_frame_view(seq.frame(t), t));
    final_state = r.state;
    if (t > 10 && r.state != SotState::Lost && iou(r.box, seq.gt(t)[1]) > 0.3f) wrong_relock = true;
  }
  EXPECT_FALSE(wrong_relock) << "a colour-mismatched lookalike must be vetoed";
  EXPECT_EQ(final_state, SotState::Lost) << "with the target gone, Lost is the honest answer";
}

}  // namespace
}  // namespace ctrk
